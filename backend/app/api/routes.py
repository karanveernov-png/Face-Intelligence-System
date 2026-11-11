from __future__ import annotations

import time
from pathlib import Path as FilePath

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, Query
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.onnx_runtime import get_active_execution_providers
from app.schemas.models import (
    DeleteResponse,
    EnrolledEntry,
    EnrollResponse,
    ModelStatus,
    WithdrawConsentResponse,
)
from app.security.auth import (
    AuthContext,
    Permission,
    get_auth_context,
    get_optional_auth_context,
    is_public_endpoint,
    require_permission,
)
from app.security.validation import (
    reject_oversized_decoded_image,
    reject_oversized_dimensions,
    validated_image_upload,
    validated_video_upload,
)
from app.services import age_estimation, detection, expressions, pipeline, recognition
from app.services.tracking import FaceTracker

router = APIRouter()


# ============================================================
# Public Endpoints (No Auth Required)
# ============================================================

@router.get("/health")
def health():
    """Health check - no authentication required."""
    return {"status": "ok", "version": settings.app_version}


@router.get("/api/models/status")
def models_status():
    """
    Model status - no authentication required (public info).
    Reports what's actually loaded for the 4-Model Accuracy-First INT8 Roster.
    """
    statuses: list[ModelStatus] = []
    dummy = np.zeros((112, 112, 3), dtype=np.uint8)
    onnx_provider = get_active_execution_providers()[0].replace("ExecutionProvider", "").lower()
    ort_backend_label = f"onnxruntime-{onnx_provider}-int8"

    try:
        detection.detect_faces(dummy)
        active_backend = detection._scrfd_model_name if detection._scrfd_ready else "unknown"
        statuses.append(
            ModelStatus(
                name="SCRFD-10G-INT8",
                purpose="face detection & 5-pt alignment (95.16% WiderFace Hard)",
                loaded=True,
                backend=active_backend,
            )
        )
    except Exception as exc:  # noqa: BLE001
        statuses.append(
            ModelStatus(
                name="SCRFD-10G-INT8",
                purpose="face detection & 5-pt alignment",
                loaded=False,
                backend="none",
                error=str(exc),
            )
        )

    for name, purpose, fn, backend in [
        (
            "ArcFace-R50-INT8",
            "512-D recognition embeddings (99.83% LFW)",
            lambda: recognition._embed(dummy),
            ort_backend_label,
        ),
        (
            "HSEmotion-ENetB0-INT8",
            "8-class expression classification (86.9% RAF-DB)",
            lambda: expressions.classify_expression(dummy),
            ort_backend_label,
        ),
        (
            "GenderAge+FairFace-Ensemble",
            "age + gender, two-model ensemble with honest uncertainty",
            age_estimation.check_model,
            ort_backend_label,
        ),
    ]:
        try:
            fn()
            statuses.append(ModelStatus(name=name, purpose=purpose, loaded=True, backend=backend))
        except Exception as exc:  # noqa: BLE001
            statuses.append(
                ModelStatus(
                    name=name,
                    purpose=purpose,
                    loaded=False,
                    backend=backend,
                    error=str(exc),
                )
            )

    return statuses


# ============================================================
# Protected Endpoints (Authentication Required)
# ============================================================

@router.post("/api/analyze/image")
async def analyze_image(
    path: FilePath = Depends(validated_image_upload),
    run_recognition: bool = Form(True),
    run_expression: bool = Form(True),
    run_age: bool = Form(False),
):
    """
    Analyze an uploaded image for faces.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="File could not be decoded as an image (corrupt or unsupported format).",
        )
    reject_oversized_decoded_image(img)

    options = pipeline.AnalysisOptions(
        run_recognition=run_recognition,
        run_expression=run_expression,
        run_age=run_age,
    )
    result = pipeline.analyze_image(img, options)
    return result.to_dict()


@router.post("/api/analyze/video")
async def analyze_video(
    path: FilePath = Depends(validated_video_upload),
    run_recognition: bool = Form(True),
    run_expression: bool = Form(True),
    run_age: bool = Form(False),
):
    """
    Analyze an uploaded video for faces.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail="File could not be opened as a video (corrupt or unsupported codec/container).",
        )

    # Check container dimensions before decoding frames
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    try:
        reject_oversized_dimensions(w=frame_w, h=frame_h)
    except HTTPException:
        cap.release()
        raise

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    tracker = FaceTracker()

    frame_idx, processed = 0, 0
    per_frame_results = []
    t0 = time.time()
    try:
        while processed < settings.max_video_frames_processed:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % max(settings.frame_skip, 1) == 0:
                options = pipeline.AnalysisOptions(
                    run_recognition=run_recognition,
                    run_expression=run_expression,
                    run_age=run_age,
                    tracker=tracker,
                    frame_index=processed,
                    decouple_detection=True,
                )
                result = pipeline.analyze_image(frame, options)
                per_frame_results.append(
                    {
                        "frame_index": frame_idx,
                        "face_count": len(result.faces),
                        "faces": result.faces,
                        "active_tracks": len(tracker.tracks),
                        "detector_backend": result.detector_meta.get("backend_used"),
                    }
                )
                processed += 1
            frame_idx += 1
    finally:
        cap.release()

    processing_seconds = time.time() - t0
    return {
        "source_fps": round(source_fps, 2),
        "frames_in_file_seen": frame_idx,
        "frames_processed": processed,
        "frame_skip": settings.frame_skip,
        "detection_interval": settings.detection_interval,
        "processing_fps": round(processed / processing_seconds, 2) if processing_seconds > 0 else None,
        "results": per_frame_results,
    }


@router.post("/api/faces/enroll", response_model=EnrollResponse)
async def enroll(
    path: FilePath = Depends(validated_image_upload),
    name: str = Form(...),
    consent_confirmed: bool = Form(...),
):
    """
    Enroll a face into the recognition gallery.
    Consent must be explicitly confirmed.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=400, detail="File could not be decoded as an image.")
    reject_oversized_decoded_image(img)

    faces, _ = detection.detect_faces(img)
    if not faces:
        raise HTTPException(status_code=422, detail="No face detected in the enrollment image.")
    if len(faces) > 1:
        raise HTTPException(
            status_code=422,
            detail="Multiple faces detected; enrollment image must contain exactly one face.",
        )

    # Consent check after face detection - fail with "no face" if no face
    if settings.consent_required and not consent_confirmed:
        raise HTTPException(
            status_code=403,
            detail="Enrollment requires explicit consent confirmation (consent_confirmed=True)."
        )

    aligned = recognition.align_face(img, faces[0])
    try:
        result = recognition.enroll_face(aligned, name=name, consent_confirmed=consent_confirmed)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return EnrollResponse(**result)


@router.get("/api/faces", response_model=list[EnrolledEntry])
def list_faces():
    """
    List all enrolled faces.
    """
    return recognition.list_enrolled()


@router.delete("/api/faces/{face_id}", response_model=DeleteResponse)
def delete_face(
    face_id: str,
):
    """
    Delete an enrolled face from the gallery.
    """
    deleted = recognition.delete_enrolled(face_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="face_id not found")
    return DeleteResponse(face_id=face_id, deleted=True)


@router.post("/api/faces/{face_id}/withdraw-consent", response_model=WithdrawConsentResponse)
def withdraw_consent(
    face_id: str,
):
    """
    Withdraw consent for a previously enrolled face.
    This permanently deletes the embedding and marks the record as withdrawn.
    """
    withdrawn = recognition.withdraw_consent(face_id)
    if not withdrawn:
        raise HTTPException(status_code=404, detail="face_id not found")
    return WithdrawConsentResponse(
        face_id=face_id,
        withdrawn=True,
        message="Consent withdrawn and biometric data permanently deleted"
    )


# ============================================================
# WebSocket Endpoint (Auth handled in ws.py)
# ============================================================
# The WebSocket endpoint is defined in app/api/ws.py and handles
# authentication during the handshake phase.