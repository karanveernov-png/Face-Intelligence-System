from __future__ import annotations

import time
from pathlib import Path as FilePath

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException

from app.core.config import settings
from app.core.onnx_runtime import get_active_execution_providers
from app.schemas.models import (
    DeleteResponse,
    EnrolledEntry,
    EnrollResponse,
    HealthResponse,
    ModelStatus,
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


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="0.3.0")


@router.get("/api/models/status", response_model=list[ModelStatus])
def models_status():
    """
    Reports what's actually loaded right now for the 4-Model Accuracy-First INT8 Roster:
      1. SCRFD-10G-INT8/Haar (Detection & 5-Point Alignment, 95.16% WiderFace Hard)
      2. ArcFace-R50-INT8 (Face Recognition, 99.83% LFW / 96.4% IJB-C)
      3. HSEmotion-ENetB0-INT8 (Emotion Analysis, 86.9% RAF-DB 8-class)
      4. GenderAge+FairFace-Ensemble (Age + Gender; two independent open models,
         averaged, with disagreement reported as uncertainty)

    Labels here must track what's actually wired up in app/core/config.py --
    this endpoint feeds the model-status badges in the UI header directly, so a
    stale label here is a user-visible accuracy claim nobody is running.
    """
    statuses: list[ModelStatus] = []
    dummy = np.zeros((112, 112, 3), dtype=np.uint8)
    onnx_provider = get_active_execution_providers()[0].replace("ExecutionProvider", "").lower()
    ort_backend_label = f"onnxruntime-{onnx_provider}-int8"

    try:
        detection.detect_faces(dummy)
        active_backend = detection._scrfd_model_name if detection._scrfd_ready else "opencv-haar"
        statuses.append(
            ModelStatus(
                name="SCRFD-10G-INT8/Haar",
                purpose="face detection & 5-pt alignment (95.16% WiderFace Hard)",
                loaded=True,
                backend=active_backend,
            )
        )
    except Exception as exc:  # noqa: BLE001
        statuses.append(
            ModelStatus(
                name="SCRFD-10G-INT8/Haar",
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


@router.post("/api/analyze/image")
async def analyze_image(
    path: FilePath = Depends(validated_image_upload),
    run_recognition: bool = Form(True),
    run_expression: bool = Form(True),
    run_age: bool = Form(False),
):
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
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail="File could not be opened as a video (corrupt or unsupported codec/container).",
        )

    # Checked against the container's own reported dimensions, before decoding
    # any frame -- same decompression-bomb concern as images (see
    # reject_oversized_dimensions), but for video a huge per-frame allocation
    # would otherwise repeat up to max_video_frames_processed times. Reading
    # CAP_PROP_FRAME_WIDTH/HEIGHT allocates nothing itself.
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
    return recognition.list_enrolled()


@router.delete("/api/faces/{face_id}", response_model=DeleteResponse)
def delete_face(face_id: str):
    deleted = recognition.delete_enrolled(face_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="face_id not found")
    return DeleteResponse(face_id=face_id, deleted=True)