"""
Orchestrates the Accuracy-First INT8 Production Stack analysis pipeline:

  1. Decoupled Detection from Recognition:
     In video and live WebSocket streams, full SCRFD-10G / SCRFD-2.5G detection
     runs on every Nth frame (`settings.detection_interval`), while intermediate
     frames propagate bounding boxes using Lucas-Kanade optical flow + centroid
     tracking (`FaceTracker.predict_inter_frame`).
  2. 5-Point Umeyama Similarity Alignment (`recognition.align_face`):
     Warps all 5 SCRFD landmarks onto the canonical 112x112 ArcFace coordinate
     frame before embedding extraction (+4-7% recognition accuracy on tilted faces).
  3. Batched Cropped Faces (`cv2.dnn.blobFromImages`):
     Stacks all face crops into a single NCHW tensor before forwarding through
     ArcFace-R50 INT8 (recognition), HSEmotion ENet-B0 INT8
     (expression), and the GenderAge + FairFace ensemble (age).
  4. Temporal EMA Track Smoothing (`FaceTracker.smooth_face_attributes`):
     Smooths per-track emotion probabilities and age estimates across frames to
     eliminate single-frame jitter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.services import age_estimation, detection, expressions, recognition


def _iou(a: detection.DetectedFace, b: detection.DetectedFace) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _iou_box(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> float:
    """IoU between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _deduplicate_faces(
    faces: list[detection.DetectedFace],
    iou_threshold: float = 0.45,
    tracker: Any | None = None,
) -> list[detection.DetectedFace]:
    """Collapse overlapping boxes down to one per real face.

    Two legitimate distinct faces essentially never overlap this heavily --
    this threshold being crossed means tracker drift (or, rarely, detector
    noise) produced more than one box for the same physical face. Keeps the
    higher-confidence box of each overlapping pair. This is a safety net on
    top of fixing the tracker's own ghost-track lifetime (see tracking.py);
    it catches duplicates regardless of what produced them.

    If a `tracker` is provided, faces matching tracks with
    `unconfirmed_predict_frames > 2` have their confidence zeroed so they're
    dropped by the existing deduplication logic.
    """
    if len(faces) <= 1:
        return faces

    # Zero out confidence for faces that match "unconfirmed" (ghost) tracks.
    if tracker is not None:
        for f in faces:
            f_box = (f.x, f.y, f.w, f.h)
            for t in tracker.tracks.values():
                if t.unconfirmed_predict_frames > 2:
                    if _iou_box(f_box, t.box) > iou_threshold:
                        f.confidence = 0.0
                        break

    order = sorted(range(len(faces)), key=lambda i: faces[i].confidence, reverse=True)
    kept: list[int] = []
    for i in order:
        if any(_iou(faces[i], faces[j]) >= iou_threshold for j in kept):
            continue
        kept.append(i)
    kept.sort()
    return [faces[i] for i in kept]


@dataclass
class AnalysisOptions:
    run_recognition: bool = True
    run_expression: bool = True
    run_age: bool = False
    tracker: Any | None = field(default=None, repr=False, compare=False)
    frame_index: int | None = field(default=None, repr=False, compare=False)
    decouple_detection: bool = field(default=False, repr=False, compare=False)


@dataclass
class PipelineResult:
    faces: list[dict[str, Any]]
    detector_meta: dict[str, Any]
    total_latency_ms: float
    task_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_count": len(self.faces),
            "faces": self.faces,
            "detector": self.detector_meta,
            "total_latency_ms": self.total_latency_ms,
            "task_errors": self.task_errors,
        }


def analyze_image(
    image_bgr: np.ndarray,
    options: AnalysisOptions = AnalysisOptions(),
) -> PipelineResult:
    t0 = time.time()

    # Step 1: Decoupled Detection vs Full SCRFD Detection
    interval = max(1, settings.detection_interval)
    use_inter_frame_tracker = (
        options.decouple_detection
        and options.tracker is not None
        and options.frame_index is not None
        and (options.frame_index % interval != 0)
        and bool(getattr(options.tracker, "tracks", None))
    )

    if use_inter_frame_tracker:
        t_track = time.time()
        detected = options.tracker.predict_inter_frame(image_bgr)
        detector_meta = {
            "fallback_used": False,
            "errors": [],
            "backend_used": "optical-centroid-tracker",
            "latency_ms": round((time.time() - t_track) * 1000, 1),
        }
    else:
        detected, detector_meta = detection.detect_faces(image_bgr)
        if options.tracker is not None:
            boxes = [(f.x, f.y, f.w, f.h) for f in detected]
            options.tracker.update(boxes, detected_faces=detected, frame_bgr=image_bgr)

    detected = _deduplicate_faces(detected, tracker=options.tracker)

    faces_out: list[dict[str, Any]] = [
        {"face_index": i, **face.to_dict()} for i, face in enumerate(detected)
    ]
    task_errors: list[str] = []

    if not detected:
        total_ms = round((time.time() - t0) * 1000, 1)
        return PipelineResult(
            faces=faces_out,
            detector_meta=detector_meta,
            total_latency_ms=total_ms,
            task_errors=task_errors,
        )

    # Step 2: Batch Cropped Faces across Recognition, Expression, and Age models
    raw_crops = [
        face.crop_bgr
        if (face.crop_bgr is not None and face.crop_bgr.size > 0)
        else image_bgr[max(face.y, 0) : face.y + face.h, max(face.x, 0) : face.x + face.w].copy()
        for face in detected
    ]

    if options.run_recognition:
        try:
            aligned_crops = [recognition.align_face(image_bgr, face) for face in detected]
            if settings.batch_inference_enabled and hasattr(recognition, "match_faces_batch"):
                match_results = recognition.match_faces_batch(aligned_crops)
                for i, res in enumerate(match_results):
                    faces_out[i]["identity"] = res.to_dict()
            else:
                for i, aligned in enumerate(aligned_crops):
                    faces_out[i]["identity"] = recognition.match_face(aligned).to_dict()
        except Exception as exc:  # noqa: BLE001
            for i, face in enumerate(detected):
                try:
                    aligned = recognition.align_face(image_bgr, face)
                    faces_out[i]["identity"] = recognition.match_face(aligned).to_dict()
                except Exception as inner_exc:  # noqa: BLE001
                    task_errors.append(f"face[{i}].recognition: {inner_exc}")
                    faces_out[i]["identity"] = None
            if not any(f.get("identity") is None for f in faces_out):
                task_errors.append(f"recognition.batch: {exc}")

    if options.run_expression:
        try:
            if settings.batch_inference_enabled and hasattr(expressions, "classify_expressions_batch"):
                expr_results = expressions.classify_expressions_batch(raw_crops)
                for i, res in enumerate(expr_results):
                    faces_out[i]["expression"] = res.to_dict()
            else:
                for i, crop in enumerate(raw_crops):
                    faces_out[i]["expression"] = expressions.classify_expression(crop).to_dict()
        except Exception:  # noqa: BLE001
            for i, crop in enumerate(raw_crops):
                try:
                    faces_out[i]["expression"] = expressions.classify_expression(crop).to_dict()
                except Exception as inner_exc:  # noqa: BLE001
                    task_errors.append(f"face[{i}].expression: {inner_exc}")
                    faces_out[i]["expression"] = None

    if options.run_age:
        try:
            if hasattr(age_estimation, "estimate_ages_batch_in_frame"):
                boxes = [(f.x, f.y, f.w, f.h) for f in detected]
                age_results = age_estimation.estimate_ages_batch_in_frame(image_bgr, boxes)
                for i, res in enumerate(age_results):
                    faces_out[i]["age"] = res.to_dict()
            elif settings.batch_inference_enabled and hasattr(age_estimation, "estimate_ages_batch"):
                age_results = age_estimation.estimate_ages_batch(raw_crops)
                for i, res in enumerate(age_results):
                    faces_out[i]["age"] = res.to_dict()
            else:
                for i, crop in enumerate(raw_crops):
                    faces_out[i]["age"] = age_estimation.estimate_age(crop).to_dict()
        except Exception:  # noqa: BLE001
            for i, crop in enumerate(raw_crops):
                try:
                    faces_out[i]["age"] = age_estimation.estimate_age(crop).to_dict()
                except Exception as inner_exc:  # noqa: BLE001
                    task_errors.append(f"face[{i}].age: {inner_exc}")
                    faces_out[i]["age"] = None

    # Step 3: Temporal EMA smoothing across video/webcam tracks
    if options.tracker is not None and hasattr(options.tracker, "smooth_face_attributes"):
        for face_entry in faces_out:
            options.tracker.smooth_face_attributes(face_entry)

    total_ms = round((time.time() - t0) * 1000, 1)
    return PipelineResult(
        faces=faces_out,
        detector_meta=detector_meta,
        total_latency_ms=total_ms,
        task_errors=task_errors,
    )


def annotate_image(image_bgr: np.ndarray, result: PipelineResult) -> np.ndarray:
    """Draw bounding boxes + labels for a human-facing preview. Never fabricates a
    face that wasn't actually detected."""
    out = image_bgr.copy()
    for f in result.faces:
        b = f["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)

        label_parts = []
        if f.get("identity"):
            label_parts.append(f["identity"]["identity"])
        if f.get("expression") and f["expression"].get("label") != "uncertain":
            label_parts.append(f["expression"]["label"])
        if f.get("age") and f["age"].get("estimated_age") is not None:
            age_label = f["age"].get("estimated_age_range") or f"~{f['age']['estimated_age']}"
            label_parts.append(f"{age_label}y")
        label = " | ".join(label_parts) if label_parts else "face"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 8), (x + tw + 6, y), (0, 200, 0), -1)
        cv2.putText(
            out,
            label,
            (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        if f.get("landmarks"):
            for pt in f["landmarks"].values():
                if pt:
                    cv2.circle(out, tuple(pt), 2, (0, 0, 255), -1)
    return out
