"""
Tests for the Accuracy-First INT8 Production Stack and Pipeline Performance Strategy:
  1. 5-Point Umeyama Similarity Alignment (`recognition.align_face` with all 5 landmarks)
  2. Temporal EMA Track Smoothing (`FaceTracker.smooth_face_attributes`)
  3. Decouple Detection from Recognition (SCRFD on every Nth frame, optical/centroid tracking in between)
  4. Batch Cropped Faces (`cv2.dnn.blobFromImages` across recognition, emotion, and age)
  5. ONNX Runtime with CPU Vectorization (`OpenVINOExecutionProvider` / `CPUExecutionProvider`)
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.config import settings
from app.core.onnx_runtime import (
    blob_from_arcface_crops,
    blob_from_mobilenet_crops,
    create_session_options,
    get_active_execution_providers,
)
from app.services import age_estimation, detection, expressions, pipeline, recognition
from app.services.detection import DetectedFace
from app.services.tracking import FaceTracker


@pytest.fixture(autouse=True)
def isolated_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(recognition, "_INDEX_PATH", tmp_path / "index.json")
    yield


def test_blob_from_images_stacks_multiple_crops():
    crops = [
        np.full((80, 70, 3), 60, dtype=np.uint8),
        np.full((120, 100, 3), 180, dtype=np.uint8),
        np.full((95, 95, 3), 120, dtype=np.uint8),
    ]
    arcface_blob = blob_from_arcface_crops(crops, size=(112, 112))
    assert arcface_blob.shape == (3, 3, 112, 112)
    assert arcface_blob.dtype == np.float32

    mobilenet_blob = blob_from_mobilenet_crops(crops, size=(224, 224))
    assert mobilenet_blob.shape == (3, 3, 224, 224)
    assert mobilenet_blob.dtype == np.float32


def test_5pt_umeyama_similarity_alignment_produces_valid_crop(blank_image):
    face = DetectedFace(
        x=100,
        y=100,
        w=112,
        h=112,
        confidence=0.98,
        backend="scrfd-10g-int8",
        left_eye=(138, 151),
        right_eye=(173, 151),
        nose=(156, 171),
        mouth_left=(141, 192),
        mouth_right=(170, 192),
        crop_bgr=blank_image[100:212, 100:212].copy(),
    )
    aligned = recognition.align_face(blank_image, face)
    assert aligned.shape == (112, 112, 3)
    assert aligned.size > 0


def test_temporal_ema_smoothing_stabilizes_age_and_expression():
    tracker = FaceTracker()
    tracker.update([(50, 50, 100, 100)])

    frame1_entry = {
        "bbox": {"x": 50, "y": 50, "w": 100, "h": 100},
        "expression": {
            "label": "happy",
            "certain": True,
            "scores": {"happy": 0.8, "neutral": 0.2},
            "note": "",
        },
        "age": {"enabled": True, "estimated_age": 30, "estimated_age_range": "25-32", "note": ""},
    }
    tracker.smooth_face_attributes(frame1_entry)
    assert frame1_entry["age"]["estimated_age"] == 30

    # Simulate a noisy 1-frame spike to age 50; EMA (alpha=0.4) should damp it to 38
    frame2_entry = {
        "bbox": {"x": 52, "y": 51, "w": 100, "h": 100},
        "expression": {
            "label": "neutral",
            "certain": True,
            "scores": {"happy": 0.2, "neutral": 0.8},
            "note": "",
        },
        "age": {"enabled": True, "estimated_age": 50, "estimated_age_range": "48-53", "note": ""},
    }
    tracker.smooth_face_attributes(frame2_entry)
    assert frame2_entry["age"]["estimated_age"] == 38
    assert frame2_entry["age"]["estimated_age_range"] == "38-43"


def test_batched_inference_across_recognition_emotion_and_age():
    crop_a = np.full((112, 112, 3), 40, dtype=np.uint8)
    crop_b = np.full((112, 112, 3), 200, dtype=np.uint8)

    recognition.enroll_face(crop_a, name="Person A", consent_confirmed=True)

    rec_results = recognition.match_faces_batch([crop_a, crop_b])
    assert len(rec_results) == 2
    assert rec_results[0].identity == "Person A"
    assert rec_results[0].distance < 0.01

    expr_results = expressions.classify_expressions_batch([crop_a, crop_b])
    assert len(expr_results) == 2
    for res in expr_results:
        d = res.to_dict()
        assert "label" in d
        assert len(d["scores"]) in (7, 8)

    age_results = age_estimation.estimate_ages_batch([crop_a, crop_b])
    assert len(age_results) == 2
    for res in age_results:
        d = res.to_dict()
        assert d["enabled"] is True
        assert d["estimated_age"] is not None
        assert d["estimated_age_range"] is not None


def test_decoupled_detection_runs_detector_only_on_interval_frames(monkeypatch):
    detector_calls = {"count": 0}

    def fake_detect(img_bgr):
        detector_calls["count"] += 1
        return (
            [
                DetectedFace(
                    x=100,
                    y=100,
                    w=80,
                    h=80,
                    confidence=0.95,
                    backend="scrfd-10g-int8",
                    left_eye=(125, 130),
                    right_eye=(155, 130),
                    crop_bgr=img_bgr[100:180, 100:180].copy(),
                )
            ],
            {"fallback_used": False, "errors": [], "backend_used": "scrfd-10g-int8", "latency_ms": 8.5},
        )

    monkeypatch.setattr(detection, "detect_faces", fake_detect)
    monkeypatch.setattr(settings, "detection_interval", 3)

    tracker = FaceTracker()
    frame = np.full((320, 320, 3), 90, dtype=np.uint8)

    res0 = pipeline.analyze_image(
        frame,
        pipeline.AnalysisOptions(
            run_recognition=True,
            run_expression=True,
            run_age=True,
            tracker=tracker,
            frame_index=0,
            decouple_detection=True,
        ),
    )
    assert detector_calls["count"] == 1
    assert res0.detector_meta["backend_used"] == "scrfd-10g-int8"
    assert len(res0.faces) == 1

    for idx in (1, 2):
        res_mid = pipeline.analyze_image(
            frame,
            pipeline.AnalysisOptions(
                run_recognition=True,
                run_expression=True,
                run_age=True,
                tracker=tracker,
                frame_index=idx,
                decouple_detection=True,
            ),
        )
        assert detector_calls["count"] == 1
        assert res_mid.detector_meta["backend_used"] == "optical-centroid-tracker"
        assert len(res_mid.faces) == 1

    res3 = pipeline.analyze_image(
        frame,
        pipeline.AnalysisOptions(
            run_recognition=True,
            run_expression=True,
            run_age=True,
            tracker=tracker,
            frame_index=3,
            decouple_detection=True,
        ),
    )
    assert detector_calls["count"] == 2
    assert res3.detector_meta["backend_used"] == "scrfd-10g-int8"


def test_onnx_runtime_cpu_vectorization_config():
    providers = get_active_execution_providers()
    assert len(providers) >= 1
    assert any(p in ("OpenVINOExecutionProvider", "CPUExecutionProvider") for p in providers)

    opts = create_session_options()
    assert opts.intra_op_num_threads >= 1
