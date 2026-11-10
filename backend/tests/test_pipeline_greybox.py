"""
GREY-BOX testing of app/services/pipeline.py: these tests use knowledge of
the internal implementation (exception handling structure, the
use_inter_frame_tracker boolean condition, _deduplicate_faces) to construct
scenarios that are difficult or impossible to trigger reliably from pure
black-box HTTP testing -- e.g. forcing a batched-recognition call to raise
partway through, to verify the documented per-face-retry fallback actually
activates rather than just being dead code.

Each test still goes through the module's public function (analyze_image,
_deduplicate_faces, _iou) rather than reaching into private state, which is
what keeps this grey-box rather than pure white-box.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from app.services import age_estimation, expressions, pipeline, recognition
from app.services.detection import DetectedFace
from app.services.pipeline import AnalysisOptions, _deduplicate_faces, _iou, analyze_image
from app.services.tracking import FaceTracker


@pytest.fixture(autouse=True)
def isolated_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(recognition, "_INDEX_PATH", tmp_path / "index.json")
    yield


def _face(x=10, y=10, w=80, h=80, conf=0.9) -> DetectedFace:
    return DetectedFace(
        x=x, y=y, w=w, h=h, confidence=conf, backend="scrfd-10g-int8",
        left_eye=(x + 20, y + 25), right_eye=(x + 55, y + 25),
        nose=(x + 37, y + 45), mouth_left=(x + 25, y + 60), mouth_right=(x + 50, y + 60),
        crop_bgr=np.full((h, w, 3), 140, dtype=np.uint8),
    )


# --- _iou: pure function edge cases ---------------------------------------

def test_iou_identical_boxes_is_one():
    a = _face(0, 0, 50, 50)
    assert _iou(a, a) == pytest.approx(1.0)


def test_iou_non_overlapping_boxes_is_zero():
    a = _face(0, 0, 50, 50)
    b = _face(1000, 1000, 50, 50)
    assert _iou(a, b) == 0.0


def test_iou_zero_area_box_does_not_divide_by_zero():
    a = _face(0, 0, 0, 0)
    b = _face(0, 0, 50, 50)
    # Must not raise ZeroDivisionError / produce nan or inf.
    result = _iou(a, b)
    assert result == 0.0


def test_iou_partial_overlap_is_between_zero_and_one():
    a = _face(0, 0, 100, 100)
    b = _face(50, 50, 100, 100)
    result = _iou(a, b)
    assert 0.0 < result < 1.0


# --- _deduplicate_faces -----------------------------------------------------

def test_deduplicate_collapses_heavily_overlapping_boxes_keeping_higher_confidence():
    low_conf = _face(0, 0, 100, 100, conf=0.5)
    high_conf = _face(2, 2, 100, 100, conf=0.95)  # near-identical box, higher confidence
    result = _deduplicate_faces([low_conf, high_conf], iou_threshold=0.45)
    assert len(result) == 1
    assert result[0].confidence == 0.95


def test_deduplicate_keeps_genuinely_distinct_faces():
    a = _face(0, 0, 100, 100)
    b = _face(500, 500, 100, 100)
    result = _deduplicate_faces([a, b])
    assert len(result) == 2


def test_deduplicate_single_face_is_a_noop():
    a = _face()
    assert _deduplicate_faces([a]) == [a]


def test_deduplicate_empty_list_is_a_noop():
    assert _deduplicate_faces([]) == []


# --- analyze_image: option flags control what's attached ------------------

def test_all_options_off_leaves_faces_with_only_detection_fields():
    with patch("app.services.detection.detect_faces", return_value=([_face()], {"backend_used": "fake"})):
        options = AnalysisOptions(run_recognition=False, run_expression=False, run_age=False)
        result = analyze_image(np.zeros((200, 200, 3), dtype=np.uint8), options)
    assert len(result.faces) == 1
    face = result.faces[0]
    assert "identity" not in face
    assert "expression" not in face
    assert "age" not in face
    assert "bbox" in face and "landmarks" in face


def test_each_option_independently_adds_its_own_key():
    with patch("app.services.detection.detect_faces", return_value=([_face()], {"backend_used": "fake"})):
        for flag, key in [("run_recognition", "identity"), ("run_expression", "expression"), ("run_age", "age")]:
            options = AnalysisOptions(run_recognition=False, run_expression=False, run_age=False)
            setattr(options, flag, True)
            result = analyze_image(np.zeros((200, 200, 3), dtype=np.uint8), options)
            assert key in result.faces[0], f"{flag}=True should add '{key}' to the face dict"


def test_no_detected_faces_short_circuits_before_any_model_call():
    """When detection finds nothing, recognition/expression/age must never
    even be invoked -- not just 'return no results', but genuinely skipped."""
    with patch("app.services.detection.detect_faces", return_value=([], {"backend_used": "fake"})):
        with patch.object(recognition, "match_faces_batch") as mock_rec, \
             patch.object(expressions, "classify_expressions_batch") as mock_expr, \
             patch.object(age_estimation, "estimate_ages_batch_in_frame") as mock_age:
            options = AnalysisOptions(run_recognition=True, run_expression=True, run_age=True)
            result = analyze_image(np.zeros((100, 100, 3), dtype=np.uint8), options)
    assert result.faces == []
    mock_rec.assert_not_called()
    mock_expr.assert_not_called()
    mock_age.assert_not_called()


# --- Forced-failure injection: verify the documented fallback paths -------

def test_recognition_batch_failure_falls_back_to_per_face_and_records_task_error():
    """White-box knowledge used here: analyze_image wraps the batched
    recognition call in try/except and retries per-face on failure. This
    forces that exact branch and checks it behaves as documented rather
    than silently losing results or crashing the whole request.

    The side_effect below deliberately fails ONLY for batch size > 1 and
    delegates to the real implementation for size 1 -- this is the actual
    documented real-world failure mode (see _embed_batch's comment on
    w600k_r50_int8.onnx not being batch-safe: N=1 works, N>1 doesn't), and
    matters here because pipeline's own per-face fallback calls
    recognition.match_face(), which itself calls match_faces_batch([face])
    (N=1) -- so a naive 'always raise' side_effect would incorrectly break
    the fallback it's supposed to be testing too."""
    real_match_faces_batch = recognition.match_faces_batch

    def flaky_batch(faces_bgr):
        if len(faces_bgr) > 1:
            raise RuntimeError("simulated batch failure")
        return real_match_faces_batch(faces_bgr)

    faces = [_face(0, 0, 80, 80), _face(200, 200, 80, 80)]
    with patch("app.services.detection.detect_faces", return_value=(faces, {"backend_used": "fake"})):
        with patch.object(recognition, "match_faces_batch", side_effect=flaky_batch):
            options = AnalysisOptions(run_recognition=True, run_expression=False, run_age=False)
            result = analyze_image(np.zeros((400, 400, 3), dtype=np.uint8), options)

    assert len(result.faces) == 2
    # Per-face retry (N=1 calls) should have succeeded despite the N>1 batch
    # failure, so both faces get a real identity dict, not None.
    for f in result.faces:
        assert f.get("identity") is not None
        assert "matched" in f["identity"]
    # And the batch-level failure must still be visible in task_errors for observability.
    assert any("recognition.batch" in e for e in result.task_errors)


def test_expression_batch_failure_falls_back_to_per_face():
    """Same N>1-fails/N=1-succeeds shape as the recognition test above, and
    for the same reason: classify_expression (pipeline's per-face fallback)
    calls classify_expressions_batch([face]) internally, so an
    unconditional failure would break the fallback it's meant to exercise."""
    real_batch = expressions.classify_expressions_batch

    def flaky_batch(faces_bgr):
        if len(faces_bgr) > 1:
            raise RuntimeError("simulated")
        return real_batch(faces_bgr)

    faces = [_face(0, 0, 80, 80), _face(200, 200, 80, 80)]
    with patch("app.services.detection.detect_faces", return_value=(faces, {"backend_used": "fake"})):
        with patch.object(expressions, "classify_expressions_batch", side_effect=flaky_batch):
            options = AnalysisOptions(run_recognition=False, run_expression=True, run_age=False)
            result = analyze_image(np.zeros((400, 400, 3), dtype=np.uint8), options)

    assert len(result.faces) == 2
    for f in result.faces:
        assert f.get("expression") is not None
        assert "label" in f["expression"]


def test_age_batch_failure_falls_back_to_per_face():
    faces = [_face(0, 0, 80, 80)]
    with patch("app.services.detection.detect_faces", return_value=(faces, {"backend_used": "fake"})):
        with patch.object(age_estimation, "estimate_ages_batch_in_frame", side_effect=RuntimeError("simulated")):
            options = AnalysisOptions(run_recognition=False, run_expression=False, run_age=True)
            result = analyze_image(np.zeros((200, 200, 3), dtype=np.uint8), options)

    assert len(result.faces) == 1
    assert result.faces[0].get("age") is not None
    assert "enabled" in result.faces[0]["age"]


def test_recognition_failure_on_every_fallback_path_too_records_per_face_errors():
    """The harder case: the batch call AND the per-face retry both fail.
    Must not crash -- each face's identity becomes None with its own
    task_errors entry, and the other faces (if any) are unaffected."""
    faces = [_face(0, 0, 80, 80)]
    with patch("app.services.detection.detect_faces", return_value=(faces, {"backend_used": "fake"})):
        with patch.object(recognition, "match_faces_batch", side_effect=RuntimeError("batch broke")), \
             patch.object(recognition, "match_face", side_effect=RuntimeError("per-face also broke")):
            options = AnalysisOptions(run_recognition=True, run_expression=False, run_age=False)
            result = analyze_image(np.zeros((200, 200, 3), dtype=np.uint8), options)

    assert len(result.faces) == 1
    assert result.faces[0]["identity"] is None
    assert any("face[0].recognition" in e for e in result.task_errors)


# --- decouple_detection / use_inter_frame_tracker boolean logic -----------

def test_decouple_detection_with_empty_tracker_falls_back_to_full_detection():
    """White-box knowledge: use_inter_frame_tracker requires
    bool(tracker.tracks) to be truthy. A brand-new tracker has no tracks yet,
    so even with decouple_detection=True and a non-interval frame_index, full
    detection must still run on the first frame -- otherwise no face would
    ever be found on frame 1 of a video."""
    tracker = FaceTracker()
    assert not tracker.tracks  # precondition: genuinely empty

    with patch("app.services.detection.detect_faces", return_value=([_face()], {"backend_used": "real-detector"})) as mock_detect:
        options = AnalysisOptions(
            run_recognition=False, run_expression=False, run_age=False,
            tracker=tracker, frame_index=1, decouple_detection=True,
        )
        result = analyze_image(np.zeros((200, 200, 3), dtype=np.uint8), options)

    mock_detect.assert_called_once()
    assert result.detector_meta["backend_used"] == "real-detector"


def test_decouple_detection_uses_tracker_on_non_interval_frame_once_tracks_exist():
    tracker = FaceTracker()
    tracker.update([(10, 10, 80, 80)])  # seed a real track
    assert tracker.tracks  # precondition

    with patch("app.services.detection.detect_faces") as mock_detect:
        options = AnalysisOptions(
            run_recognition=False, run_expression=False, run_age=False,
            tracker=tracker, frame_index=1, decouple_detection=True,  # 1 % detection_interval(3) != 0
        )
        result = analyze_image(np.zeros((200, 200, 3), dtype=np.uint8), options)

    mock_detect.assert_not_called()
    assert result.detector_meta["backend_used"] == "optical-centroid-tracker"


def test_decouple_detection_runs_full_detection_on_interval_frame_even_with_tracks():
    from app.core.config import settings

    tracker = FaceTracker()
    tracker.update([(10, 10, 80, 80)])

    with patch("app.services.detection.detect_faces", return_value=([_face()], {"backend_used": "real-detector"})) as mock_detect:
        options = AnalysisOptions(
            run_recognition=False, run_expression=False, run_age=False,
            tracker=tracker, frame_index=settings.detection_interval * 2,  # an exact interval multiple
            decouple_detection=True,
        )
        analyze_image(np.zeros((200, 200, 3), dtype=np.uint8), options)

    mock_detect.assert_called_once()
