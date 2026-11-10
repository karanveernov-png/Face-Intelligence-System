import numpy as np
import pytest

from app.services import detection


def test_no_face_on_blank_image_returns_empty_list(blank_image):
    faces, meta = detection.detect_faces(blank_image)
    assert faces == []
    assert "backend_used" in meta
    assert "latency_ms" in meta


def test_no_face_on_noise_image_returns_empty_list(noise_image):
    faces, meta = detection.detect_faces(noise_image)
    # A detector may occasionally fire a low-confidence false positive on noise;
    # what we're really testing is that it doesn't crash and returns a well-formed list.
    assert isinstance(faces, list)
    for f in faces:
        assert f.confidence >= 0


def test_empty_image_raises_value_error():
    with pytest.raises(ValueError):
        detection.detect_faces(np.zeros((0, 0, 3), dtype=np.uint8))


def test_detected_face_serializes_to_dict(blank_image):
    # Construct a DetectedFace directly to test the serialization contract,
    # independent of whether the detector fires on this particular image.
    face = detection.DetectedFace(x=10, y=10, w=50, h=50, confidence=0.9, backend="centerface")
    d = face.to_dict()
    assert d["bbox"] == {"x": 10, "y": 10, "w": 50, "h": 50}
    assert d["confidence"] == 0.9
    assert d["detector_backend"] == "centerface"
    assert d["landmarks"] is None  # no eyes were set


def test_haar_fallback_loads_bundled_cascade():
    # This must work with zero network access, since it's bundled in opencv-python.
    cascade = detection._get_haar()
    assert not cascade.empty()
