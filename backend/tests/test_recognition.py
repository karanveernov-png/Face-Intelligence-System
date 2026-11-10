import numpy as np
import pytest

from app.services import recognition


@pytest.fixture(autouse=True)
def isolated_gallery(tmp_path, monkeypatch):
    """Point the recognition module at a throwaway gallery dir for every test,
    so tests never read/write the real data/embeddings/index.json."""
    monkeypatch.setattr(recognition, "_INDEX_PATH", tmp_path / "index.json")
    yield


def test_enrollment_requires_consent(blank_image):
    with pytest.raises(PermissionError):
        recognition.enroll_face(blank_image, name="Test Person", consent_confirmed=False)


def test_enrollment_requires_nonempty_name(blank_image):
    with pytest.raises(ValueError):
        recognition.enroll_face(blank_image, name="   ", consent_confirmed=True)


def test_match_returns_unknown_when_gallery_is_empty(blank_image):
    result = recognition.match_face(blank_image)
    assert result.identity is None
    assert result.to_dict()["identity"] == "Unknown"


def test_enroll_then_match_same_image_is_recognized(blank_image):
    enrolled = recognition.enroll_face(blank_image, name="Test Person", consent_confirmed=True)
    assert enrolled["name"] == "Test Person"

    result = recognition.match_face(blank_image)
    # ONNX inference is deterministic for a fixed input, so re-embedding the exact
    # same pixels must land at distance ~0 from itself and clear the threshold.
    assert result.identity == "Test Person"
    assert result.distance < 0.01


def test_enroll_then_list_then_delete(blank_image):
    enrolled = recognition.enroll_face(blank_image, name="Test Person", consent_confirmed=True)
    entries = recognition.list_enrolled()
    assert any(e["face_id"] == enrolled["face_id"] for e in entries)

    deleted = recognition.delete_enrolled(enrolled["face_id"])
    assert deleted is True
    assert recognition.delete_enrolled(enrolled["face_id"]) is False  # already gone

    entries_after = recognition.list_enrolled()
    assert all(e["face_id"] != enrolled["face_id"] for e in entries_after)


def test_cosine_distance_of_identical_unit_vectors_is_zero():
    v = np.array([1.0, 0.0, 0.0])
    assert recognition._cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_of_orthogonal_unit_vectors_is_one():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert recognition._cosine_distance(a, b) == pytest.approx(1.0, abs=1e-9)
