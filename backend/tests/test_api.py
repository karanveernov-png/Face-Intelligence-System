import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import recognition


@pytest.fixture(autouse=True)
def isolated_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(recognition, "_INDEX_PATH", tmp_path / "index.json")
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def blank_jpg_bytes(blank_image) -> bytes:
    ok, buf = cv2.imencode(".jpg", blank_image)
    assert ok
    return buf.tobytes()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models_status_reports_real_state(client):
    resp = client.get("/api/models/status")
    assert resp.status_code == 200
    body = resp.json()
    names = {m["name"] for m in body}
    assert {
        "SCRFD-10G-INT8/Haar",
        "ArcFace-R50-INT8",
        "HSEmotion-ENetB0-INT8",
        "GenderAge+FairFace-Ensemble",
    }.issubset(names)
    for m in body:
        assert "loaded" in m


def test_analyze_image_rejects_wrong_extension(client):
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400


def test_analyze_image_rejects_corrupt_file(client):
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("broken.jpg", b"\x00\x01not a real jpeg", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_analyze_image_on_blank_image_returns_zero_faces(client, blank_jpg_bytes):
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("blank.jpg", blank_jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["face_count"] == 0
    assert body["faces"] == []
    assert "total_latency_ms" in body


def test_enroll_consent_false_on_faceless_image_is_422_for_the_face_check_not_consent(client, blank_jpg_bytes):
    """BUG FOUND IN TESTING (fixed by clarifying, not changing behavior): this
    test used to be named test_enroll_requires_consent_field and asserted
    422, which happens to be correct -- but for the wrong reason. blank_jpg
    has no face, so the route's own "no face detected" check (422) fires
    before the consent check (403) is ever reached. The test passed while
    giving zero regression coverage to consent enforcement itself. See
    test_security_blackbox.py::test_enroll_without_consent_on_a_real_detected_face_is_403_forbidden
    for the test that actually exercises the 403 path, using a mocked
    detection result so a face really is present when consent is checked.
    This test is kept, renamed, to document that the 422-on-faceless-image
    behavior is itself still correct and intentional."""
    resp = client.post(
        "/api/faces/enroll",
        data={"name": "Test Person", "consent_confirmed": "false"},
        files={"file": ("blank.jpg", blank_jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "no face" in resp.json()["detail"].lower()


def test_faces_list_starts_empty(client):
    resp = client.get("/api/faces")
    assert resp.status_code == 200
    assert resp.json() == []


def test_delete_nonexistent_face_returns_404(client):
    resp = client.delete("/api/faces/does-not-exist")
    assert resp.status_code == 404
