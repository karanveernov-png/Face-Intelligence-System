"""
BLACK-BOX testing: every test here talks to the API the way an external
client (or attacker) would -- HTTP in, HTTP out, no knowledge of internals
assumed beyond what the API itself documents. Where a test needs a specific
internal code path to fire (e.g. "a face really was detected"), it uses
monkeypatching only to construct that external scenario deterministically;
it never asserts on internal state directly.

Organized by attack surface / edge case:
  1. Upload validation (size, extension, corruption, emptiness)
  2. Decompression-bomb protection (decoded-dimension limits)
  3. Path-traversal / filename-injection attempts
  4. Enrollment input validation (consent, name, face count)
  5. Malformed / missing request fields
  6. Injection-style strings in free-text fields
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.services import recognition
from app.services.detection import DetectedFace


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


def _fake_detected_face(img_shape=(300, 300)) -> DetectedFace:
    """A deterministic 'a real face was detected here' stand-in, used only to
    reach code paths (consent check, name validation) that sit downstream of
    detection -- this project's own synthetic images can't reliably trigger
    the real SCRFD detector (see conftest.py), so black-box tests of that
    downstream logic need this to avoid being accidentally gated on
    detection succeeding or failing instead of the thing actually being
    tested. The crop content itself is irrelevant to these tests."""
    crop = np.full((200, 200, 3), 150, dtype=np.uint8)
    return DetectedFace(
        x=20, y=20, w=200, h=200, confidence=0.99, backend="scrfd-10g-int8",
        left_eye=(80, 90), right_eye=(160, 90), nose=(120, 130),
        mouth_left=(90, 170), mouth_right=(150, 170),
        crop_bgr=crop,
    )


# --- 1. Upload validation -------------------------------------------------

def test_analyze_image_rejects_wrong_extension(client):
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400


def test_analyze_image_rejects_corrupt_file_with_valid_extension(client, invalid_image_bytes):
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("looks_legit.jpg", invalid_image_bytes, "image/jpeg")},
    )
    assert resp.status_code == 400


def test_analyze_image_rejects_empty_file(client):
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_analyze_image_rejects_oversized_upload(client, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "max_upload_size_mb", 1)
    oversized = b"\xff" * (2 * 1024 * 1024)  # 2MB body, 1MB cap
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("big.jpg", oversized, "image/jpeg")},
    )
    assert resp.status_code == 413


def test_analyze_image_rejects_uppercase_and_double_extensions(client):
    # Case-insensitivity: a real image under a SHOUTY extension should still
    # be accepted at the validation layer (then succeed or fail on content).
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("photo.JPG", b"not a real jpeg", "image/jpeg")},
    )
    assert resp.status_code == 400  # rejected for content, not extension case
    assert "could not be decoded" in resp.json()["detail"].lower()

    # A disguised executable (double extension, real extension is .exe) must
    # be rejected at the extension gate, regardless of the first-looking part.
    resp2 = client.post(
        "/api/analyze/image",
        files={"file": ("photo.jpg.exe", b"MZ\x90\x00fake-pe-header", "application/octet-stream")},
    )
    assert resp2.status_code == 400
    assert "unsupported file type" in resp2.json()["detail"].lower()


def test_analyze_image_rejects_missing_filename(client):
    # Some HTTP clients can be coerced into sending no filename at all.
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("", b"some bytes", "application/octet-stream")},
    )
    assert resp.status_code in (400, 422)  # either our check or FastAPI's own


# --- 2. Decompression-bomb protection ------------------------------------

def test_analyze_image_rejects_decompression_bomb(client):
    """A file that's small on disk but enormous once decoded must be
    rejected before it reaches any model -- see
    app/security/validation.py:reject_oversized_decoded_image."""
    huge = np.full((15000, 15000, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", huge, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok
    file_bytes = buf.tobytes()
    assert len(file_bytes) < 5 * 1024 * 1024  # proves it's small on disk

    resp = client.post(
        "/api/analyze/image",
        files={"file": ("bomb.png", file_bytes, "image/png")},
    )
    assert resp.status_code == 413
    assert "decompression" in resp.json()["detail"].lower() or "exceeds" in resp.json()["detail"].lower()


def test_analyze_image_accepts_a_large_but_reasonable_photo(client):
    """The fix must not be so aggressive it rejects a legitimate large photo
    (e.g. a modern phone/DSLR shot) -- this one sits just under both caps."""
    from app.core.config import settings

    w = h = int((settings.max_decoded_image_pixels ** 0.5) * 0.9)
    w, h = min(w, settings.max_decoded_image_dimension_px - 100), min(h, settings.max_decoded_image_dimension_px - 100)
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    resp = client.post(
        "/api/analyze/image",
        files={"file": ("big_but_fine.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert resp.status_code == 200


def test_enroll_image_also_rejects_decompression_bomb(client):
    huge = np.full((15000, 15000, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", huge, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok

    resp = client.post(
        "/api/faces/enroll",
        files={"file": ("bomb.png", buf.tobytes(), "image/png")},
        data={"name": "Someone", "consent_confirmed": "true"},
    )
    assert resp.status_code == 413


# --- 3. Path-traversal / filename-injection attempts ----------------------

@pytest.mark.parametrize(
    "malicious_filename",
    [
        "../../../etc/passwd.jpg",
        "..\\..\\windows\\system32\\config.jpg",
        "/etc/passwd.jpg",
        "....//....//etc/passwd.jpg",
    ],
)
def test_uploaded_filename_cannot_escape_temp_dir(client, blank_jpg_bytes, malicious_filename):
    """The server must generate its own temp filename regardless of what the
    client claims the file is called -- see validation.py's docstring point
    2. We can't inspect the server's filesystem from here (that would be
    white-box), so the black-box assertion is simply: the request either
    succeeds normally (proving the traversal-looking name was treated as
    inert data) or fails with an ordinary validation error -- never a 500,
    and never any sign the path was honored."""
    resp = client.post(
        "/api/analyze/image",
        files={"file": (malicious_filename, blank_jpg_bytes, "image/jpeg")},
    )
    assert resp.status_code in (200, 400, 413)


@pytest.mark.parametrize(
    "malicious_face_id",
    ["..", "...", "..%2F..%2Fetc%2Fpasswd", "' OR '1'='1", "<script>alert(1)</script>", "a" * 5000],
)
def test_delete_with_malicious_face_id_is_safely_not_found(client, malicious_face_id):
    """None of these should ever match a real enrolled UUID, so the only
    correct behavior is 404 -- never a 500, never a different status that
    would suggest the string was interpreted as a path or query fragment."""
    resp = client.delete(f"/api/faces/{malicious_face_id}")
    assert resp.status_code == 404


# --- 4. Enrollment input validation ---------------------------------------

def test_enroll_without_consent_on_a_real_detected_face_is_403_forbidden(client):
    """Regression-targeted fix: the pre-existing test with a similar name
    asserted 422, but on a blank (faceless) image -- so it was actually
    exercising the 'no face detected' branch, not consent enforcement at
    all. This test forces a face to be 'detected' so the request reaches the
    actual consent check and exercises the real 403 branch."""
    img = np.full((300, 300, 3), 150, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    with patch("app.api.routes.detection.detect_faces", return_value=([_fake_detected_face()], {})):
        resp = client.post(
            "/api/faces/enroll",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
            data={"name": "Test Person", "consent_confirmed": "false"},
        )
    assert resp.status_code == 403


def test_enroll_missing_consent_field_entirely_is_422(client, blank_jpg_bytes):
    """consent_confirmed has no default (Form(...)) -- omitting it entirely
    must be a pydantic/FastAPI validation error, distinct from sending an
    explicit 'false'."""
    resp = client.post(
        "/api/faces/enroll",
        files={"file": ("blank.jpg", blank_jpg_bytes, "image/jpeg")},
        data={"name": "Test Person"},
    )
    assert resp.status_code == 422


def test_enroll_with_empty_name_on_real_face_is_400(client):
    img = np.full((300, 300, 3), 150, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    with patch("app.api.routes.detection.detect_faces", return_value=([_fake_detected_face()], {})):
        resp = client.post(
            "/api/faces/enroll",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
            data={"name": "   ", "consent_confirmed": "true"},
        )
    assert resp.status_code == 400


def test_enroll_with_zero_faces_is_422(client, blank_jpg_bytes):
    resp = client.post(
        "/api/faces/enroll",
        files={"file": ("blank.jpg", blank_jpg_bytes, "image/jpeg")},
        data={"name": "Nobody", "consent_confirmed": "true"},
    )
    assert resp.status_code == 422
    assert "no face" in resp.json()["detail"].lower()


def test_enroll_with_two_faces_is_422(client):
    img = np.full((300, 600, 3), 150, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    two_faces = [_fake_detected_face(), _fake_detected_face()]
    with patch("app.api.routes.detection.detect_faces", return_value=(two_faces, {})):
        resp = client.post(
            "/api/faces/enroll",
            files={"file": ("two.jpg", buf.tobytes(), "image/jpeg")},
            data={"name": "Ambiguous", "consent_confirmed": "true"},
        )
    assert resp.status_code == 422
    assert "multiple faces" in resp.json()["detail"].lower()


def test_enroll_then_delete_round_trip_with_real_detected_face(client):
    img = np.full((300, 300, 3), 150, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    with patch("app.api.routes.detection.detect_faces", return_value=([_fake_detected_face()], {})):
        enroll_resp = client.post(
            "/api/faces/enroll",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
            data={"name": "Round Trip", "consent_confirmed": "true"},
        )
    assert enroll_resp.status_code == 200
    face_id = enroll_resp.json()["face_id"]

    listed = client.get("/api/faces").json()
    assert any(e["face_id"] == face_id for e in listed)

    del_resp = client.delete(f"/api/faces/{face_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    del_again = client.delete(f"/api/faces/{face_id}")
    assert del_again.status_code == 404  # already gone


# --- 5. Malformed / missing request fields --------------------------------

def test_analyze_image_missing_file_field_is_422(client):
    resp = client.post("/api/analyze/image", data={"run_recognition": "true"})
    assert resp.status_code == 422


def test_analyze_image_garbage_boolean_form_values_do_not_500(client, blank_jpg_bytes):
    """FastAPI/pydantic should reject or coerce nonsense boolean strings, but
    must never surface as an unhandled 500."""
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("blank.jpg", blank_jpg_bytes, "image/jpeg")},
        data={"run_recognition": "banana"},
    )
    assert resp.status_code in (200, 422)


def test_analyze_video_rejects_corrupt_file(client):
    resp = client.post(
        "/api/analyze/video",
        files={"file": ("broken.mp4", b"not a real mp4 container", "video/mp4")},
    )
    assert resp.status_code == 400


def test_analyze_video_rejects_wrong_extension(client):
    resp = client.post(
        "/api/analyze/video",
        files={"file": ("video.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


# --- 6. Injection-style strings in free-text fields ------------------------

@pytest.mark.parametrize(
    "injection_name",
    [
        "Robert'); DROP TABLE faces;--",
        '{"$ne": null}',
        "<img src=x onerror=alert(1)>",
        "../../etc/passwd",
        "名前\u0000\u200b",  # unicode + embedded null + zero-width space
        "😀" * 50,
    ],
)
def test_enroll_name_with_injection_style_strings_is_stored_as_inert_text(client, injection_name):
    """There's no SQL or template engine in this path (the gallery is a flat
    JSON index), but the name does round-trip through json.dump/json.load
    and back out over HTTP -- this proves that pipeline is robust to hostile
    strings: no crash, and the string comes back out exactly as sent, never
    executed, reinterpreted, or silently mangled."""
    img = np.full((300, 300, 3), 150, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    with patch("app.api.routes.detection.detect_faces", return_value=([_fake_detected_face()], {})):
        resp = client.post(
            "/api/faces/enroll",
            files={"file": ("face.jpg", buf.tobytes(), "image/jpeg")},
            data={"name": injection_name, "consent_confirmed": "true"},
        )
    assert resp.status_code == 200
    returned_name = resp.json()["name"]
    assert returned_name == injection_name.strip()

    listed = client.get("/api/faces").json()
    assert any(e["name"] == injection_name.strip() for e in listed)
