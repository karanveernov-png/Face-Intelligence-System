import threading

import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import pipeline, recognition
from app.security.auth import create_api_key, Role


@pytest.fixture(autouse=True)
def isolated_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(recognition, "_INDEX_PATH", tmp_path / "index.json")
    yield


@pytest.fixture
def ws_token():
    """Create a test API key with Operator role for WebSocket testing."""
    plaintext, record = create_api_key(
        name="test-ws-operator",
        role=Role.OPERATOR,
        expires_in_days=1,
        metadata={"test": True}
    )
    return plaintext


@pytest.fixture
def ws_admin_token():
    """Create a test API key with Admin role for WebSocket testing."""
    plaintext, record = create_api_key(
        name="test-ws-admin",
        role=Role.ADMIN,
        expires_in_days=1,
        metadata={"test": True}
    )
    return plaintext


@pytest.fixture
def blank_jpeg_bytes(blank_image) -> bytes:
    ok, buf = cv2.imencode(".jpg", blank_image)
    assert ok
    return buf.tobytes()


def test_ws_connects_and_analyzes_one_frame(ws_token, blank_jpeg_bytes):
    client = TestClient(app)
    with client.websocket_connect(f"/ws/analyze?token={ws_token}") as ws:
        ws.send_bytes(blank_jpeg_bytes)
        result = ws.receive_json()
        assert "face_count" in result
        assert result["face_count"] == 0  # blank synthetic image, same as the REST equivalent
        assert "active_tracks" in result


def test_ws_rejects_decompression_bomb_frame_without_closing(ws_token, blank_jpeg_bytes):
    """Same decompression-bomb concern as the HTTP routes (see
    test_security_blackbox.py) -- a frame small enough to pass
    _MAX_FRAME_BYTES but which decodes to an enormous array must be
    rejected with an {"error": ...} message, and the connection must stay
    open and usable for the next frame."""
    import numpy as np

    huge = np.full((15000, 15000, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", huge, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok
    bomb_bytes = buf.tobytes()
    assert len(bomb_bytes) < 5 * 1024 * 1024  # small on disk, huge once decoded

    client = TestClient(app)
    with client.websocket_connect(f"/ws/analyze?token={ws_token}") as ws:
        ws.send_bytes(bomb_bytes)
        result = ws.receive_json()
        assert "error" in result
        assert "exceeds" in result["error"].lower() or "decompression" in result["error"].lower()

        # Connection must still be usable afterward.
        ws.send_bytes(blank_jpeg_bytes)
        result2 = ws.receive_json()
        assert "face_count" in result2


def test_ws_recovers_after_a_bad_frame_without_closing(ws_token, blank_jpeg_bytes):
    client = TestClient(app)
    with client.websocket_connect(f"/ws/analyze?token={ws_token}") as ws:
        ws.send_bytes(b"not a real jpeg at all")
        bad_result = ws.receive_json()
        assert "error" in bad_result

        # the connection must still be alive and usable after a bad frame
        ws.send_bytes(blank_jpeg_bytes)
        good_result = ws.receive_json()
        assert "face_count" in good_result


def test_ws_rejects_empty_frame_without_closing(ws_token, blank_jpeg_bytes):
    client = TestClient(app)
    with client.websocket_connect(f"/ws/analyze?token={ws_token}") as ws:
        ws.send_bytes(b"")
        resp = ws.receive_json()
        assert "error" in resp

        ws.send_bytes(blank_jpeg_bytes)
        good_result = ws.receive_json()
        assert "face_count" in good_result


def test_ws_track_ids_persist_across_frames_in_one_connection(ws_token, blank_jpeg_bytes):
    # With zero faces in a blank frame there's nothing to track, but this
    # confirms the endpoint accepts a sequence of frames on one connection
    # without error -- the per-connection FaceTracker lifecycle itself.
    client = TestClient(app)
    with client.websocket_connect(f"/ws/analyze?token={ws_token}") as ws:
        for _ in range(3):
            ws.send_bytes(blank_jpeg_bytes)
            result = ws.receive_json()
            assert result["active_tracks"] == 0


def test_ws_respects_query_param_options(ws_token, blank_jpeg_bytes):
    client = TestClient(app)
    with client.websocket_connect(
        f"/ws/analyze?token={ws_token}&run_recognition=false&run_expression=false&run_age=false"
    ) as ws:
        ws.send_bytes(blank_jpeg_bytes)
        result = ws.receive_json()
        assert "face_count" in result  # still responds normally with all tasks disabled


def test_ws_age_is_disabled_by_default(ws_token, blank_jpeg_bytes, monkeypatch):
    seen_options = []

    def fake_analyze(_frame, options):
        seen_options.append(options)
        return pipeline.PipelineResult(faces=[], detector_meta={}, total_latency_ms=0)

    monkeypatch.setattr(pipeline, "analyze_image", fake_analyze)
    client = TestClient(app)
    with client.websocket_connect(f"/ws/analyze?token={ws_token}") as ws:
        ws.send_bytes(blank_jpeg_bytes)
        assert ws.receive_json()["face_count"] == 0

    assert len(seen_options) == 1
    assert seen_options[0].run_age is False


def test_ws_option_change_applies_after_in_flight_frame(ws_token, blank_jpeg_bytes, monkeypatch):
    analysis_started = threading.Event()
    finish_analysis = threading.Event()
    seen_age_options = []

    def fake_analyze(_frame, options):
        seen_age_options.append(options.run_age)
        if len(seen_age_options) == 1:
            analysis_started.set()
            assert finish_analysis.wait(timeout=5)
        return pipeline.PipelineResult(faces=[], detector_meta={}, total_latency_ms=0)

    monkeypatch.setattr(pipeline, "analyze_image", fake_analyze)
    client = TestClient(app)
    with client.websocket_connect(f"/ws/analyze?token={ws_token}&run_age=true") as ws:
        ws.send_bytes(blank_jpeg_bytes)
        assert analysis_started.wait(timeout=2)

        ws.send_json({
            "type": "options",
            "run_recognition": True,
            "run_expression": True,
            "run_age": False,
        })
        assert ws.receive_json() == {"type": "options_applied"}
        finish_analysis.set()
        assert ws.receive_json()["face_count"] == 0

        ws.send_bytes(blank_jpeg_bytes)
        assert ws.receive_json()["face_count"] == 0

    assert seen_age_options == [True, False]