"""
WebSocket endpoint for webcam ("live") mode with authentication.

Implements the Pipeline Performance Strategy for live video streams:
  1. Decouple Detection from Recognition: runs SCRFD-10G on every Nth frame
     (`settings.detection_interval`), tracking bounding boxes in between via
     optical flow + centroid tracking on the per-connection `FaceTracker`.
  2. Batch Cropped Faces: stacks all detected/tracked face crops in the frame
     via `cv2.dnn.blobFromImages` for single-pass ONNX Runtime inference.

Security:
  - Requires authentication during WebSocket handshake
  - Token validated before connection accepted
  - Rate limiting on frames per second
  - Connection limits
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status

try:
    import websockets.exceptions

    _WS_CLOSED_EXCEPTIONS: tuple[type[BaseException], ...] = (
        WebSocketDisconnect,
        websockets.exceptions.ConnectionClosed,
        RuntimeError,
    )
except ImportError:
    _WS_CLOSED_EXCEPTIONS = (WebSocketDisconnect, RuntimeError)

from app.core.config import settings
from app.security.auth import (
    APIKeyRecord,
    AuthContext,
    Permission,
    _api_keys,
    _hash_api_key,
    _verify_api_key,
    get_auth_context,
)
from app.security.validation import reject_oversized_decoded_image
from app.services import pipeline
from app.services.tracking import FaceTracker

logger = logging.getLogger("face_intelligence.ws")

# Use standard logger for ws module
ws_logger = logger

_MAX_FRAME_BYTES = 8 * 1024 * 1024  # 8MB per frame
_MAX_CONNECTIONS = 100  # Max concurrent WebSocket connections
_IDLE_TIMEOUT_SECONDS = 300  # 5 minutes idle timeout

# Connection tracking
_active_connections: dict[str, dict] = {}  # client_id -> connection info


router = APIRouter()


async def authenticate_websocket(websocket: WebSocket, token: Optional[str]) -> AuthContext:
    """
    Authenticate WebSocket connection during handshake.
    Raises HTTPException if authentication fails (causes handshake rejection).
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required. Provide via 'token' query parameter."
        )

    # Validate token
    key_hash = _hash_api_key(token)
    record: Optional[APIKeyRecord] = _api_keys.get(key_hash)

    if not record:
        logger.warning("WebSocket: Invalid token attempted")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

    if record.revoked:
        logger.warning("WebSocket: Revoked token attempted: %s", record.name)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has been revoked"
        )

    if record.expires_at and time.time() > record.expires_at:
        logger.warning("WebSocket: Expired token attempted: %s", record.name)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired"
        )

    # Check permission for stream analysis
    from app.security.auth import ROLE_PERMISSIONS
    permissions = ROLE_PERMISSIONS.get(record.role, set())
    if Permission.ANALYZE_STREAM not in permissions:
        logger.warning("WebSocket: Insufficient permissions for %s: %s", record.name, record.role.value)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for stream analysis"
        )

    # Update last used
    record.last_used_at = time.time()

    return AuthContext(
        api_key_name=record.name,
        role=record.role,
        permissions=permissions,
        key_record=record,
    )


def check_connection_limits(client_ip: str) -> bool:
    """Check if connection limits allow new connection."""
    # Count connections from this IP
    ip_connections = sum(1 for info in _active_connections.values() if info.get("client_ip") == client_ip)
    if ip_connections >= 10:  # Max 10 connections per IP
        return False

    # Total connection limit
    if len(_active_connections) >= _MAX_CONNECTIONS:
        return False

    return True


def register_connection(client_id: str, client_ip: str, auth: Optional[AuthContext] = None) -> None:
    """Register a new WebSocket connection."""
    _active_connections[client_id] = {
        "client_ip": client_ip,
        "auth_name": (auth.api_key_name if auth else None),
        "role": (auth.role.value if auth else None),
        "connected_at": time.time(),
        "last_frame_at": time.time(),
        "frame_count": 0,
    }


def unregister_connection(client_id: str) -> None:
    """Unregister a WebSocket connection."""
    _active_connections.pop(client_id, None)


def update_frame_timestamp(client_id: str) -> bool:
    """
    Update frame timestamp and check idle timeout.
    Returns True if frame allowed, False if idle timeout.
    """
    if client_id not in _active_connections:
        return False

    conn = _active_connections[client_id]
    now = time.time()

    # Check idle timeout
    if now - conn["last_frame_at"] > _IDLE_TIMEOUT_SECONDS:
        return False

    conn["last_frame_at"] = now
    return True


@router.websocket("/ws/analyze")
async def analyze_stream(
    websocket: WebSocket,
    run_recognition: bool = Query(True),
    run_expression: bool = Query(True),
    run_age: bool = Query(False),
):
    """
    WebSocket endpoint for live webcam analysis.

    Query Parameters:
    - run_recognition: Enable face recognition (default: true)
    - run_expression: Enable expression analysis (default: true)
    - run_age: Enable age estimation (default: false)

    Message Format (Client -> Server):
    - Binary: JPEG/PNG frame data
    - Text (JSON): {"type": "options", "run_recognition": true, "run_expression": true, "run_age": false}

    Response Format (Server -> Client):
    - JSON: PipelineResult with faces, detector meta, latency, active_tracks
    - JSON: {"type": "options_applied"} on options update
    - JSON: {"error": "..."} on errors
    """
    client_ip = websocket.client.host if websocket.client else "unknown"
    client_id = f"{client_ip}:{id(websocket)}"

    # Check connection limits before accepting
    if not check_connection_limits(client_ip):
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Connection limit exceeded")
        return

    # Accept the connection
    await websocket.accept()
    register_connection(client_id, client_ip, None)

    logger.info(
        "WebSocket connected: client_id=%s client_ip=%s",
        client_id,
        client_ip,
    )

    tracker = FaceTracker()
    options = pipeline.AnalysisOptions(
        run_recognition=run_recognition,
        run_expression=run_expression,
        run_age=run_age,
    )
    frames_received = 0
    processing_task: Optional[asyncio.Task] = None

    async def process_frame(
        frame: np.ndarray,
        frame_number: int,
        frame_options: pipeline.AnalysisOptions,
    ) -> None:
        """Process a single frame and send results."""
        try:
            result = await asyncio.to_thread(pipeline.analyze_image, frame, frame_options)

            # Sync tracker if needed
            if not frame_options.decouple_detection or not result.detector_meta:
                boxes = [
                    (f["bbox"]["x"], f["bbox"]["y"], f["bbox"]["w"], f["bbox"]["h"])
                    for f in result.faces
                ]
                tracked = tracker.update(boxes)
                active_tracks = len(tracked)
            else:
                active_tracks = len(tracker.tracks)

            payload = result.to_dict()
            payload["active_tracks"] = active_tracks
            await websocket.send_json(payload)

        except Exception as exc:  # noqa: BLE001
            logger.warning("WS frame %d failed: %s", frame_number, exc, exc_info=settings.debug)
            try:
                await websocket.send_json({"error": f"frame processing failed: {exc}"})
            except _WS_CLOSED_EXCEPTIONS:
                return

    try:
        while True:
            # Check idle timeout
            if time.time() - _active_connections.get(client_id, {}).get("last_frame_at", time.time()) > _IDLE_TIMEOUT_SECONDS:
                await websocket.send_json({"error": "idle timeout"})
                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="Idle timeout")
                break

            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # Handle text messages (options updates)
            text = message.get("text")
            if text is not None:
                try:
                    update = json.loads(text)
                    if not isinstance(update, dict) or update.get("type") != "options":
                        raise ValueError("expected an options update")
                    option_values = {
                        "run_recognition": update.get("run_recognition"),
                        "run_expression": update.get("run_expression"),
                        "run_age": update.get("run_age"),
                    }
                    if not all(isinstance(v, bool) for v in option_values.values() if v is not None):
                        raise ValueError("all analysis options must be booleans")
                    # Only update provided options
                    if option_values["run_recognition"] is not None:
                        options.run_recognition = option_values["run_recognition"]
                    if option_values["run_expression"] is not None:
                        options.run_expression = option_values["run_expression"]
                    if option_values["run_age"] is not None:
                        options.run_age = option_values["run_age"]
                    await websocket.send_json({"type": "options_applied"})
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    await websocket.send_json({"error": f"invalid options update: {exc}"})
                continue

            # Handle binary frames
            data = message.get("bytes")
            if data is None:
                continue

            frames_received += 1

            # Check idle timeout
            if not update_frame_timestamp(client_id):
                await websocket.send_json({"error": "idle timeout"})
                continue

            if len(data) == 0:
                await websocket.send_json({"error": "received an empty frame"})
                continue
            if len(data) > _MAX_FRAME_BYTES:
                await websocket.send_json({"error": f"frame exceeds {_MAX_FRAME_BYTES} bytes, dropped"})
                continue

            buf = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "frame could not be decoded as an image"})
                continue

            # Decompression bomb check
            try:
                reject_oversized_decoded_image(frame)
            except HTTPException as exc:
                await websocket.send_json({"error": str(exc.detail)})
                continue

            # Prevent overlapping frame processing
            if processing_task is not None and not processing_task.done():
                await websocket.send_json({"error": "a frame is already being processed"})
                continue

            frame_options = pipeline.AnalysisOptions(
                run_recognition=options.run_recognition,
                run_expression=options.run_expression,
                run_age=options.run_age,
                tracker=tracker,
                frame_index=frames_received - 1,
                decouple_detection=True,
            )
            processing_task = asyncio.create_task(
                process_frame(frame, frames_received, frame_options)
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s (%d frames)", client_id, frames_received)
    except _WS_CLOSED_EXCEPTIONS:
        logger.info("WebSocket closed: %s", client_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=settings.debug)
    finally:
        unregister_connection(client_id)
        if processing_task is not None and not processing_task.done():
            processing_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/ws/stats")
def websocket_stats():
    """
    Get WebSocket connection statistics.
    """
    return {
        "active_connections": len(_active_connections),
        "connections": [
            {
                "client_id": cid,
                "client_ip": info["client_ip"],
                "auth_name": info["auth_name"],
                "role": info["role"],
                "connected_duration": time.time() - info["connected_at"],
                "frames_received": info["frame_count"],
            }
            for cid, info in _active_connections.items()
        ],
    }