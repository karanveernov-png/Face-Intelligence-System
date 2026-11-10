"""
WebSocket endpoint for webcam ("live") mode.

Implements the Pipeline Performance Strategy for live video streams:
  1. Decouple Detection from Recognition: runs SCRFD-2.5G on every Nth frame
     (`settings.detection_interval`), tracking bounding boxes in between via
     optical flow + centroid tracking on the per-connection `FaceTracker`.
  2. Batch Cropped Faces: stacks all detected/tracked face crops in the frame
     via `cv2.dnn.blobFromImages` for single-pass ONNX Runtime inference.
"""
from __future__ import annotations

import asyncio
import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

try:
    import websockets.exceptions

    _WS_CLOSED_EXCEPTIONS: tuple[type[BaseException], ...] = (
        WebSocketDisconnect,
        websockets.exceptions.ConnectionClosed,
        RuntimeError,
    )
except ImportError:
    _WS_CLOSED_EXCEPTIONS = (WebSocketDisconnect, RuntimeError)

from app.security.validation import reject_oversized_decoded_image
from app.services import pipeline
from app.services.tracking import FaceTracker

logger = logging.getLogger("face_intelligence.ws")

router = APIRouter()

_MAX_FRAME_BYTES = 8 * 1024 * 1024  # a single JPEG frame should never legitimately be this big


@router.websocket("/ws/analyze")
async def analyze_stream(
    websocket: WebSocket,
    run_recognition: bool = True,
    run_expression: bool = True,
    run_age: bool = False,
):
    await websocket.accept()
    tracker = FaceTracker()  # one per connection, so track IDs stay stable for this stream
    options = pipeline.AnalysisOptions(
        run_recognition=run_recognition, run_expression=run_expression, run_age=run_age
    )
    frames_received = 0
    processing_task: asyncio.Task[None] | None = None

    async def process_frame(
        frame: np.ndarray,
        frame_number: int,
        frame_options: pipeline.AnalysisOptions,
    ) -> None:
        try:
            result = await asyncio.to_thread(pipeline.analyze_image, frame, frame_options)
            # If `pipeline.analyze_image` was monkeypatched in tests or didn't
            # already update the tracker in-place, ensure tracker state stays synced.
            if not frame_options.decouple_detection or result.detector_meta == {}:
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
        except Exception as exc:  # noqa: BLE001 — one bad frame must not kill the stream
            logger.warning("ws frame %d failed to process: %s", frame_number, exc)
            try:
                await websocket.send_json({"error": f"frame processing failed: {exc}"})
            except _WS_CLOSED_EXCEPTIONS:
                return

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

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
                    if not all(type(value) is bool for value in option_values.values()):
                        raise ValueError("all analysis options must be booleans")
                    options = pipeline.AnalysisOptions(**option_values)
                    await websocket.send_json({"type": "options_applied"})
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    await websocket.send_json({"error": f"invalid options update: {exc}"})
                continue

            data = message.get("bytes")
            if data is None:
                continue
            frames_received += 1

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

            # Same decompression-bomb concern as the HTTP image/video routes
            # (see app/security/validation.py): _MAX_FRAME_BYTES bounds the
            # compressed frame, not its decoded size.
            try:
                reject_oversized_decoded_image(frame)
            except HTTPException as exc:
                await websocket.send_json({"error": str(exc.detail)})
                continue

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
        logger.info("webcam stream closed after %d frames", frames_received)
    finally:
        if processing_task is not None and not processing_task.done():
            processing_task.cancel()