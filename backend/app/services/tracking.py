"""
Face tracking across video frames with Temporal EMA Smoothing & Decoupled Optical Flow.

Combines three accuracy + latency optimizations for video and live webcam streams:
  1. IoU + Centroid Matching (`update`): links detections across frames into
     persistent `Track` IDs.
  2. Pyramidal Lucas-Kanade Optical Flow + Centroid Velocity (`predict_inter_frame`):
     propagates bounding boxes and 5-point landmarks on intermediate frames when
     SCRFD detection is decoupled (`settings.detection_interval`).
  3. Temporal Exponential Moving Average (EMA) Smoothing (`smooth_face_attributes`):
     eliminates single-frame flicker/jitter in expression probabilities and age
     estimates across a track's lifetime (`settings.temporal_ema_alpha`).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.services.age_estimation import _bucket_for_age
from app.services.detection import DetectedFace
from app.services.expressions import result_from_scores_dict


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
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


def _centroid(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    missed_frames: int = 0
    age_frames: int = 1
    velocity: tuple[float, float] = (0.0, 0.0)
    confidence: float = 1.0
    backend: str = "scrfd-10g-int8"
    # Frames predicted purely by optical flow since this track last matched a
    # *real* SCRFD detection. This is what lets a drifted "ghost" track stop
    # being rendered quickly instead of lingering for `tracker_max_missed_frames`
    # full detection cycles (see predict_inter_frame / update below).
    unconfirmed_predict_frames: int = 0
    left_eye: tuple[int, int] | None = None
    right_eye: tuple[int, int] | None = None
    nose: tuple[int, int] | None = None
    mouth_left: tuple[int, int] | None = None
    mouth_right: tuple[int, int] | None = None
    ema_expression_scores: dict[str, float] | None = field(default=None, repr=False)
    ema_age: float | None = field(default=None, repr=False)
    cached_identity: dict[str, Any] | None = field(default=None, repr=False)


class FaceTracker:
    """One instance per video stream. Supports IoU detection matching, inter-frame
    optical-flow + centroid propagation, and temporal EMA attribute smoothing."""

    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self._id_counter = itertools.count(1)
        self._prev_gray: np.ndarray | None = None

    def update(
        self,
        detections: list[tuple[int, int, int, int]],
        detected_faces: list[DetectedFace] | None = None,
        frame_bgr: np.ndarray | None = None,
    ) -> dict[int, tuple[int, int, int, int]]:
        """
        detections: list of (x, y, w, h) boxes for the current frame.
        Returns {track_id: box} for every currently active track.
        """
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())

        candidate_pairs = []
        for tid in unmatched_tracks:
            for di in unmatched_detections:
                iou = _iou(self.tracks[tid].box, detections[di])
                if iou >= settings.tracker_iou_threshold:
                    candidate_pairs.append((iou, tid, di))
        candidate_pairs.sort(reverse=True, key=lambda p: p[0])

        matched_tracks, matched_detections = set(), set()
        for iou, tid, di in candidate_pairs:
            if tid in matched_tracks or di in matched_detections:
                continue
            old_cx, old_cy = _centroid(self.tracks[tid].box)
            new_cx, new_cy = _centroid(detections[di])
            vx = 0.6 * (new_cx - old_cx) + 0.4 * self.tracks[tid].velocity[0]
            vy = 0.6 * (new_cy - old_cy) + 0.4 * self.tracks[tid].velocity[1]

            t = self.tracks[tid]
            t.box = detections[di]
            t.velocity = (vx, vy)
            t.missed_frames = 0
            t.age_frames += 1
            t.unconfirmed_predict_frames = 0

            if detected_faces and di < len(detected_faces):
                df = detected_faces[di]
                t.confidence = df.confidence
                t.backend = df.backend
                t.left_eye = df.left_eye
                t.right_eye = df.right_eye
                t.nose = df.nose
                t.mouth_left = df.mouth_left
                t.mouth_right = df.mouth_right

            matched_tracks.add(tid)
            matched_detections.add(di)

        for di in sorted(unmatched_detections - matched_detections):
            new_id = next(self._id_counter)
            df = detected_faces[di] if (detected_faces and di < len(detected_faces)) else None
            self.tracks[new_id] = Track(
                track_id=new_id,
                box=detections[di],
                confidence=df.confidence if df else 1.0,
                backend=df.backend if df else settings.detector_backend,
                left_eye=df.left_eye if df else None,
                right_eye=df.right_eye if df else None,
                nose=df.nose if df else None,
                mouth_left=df.mouth_left if df else None,
                mouth_right=df.mouth_right if df else None,
            )

        for tid in unmatched_tracks - matched_tracks:
            t = self.tracks[tid]
            t.missed_frames += 1
            t.age_frames += 1
            # If this track was predicted by optical flow (not a fresh detection),
            # also count those unconfirmed frames towards its deletion threshold
            # to accelerate purging of drifted/ghost tracks.
            if t.unconfirmed_predict_frames > 0:
                t.missed_frames += t.unconfirmed_predict_frames

        expired = [
            tid
            for tid, t in self.tracks.items()
            if t.missed_frames > settings.tracker_max_missed_frames
        ]
        for tid in expired:
            del self.tracks[tid]

        if frame_bgr is not None and frame_bgr.size > 0:
            self._prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        return {tid: t.box for tid, t in self.tracks.items()}

    def predict_inter_frame(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """
        Propagate active face bounding boxes on intermediate frames using
        Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) with centroid velocity fallback.
        """
        if frame_bgr is None or frame_bgr.size == 0 or not self.tracks:
            return []

        im_h, im_w = frame_bgr.shape[:2]
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        predicted_faces: list[DetectedFace] = []

        for t in list(self.tracks.values()):
            x, y, w, h = t.box
            dx, dy = t.velocity

            if self._prev_gray is not None and self._prev_gray.shape == curr_gray.shape:
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(im_w, x + w), min(im_h, y + h)
                if x1 - x0 >= 12 and y1 - y0 >= 12:
                    grid_xs = np.linspace(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0), 4)
                    grid_ys = np.linspace(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0), 4)
                    pts0 = np.array(
                        [[[gx, gy]] for gy in grid_ys for gx in grid_xs],
                        dtype=np.float32,
                    )
                    pts1, status, _ = cv2.calcOpticalFlowPyrLK(
                        self._prev_gray,
                        curr_gray,
                        pts0,
                        None,
                        winSize=(15, 15),
                        maxLevel=2,
                    )
                    if pts1 is not None and status is not None:
                        good_mask = status.flatten() == 1
                        if np.any(good_mask):
                            shifts = (pts1[good_mask] - pts0[good_mask]).reshape(-1, 2)
                            dx = float(np.median(shifts[:, 0]))
                            dy = float(np.median(shifts[:, 1]))
                            t.velocity = (dx, dy)
                        # else: no reliable flow this frame (e.g. a flat/low-
                        # texture region -- a blank wall, or just a very still,
                        # uniformly-lit scene). That's not evidence the track
                        # is wrong, only that LK had nothing to lock onto, so
                        # we keep the last known velocity rather than treat it
                        # as a sign of drift -- only an *un-reconciled* track
                        # (see unconfirmed_predict_frames below) is suspect.

            t.unconfirmed_predict_frames += 1
            # A track that has gone this long without a *real* SCRFD detection
            # confirming it is more likely a drifted ghost than a real face
            # that's merely between detection frames -- stop rendering it
            # (it stays in self.tracks a little longer so a late-arriving
            # real detection can still reclaim it via IoU matching in update()).
            # Fixed: constant 2 here means optical flow can bridge *one* missed
            # frame, but two in a row (or one full failure as above) hides it.
            max_unconfirmed = 2 # was: max(1, settings.detection_interval) + 2
            if t.unconfirmed_predict_frames > max_unconfirmed:
                continue

            nx = int(np.clip(round(x + dx), 0, max(0, im_w - w)))
            ny = int(np.clip(round(y + dy), 0, max(0, im_h - h)))
            nw = min(w, im_w - nx)
            nh = min(h, im_h - ny)
            if nw <= 0 or nh <= 0:
                continue

            shift_x, shift_y = nx - x, ny - y
            t.box = (nx, ny, nw, nh)
            t.age_frames += 1

            def _shift_pt(pt: tuple[int, int] | None) -> tuple[int, int] | None:
                if pt is None:
                    return None
                return (int(pt[0] + shift_x), int(pt[1] + shift_y))

            t.left_eye = _shift_pt(t.left_eye)
            t.right_eye = _shift_pt(t.right_eye)
            t.nose = _shift_pt(t.nose)
            t.mouth_left = _shift_pt(t.mouth_left)
            t.mouth_right = _shift_pt(t.mouth_right)

            predicted_faces.append(
                DetectedFace(
                    x=nx,
                    y=ny,
                    w=nw,
                    h=nh,
                    confidence=t.confidence,
                    backend=f"{t.backend}+optical-track",
                    left_eye=t.left_eye,
                    right_eye=t.right_eye,
                    nose=t.nose,
                    mouth_left=t.mouth_left,
                    mouth_right=t.mouth_right,
                    crop_bgr=frame_bgr[ny : ny + nh, nx : nx + nw].copy(),
                )
            )

        self._prev_gray = curr_gray
        return predicted_faces

    def smooth_face_attributes(self, face_entry: dict[str, Any]) -> None:
        """
        Apply Temporal Exponential Moving Average (EMA) smoothing to a face's
        expression score distribution and estimated age when it matches an
        active track. Eliminates single-frame flicker in video/webcam streams.
        """
        bbox = face_entry.get("bbox")
        if not bbox or not self.tracks:
            return
        box_tuple = (bbox["x"], bbox["y"], bbox["w"], bbox["h"])

        best_track: Track | None = None
        best_iou = 0.0
        for t in self.tracks.values():
            score = _iou(t.box, box_tuple)
            if score > best_iou:
                best_iou = score
                best_track = t

        if best_track is None or best_iou < settings.tracker_iou_threshold:
            return

        alpha = float(np.clip(settings.temporal_ema_alpha, 0.05, 1.0))

        expr = face_entry.get("expression")
        if expr and isinstance(expr.get("scores"), dict):
            curr_scores: dict[str, float] = expr["scores"]
            if best_track.ema_expression_scores is None:
                best_track.ema_expression_scores = dict(curr_scores)
            else:
                smoothed = {}
                for k, v in curr_scores.items():
                    prev_v = best_track.ema_expression_scores.get(k, v)
                    smoothed[k] = alpha * float(v) + (1.0 - alpha) * float(prev_v)
                total = sum(smoothed.values()) or 1.0
                best_track.ema_expression_scores = {k: v / total for k, v in smoothed.items()}
            smoothed_res = result_from_scores_dict(best_track.ema_expression_scores)
            face_entry["expression"] = smoothed_res.to_dict()

        age_info = face_entry.get("age")
        if age_info and age_info.get("enabled") and age_info.get("estimated_age") is not None:
            curr_age = float(age_info["estimated_age"])
            if best_track.ema_age is None:
                best_track.ema_age = curr_age
            else:
                best_track.ema_age = alpha * curr_age + (1.0 - alpha) * best_track.ema_age
            smoothed_age_int = int(round(best_track.ema_age))
            age_info["estimated_age"] = smoothed_age_int
            age_info["estimated_age_range"] = _bucket_for_age(smoothed_age_int)

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "track_id": t.track_id,
                "box": t.box,
                "age_frames": t.age_frames,
                "missed_frames": t.missed_frames,
            }
            for t in self.tracks.values()
        ]
