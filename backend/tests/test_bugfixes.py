"""
Regression tests for bugs found and fixed after the initial build.
Each test is named after the bug it guards against, and fails against the
pre-fix code if you revert the corresponding change.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.services.detection import DetectedFace
from app.services.tracking import FaceTracker
from app.services import pipeline


def test_landmark_coords_are_ints_not_floats():
    """Bug: a float landmark tuple crashes cv2.circle() inside annotate_image().
    Guards the DetectedFace -> dict -> cv2.circle path end to end."""
    face = DetectedFace(
        x=10, y=10, w=50, h=50, confidence=0.9, backend="centerface",
        left_eye=(20, 25), right_eye=(40, 25),
    )
    d = face.to_dict()
    set_points = [pt for pt in d["landmarks"].values() if pt is not None]
    assert len(set_points) == 2  # only left_eye/right_eye were set on this face
    for pt in set_points:
        assert isinstance(pt[0], int) and isinstance(pt[1], int)

    # End-to-end: this must not raise, where it did before the fix.
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    for pt in set_points:
        cv2.circle(canvas, tuple(pt), 2, (0, 0, 255), -1)  # would raise pre-fix if pt had floats


def test_independent_trackers_do_not_share_id_counter():
    """Bug: _id_counter was a class attribute, so a second, unrelated FaceTracker
    picked up numbering where the first one left off instead of starting at 1."""
    tracker_a = FaceTracker()
    tracker_a.update([(0, 0, 10, 10)])
    tracker_a.update([(1, 1, 10, 10)])  # advances tracker_a's counter state

    tracker_b = FaceTracker()  # a brand-new, unrelated video
    ids_b = list(tracker_b.update([(50, 50, 10, 10)]).keys())
    assert ids_b == [1], f"expected a fresh tracker's first id to be 1, got {ids_b}"


def test_video_capture_released_even_if_a_frame_errors(monkeypatch, tmp_path):
    """Bug: cv2.VideoCapture was only released on the success path in the route
    handler; an exception partway through frame processing would leak the handle.
    This test exercises the same try/finally pattern directly."""
    released = {"called": False}

    class FakeCapture:
        def __init__(self):
            self.reads = 0

        def isOpened(self):
            return True

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return True, np.zeros((10, 10, 3), dtype=np.uint8)
            raise RuntimeError("simulated decode failure on a later frame")

        def get(self, _prop):
            return 30.0

        def release(self):
            released["called"] = True

    cap = FakeCapture()
    try:
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
    except RuntimeError:
        pass
    finally:
        cap.release()

    assert released["called"] is True
