import numpy as np
import pytest

from app.services import recognition
from app.services.detection import DetectedFace


def _fake_face(left_eye=None, right_eye=None, x=100, y=100, w=100, h=100):
    return DetectedFace(
        x=x, y=y, w=w, h=h, confidence=0.9, backend="centerface",
        left_eye=left_eye, right_eye=right_eye,
        crop_bgr=np.full((h, w, 3), 50, dtype=np.uint8),
    )


def test_align_falls_back_to_plain_crop_without_landmarks(blank_image):
    face = _fake_face(left_eye=None, right_eye=None)
    result = recognition.align_face(blank_image, face)
    assert np.array_equal(result, face.crop_bgr)


def test_align_with_level_eyes_is_a_near_noop_rotation(blank_image):
    # Eyes already on a horizontal line -> rotation angle should be ~0,
    # so the aligned crop should closely match a plain crop of the same box.
    face = _fake_face(left_eye=(120, 140), right_eye=(180, 140))
    aligned = recognition.align_face(blank_image, face)
    plain_crop = blank_image[face.y:face.y + face.h, face.x:face.x + face.w]
    assert aligned.shape == plain_crop.shape
    # Allow small pixel differences from the (near-zero-angle) affine warp's
    # interpolation; this should still be close to identical.
    diff = np.abs(aligned.astype(int) - plain_crop.astype(int)).mean()
    assert diff < 5.0


def test_align_with_tilted_eyes_produces_correctly_shaped_output(blank_image):
    # A visibly tilted eye line (not level) -> still must return a valid,
    # correctly-shaped crop rather than crashing or returning empty data.
    face = _fake_face(left_eye=(110, 160), right_eye=(190, 120))
    aligned = recognition.align_face(blank_image, face)
    assert aligned.shape == (face.h, face.w, 3)
    assert aligned.size > 0


def test_align_degenerate_box_falls_back_to_plain_crop():
    # A face box positioned entirely outside the frame after rotation would
    # produce an empty array -- must fall back rather than propagate that.
    tiny_image = np.zeros((50, 50, 3), dtype=np.uint8)
    face = _fake_face(left_eye=(5, 5), right_eye=(10, 45), x=200, y=200, w=50, h=50)
    result = recognition.align_face(tiny_image, face)
    assert result.size > 0
    assert np.array_equal(result, face.crop_bgr)
