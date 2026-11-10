"""
Shared pytest fixtures.

Deliberately uses only SYNTHETIC, procedurally-generated images — never a
downloaded photo of a real person — because this repository does not ship
with a consented face dataset (see docs/TROUBLESHOOTING.md, "Why are there
no real face images in samples/?"). These fixtures are sufficient to test
error handling, I/O plumbing, the tracker, and the recognition pipeline's
match-logic end to end (same input -> same embedding -> matches itself),
but they CANNOT validate true-positive face-detection accuracy. That
requires a real, consented photo supplied by whoever runs the system — see
tests/test_detection_requires_real_face.py, which is skipped by default for
exactly this reason.
"""
from __future__ import annotations

import numpy as np
import pytest
import cv2


@pytest.fixture
def blank_image() -> np.ndarray:
    """A plain gray rectangle on black — contains no face-like structure."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (590, 430), (120, 120, 120), -1)
    return img


@pytest.fixture
def noise_image() -> np.ndarray:
    """Random noise — a harder negative case than a flat rectangle."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def blank_image_path(tmp_path, blank_image) -> str:
    path = tmp_path / "blank.jpg"
    cv2.imwrite(str(path), blank_image)
    return str(path)


@pytest.fixture
def invalid_image_bytes() -> bytes:
    """Bytes that are not a valid image of any kind."""
    return b"this is not an image file, just plain bytes \x00\x01\x02"
