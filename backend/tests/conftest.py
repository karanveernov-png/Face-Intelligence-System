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
import time
from fastapi.testclient import TestClient

from app.main import app
from app.security.auth import create_api_key, Role


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


@pytest.fixture
def test_api_key():
    """Create a test API key with Operator role for testing."""
    plaintext, record = create_api_key(
        name="test-operator",
        role=Role.OPERATOR,
        expires_in_days=1,
        metadata={"test": True}
    )
    return plaintext


@pytest.fixture
def test_admin_key():
    """Create a test API key with Admin role for testing."""
    plaintext, record = create_api_key(
        name="test-admin",
        role=Role.ADMIN,
        expires_in_days=1,
        metadata={"test": True}
    )
    return plaintext


@pytest.fixture
def test_service_key():
    """Create a test API key with Service role for testing."""
    plaintext, record = create_api_key(
        name="test-service",
        role=Role.SERVICE,
        expires_in_days=1,
        metadata={"test": True}
    )
    return plaintext


@pytest.fixture
def auth_headers(test_api_key):
    """Headers with API key for authenticated requests."""
    return {"X-API-Key": test_api_key}


@pytest.fixture
def admin_headers(test_admin_key):
    """Headers with Admin API key for authenticated requests."""
    return {"X-API-Key": test_admin_key}


@pytest.fixture
def service_headers(test_service_key):
    """Headers with Service API key for authenticated requests."""
    return {"X-API-Key": test_service_key}


@pytest.fixture
def client():
    """Test client without authentication (for public endpoints)."""
    return TestClient(app)


@pytest.fixture
def auth_client(auth_headers):
    """Test client with Operator authentication."""
    return TestClient(app, headers=auth_headers)


@pytest.fixture
def admin_client(admin_headers):
    """Test client with Admin authentication."""
    return TestClient(app, headers=admin_headers)


@pytest.fixture
def service_client(service_headers):
    """Test client with Service authentication."""
    return TestClient(app, headers=service_headers)
