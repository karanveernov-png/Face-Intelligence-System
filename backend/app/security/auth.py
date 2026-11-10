"""
Authentication and Authorization Module

Provides API key authentication, role-based access control, and secure
credential management for the Face Intelligence System.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fastapi import Depends, HTTPException, Header, Request, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from app.core.config import settings

logger = logging.getLogger("face_intelligence.auth")

# API Key header and query parameter names
API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_QUERY_NAME = "api_key"

# WebSocket authentication uses query parameter
WS_AUTH_QUERY_NAME = "token"

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY_NAME, auto_error=False)


class Role(str, Enum):
    """Role definitions for RBAC."""
    ADMIN = "admin"           # Full access: enroll, list, delete, configure
    OPERATOR = "operator"     # Analysis access: analyze, recognize
    VIEWER = "viewer"         # Read-only: list, status
    SERVICE = "service"       # Service-to-service: full API access


class Permission(str, Enum):
    """Granular permissions."""
    # Face management
    ENROLL_FACE = "enroll_face"
    LIST_FACES = "list_faces"
    DELETE_FACE = "delete_face"
    # Analysis
    ANALYZE_IMAGE = "analyze_image"
    ANALYZE_VIDEO = "analyze_video"
    ANALYZE_STREAM = "analyze_stream"
    # Recognition
    RECOGNIZE = "recognize"
    # System
    MODEL_STATUS = "model_status"
    HEALTH_CHECK = "health_check"
    ADMIN_CONFIG = "admin_config"


# Role -> Permission mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: {
        Permission.ENROLL_FACE,
        Permission.LIST_FACES,
        Permission.DELETE_FACE,
        Permission.ANALYZE_IMAGE,
        Permission.ANALYZE_VIDEO,
        Permission.ANALYZE_STREAM,
        Permission.RECOGNIZE,
        Permission.MODEL_STATUS,
        Permission.HEALTH_CHECK,
        Permission.ADMIN_CONFIG,
    },
    Role.OPERATOR: {
        Permission.ANALYZE_IMAGE,
        Permission.ANALYZE_VIDEO,
        Permission.ANALYZE_STREAM,
        Permission.RECOGNIZE,
        Permission.MODEL_STATUS,
        Permission.HEALTH_CHECK,
    },
    Role.VIEWER: {
        Permission.LIST_FACES,
        Permission.MODEL_STATUS,
        Permission.HEALTH_CHECK,
    },
    Role.SERVICE: {
        Permission.ENROLL_FACE,
        Permission.LIST_FACES,
        Permission.DELETE_FACE,
        Permission.ANALYZE_IMAGE,
        Permission.ANALYZE_VIDEO,
        Permission.ANALYZE_STREAM,
        Permission.RECOGNIZE,
        Permission.MODEL_STATUS,
        Permission.HEALTH_CHECK,
    },
}


@dataclass
class APIKeyRecord:
    """Stored API key record (hashed)."""
    key_hash: str
    name: str
    role: Role
    created_at: float
    expires_at: Optional[float] = None
    revoked: bool = False
    last_used_at: Optional[float] = None
    metadata: Optional[dict] = None


@dataclass
class AuthContext:
    """Authentication context for request."""
    api_key_name: str
    role: Role
    permissions: set[Permission]
    key_record: APIKeyRecord


# In-memory API key store (in production, use Redis or database)
_api_keys: dict[str, APIKeyRecord] = {}


def _hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA-256 with salt."""
    salt = settings.api_key_salt.get_secret_value().encode()
    return hashlib.pbkdf2_hmac(
        "sha256",
        api_key.encode(),
        salt,
        100_000
    ).hex()


def _verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """Constant-time verification of API key against stored hash."""
    provided_hash = _hash_api_key(provided_key)
    return hmac.compare_digest(provided_hash, stored_hash)


def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key.
    Returns (plaintext_key, key_hash).
    """
    # Format: fi_<random> for easy identification
    plaintext = f"fi_{secrets.token_urlsafe(32)}"
    key_hash = _hash_api_key(plaintext)
    return plaintext, key_hash


def create_api_key(
    name: str,
    role: Role,
    expires_in_days: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> tuple[str, APIKeyRecord]:
    """
    Create a new API key record.
    Returns (plaintext_key, record).
    """
    plaintext, key_hash = generate_api_key()

    expires_at = None
    if expires_in_days:
        expires_at = time.time() + (expires_in_days * 86400)

    record = APIKeyRecord(
        key_hash=key_hash,
        name=name,
        role=role,
        created_at=time.time(),
        expires_at=expires_at,
        metadata=metadata or {},
    )

    _api_keys[key_hash] = record
    logger.info("Created API key '%s' with role %s", name, role.value)
    return plaintext, record


def get_api_key_record(key_hash: str) -> Optional[APIKeyRecord]:
    """Get API key record by hash."""
    return _api_keys.get(key_hash)


def revoke_api_key(key_hash: str) -> bool:
    """Revoke an API key."""
    if key_hash in _api_keys:
        _api_keys[key_hash].revoked = True
        logger.info("Revoked API key: %s", key_hash[:16] + "...")
        return True
    return False


def list_api_keys() -> list[dict]:
    """List all API keys (without hashes)."""
    return [
        {
            "name": record.name,
            "role": record.role.value,
            "created_at": record.created_at,
            "expires_at": record.expires_at,
            "revoked": record.revoked,
            "last_used_at": record.last_used_at,
            "metadata": record.metadata,
        }
        for record in _api_keys.values()
    ]


def load_api_keys_from_env() -> int:
    """
    Load API keys from environment variable.
    Format: FACE_API_KEYS="name:role:key_hash,name2:role2:key_hash2"
    Returns count of loaded keys.
    """
    env_keys = os.environ.get("FACE_API_KEYS", "")
    if not env_keys:
        return 0

    count = 0
    for entry in env_keys.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            logger.warning("Invalid FACE_API_KEYS entry: %s", entry)
            continue
        name, role_str, key_hash = parts
        try:
            role = Role(role_str)
        except ValueError:
            logger.warning("Invalid role in FACE_API_KEYS: %s", role_str)
            continue

        record = APIKeyRecord(
            key_hash=key_hash,
            name=name,
            role=role,
            created_at=time.time(),
        )
        _api_keys[key_hash] = record
        count += 1

    logger.info("Loaded %d API keys from environment", count)
    return count


def require_permission(permission: Permission):
    """Dependency factory for requiring a specific permission."""
    async def check_permission(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if permission not in auth.permissions:
            logger.warning(
                "Permission denied: %s required, user has %s",
                permission.value,
                ", ".join(p.value for p in auth.permissions)
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value} required"
            )
        return auth
    return check_permission


async def get_api_key(
    request: Request,
    api_key_header: Optional[str] = Depends(api_key_header),
    api_key_query: Optional[str] = Depends(api_key_query),
) -> str:
    """
    Extract API key from header or query parameter.
    Header takes precedence over query parameter.
    """
    # Check header first
    if api_key_header:
        return api_key_header

    # Check query parameter
    if api_key_query:
        logger.warning("API key provided in query parameter (less secure)")
        return api_key_query

    # Check for Authorization header (Bearer token)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="API key required. Provide via X-API-Key header, Authorization: Bearer, or api_key query parameter.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_auth_context(api_key: str = Depends(get_api_key)) -> AuthContext:
    """
    Validate API key and return authentication context.
    """
    key_hash = _hash_api_key(api_key)
    record = _api_keys.get(key_hash)

    if not record:
        logger.warning("Invalid API key attempted: %s...", api_key[:8] + "...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Check revocation
    if record.revoked:
        logger.warning("Revoked API key attempted: %s", record.name)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has been revoked"
        )

    # Check expiration
    if record.expires_at and time.time() > record.expires_at:
        logger.warning("Expired API key attempted: %s", record.name)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired"
        )

    # Update last used timestamp
    record.last_used_at = time.time()

    permissions = ROLE_PERMISSIONS.get(record.role, set())

    return AuthContext(
        api_key_name=record.name,
        role=record.role,
        permissions=permissions,
        key_record=record,
    )


async def get_optional_auth_context(
    request: Request,
    api_key_header: Optional[str] = Depends(api_key_header),
    api_key_query: Optional[str] = Depends(api_key_query),
) -> Optional[AuthContext]:
    """
    Get auth context if API key provided, otherwise return None.
    For endpoints that support both authenticated and anonymous access.
    """
    api_key = None
    if api_key_header:
        api_key = api_key_header
    elif api_key_query:
        api_key = api_key_query
    else:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]

    if not api_key:
        return None

    try:
        return await get_auth_context(api_key)
    except HTTPException:
        return None


# Public endpoint permissions (no auth required)
PUBLIC_PERMISSIONS = {
    Permission.HEALTH_CHECK,
    Permission.MODEL_STATUS,
}

def is_public_endpoint(permission: Permission) -> bool:
    """Check if an endpoint permission is public."""
    return permission in PUBLIC_PERMISSIONS


# Initialize default keys from environment on module load
load_api_keys_from_env()