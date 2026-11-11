"""
AI Face Intelligence System - FastAPI Application Entry Point

Production-ready configuration with authentication, security headers,
structured logging, and health checks.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.routes import router
from app.api.ws import router as ws_router
from app.core.config import settings
from app.security.auth import load_api_keys_from_env
from app.security.model_integrity import verify_model_integrity, generate_manifest, save_manifest

# ============================================================
# Structured Logging Setup
# ============================================================
class StructuredLogger:
    """JSON-structured logger for production."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log(self, level: int, message: str, **kwargs):
        # Create structured message with key=value pairs
        extra_data = {
            "timestamp": time.time(),
            "level": logging.getLevelName(level),
            "service": "face-intelligence",
            **kwargs,
        }
        # Format as "message key1=value1 key2=value2 ..."
        extra_str = " ".join(f"{k}={v}" for k, v in extra_data.items() if k not in ("timestamp", "level", "service"))
        formatted_msg = f"{message} {extra_str}" if extra_str else message
        self.logger.log(level, formatted_msg, extra={"structured": extra_data})

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)


# Configure root logger
log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(message)s",
    stream=sys.stdout,
)

# Suppress noisy loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Application logger
app_logger = StructuredLogger("face_intelligence.app")


# ============================================================
# Rate Limiter
# ============================================================
def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting (IP + API key if present)."""
    # Try to get API key from header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key[:16]}"
    # Fall back to IP
    return get_remote_address(request)


limiter = Limiter(
    key_func=get_client_identifier,
    default_limits=[],
    storage_uri=settings.redis_url,
)


# ============================================================
# Application Lifecycle
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    app_logger.info("Starting Face Intelligence System", version=settings.app_version, environment=settings.environment)

    # Load API keys from environment
    key_count = load_api_keys_from_env()
    app_logger.info("Loaded API keys from environment", count=key_count)

    # Validate production configuration
    if settings.environment == "production":
        issues = settings.validate_production_config()
        if issues:
            for issue in issues:
                app_logger.warning("Production config issue", issue=issue)

    # Initialize encryption
    if settings.encryption_key:
        try:
            from app.security.crypto import initialize_encryption
            master_key = base64.b64decode(settings.encryption_key.get_secret_value())
            salt = base64.b64decode(settings.encryption_salt.get_secret_value()) if settings.encryption_salt else None
            initialize_encryption(master_key, salt)
            app_logger.info("Encryption initialized successfully")
        except Exception as e:
            app_logger.error("Failed to initialize encryption", error=str(e))
            if settings.environment == "production":
                raise
    else:
        app_logger.warning("Encryption not initialized - FACE_ENCRYPTION_KEY not set")

    # Verify model integrity
    if settings.environment == "production":
        try:
            ok, issues = verify_model_integrity()
            if not ok:
                for issue in issues:
                    app_logger.error("Model integrity check failed", issue=issue)
                # In production, fail fast on model integrity issues
                raise RuntimeError(f"Model integrity verification failed: {len(issues)} issues")
            else:
                app_logger.info("Model integrity verification passed")
        except FileNotFoundError:
            app_logger.warning("Model manifest not found; run model_integrity.generate to create one")
        except Exception as e:
            app_logger.error("Model integrity check error", error=str(e))
            if settings.environment == "production":
                raise

    yield

    # Shutdown
    app_logger.info("Shutting down Face Intelligence System")


import base64  # for lifespan encryption init


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title=settings.app_name,
    description=(
        "Local-first face detection, consent-based recognition, expression "
        "classification, and approximate age estimation. See /docs for the "
        "interactive API reference, and docs/PRIVACY.md in the repo for the "
        "consent and data-retention model."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" or settings.debug else None,
    redoc_url="/redoc" if settings.environment != "production" or settings.debug else None,
    openapi_url="/openapi.json" if settings.environment != "production" or settings.debug else None,
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    start_time = time.time()

    # Generate request ID
    request_id = request.headers.get("X-Request-ID") or secrets.token_hex(8)
    request.state.request_id = request_id

    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{(time.time() - start_time) * 1000:.2f}ms"

    # CSP for API
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"

    # HSTS for production
    if settings.environment == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests with structured data."""
    if not settings.log_requests:
        return await call_next(request)

    start_time = time.time()
    client_ip = get_remote_address(request)

    app_logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        client_ip=client_ip,
        request_id=getattr(request.state, "request_id", None),
    )

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000
    app_logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
        request_id=getattr(request.state, "request_id", None),
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler - never leak stack traces in production."""
    request_id = getattr(request.state, "request_id", None)

    app_logger.error(
        "Unhandled exception",
        path=request.url.path,
        error_type=type(exc).__name__,
        error=str(exc),
        request_id=request_id,
        exc_info=settings.debug,
    )

    if settings.debug:
        raise

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
        },
    )


# Include routers
app.include_router(router)
app.include_router(ws_router)


# ============================================================
# Public Endpoints (No Auth Required)
# ============================================================
@app.get("/")
def root():
    return {
        "message": "AI Face Intelligence System API",
        "version": settings.app_version,
        "docs": "/docs" if settings.environment != "production" or settings.debug else "disabled",
    }


@app.get("/health")
def health():
    """Health check endpoint - no auth required."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.environment,
    }


@app.get("/ready")
def readiness():
    """
    Readiness check - verifies models and dependencies are loaded.
    Used by Kubernetes readiness probe.
    """
    from app.core.onnx_runtime import get_active_execution_providers
    from app.services import detection, recognition, expressions, age_estimation

    checks = {
        "api": "ok",
        "models_dir": settings.MODELS_DIR.exists(),
        "data_dir": settings.DATA_DIR.exists(),
        "execution_providers": get_active_execution_providers(),
    }

    # Check each model
    try:
        dummy = __import__("numpy").zeros((112, 112, 3), dtype=__import__("numpy").uint8)
        detection.detect_faces(dummy)
        checks["detection"] = "ok"
    except Exception as e:
        checks["detection"] = f"error: {e}"

    try:
        recognition._embed(dummy)
        checks["recognition"] = "ok"
    except Exception as e:
        checks["recognition"] = f"error: {e}"

    try:
        expressions.classify_expression(dummy)
        checks["expression"] = "ok"
    except Exception as e:
        checks["expression"] = f"error: {e}"

    try:
        age_estimation.check_model()
        checks["age_estimation"] = "ok"
    except Exception as e:
        checks["age_estimation"] = f"error: {e}"

    all_ready = all(v == "ok" or (isinstance(v, list) and v) for v in checks.values())

    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "ready": all_ready,
            "checks": checks,
        },
    )


import secrets  # for request ID generation