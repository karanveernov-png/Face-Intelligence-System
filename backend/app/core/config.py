"""
Configuration Management

Environment-driven settings with validation using Pydantic Settings.
Supports development and production profiles with secure defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.types import SecretStr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'
GALLERY_DIR = DATA_DIR / 'gallery'
EMBEDDINGS_DIR = DATA_DIR / 'embeddings'
TMP_UPLOAD_DIR = DATA_DIR / 'tmp_uploads'

# Ensure directories exist
for d in (MODELS_DIR, DATA_DIR, GALLERY_DIR, EMBEDDINGS_DIR, TMP_UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============================================================
    # Application
    # ============================================================
    app_name: str = "Face Intelligence System"
    app_version: str = "0.3.0"
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    debug: bool = Field(default=False, validation_alias="DEBUG")

    # ============================================================
    # Security
    # ============================================================
    # API Key salt for hashing (generate with: python -c "import secrets; print(secrets.token_hex(32))")
    api_key_salt: SecretStr = Field(
        default=SecretStr("dev-salt-change-in-production"),
        validation_alias="API_KEY_SALT"
    )

    # Master encryption key for biometric data (base64-encoded 32 bytes)
    # Generate with: python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
    encryption_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias="FACE_ENCRYPTION_KEY"
    )

    # Encryption salt (base64-encoded 32 bytes)
    encryption_salt: Optional[SecretStr] = Field(
        default=None,
        validation_alias="FACE_ENCRYPTION_SALT"
    )

    # CORS origins (comma-separated)
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        validation_alias="CORS_ORIGINS"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, validation_alias="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=60, validation_alias="RATE_LIMIT_RPM")
    rate_limit_burst: int = Field(default=10, validation_alias="RATE_LIMIT_BURST")

    # Redis for distributed rate limiting
    redis_url: Optional[str] = Field(default=None, validation_alias="REDIS_URL")

    # ============================================================
    # Model Configuration
    # ============================================================
    # 1. Detection & Alignment
    detector_backend: str = "scrfd-10g-int8"
    detector_model_file: str = "scrfd_10g_bnkps_int8.onnx"
    detector_input_size: tuple[int, int] = (640, 640)
    detection_confidence_threshold: float = 0.5
    detection_nms_threshold: float = 0.4

    # 2. Face Recognition
    recognition_model: str = "ArcFace-R50-INT8"
    recognition_model_file: str = "w600k_r50_int8.onnx"
    recognition_distance_threshold: float = 0.68
    use_5pt_similarity_alignment: bool = True

    # 3. Emotion Analysis
    expression_model: str = "HSEmotion-ENetB0-INT8"
    expression_model_file: str = "models/enet_b0_8_best_vgaf.onnx"
    expression_min_confidence: float = 0.3

    # 4. Age & Gender
    age_model: str = "GenderAge+FairFace-Ensemble"
    age_model_file: str = "models/fairface.onnx"
    gender_model_file: str = "models/fairface.onnx"
    landmark_predictor_file: str = "shape_predictor_5_face_landmarks.dat"
    age_estimation_enabled: bool = True

    # ============================================================
    # Pipeline Performance
    # ============================================================
    auto_int8_quantize: bool = True
    detection_interval: int = 3
    temporal_ema_alpha: float = 0.4
    batch_inference_enabled: bool = True
    onnx_execution_providers: tuple[str, ...] = (
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    )
    onnx_intra_op_threads: int = Field(
        default_factory=lambda: max(1, min(os.cpu_count() or 4, 8))
    )
    onnx_inter_op_threads: int = 1

    # ============================================================
    # Tracker & Media Upload
    # ============================================================
    tracker_max_missed_frames: int = 4
    tracker_iou_threshold: float = 0.3
    max_upload_size_mb: int = 25
    max_decoded_image_pixels: int = 40_000_000
    max_decoded_image_dimension_px: int = 12_000
    allowed_image_ext: tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    allowed_video_ext: tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv')
    frame_skip: int = 1
    max_video_frames_processed: int = 300
    retain_enrollment_images: bool = False

    # ============================================================
    # Biometric Privacy
    # ============================================================
    consent_required: bool = True
    consent_version: str = "1.0"
    retention_days: int = 365
    auto_delete_expired: bool = True

    # ============================================================
    # Logging
    # ============================================================
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    log_requests: bool = True

    # ============================================================
    # Validators
    # ============================================================
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    @field_validator("cors_origins")
    @classmethod
    def parse_cors_origins(cls, v: str) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("detector_input_size")
    @classmethod
    def parse_detector_input_size(cls, v: str | tuple[int, int]) -> tuple[int, int]:
        if isinstance(v, str):
            parts = v.split(",")
            if len(parts) != 2:
                raise ValueError("detector_input_size must be 'width,height'")
            return (int(parts[0]), int(parts[1]))
        return v

    @field_validator("onnx_execution_providers")
    @classmethod
    def parse_execution_providers(cls, v: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(v, str):
            return tuple(p.strip() for p in v.split(",") if p.strip())
        return v

    @field_validator("allowed_image_ext", "allowed_video_ext")
    @classmethod
    def parse_extensions(cls, v: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(v, str):
            return tuple(ext.strip() for ext in v.split(",") if ext.strip())
        return v

    def get_cors_origins_list(self) -> list[str]:
        """Get CORS origins as list."""
        return self.parse_cors_origins(self.cors_origins)

    def validate_production_config(self) -> list[str]:
        """
        Validate configuration for production deployment.
        Returns list of warnings/errors.
        """
        issues = []

        if self.environment == "production":
            if self.debug:
                issues.append("DEBUG must be false in production")

            if self.api_key_salt.get_secret_value() == "dev-salt-change-in-production":
                issues.append("API_KEY_SALT must be changed from default in production")

            if not self.encryption_key:
                issues.append("FACE_ENCRYPTION_KEY must be set in production")

            if not self.encryption_salt:
                issues.append("FACE_ENCRYPTION_SALT must be set in production")

            if "localhost" in str(self.cors_origins):
                issues.append("CORS_ORIGINS should not contain localhost in production")

            if not self.rate_limit_enabled:
                issues.append("RATE_LIMIT_ENABLED should be true in production")

            if not self.redis_url and self.environment == "production":
                issues.append("REDIS_URL should be set for distributed rate limiting in production")

        return issues


# Global settings instance
settings = Settings()

# Validate on import (will raise on critical issues)
if settings.environment == "production":
    issues = settings.validate_production_config()
    if issues:
        import logging
        logger = logging.getLogger("face_intelligence.config")
        for issue in issues:
            logger.warning("Production config issue: %s", issue)