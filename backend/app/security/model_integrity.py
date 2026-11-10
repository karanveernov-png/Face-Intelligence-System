"""
Model Integrity Verification Module

Provides SHA-256 manifest-based verification for ONNX model files
to detect supply chain tampering and unauthorized modifications.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.config import settings, MODELS_DIR

logger = logging.getLogger("face_intelligence.model_integrity")


@dataclass
class ModelManifestEntry:
    """Single model file entry in the manifest."""
    filename: str
    sha256: str
    size: int
    purpose: str
    required: bool = True


@dataclass
class ModelManifest:
    """Complete manifest of all model files with their expected hashes."""
    version: str = "1.0"
    generated_at: str = ""
    models: dict[str, ModelManifestEntry] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "models": {
                name: {
                    "filename": entry.filename,
                    "sha256": entry.sha256,
                    "size": entry.size,
                    "purpose": entry.purpose,
                    "required": entry.required,
                }
                for name, entry in self.models.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelManifest":
        manifest = cls(
            version=data.get("version", "1.0"),
            generated_at=data.get("generated_at", ""),
        )
        for name, entry_data in data.get("models", {}).items():
            manifest.models[name] = ModelManifestEntry(**entry_data)
        return manifest


# Expected model files and their purposes
MODEL_REGISTRY = {
    "scrfd_10g_bnkps_int8.onnx": {
        "purpose": "Face detection & 5-pt alignment (SCRFD-10G INT8)",
        "required": True,
    },
    "w600k_r50_int8.onnx": {
        "purpose": "Face recognition embeddings (ArcFace-R50 INT8)",
        "required": True,
    },
    "enet_b0_8_best_vgaf.onnx": {
        "purpose": "Expression classification (HSEmotion-ENetB0 INT8)",
        "required": True,
    },
    "fairface.onnx": {
        "purpose": "Age & gender estimation (FairFace ensemble)",
        "required": True,
    },
    "mobilefacenet_arcface_int8.onnx": {
        "purpose": "Face recognition fallback (MobileFaceNet INT8)",
        "required": False,
    },
    "shape_predictor_5_face_landmarks.dat": {
        "purpose": "Dlib 5-point landmark predictor (legacy age model)",
        "required": False,
    },
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_manifest(models_dir: Optional[Path] = None) -> ModelManifest:
    """
    Generate a manifest of all model files with their current hashes.

    This should be run once after deploying trusted models to create
    the reference manifest for future integrity checks.
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    manifest = ModelManifest()
    from datetime import datetime
    manifest.generated_at = datetime.utcnow().isoformat() + "Z"

    for filename, meta in MODEL_REGISTRY.items():
        filepath = models_dir / filename
        if filepath.exists():
            sha256 = compute_sha256(filepath)
            size = filepath.stat().st_size
            manifest.models[filename] = ModelManifestEntry(
                filename=filename,
                sha256=sha256,
                size=size,
                purpose=meta["purpose"],
                required=meta["required"],
            )
            logger.info("Manifest entry: %s -> %s", filename, sha256[:16] + "...")
        elif meta["required"]:
            logger.warning("Required model file missing: %s", filename)
        else:
            logger.info("Optional model file not present: %s", filename)

    return manifest


def save_manifest(manifest: ModelManifest, output_path: Optional[Path] = None) -> Path:
    """Save manifest to JSON file."""
    if output_path is None:
        output_path = MODELS_DIR / "model_manifest.json"

    with open(output_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info("Saved model manifest to %s", output_path)
    return output_path


def load_manifest(manifest_path: Optional[Path] = None) -> ModelManifest:
    """Load manifest from JSON file."""
    if manifest_path is None:
        manifest_path = MODELS_DIR / "model_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Model manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        data = json.load(f)

    return ModelManifest.from_dict(data)


def verify_model_integrity(
    manifest_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> tuple[bool, list[str]]:
    """
    Verify all model files against the manifest.

    Returns:
        Tuple of (all_verified, list_of_issues)
    """
    if manifest_path is None:
        manifest_path = MODELS_DIR / "model_manifest.json"
    if models_dir is None:
        models_dir = MODELS_DIR

    if not manifest_path.exists():
        issues = [f"Model manifest not found at {manifest_path}"]
        logger.error("Model manifest not found")
        return False, issues

    manifest = load_manifest(manifest_path)
    issues = []

    for name, entry in manifest.models.items():
        filepath = models_dir / name

        if not filepath.exists():
            if entry.required:
                issues.append(f"Required model missing: {name} ({entry.purpose})")
                logger.error("Required model missing: %s", name)
            else:
                logger.info("Optional model not present: %s", name)
            continue

        # Check file size
        actual_size = filepath.stat().st_size
        if actual_size != entry.size:
            issues.append(f"Size mismatch for {name}: expected {entry.size}, got {actual_size}")
            logger.error("Size mismatch for %s", name)
            continue

        # Check SHA-256 hash
        actual_hash = compute_sha256(filepath)
        if actual_hash != entry.sha256:
            issues.append(f"HASH MISMATCH for {name}: expected {entry.sha256[:16]}..., got {actual_hash[:16]}...")
            logger.error("HASH MISMATCH for %s", name)
            continue

        logger.info("Model verified: %s (%s)", name, entry.purpose)

    all_verified = len(issues) == 0
    if all_verified:
        logger.info("All model integrity checks passed")
    else:
        logger.error("Model integrity verification FAILED: %d issues", len(issues))
        for issue in issues:
            logger.error("  - %s", issue)

    return all_verified, issues


def verify_model_on_load(filename: str) -> bool:
    """
    Verify a specific model file on load.
    Called by the ONNX runtime loader to ensure model hasn't been tampered with.
    """
    manifest_path = MODELS_DIR / "model_manifest.json"
    if not manifest_path.exists():
        # No manifest yet - log warning but allow load (first run)
        logger.warning("No model manifest found; skipping integrity check for %s", filename)
        return True

    try:
        manifest = load_manifest(manifest_path)
        if filename not in manifest.models:
            # Not in manifest - could be a new/unknown model
            logger.warning("Model %s not in manifest; skipping integrity check", filename)
            return True

        entry = manifest.models[filename]
        filepath = MODELS_DIR / filename

        if not filepath.exists():
            logger.error("Model file %s not found", filename)
            return False

        actual_hash = compute_sha256(filepath)
        if actual_hash != entry.sha256:
            logger.error("INTEGRITY CHECK FAILED for %s: hash mismatch", filename)
            return False

        return True

    except Exception as e:
        logger.error("Integrity check error for %s: %s", filename, e)
        return False


# CLI entry point for generating manifest
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        manifest = generate_manifest()
        save_manifest(manifest)
        print("Manifest generated successfully")
    elif len(sys.argv) > 1 and sys.argv[1] == "verify":
        ok, issues = verify_model_integrity()
        if ok:
            print("All models verified")
            sys.exit(0)
        else:
            print(f"Verification failed: {len(issues)} issues")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
    else:
        print("Usage: python -m app.security.model_integrity [generate|verify]")
        sys.exit(1)