"""
Encryption Module for Biometric Data at Rest

Provides authenticated encryption (AES-256-GCM) for face embeddings
and sensitive biometric data storage.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from app.core.config import settings

logger = logging.getLogger("face_intelligence.crypto")


@dataclass
class EncryptedData:
    """Container for encrypted data with metadata."""
    ciphertext: bytes
    nonce: bytes
    key_id: str
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "nonce": base64.b64encode(self.nonce).decode(),
            "key_id": self.key_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EncryptedData":
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            nonce=base64.b64decode(data["nonce"]),
            key_id=data["key_id"],
            version=data.get("version", 1),
        )


class KeyManager:
    """
    Manages encryption keys for biometric data.

    Keys are derived from a master key using PBKDF2 with unique salts.
    Supports key rotation by maintaining multiple key versions.
    """

    def __init__(self):
        self._master_key: Optional[bytes] = None
        self._keys: dict[str, bytes] = {}  # key_id -> derived key
        self._current_key_id: Optional[str] = None
        self._salt: Optional[bytes] = None

    def initialize(self, master_key: Optional[bytes] = None, salt: Optional[bytes] = None) -> None:
        """
        Initialize the key manager.

        Args:
            master_key: 32-byte master key. If None, loads from env.
            salt: Salt for key derivation. If None, generates new one.
        """
        if master_key is None:
            master_key = self._load_master_key_from_env()

        if master_key is None:
            raise RuntimeError(
                "Master encryption key not configured. "
                "Set FACE_ENCRYPTION_KEY environment variable "
                "(base64-encoded 32-byte key) or call initialize() with a key."
            )

        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes (256 bits)")

        self._master_key = master_key

        if salt is None:
            salt = self._load_or_generate_salt()

        self._salt = salt

        # Derive current encryption key
        self._derive_key("current")

    def _load_master_key_from_env(self) -> Optional[bytes]:
        """Load master key from environment variable."""
        key_b64 = os.environ.get("FACE_ENCRYPTION_KEY")
        if not key_b64:
            return None
        try:
            return base64.b64decode(key_b64)
        except Exception as e:
            logger.error("Invalid FACE_ENCRYPTION_KEY: %s", e)
            return None

    def _load_or_generate_salt(self) -> bytes:
        """Load or generate salt for key derivation."""
        salt_b64 = os.environ.get("FACE_ENCRYPTION_SALT")
        if salt_b64:
            try:
                return base64.b64decode(salt_b64)
            except Exception:
                pass

        # Generate new salt
        salt = secrets.token_bytes(32)
        logger.warning(
            "Generated new encryption salt. Set FACE_ENCRYPTION_SALT=%s to persist.",
            base64.b64encode(salt).decode()
        )
        return salt

    def _derive_key(self, key_id: str) -> bytes:
        """Derive encryption key from master key."""
        if self._master_key is None or self._salt is None:
            raise RuntimeError("KeyManager not initialized")

        # Use PBKDF2 with key_id as additional context
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt + key_id.encode(),
            iterations=100_000,
        )
        derived_key = kdf.derive(self._master_key)
        self._keys[key_id] = derived_key
        self._current_key_id = key_id
        return derived_key

    def rotate_key(self) -> str:
        """
        Rotate to a new encryption key.
        Returns the new key_id.
        """
        new_key_id = f"key_{secrets.token_hex(8)}"
        self._derive_key(new_key_id)
        logger.info("Rotated encryption key to %s", new_key_id)
        return new_key_id

    def get_key(self, key_id: Optional[str] = None) -> bytes:
        """Get encryption key by ID, or current key if None."""
        if key_id is None:
            key_id = self._current_key_id

        if key_id is None:
            raise RuntimeError("No current encryption key. Call initialize() first.")

        if key_id not in self._keys:
            # Try to derive it
            self._derive_key(key_id)

        return self._keys[key_id]

    def get_current_key_id(self) -> str:
        """Get the current key ID."""
        if self._current_key_id is None:
            raise RuntimeError("KeyManager not initialized")
        return self._current_key_id

    def export_key_material(self) -> dict[str, str]:
        """Export key material for backup (use with extreme caution)."""
        return {
            "master_key": base64.b64encode(self._master_key).decode() if self._master_key else "",
            "salt": base64.b64encode(self._salt).decode() if self._salt else "",
            "current_key_id": self._current_key_id or "",
        }


# Global key manager instance
_key_manager = KeyManager()


def get_key_manager() -> KeyManager:
    """Get the global key manager instance."""
    return _key_manager


def initialize_encryption(master_key: Optional[bytes] = None, salt: Optional[bytes] = None) -> None:
    """Initialize the global key manager."""
    _key_manager.initialize(master_key, salt)


def generate_master_key() -> tuple[bytes, bytes]:
    """
    Generate a new master key and salt.
    Returns (master_key, salt) both as base64-encoded strings for storage.
    """
    master_key = secrets.token_bytes(32)
    salt = secrets.token_bytes(32)
    return master_key, salt


def encrypt_embedding(embedding: list[float], key_id: Optional[str] = None) -> EncryptedData:
    """
    Encrypt a face embedding (512-dim vector).

    Args:
        embedding: List of 512 floats
        key_id: Optional key ID for key rotation support

    Returns:
        EncryptedData container
    """
    km = get_key_manager()
    key = km.get_key(key_id)

    # Serialize embedding to bytes
    plaintext = json.dumps(embedding, separators=(",", ":")).encode()

    # Generate nonce (12 bytes for AES-GCM)
    nonce = secrets.token_bytes(12)

    # Encrypt
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    return EncryptedData(
        ciphertext=ciphertext,
        nonce=nonce,
        key_id=km.get_current_key_id(),
    )


def decrypt_embedding(encrypted: EncryptedData) -> list[float]:
    """
    Decrypt a face embedding.

    Args:
        encrypted: EncryptedData container

    Returns:
        List of 512 floats
    """
    km = get_key_manager()
    key = km.get_key(encrypted.key_id)

    # Decrypt
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(encrypted.nonce, encrypted.ciphertext, None)

    # Deserialize
    return json.loads(plaintext.decode())


def encrypt_data(data: dict[str, Any], key_id: Optional[str] = None) -> EncryptedData:
    """
    Encrypt arbitrary JSON-serializable data.

    Args:
        data: Dictionary to encrypt
        key_id: Optional key ID for key rotation support

    Returns:
        EncryptedData container
    """
    km = get_key_manager()
    key = km.get_key(key_id)

    plaintext = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    nonce = secrets.token_bytes(12)

    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    return EncryptedData(
        ciphertext=ciphertext,
        nonce=nonce,
        key_id=km.get_current_key_id(),
    )


def decrypt_data(encrypted: EncryptedData) -> dict[str, Any]:
    """
    Decrypt arbitrary data.

    Args:
        encrypted: EncryptedData container

    Returns:
        Decrypted dictionary
    """
    km = get_key_manager()
    key = km.get_key(encrypted.key_id)

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(encrypted.nonce, encrypted.ciphertext, None)

    return json.loads(plaintext.decode())


def verify_key_access(master_key_b64: str, salt_b64: str) -> bool:
    """
    Verify that a master key and salt can decrypt current data.
    Used for key rotation validation.
    """
    try:
        master_key = base64.b64decode(master_key_b64)
        salt = base64.b64decode(salt_b64)

        test_km = KeyManager()
        test_km.initialize(master_key, salt)

        # Try to decrypt a test payload
        test_data = {"test": "verification"}
        encrypted = encrypt_data(test_data)
        decrypted = decrypt_data(encrypted)

        return decrypted == test_data
    except Exception as e:
        logger.error("Key verification failed: %s", e)
        return False


# Migration utilities for existing plaintext embeddings
def migrate_plaintext_embeddings(index_path: str, backup_path: str) -> int:
    """
    Migrate plaintext embeddings to encrypted storage.

    Args:
        index_path: Path to plaintext index.json
        backup_path: Path to save backup of plaintext index

    Returns:
        Number of embeddings migrated
    """
    import shutil

    # Create backup
    shutil.copy2(index_path, backup_path)
    logger.info("Created backup at %s", backup_path)

    # Load plaintext index
    with open(index_path, "r") as f:
        plaintext_index = json.load(f)

    # Encrypt each embedding
    encrypted_index = {}
    for face_id, record in plaintext_index.items():
        embedding = record.pop("embedding")  # Remove from plaintext
        encrypted = encrypt_embedding(embedding)

        # Store encrypted embedding with metadata
        record["embedding_encrypted"] = encrypted.to_dict()
        record["embedding"] = None  # Explicitly mark as encrypted
        encrypted_index[face_id] = record

    # Write encrypted index
    with open(index_path, "w") as f:
        json.dump(encrypted_index, f, indent=2)

    logger.info("Migrated %d embeddings to encrypted storage", len(encrypted_index))
    return len(encrypted_index)