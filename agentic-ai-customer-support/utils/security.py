"""Password and token helpers for local authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash a password using PBKDF2 with a per-password salt."""

    salt_bytes = salt or secrets.token_bytes(16)
    derived_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 100_000)
    return f"{base64.b64encode(salt_bytes).decode('utf-8')}${base64.b64encode(derived_key).decode('utf-8')}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored PBKDF2 hash."""

    salt_text, hash_text = stored_hash.split("$", maxsplit=1)
    salt_bytes = base64.b64decode(salt_text.encode("utf-8"))
    expected_hash = hash_password(password, salt=salt_bytes)
    return hmac.compare_digest(expected_hash, stored_hash)


def hash_token(token: str) -> str:
    """Return a stable SHA-256 hash for an access token."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()
