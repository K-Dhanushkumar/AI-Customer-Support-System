"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency boundary
    def load_dotenv(*_args, **_kwargs):
        """Fallback no-op when python-dotenv is unavailable."""

        return False


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback."""

    value = os.getenv(name)
    return int(value) if value and value.isdigit() else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    base_dir: Path = BASE_DIR
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    embeddings_dir: Path = BASE_DIR / "embeddings"
    vectorstore_dir: Path = BASE_DIR / "vectorstore"
    database_dir: Path = BASE_DIR / "data"
    database_path: Path = BASE_DIR / "data" / "app.db"
    logs_dir: Path = BASE_DIR / "logs"
    log_file: Path = BASE_DIR / "logs" / "app.log"
    embedding_model_name: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model_name: str = os.getenv("HF_LLM_MODEL", "google/flan-t5-small")
    admin_username: str = os.getenv("ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "admin123")
    token_ttl_minutes: int = _get_int("TOKEN_TTL_MINUTES", 720)
    conversation_history_limit: int = _get_int("CONVERSATION_HISTORY_LIMIT", 6)
    chunk_size: int = _get_int("CHUNK_SIZE", 500)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 80)
    top_k: int = _get_int("TOP_K", 3)
    max_new_tokens: int = _get_int("MAX_NEW_TOKENS", 160)


settings = Settings()
settings.database_dir.mkdir(parents=True, exist_ok=True)
settings.logs_dir.mkdir(parents=True, exist_ok=True)
settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
