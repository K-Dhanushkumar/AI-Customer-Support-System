"""Sentence-transformers embedding helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from utils.config import settings


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str | None = None):
    """Load and cache the embedding model."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("sentence-transformers is required to build embeddings") from exc

    resolved_name = model_name or settings.embedding_model_name
    return SentenceTransformer(resolved_name)


def embed_texts(texts: Sequence[str], model_name: str | None = None):
    """Convert text inputs into normalized embedding vectors."""

    import numpy as np

    model = get_embedding_model(model_name)
    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query: str, model_name: str | None = None):
    """Embed a single query string."""

    return embed_texts([query], model_name=model_name)[0]
