"""FAISS-based vector storage and retrieval."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from utils.schema import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    """A retrieved document chunk and its similarity score."""

    chunk: DocumentChunk
    score: float


def _load_faiss():
    """Import FAISS lazily and raise a clear error if it is unavailable."""

    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("faiss-cpu is required for vector store operations") from exc
    return faiss


def build_index(embeddings):
    """Build an inner-product FAISS index from normalized embeddings."""

    faiss = _load_faiss()
    import numpy as np

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings must not be empty")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def save_vectorstore(index, chunks: list[DocumentChunk], vectorstore_dir: Path) -> None:
    """Persist the FAISS index and chunk metadata to disk."""

    faiss = _load_faiss()
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(vectorstore_dir / "index.faiss"))
    with (vectorstore_dir / "chunks.pkl").open("wb") as handle:
        pickle.dump(chunks, handle)


def load_vectorstore(vectorstore_dir: Path):
    """Load the FAISS index and chunk metadata from disk."""

    faiss = _load_faiss()
    index_path = vectorstore_dir / "index.faiss"
    chunks_path = vectorstore_dir / "chunks.pkl"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("vector store files are missing")

    index = faiss.read_index(str(index_path))
    with chunks_path.open("rb") as handle:
        chunks = pickle.load(handle)
    return index, chunks


def search(index, chunks: list[DocumentChunk], query_embedding, top_k: int) -> list[SearchResult]:
    """Return the top-k most similar chunks for a query embedding."""

    import numpy as np

    if query_embedding.ndim != 1:
        raise ValueError("query_embedding must be a 1D vector")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    scores, indices = index.search(np.asarray([query_embedding], dtype=np.float32), top_k)
    if scores.shape != indices.shape:
        raise RuntimeError("FAISS returned mismatched score and index shapes")

    results: list[SearchResult] = []
    for position in range(indices.shape[1]):
        score = float(scores[0][position])
        index_position = int(indices[0][position])
        if index_position == -1:
            continue
        if index_position >= len(chunks):
            raise IndexError("FAISS returned an out-of-range chunk index")
        results.append(SearchResult(chunk=chunks[index_position], score=float(score)))
    return results
