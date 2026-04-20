"""Project orchestration for data, embeddings, retrieval, and generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from embeddings.embedding_model import embed_texts
from llm.hf_llm import generate_response
from utils.config import settings
from utils.logging import get_logger
from utils.schema import DocumentChunk
from utils.text import load_raw_documents
from vectorstore.faiss_store import build_index, load_vectorstore, save_vectorstore, search


@dataclass(frozen=True)
class SupportSystem:
    """Container for the FAISS index and its corresponding chunks."""

    index: object
    chunks: list[DocumentChunk]


def build_knowledge_base(raw_data_dir: Path | None = None, vectorstore_dir: Path | None = None) -> SupportSystem:
    """Load raw data, build embeddings, and persist the FAISS index."""

    raw_dir = raw_data_dir or settings.raw_data_dir
    store_dir = vectorstore_dir or settings.vectorstore_dir
    logger = get_logger(__name__)

    chunks = load_raw_documents(raw_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError(f"No .txt files found in {raw_dir}")

    embeddings = embed_texts([chunk.text for chunk in chunks])
    index = build_index(np.asarray(embeddings, dtype=np.float32))
    save_vectorstore(index, chunks, store_dir)
    logger.info("knowledge_base_built chunks=%s", len(chunks))
    return SupportSystem(index=index, chunks=chunks)


@lru_cache(maxsize=1)
def build_support_system() -> SupportSystem:
    """Load a persisted vector store or build one from raw data."""

    store_dir = settings.vectorstore_dir
    try:
        index, chunks = load_vectorstore(store_dir)
        if not chunks:
            raise ValueError("vector store contains no chunks")
        return SupportSystem(index=index, chunks=chunks)
    except FileNotFoundError:
        return build_knowledge_base()


def answer_support_query(query: str, top_k: int | None = None) -> str:
    """Answer a support query using retrieval-augmented generation."""

    system = build_support_system()
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    query_embedding = embed_texts([normalized_query])[0]
    results = search(system.index, system.chunks, query_embedding, top_k or settings.top_k)
    if results:
        context = "\n\n".join(f"Source: {result.chunk.source}\n{result.chunk.text}" for result in results)
        return generate_response(normalized_query, context=context)
    return generate_response(normalized_query)
