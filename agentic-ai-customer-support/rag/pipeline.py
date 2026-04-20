"""RAG prompt construction and answer generation."""

from __future__ import annotations

import logging
from typing import Iterable

from llm.hf_llm import generate_response
from rag.memory import build_memory_context
from utils.schema import DocumentChunk
from utils.text import combine_chunks


logger = logging.getLogger(__name__)


def build_context(chunks: Iterable[DocumentChunk]) -> str:
    """Build a readable context block from retrieved chunks."""

    formatted_chunks = []
    for chunk in chunks:
        formatted_chunks.append(f"Source: {chunk.source} | Chunk: {chunk.chunk_id}\n{chunk.text}")
    if not formatted_chunks:
        logger.info("retrieval_returned_no_context")
    return combine_chunks(formatted_chunks)


def answer_with_rag(question: str, chunks: list[DocumentChunk], memory_context: str = "") -> str:
    """Generate an answer from retrieved context and a question."""

    retrieved_context = build_context(chunks)
    context = combine_chunks([memory_context, retrieved_context]) if memory_context else retrieved_context
    return generate_response(question=question, context=context)


def build_prompt_context(chunks: list[DocumentChunk], messages: list[dict] | None = None) -> str:
    """Combine retrieved context with optional conversation memory."""

    memory_context = build_memory_context(messages or [])
    retrieved_context = build_context(chunks)
    return combine_chunks([memory_context, retrieved_context])
