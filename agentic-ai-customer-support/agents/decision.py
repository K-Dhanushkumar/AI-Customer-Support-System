"""Simple, explainable agent routing logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from typing import Callable

from embeddings.embedding_model import embed_query
from rag.pipeline import answer_with_rag
from utils.config import settings
from vectorstore.faiss_store import search


logger = logging.getLogger(__name__)


KNOWLEDGE_KEYWORDS = (
    "how",
    "what",
    "when",
    "where",
    "why",
    "can you",
    "do you",
    "help",
    "support",
    "policy",
    "refund",
    "billing",
    "password",
    "account",
    "error",
)


@dataclass(frozen=True)
class SupportSystemState:
    """Container for runtime retrieval and generation components."""

    index: object
    chunks: list
    top_k: int = settings.top_k


def requires_retrieval(query: str) -> bool:
    """Decide whether a query should use retrieval."""

    normalized_query = query.lower().strip()
    return "?" in normalized_query or any(keyword in normalized_query for keyword in KNOWLEDGE_KEYWORDS)


def retrieve_relevant_chunks(query: str, state: SupportSystemState) -> list:
    """Retrieve the most relevant chunks for a query."""

    query_embedding = embed_query(query)
    results = search(state.index, state.chunks, query_embedding, state.top_k)
    if not results:
        logger.info("no_retrieval_results query=%s", query)
    return [result.chunk for result in results]


def _call_direct_response(direct_response_fn: Callable[..., str], query: str, memory_context: str) -> str:
    """Invoke the direct response function with memory when supported."""

    try:
        return direct_response_fn(query, memory_context=memory_context)
    except TypeError:
        return direct_response_fn(query)


def answer_query(
    query: str,
    state: SupportSystemState,
    direct_response_fn: Callable[..., str],
    memory_context: str = "",
) -> str:
    """Route a query through retrieval when needed, otherwise use the direct LLM response."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")

    if requires_retrieval(normalized_query):
        chunks = retrieve_relevant_chunks(normalized_query, state)
        if chunks:
            return answer_with_rag(normalized_query, chunks, memory_context=memory_context)
    return _call_direct_response(direct_response_fn, normalized_query, memory_context)
