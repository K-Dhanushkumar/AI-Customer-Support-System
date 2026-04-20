"""Shared data structures for the system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """A processed text chunk with source metadata."""

    text: str
    source: str
    chunk_id: int
