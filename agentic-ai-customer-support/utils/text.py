"""Text loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from utils.schema import DocumentChunk


def clean_text(text: str) -> str:
    """Normalize whitespace and strip empty lines."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = end - chunk_overlap

    return chunks


def load_text_file(file_path: Path) -> str:
    """Load a UTF-8 text file from disk."""

    return file_path.read_text(encoding="utf-8")


def load_raw_documents(raw_data_dir: Path, chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
    """Load, clean, and chunk every text file in the raw data directory."""

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")

    documents: list[DocumentChunk] = []
    for file_path in sorted(raw_data_dir.glob("*.txt")):
        raw_text = load_text_file(file_path)
        normalized_text = clean_text(raw_text)
        for index, chunk in enumerate(split_text(normalized_text, chunk_size, chunk_overlap)):
            documents.append(DocumentChunk(text=chunk, source=file_path.name, chunk_id=index))
    return documents


def combine_chunks(chunks: Iterable[str]) -> str:
    """Combine chunks into a retrieval context block."""

    return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
