"""Core unit tests for non-ML utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.decision import requires_retrieval
from evaluation.metrics import evaluate_exact_match
from rag.pipeline import build_context
from utils.schema import DocumentChunk
from utils.text import split_text


class CoreTests(unittest.TestCase):
    """Basic deterministic tests."""

    def test_split_text_returns_overlapping_chunks(self) -> None:
        """Chunking should create more than one chunk for long input."""

        chunks = split_text("a" * 30, chunk_size=10, chunk_overlap=2)
        self.assertGreaterEqual(len(chunks), 3)

    def test_requires_retrieval_for_question(self) -> None:
        """Questions about support content should route to retrieval."""

        self.assertTrue(requires_retrieval("How do I reset my password?"))

    def test_exact_match_is_case_insensitive(self) -> None:
        """Normalized answers should compare case-insensitively."""

        self.assertTrue(evaluate_exact_match("Hello", "hello"))

    def test_build_context_formats_chunks(self) -> None:
        """RAG context builder should include source metadata."""

        chunks = [DocumentChunk(text="Reset by email", source="faq.txt", chunk_id=0)]
        context = build_context(chunks)
        self.assertIn("faq.txt", context)
        self.assertIn("Reset by email", context)


if __name__ == "__main__":
    unittest.main()
