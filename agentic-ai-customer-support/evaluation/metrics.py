"""Simple evaluation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """Single-query evaluation output."""

    query: str
    expected_answer: str
    predicted_answer: str
    exact_match: bool


def evaluate_exact_match(expected_answer: str, predicted_answer: str) -> bool:
    """Return whether two normalized strings match exactly."""

    return expected_answer.strip().lower() == predicted_answer.strip().lower()


def log_evaluation_results(results: list[EvaluationResult]) -> None:
    """Log evaluation results using the standard logging module."""

    logger = logging.getLogger(__name__)
    for result in results:
        logger.info(
            "query=%s exact_match=%s expected=%s predicted=%s",
            result.query,
            result.exact_match,
            result.expected_answer,
            result.predicted_answer,
        )
