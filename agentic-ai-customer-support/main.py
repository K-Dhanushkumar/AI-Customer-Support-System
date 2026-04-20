"""Command-line entrypoint for the customer support system."""

from __future__ import annotations

import argparse
import sys

from agents.decision import SupportSystemState, answer_query
from llm.hf_llm import generate_response
from rag.service import build_support_system
from utils.config import settings
from utils.logging import get_logger, setup_logging
from utils.storage import initialize_database


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Agentic AI Customer Support")
    parser.add_argument(
        "--query",
        default="How do I reset my password?",
        help="Support question to answer",
    )
    return parser.parse_args()


def main() -> None:
    """Build the support system and print one answer."""

    setup_logging(settings.log_file)
    logger = get_logger(__name__)
    args = parse_args()

    try:
        initialize_database()
        system = build_support_system()
        state = SupportSystemState(index=system.index, chunks=system.chunks, top_k=settings.top_k)
        answer = answer_query(
            args.query,
            state,
            direct_response_fn=lambda query, memory_context="": generate_response(query, context=memory_context),
        )
        logger.info("query_processed")
        print(answer)
    except Exception as exc:
        logger.exception("startup_or_runtime_failure")
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
