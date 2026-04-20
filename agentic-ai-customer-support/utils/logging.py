"""Logging configuration for the application."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_file: Path) -> None:
    """Configure root logging to write to stdout and a log file."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""

    return logging.getLogger(name)
