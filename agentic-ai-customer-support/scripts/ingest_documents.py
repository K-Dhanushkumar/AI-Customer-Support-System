"""Ingest new support documents into the raw data folder and rebuild the vector store."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rag.service import build_knowledge_base
from utils.config import settings
from utils.logging import get_logger, setup_logging
from rag.service import build_support_system


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Ingest support documents")
    parser.add_argument("source", help="Directory containing .txt files to ingest")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the FAISS vector store after ingestion",
    )
    return parser.parse_args()


def copy_text_files(source_dir: Path, destination_dir: Path) -> int:
    """Copy .txt files from the source directory into the raw data folder."""

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    for file_path in sorted(source_dir.glob("*.txt")):
        shutil.copy2(file_path, destination_dir / file_path.name)
        copied_files += 1
    return copied_files


def main() -> None:
    """Ingest documents and optionally rebuild the vector store."""

    setup_logging(settings.log_file)
    logger = get_logger(__name__)
    args = parse_args()

    source_dir = Path(args.source).resolve()
    copied_files = copy_text_files(source_dir, settings.raw_data_dir)
    logger.info("documents_ingested count=%s source=%s", copied_files, source_dir)

    if args.rebuild:
        build_support_system.cache_clear()
        build_knowledge_base()
        logger.info("vectorstore_rebuilt")


if __name__ == "__main__":
    main()
