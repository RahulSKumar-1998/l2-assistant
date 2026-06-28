"""Bootstrap historical ticket indexing into the vector store.

CLI script for one-time bulk indexing of resolved incidents and KB
articles from ServiceNow into the vector store. Supports dry-run mode
for validation without writes.

Usage:
    python scripts/bootstrap_index.py
    python scripts/bootstrap_index.py --dry-run
    python scripts/bootstrap_index.py --batch-size 50 --limit 1000
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


async def fetch_resolved_incidents(
    limit: int,
    offset: int = 0,
) -> list[dict]:
    """Fetch resolved incidents from ServiceNow.

    Args:
        limit: Maximum number of incidents to fetch.
        offset: Pagination offset.

    Returns:
        List of incident records.
    """
    # In production, this would use the ServiceNow client:
    # from app.config import get_settings
    # settings = get_settings()
    # client = ServiceNowClient(settings.servicenow)
    # params = IncidentQueryParams(state="6", limit=limit, offset=offset)
    # return await client.list_incidents(params)
    logger.info("fetch_resolved_incidents", limit=limit, offset=offset)
    return []


async def fetch_kb_articles(limit: int, offset: int = 0) -> list[dict]:
    """Fetch published KB articles from ServiceNow.

    Args:
        limit: Maximum number of articles to fetch.
        offset: Pagination offset.

    Returns:
        List of KB article records.
    """
    logger.info("fetch_kb_articles", limit=limit, offset=offset)
    return []


async def process_and_chunk(records: list[dict], record_type: str) -> list[dict]:
    """Process records through the NLP pipeline and chunk for indexing.

    Args:
        records: Raw records from ServiceNow.
        record_type: "incident" or "kb_article".

    Returns:
        List of text chunks ready for embedding.
    """
    # In production, this would use the ticket processor:
    # from app.ingestion.ticket_processor import TicketProcessor
    # processor = TicketProcessor()
    # chunks = []
    # for record in records:
    #     processed = processor.process(record)
    #     chunks.extend(processor.chunk(processed))
    # return chunks
    logger.info("process_and_chunk", record_type=record_type, count=len(records))
    return []


async def index_chunks(chunks: list[dict], dry_run: bool = False) -> int:
    """Index text chunks into the vector store.

    Args:
        chunks: Text chunks with embeddings to index.
        dry_run: If True, skip actual indexing.

    Returns:
        Number of chunks indexed.
    """
    if dry_run:
        logger.info("dry_run_index", chunk_count=len(chunks))
        return len(chunks)

    # In production:
    # from app.core.vector_store import VectorStore
    # store = VectorStore(get_settings().vector_store)
    # result = await store.upsert(chunks)
    # return result["upserted_count"]
    logger.info("index_chunks", chunk_count=len(chunks))
    return len(chunks)


async def bootstrap(
    batch_size: int = 100,
    limit: int = 0,
    dry_run: bool = False,
) -> None:
    """Run the bootstrap indexing pipeline.

    Args:
        batch_size: Number of records to process per batch.
        limit: Maximum total records (0 = unlimited).
        dry_run: If True, validate without writing to vector store.
    """
    start_time = time.monotonic()
    total_indexed = 0
    total_errors = 0

    logger.info(
        "bootstrap_start",
        batch_size=batch_size,
        limit=limit or "unlimited",
        dry_run=dry_run,
    )

    # ── Phase 1: Index resolved incidents ────────────────────────────────
    logger.info("phase_1_start", phase="resolved_incidents")
    offset = 0
    incident_count = 0

    while True:
        current_limit = batch_size
        if limit > 0:
            remaining = limit - incident_count
            if remaining <= 0:
                break
            current_limit = min(batch_size, remaining)

        incidents = await fetch_resolved_incidents(
            limit=current_limit, offset=offset
        )

        if not incidents:
            break

        try:
            chunks = await process_and_chunk(incidents, "incident")
            indexed = await index_chunks(chunks, dry_run=dry_run)
            total_indexed += indexed
            incident_count += len(incidents)
        except Exception as e:
            total_errors += len(incidents)
            logger.error("batch_error", error=str(e), offset=offset)

        offset += len(incidents)

        # Progress and ETA
        elapsed = time.monotonic() - start_time
        rate = incident_count / elapsed if elapsed > 0 else 0
        if limit > 0 and rate > 0:
            remaining_records = limit - incident_count
            eta_seconds = remaining_records / rate
            logger.info(
                "progress",
                phase="incidents",
                processed=incident_count,
                total=limit or "unknown",
                rate=f"{rate:.1f} records/sec",
                eta=f"{eta_seconds:.0f}s",
            )
        else:
            logger.info(
                "progress",
                phase="incidents",
                processed=incident_count,
                rate=f"{rate:.1f} records/sec",
            )

    # ── Phase 2: Index KB articles ───────────────────────────────────────
    logger.info("phase_2_start", phase="kb_articles")
    offset = 0
    kb_count = 0

    while True:
        articles = await fetch_kb_articles(limit=batch_size, offset=offset)

        if not articles:
            break

        try:
            chunks = await process_and_chunk(articles, "kb_article")
            indexed = await index_chunks(chunks, dry_run=dry_run)
            total_indexed += indexed
            kb_count += len(articles)
        except Exception as e:
            total_errors += len(articles)
            logger.error("batch_error", error=str(e), offset=offset)

        offset += len(articles)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    logger.info(
        "bootstrap_complete",
        total_incidents=incident_count,
        total_kb_articles=kb_count,
        total_chunks_indexed=total_indexed,
        total_errors=total_errors,
        elapsed_seconds=f"{elapsed:.1f}",
        dry_run=dry_run,
    )

    if total_errors > 0:
        logger.warning("bootstrap_had_errors", error_count=total_errors)


def main() -> None:
    """CLI entry point for bootstrap indexing."""
    parser = argparse.ArgumentParser(
        description="Bootstrap historical ticket indexing into the vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/bootstrap_index.py                    # Index all resolved incidents
  python scripts/bootstrap_index.py --dry-run          # Validate without writing
  python scripts/bootstrap_index.py --batch-size 50    # Process 50 records at a time
  python scripts/bootstrap_index.py --limit 1000       # Index at most 1000 incidents
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate pipeline without writing to vector store",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records per batch (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum total records to process (0 = unlimited)",
    )

    args = parser.parse_args()

    logger.info(
        "bootstrap_cli_start",
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        limit=args.limit,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    asyncio.run(
        bootstrap(
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
