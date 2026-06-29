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
from pathlib import Path

import structlog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Patch requests to bypass SSL verification for tiktoken downloads
import requests
original_get = requests.get
def patched_get(*args, **kwargs):
    if len(args) > 0 and "openaipublic" in args[0]:
        kwargs["verify"] = False
    elif "url" in kwargs and "openaipublic" in kwargs["url"]:
        kwargs["verify"] = False
    return original_get(*args, **kwargs)
requests.get = patched_get

from app.config import get_settings
from app.ingestion.mock_client import MockServiceNowClient
from app.ingestion.pipeline import process_and_index, store_incident_record
from app.ingestion.servicenow_client import ServiceNowClient
from app.models.incident import IncidentQueryParams, IncidentRecord
from app.storage.postgres import get_session_factory

logger = structlog.get_logger(__name__)


async def fetch_resolved_incidents(
    limit: int,
    offset: int = 0,
    *,
    use_mock: bool = False,
) -> list[IncidentRecord]:
    """Fetch resolved incidents from ServiceNow.

    Args:
        limit: Maximum number of incidents to fetch.
        offset: Pagination offset.

    Returns:
        List of incident records.
    """
    logger.info(
        "fetch_resolved_incidents",
        limit=limit,
        offset=offset,
        source="mock" if use_mock else "servicenow",
    )
    client_cls = MockServiceNowClient if use_mock else ServiceNowClient
    params = IncidentQueryParams(state="6,7", limit=limit, offset=offset)
    async with client_cls() as client:  # type: ignore[call-arg]
        return await client.list_incidents(params)


async def fetch_kb_articles(limit: int, offset: int = 0) -> list[dict]:
    """Fetch published KB articles from ServiceNow.

    Args:
        limit: Maximum number of articles to fetch.
        offset: Pagination offset.

    Returns:
        List of KB article records.
    """
    logger.info(
        "fetch_kb_articles_not_implemented",
        limit=limit,
        offset=offset,
    )
    return []


async def store_and_optionally_index(
    incidents: list[IncidentRecord],
    *,
    dry_run: bool = False,
    store_only: bool = False,
) -> tuple[int, int]:
    """Store incidents in PostgreSQL and optionally vectorize resolved ones.

    Args:
        incidents: ServiceNow-style incident records.
        dry_run: If True, do not write to DB or vector store.
        store_only: If True, skip embedding/vector upsert after DB persistence.

    Returns:
        Tuple of (stored_count, indexed_count).
    """
    if dry_run:
        logger.info(
            "dry_run_store_and_index",
            incident_count=len(incidents),
            store_only=store_only,
        )
        return len(incidents), 0 if store_only else len(incidents)

    session_factory = get_session_factory()
    stored_count = 0
    indexed_count = 0
    from sqlalchemy import text

    for incident_record in incidents:
        async with session_factory() as session:
            incident_db = await store_incident_record(incident_record, session=session)
            await session.commit()
            stored_count += 1

        if not store_only:
            result = await process_and_index(incident_db)
            if int(result["upserted_count"]) > 0 and int(result["failed_count"]) == 0:
                indexed_count += 1
                async with session_factory() as session:
                    await session.execute(
                        text("UPDATE incidents SET is_indexed = 1, updated_at = :now WHERE id = :id"),
                        {"now": datetime.now(timezone.utc), "id": str(incident_db.id)}
                    )
                    await session.commit()

    logger.info(
        "store_and_index_batch_completed",
        stored_count=stored_count,
        indexed_count=indexed_count,
        store_only=store_only,
    )
    return stored_count, indexed_count


async def bootstrap(
    batch_size: int = 100,
    limit: int = 0,
    dry_run: bool = False,
    store_only: bool = False,
    use_mock: bool = False,
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
        store_only=store_only,
        use_mock=use_mock,
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
            limit=current_limit,
            offset=offset,
            use_mock=use_mock,
        )

        if not incidents:
            break

        try:
            _, indexed = await store_and_optionally_index(
                incidents,
                dry_run=dry_run,
                store_only=store_only,
            )
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

    # ── Phase 2: KB articles (still optional / not yet wired here) ──────
    logger.info("phase_2_start", phase="kb_articles")
    offset = 0
    kb_count = 0

    while True:
        articles = await fetch_kb_articles(limit=batch_size, offset=offset)

        if not articles:
            break

        kb_count += len(articles)

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
  python scripts/bootstrap_index.py --store-only       # Persist incidents without vectorization
  python scripts/bootstrap_index.py --use-mock         # Load fixture incidents locally
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
    parser.add_argument(
        "--store-only",
        action="store_true",
        default=False,
        help="Persist incidents to PostgreSQL but skip embedding/vector indexing",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        default=False,
        help="Use the bundled mock ServiceNow client instead of a live instance",
    )

    args = parser.parse_args()

    logger.info(
        "bootstrap_cli_start",
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        limit=args.limit,
        store_only=args.store_only,
        use_mock=args.use_mock,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    asyncio.run(
        bootstrap(
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
            store_only=args.store_only,
            use_mock=args.use_mock,
        )
    )


if __name__ == "__main__":
    main()
