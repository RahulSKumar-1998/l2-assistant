"""Celery tasks for scheduled maintenance jobs.

Nightly tasks for:
- Reindexing resolved tickets that were missed during the day.
- Rebuilding the BM25 sparse retrieval index.
- Processing accumulated feedback to adjust source quality scores.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from celery import Task

from app.workers.celery_app import celery_app

logger = structlog.get_logger(__name__)


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get the current event loop or create a new one for sync Celery workers.

    Returns:
        An asyncio event loop.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


# ── Async Implementations ──────────────────────────────────────────────────


async def _nightly_reindex_impl() -> dict[str, Any]:
    """Index all resolved tickets that haven't been indexed yet.

    Queries for incidents with state='6' (resolved) or state='7' (closed)
    that have is_indexed=False, then processes each through the ingestion
    pipeline.

    Returns:
        Dictionary with total processed, success, and failure counts.
    """
    from sqlalchemy import select, and_

    from app.storage.postgres import IncidentDB, get_db_session

    log = logger.bind(task="nightly_reindex")
    log.info("nightly_reindex_started")

    processed = 0
    indexed = 0
    errors = 0

    async for session in get_db_session():
        # Find all resolved/closed tickets not yet indexed
        stmt = select(IncidentDB).where(
            and_(
                IncidentDB.state.in_(["6", "7"]),  # Resolved or Closed
                IncidentDB.is_indexed == False,  # noqa: E712
            )
        ).limit(500)  # Process in batches

        result = await session.execute(stmt)
        unindexed = result.scalars().all()

        log.info("unindexed_tickets_found", count=len(unindexed))

        for incident in unindexed:
            processed += 1
            try:
                from app.ingestion.pipeline import process_and_index

                await process_and_index(incident)

                incident.is_indexed = True
                indexed += 1

                log.debug(
                    "ticket_reindexed",
                    incident_number=incident.number,
                    snow_sys_id=incident.snow_sys_id,
                )
            except Exception as exc:
                errors += 1
                log.error(
                    "ticket_reindex_failed",
                    incident_number=incident.number,
                    error=str(exc),
                    exc_info=True,
                )

        await session.flush()

    log.info(
        "nightly_reindex_completed",
        processed=processed,
        indexed=indexed,
        errors=errors,
    )

    return {
        "status": "completed",
        "processed": processed,
        "indexed": indexed,
        "errors": errors,
    }


async def _rebuild_bm25_impl() -> dict[str, Any]:
    """Rebuild the BM25 sparse retrieval index from PostgreSQL.

    Fetches all indexed incident text from the database and rebuilds
    the BM25 index for hybrid retrieval (dense + sparse).

    Returns:
        Dictionary with rebuild status and document count.
    """
    from sqlalchemy import select

    from app.core.embedder import Embedder
    from app.core.retriever import HybridRetriever
    from app.storage.postgres import IncidentDB, get_db_session

    log = logger.bind(task="rebuild_bm25")
    log.info("bm25_rebuild_started")

    doc_count = 0

    async for session in get_db_session():
        # Fetch all indexed incidents
        stmt = select(IncidentDB).where(
            IncidentDB.is_indexed == True  # noqa: E712
        )
        result = await session.execute(stmt)
        incidents = result.scalars().all()

        doc_count = len(incidents)
        log.info("bm25_documents_loaded", count=doc_count)

        if doc_count == 0:
            log.warning("no_documents_for_bm25_rebuild")
            return {
                "status": "completed",
                "documents": 0,
                "detail": "No indexed documents found",
            }

        # Build corpus for BM25
        corpus: list[dict[str, str]] = []
        for inc in incidents:
            text_parts = [inc.short_description or "", inc.description or ""]
            if inc.resolution_notes:
                text_parts.append(inc.resolution_notes)
            if inc.root_cause:
                text_parts.append(inc.root_cause)

            corpus.append({
                "chunk_id": inc.snow_sys_id,
                "chunk_text": " ".join(text_parts),
                "chunk_type": "description",
                "source_id": inc.number,
                "source_type": "incident",
                "metadata": {
                    "category": inc.category or "",
                    "cmdb_ci": inc.cmdb_ci or "",
                    "resolved_at": inc.resolved_at.isoformat() if inc.resolved_at else None,
                    "resolution_notes": inc.resolution_notes or "",
                },
            })

        retriever = HybridRetriever(embedder=Embedder(), vector_store=None)
        await retriever.rebuild_bm25_index(corpus)
        log.info("bm25_index_rebuilt_successfully")

    log.info("bm25_rebuild_completed", documents=doc_count)
    return {
        "status": "completed",
        "documents": doc_count,
    }


async def _process_feedback_impl() -> dict[str, Any]:
    """Process accumulated feedback to update source quality scores.

    Aggregates all unprocessed feedback records, computes quality scores
    for each source document referenced in recommendations, and updates
    the feedback_weights table.

    Returns:
        Dictionary with processing statistics.
    """
    from sqlalchemy import select, func

    from app.models.feedback import FeedbackStats
    from app.storage.postgres import (
        FeedbackDB,
        FeedbackWeightDB,
        RecommendationDB,
        get_db_session,
    )

    log = logger.bind(task="process_feedback")
    log.info("feedback_processing_started")

    total_feedback = 0
    positive_count = 0
    negative_count = 0
    sources_updated = 0

    async for session in get_db_session():
        # Get feedback from the last 24 hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        stmt = (
            select(FeedbackDB)
            .where(FeedbackDB.created_at >= cutoff)
        )
        result = await session.execute(stmt)
        feedback_records = result.scalars().all()

        total_feedback = len(feedback_records)
        log.info("feedback_records_found", count=total_feedback)

        if total_feedback == 0:
            return {
                "status": "completed",
                "total_feedback": 0,
                "sources_updated": 0,
            }

        # Aggregate feedback by recommendation
        rec_feedback: dict[str, list[int]] = {}
        for fb in feedback_records:
            rec_id_str = str(fb.recommendation_id)
            if rec_id_str not in rec_feedback:
                rec_feedback[rec_id_str] = []
            rec_feedback[rec_id_str].append(fb.rating)
            if fb.rating >= 4:
                positive_count += 1
            elif fb.rating <= 2:
                negative_count += 1

        # For each recommendation, update source quality weights
        for rec_id_str, ratings in rec_feedback.items():
            try:
                rec_uuid = __import__("uuid").UUID(rec_id_str)
            except ValueError:
                continue

            rec_stmt = select(RecommendationDB).where(
                RecommendationDB.id == rec_uuid
            )
            rec_result = await session.execute(rec_stmt)
            recommendation = rec_result.scalar_one_or_none()

            if not recommendation or not recommendation.similar_incidents:
                continue

            # Extract source IDs from similar incidents
            source_ids: list[str] = []
            similar = recommendation.similar_incidents or []
            for sim in similar:
                if isinstance(sim, dict) and "number" in sim:
                    source_ids.append(sim["number"])

            avg_rating = sum(ratings) / len(ratings)
            # Scale: rating 1-5 → quality multiplier 0.5-1.5
            quality_delta = (avg_rating - 3.0) / 4.0

            for source_id in source_ids:
                weight_stmt = select(FeedbackWeightDB).where(
                    FeedbackWeightDB.source_id == source_id
                )
                weight_result = await session.execute(weight_stmt)
                weight = weight_result.scalar_one_or_none()

                if weight:
                    # Update existing weight
                    new_score = max(0.1, min(2.0, weight.quality_score + quality_delta))
                    weight.quality_score = new_score
                    if avg_rating >= 4:
                        weight.positive_signals += len(ratings)
                    elif avg_rating <= 2:
                        weight.negative_signals += len(ratings)
                else:
                    # Create new weight record
                    new_weight = FeedbackWeightDB(
                        source_id=source_id,
                        source_type="incident",
                        quality_score=max(0.1, min(2.0, 1.0 + quality_delta)),
                        positive_signals=len([r for r in ratings if r >= 4]),
                        negative_signals=len([r for r in ratings if r <= 2]),
                    )
                    session.add(new_weight)

                sources_updated += 1

        await session.flush()

    log.info(
        "feedback_processing_completed",
        total=total_feedback,
        positive=positive_count,
        negative=negative_count,
        sources_updated=sources_updated,
    )

    return {
        "status": "completed",
        "total_feedback": total_feedback,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "sources_updated": sources_updated,
    }


# ── Celery Tasks ────────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="app.workers.reindex_worker.nightly_reindex",
    max_retries=2,
    default_retry_delay=300,
    acks_late=True,
    track_started=True,
)
def nightly_reindex(self: Task) -> dict[str, Any]:
    """Nightly task: index all resolved tickets since the last run.

    Scheduled at 02:00 UTC via Celery Beat. Finds all unindexed
    resolved/closed tickets and processes them through the ingestion
    pipeline.

    Args:
        self: Celery Task instance.

    Returns:
        Dictionary with reindex statistics.
    """
    log = logger.bind(task_id=self.request.id, task="nightly_reindex")
    log.info("nightly_reindex_task_started")

    try:
        loop = _get_or_create_event_loop()
        result = loop.run_until_complete(_nightly_reindex_impl())
        log.info("nightly_reindex_task_completed", **result)
        return result
    except Exception as exc:
        log.error("nightly_reindex_task_failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=300)


@celery_app.task(
    bind=True,
    name="app.workers.reindex_worker.rebuild_bm25",
    max_retries=2,
    default_retry_delay=300,
    acks_late=True,
    track_started=True,
)
def rebuild_bm25(self: Task) -> dict[str, Any]:
    """Nightly task: rebuild the BM25 sparse retrieval index.

    Scheduled at 03:00 UTC via Celery Beat. Loads all indexed
    incident text from PostgreSQL and rebuilds the BM25 index
    for hybrid (dense + sparse) retrieval.

    Args:
        self: Celery Task instance.

    Returns:
        Dictionary with rebuild statistics.
    """
    log = logger.bind(task_id=self.request.id, task="rebuild_bm25")
    log.info("bm25_rebuild_task_started")

    try:
        loop = _get_or_create_event_loop()
        result = loop.run_until_complete(_rebuild_bm25_impl())
        log.info("bm25_rebuild_task_completed", **result)
        return result
    except Exception as exc:
        log.error("bm25_rebuild_task_failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=300)


@celery_app.task(
    bind=True,
    name="app.workers.reindex_worker.process_feedback_nightly",
    max_retries=2,
    default_retry_delay=300,
    acks_late=True,
    track_started=True,
)
def process_feedback_nightly(self: Task) -> dict[str, Any]:
    """Nightly task: process accumulated feedback to adjust quality scores.

    Scheduled at 04:00 UTC via Celery Beat. Aggregates all feedback
    from the last 24 hours and updates source document quality weights
    in the feedback_weights table.

    Args:
        self: Celery Task instance.

    Returns:
        Dictionary with processing statistics.
    """
    log = logger.bind(task_id=self.request.id, task="process_feedback")
    log.info("feedback_processing_task_started")

    try:
        loop = _get_or_create_event_loop()
        result = loop.run_until_complete(_process_feedback_impl())
        log.info("feedback_processing_task_completed", **result)
        return result
    except Exception as exc:
        log.error("feedback_processing_task_failed", error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=300)
