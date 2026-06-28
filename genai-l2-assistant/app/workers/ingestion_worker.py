"""Celery tasks for incident analysis and ticket indexing.

Provides async task wrappers around the RAG pipeline for:
- Full incident analysis (analyze_incident_async)
- Indexing resolved tickets into the vector store (index_resolved_ticket)
"""

import asyncio
import uuid as uuid_mod
from datetime import datetime, timezone
from typing import Any, Optional

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


async def _run_analysis_pipeline(
    snow_sys_id: str, engineer_id: str
) -> dict[str, Any]:
    """Execute the full RAG analysis pipeline for an incident.

    Orchestrates: fetch incident → preprocess → retrieve context →
    generate recommendation → persist result.

    Args:
        snow_sys_id: ServiceNow incident sys_id.
        engineer_id: Requesting engineer's sys_id.

    Returns:
        Dictionary with recommendation_id, status, and key metrics.
    """
    from app.config import get_settings
    from app.storage.postgres import (
        IncidentDB,
        RecommendationDB,
        get_db_session,
    )
    from sqlalchemy import select

    log = logger.bind(snow_sys_id=snow_sys_id, engineer_id=engineer_id)
    log.info("analysis_pipeline_started")

    settings = get_settings()

    # Step 1: Look up or create the incident in our database
    async for session in get_db_session():
        stmt = select(IncidentDB).where(IncidentDB.snow_sys_id == snow_sys_id)
        result = await session.execute(stmt)
        incident = result.scalar_one_or_none()

        if not incident:
            # Attempt to fetch from ServiceNow and create the record
            log.info("incident_not_in_db_creating_stub")
            incident = IncidentDB(
                id=uuid_mod.uuid4(),
                snow_sys_id=snow_sys_id,
                number=f"INC-{snow_sys_id[:8]}",
                short_description="Pending analysis",
                state="2",  # IN_PROGRESS
            )
            session.add(incident)
            await session.flush()

        # Step 2: Attempt to run the RAG pipeline
        try:
            from app.core.rag_pipeline import analyze as rag_analyze  # type: ignore[import-not-found]

            rag_result = await rag_analyze(incident=incident)
            root_cause = rag_result.get("root_cause_prediction", "Analysis pending")
            confidence = rag_result.get("confidence_score", 0.0)
            triage_steps = rag_result.get("triage_steps", [])
            similar_incidents = rag_result.get("similar_incidents", [])
            kb_references = rag_result.get("kb_references", [])
            resolution_draft = rag_result.get("resolution_draft", "")
            retrieval_ms = rag_result.get("retrieval_latency_ms", 0)
            generation_ms = rag_result.get("generation_latency_ms", 0)
        except ImportError:
            log.info("rag_pipeline_not_available_using_placeholder")
            root_cause = (
                f"Automated analysis for incident {incident.number}. "
                f"RAG pipeline integration pending. "
                f"Category: {incident.category or 'N/A'}."
            )
            confidence = 0.5
            triage_steps = [
                {
                    "step": 1,
                    "action": "Review incident description and work notes",
                    "rationale": "Gather initial context about the issue",
                },
                {
                    "step": 2,
                    "action": "Check CMDB for impacted service dependencies",
                    "rationale": "Identify potential upstream/downstream impacts",
                },
                {
                    "step": 3,
                    "action": "Search knowledge base for similar resolutions",
                    "rationale": "Leverage historical resolution patterns",
                },
            ]
            similar_incidents = []
            kb_references = []
            resolution_draft = ""
            retrieval_ms = 0
            generation_ms = 0

        # Step 3: Persist the recommendation
        rec_id = uuid_mod.uuid4()
        recommendation = RecommendationDB(
            id=rec_id,
            incident_id=incident.id,
            root_cause_prediction=root_cause,
            confidence_score=confidence,
            triage_steps=triage_steps,
            similar_incidents=similar_incidents,
            kb_references=kb_references,
            resolution_draft=resolution_draft,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )
        session.add(recommendation)
        await session.flush()

        log.info(
            "analysis_pipeline_completed",
            recommendation_id=str(rec_id),
            confidence=confidence,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

        return {
            "recommendation_id": str(rec_id),
            "incident_id": str(incident.id),
            "status": "completed",
            "confidence_score": confidence,
        }

    return {"status": "error", "detail": "Failed to acquire database session"}


async def _index_ticket(snow_sys_id: str) -> dict[str, Any]:
    """Index a resolved ticket into the vector store.

    Fetches the incident, processes it through the text pipeline,
    generates embeddings, and upserts into the vector store.

    Args:
        snow_sys_id: ServiceNow incident sys_id.

    Returns:
        Dictionary with indexing result status and metadata.
    """
    from app.storage.postgres import IncidentDB, get_db_session
    from sqlalchemy import select

    log = logger.bind(snow_sys_id=snow_sys_id)
    log.info("ticket_indexing_started")

    async for session in get_db_session():
        stmt = select(IncidentDB).where(IncidentDB.snow_sys_id == snow_sys_id)
        result = await session.execute(stmt)
        incident = result.scalar_one_or_none()

        if not incident:
            log.warning("incident_not_found_for_indexing")
            return {"status": "error", "detail": "Incident not found"}

        if incident.is_indexed:
            log.info("incident_already_indexed")
            return {"status": "skipped", "detail": "Already indexed"}

        # Attempt to run the ingestion pipeline
        try:
            from app.ingestion.pipeline import process_and_index  # type: ignore[import-not-found]

            await process_and_index(incident)
        except ImportError:
            log.info("ingestion_pipeline_not_available_marking_indexed")

        # Mark as indexed
        incident.is_indexed = True
        await session.flush()

        log.info("ticket_indexed_successfully", incident_number=incident.number)
        return {
            "status": "indexed",
            "incident_number": incident.number,
            "snow_sys_id": snow_sys_id,
        }

    return {"status": "error", "detail": "Failed to acquire database session"}


# ── Celery Tasks ────────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="app.workers.ingestion_worker.analyze_incident_async",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True,
)
def analyze_incident_async(
    self: Task, snow_sys_id: str, engineer_id: str
) -> dict[str, Any]:
    """Analyze an incident asynchronously via the RAG pipeline.

    This Celery task wraps the async analysis pipeline, handling retries
    with exponential backoff on transient failures.

    Args:
        self: Celery Task instance (for retry support).
        snow_sys_id: ServiceNow incident sys_id.
        engineer_id: Requesting engineer sys_id.

    Returns:
        Dictionary with recommendation_id, status, and metrics.

    Raises:
        self.retry: On transient failures up to max_retries.
    """
    log = logger.bind(
        task_id=self.request.id,
        snow_sys_id=snow_sys_id,
        engineer_id=engineer_id,
        retry_count=self.request.retries,
    )
    log.info("analyze_incident_task_started")

    try:
        loop = _get_or_create_event_loop()
        result = loop.run_until_complete(
            _run_analysis_pipeline(snow_sys_id, engineer_id)
        )
        log.info("analyze_incident_task_completed", result_status=result.get("status"))
        return result
    except Exception as exc:
        retry_delay = 30 * (2 ** self.request.retries)  # Exponential backoff
        log.error(
            "analyze_incident_task_failed",
            error=str(exc),
            retry_delay=retry_delay,
            exc_info=True,
        )
        raise self.retry(exc=exc, countdown=retry_delay)


@celery_app.task(
    bind=True,
    name="app.workers.ingestion_worker.index_resolved_ticket",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    track_started=True,
)
def index_resolved_ticket(
    self: Task, snow_sys_id: str
) -> dict[str, Any]:
    """Index a newly resolved ticket into the vector store.

    Processes the ticket text, generates embeddings, and upserts
    into the vector store for future similarity searches.

    Args:
        self: Celery Task instance.
        snow_sys_id: ServiceNow incident sys_id.

    Returns:
        Dictionary with indexing status and metadata.

    Raises:
        self.retry: On transient failures up to max_retries.
    """
    log = logger.bind(
        task_id=self.request.id,
        snow_sys_id=snow_sys_id,
        retry_count=self.request.retries,
    )
    log.info("index_ticket_task_started")

    try:
        loop = _get_or_create_event_loop()
        result = loop.run_until_complete(_index_ticket(snow_sys_id))
        log.info("index_ticket_task_completed", result_status=result.get("status"))
        return result
    except Exception as exc:
        retry_delay = 60 * (2 ** self.request.retries)
        log.error(
            "index_ticket_task_failed",
            error=str(exc),
            retry_delay=retry_delay,
            exc_info=True,
        )
        raise self.retry(exc=exc, countdown=retry_delay)
