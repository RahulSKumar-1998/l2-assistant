"""Celery tasks for incident analysis and ticket indexing.

Provides async task wrappers around the RAG pipeline for:
- Full incident analysis (analyze_incident_async)
- Indexing resolved tickets into the vector store (index_resolved_ticket)
"""

import asyncio
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
    from app.core.context_assembler import ContextAssembler
    from app.core.embedder import Embedder
    from app.core.llm_client import LLMClient
    from app.core.rag_pipeline import RAGPipeline
    from app.core.reranker import Reranker
    from app.core.retriever import HybridRetriever
    from app.ingestion.mock_client import MockServiceNowClient
    from app.ingestion.pipeline import store_incident_record
    from app.ingestion.servicenow_client import ServiceNowClient
    from app.storage.postgres import (
        IncidentDB,
        get_db_session,
        get_session_factory,
    )
    from app.storage.vector_store import get_vector_store
    from sqlalchemy import select

    log = logger.bind(snow_sys_id=snow_sys_id, engineer_id=engineer_id)
    log.info("analysis_pipeline_started")

    settings = get_settings()

    # Choose a real ServiceNow client when credentials exist, otherwise use fixtures.
    use_mock_client = not settings.servicenow.username and not settings.servicenow.client_id
    snow_client_cls = MockServiceNowClient if use_mock_client else ServiceNowClient

    # Step 1: Look up or create the incident in our database
    async for session in get_db_session():
        stmt = select(IncidentDB).where(IncidentDB.snow_sys_id == snow_sys_id)
        result = await session.execute(stmt)
        incident = result.scalar_one_or_none()

        if not incident:
            log.info("incident_not_in_db_fetching_from_source", source="mock" if use_mock_client else "servicenow")
            async with snow_client_cls() as snow_client:  # type: ignore[call-arg]
                incident_record = await snow_client.get_incident(snow_sys_id)
            incident = await store_incident_record(incident_record, session=session)

        # Refresh full current incident data if possible before analysis.
        try:
            async with snow_client_cls() as snow_client:  # type: ignore[call-arg]
                latest_record = await snow_client.get_incident(snow_sys_id)
            incident = await store_incident_record(latest_record, session=session)
        except Exception:
            log.warning("incident_refresh_failed_using_db_copy", exc_info=True)

        from app.ingestion.pipeline import incident_db_to_record
        from app.ingestion.ticket_processor import TicketPreprocessor

        vector_store = get_vector_store()
        embedder = Embedder()
        retriever = HybridRetriever(embedder=embedder, vector_store=vector_store)
        reranker = Reranker()
        context_assembler = ContextAssembler()
        llm_client = LLMClient()
        rag_pipeline = RAGPipeline(
            embedder=embedder,
            retriever=retriever,
            reranker=reranker,
            context_assembler=context_assembler,
            llm_client=llm_client,
            db_session_factory=get_session_factory(),
        )

        incident_record = incident_db_to_record(incident)
        processed_ticket = TicketPreprocessor().preprocess(incident_record)
        recommendation = await rag_pipeline.analyze_incident(
            ticket=processed_ticket,
            incident_db_id=incident.id,
        )

        log.info(
            "analysis_pipeline_completed",
            recommendation_id=str(recommendation.id),
            confidence=recommendation.confidence_score,
            retrieval_ms=recommendation.retrieval_latency_ms,
            generation_ms=recommendation.generation_latency_ms,
        )

        return {
            "recommendation_id": str(recommendation.id),
            "incident_id": str(incident.id),
            "status": "completed",
            "confidence_score": recommendation.confidence_score,
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

        from app.ingestion.pipeline import process_and_index

        result = await process_and_index(incident)

        # Mark as indexed
        incident.is_indexed = True
        await session.flush()

        log.info("ticket_indexed_successfully", incident_number=incident.number)
        return {
            "status": "indexed",
            "incident_number": incident.number,
            "snow_sys_id": snow_sys_id,
            "total_chunks": result["total_chunks"],
            "upserted_count": result["upserted_count"],
            "failed_count": result["failed_count"],
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
