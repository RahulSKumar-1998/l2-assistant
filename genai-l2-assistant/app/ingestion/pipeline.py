"""High-level ingestion helpers for storing incidents and indexing them.

This module bridges the existing ingestion, embedding, and storage layers so
workers and scripts can use one consistent API for:
- storing previous incidents in PostgreSQL
- converting stored incidents into processed/chunked documents
- vectorizing resolved incidents into the configured vector store
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.ticket_processor import TicketChunker, TicketPreprocessor
from app.models.incident import IncidentRecord
from app.storage.postgres import IncidentDB, get_session_factory

logger = structlog.get_logger(__name__)


async def store_incident_record(
    incident_record: IncidentRecord,
    *,
    session: AsyncSession | None = None,
) -> IncidentDB:
    """Insert or update an incident row from a ServiceNow-style incident record.

    This is the primary persistence entrypoint for historical incidents before
    they are later indexed/vectorized.
    """
    owns_session = session is None
    if session is None:
        factory = get_session_factory()
        session = factory()

    try:
        stmt = select(IncidentDB).where(IncidentDB.snow_sys_id == incident_record.sys_id)
        result = await session.execute(stmt)
        incident = result.scalar_one_or_none()

        if incident is None:
            incident = IncidentDB(
                snow_sys_id=incident_record.sys_id,
                number=incident_record.number,
                short_description=incident_record.short_description,
                description=incident_record.description,
                category=incident_record.category,
                subcategory=incident_record.subcategory,
                priority=incident_record.priority,
                state=incident_record.state,
                assignment_group=incident_record.assignment_group,
                assigned_to=incident_record.assigned_to,
                cmdb_ci=incident_record.cmdb_ci,
                opened_at=incident_record.opened_at,
                resolved_at=incident_record.resolved_at,
                resolution_notes=incident_record.resolution_notes,
                root_cause=incident_record.root_cause,
                is_indexed=False,
            )
            session.add(incident)
        else:
            incident.number = incident_record.number
            incident.short_description = incident_record.short_description
            incident.description = incident_record.description
            incident.category = incident_record.category
            incident.subcategory = incident_record.subcategory
            incident.priority = incident_record.priority
            incident.state = incident_record.state
            incident.assignment_group = incident_record.assignment_group
            incident.assigned_to = incident_record.assigned_to
            incident.cmdb_ci = incident_record.cmdb_ci
            incident.opened_at = incident_record.opened_at
            incident.resolved_at = incident_record.resolved_at
            incident.resolution_notes = incident_record.resolution_notes
            incident.root_cause = incident_record.root_cause
            incident.updated_at = datetime.now(timezone.utc)

        await session.flush()
        if owns_session:
            await session.commit()

        logger.info(
            "incident_record_stored",
            snow_sys_id=incident.snow_sys_id,
            number=incident.number,
            state=incident.state,
        )
        return incident
    finally:
        if owns_session:
            await session.close()


def incident_db_to_record(incident: IncidentDB) -> IncidentRecord:
    """Convert a stored DB incident row back into the domain incident model."""
    return IncidentRecord(
        sys_id=incident.snow_sys_id,
        number=incident.number,
        short_description=incident.short_description or "",
        description=incident.description or "",
        category=incident.category or "",
        subcategory=incident.subcategory or "",
        priority=incident.priority or 4,
        state=incident.state or "1",
        assignment_group=incident.assignment_group or "",
        assigned_to=incident.assigned_to or "",
        cmdb_ci=incident.cmdb_ci or "",
        opened_at=incident.opened_at,
        resolved_at=incident.resolved_at,
        work_notes="",
        resolution_notes=incident.resolution_notes,
        root_cause=incident.root_cause,
    )


async def process_and_index(
    incident: IncidentDB,
    *,
    namespace: str = "",
) -> dict[str, int | str]:
    """Process a stored resolved incident and upsert its chunks into the vector store.

    This is the worker-facing helper that makes the documented
    historical-incident → embedding → vectorization path runnable.
    """
    incident_record = incident_db_to_record(incident)

    preprocessor = TicketPreprocessor()
    chunker = TicketChunker()
    processed = preprocessor.preprocess(incident_record)
    chunks = chunker.chunk(processed)

    from app.ingestion.embedding_pipeline import EmbeddingPipeline
    from app.storage.vector_store import get_vector_store

    vector_store = get_vector_store()
    pipeline = EmbeddingPipeline(vector_store=vector_store)
    result = await pipeline.run_batch(chunks, namespace=namespace)

    incident.is_indexed = result.upserted_count > 0 and result.failed_count == 0

    logger.info(
        "incident_processed_and_indexed",
        number=incident.number,
        snow_sys_id=incident.snow_sys_id,
        total_chunks=result.total_chunks,
        upserted_count=result.upserted_count,
        failed_count=result.failed_count,
    )

    return {
        "total_chunks": result.total_chunks,
        "upserted_count": result.upserted_count,
        "failed_count": result.failed_count,
    }



