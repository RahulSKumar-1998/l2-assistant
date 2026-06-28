"""Recommendation retrieval API routes.

Provides a single endpoint for fetching the full recommendation result
for a given incident, identified by its internal UUID.
"""

from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recommendation import RecommendationResult
from app.storage.postgres import (
    IncidentDB,
    RecommendationDB,
    get_db_session,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
logger = structlog.get_logger(__name__)


@router.get(
    "/{incident_id}",
    response_model=RecommendationResult,
    summary="Get recommendation by incident ID",
    description="Returns the most recent AI recommendation for the given incident UUID.",
)
async def get_recommendation(incident_id: UUID) -> RecommendationResult:
    """Fetch the latest recommendation for an incident.

    Args:
        incident_id: The internal incident UUID (from the incidents table).

    Returns:
        The full RecommendationResult including root cause, triage steps,
        similar incidents, KB references, and performance metrics.

    Raises:
        HTTPException: 404 if the incident or recommendation is not found.
    """
    log = logger.bind(incident_id=str(incident_id))

    async for session in get_db_session():
        # Verify the incident exists
        incident_stmt = select(IncidentDB).where(IncidentDB.id == incident_id)
        incident_result = await session.execute(incident_stmt)
        incident = incident_result.scalar_one_or_none()

        if not incident:
            log.warning("incident_not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found",
            )

        # Fetch the latest recommendation
        rec_stmt = (
            select(RecommendationDB)
            .where(RecommendationDB.incident_id == incident_id)
            .order_by(RecommendationDB.created_at.desc())
            .limit(1)
        )
        rec_result = await session.execute(rec_stmt)
        rec = rec_result.scalar_one_or_none()

        if not rec:
            log.info("no_recommendation_for_incident")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No recommendation found for incident {incident_id}",
            )

        log.info(
            "recommendation_retrieved",
            recommendation_id=str(rec.id),
            confidence=rec.confidence_score,
        )

        return RecommendationResult(
            id=rec.id,
            incident_id=rec.incident_id,
            root_cause_prediction=rec.root_cause_prediction,
            confidence_score=rec.confidence_score,
            triage_steps=rec.triage_steps or [],
            similar_incidents=rec.similar_incidents or [],
            kb_references=rec.kb_references or [],
            resolution_draft=rec.resolution_draft,
            retrieval_latency_ms=rec.retrieval_latency_ms,
            generation_latency_ms=rec.generation_latency_ms,
            created_at=rec.created_at,
        )

    # Should never reach here
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error retrieving recommendation.",
    )
