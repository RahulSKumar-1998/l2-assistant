"""Incident analysis API routes.

Provides endpoints for:
- Triggering RAG pipeline analysis on incidents.
- Retrieving cached or fresh recommendations.
- Finding similar historical incidents.
- Receiving ServiceNow webhook notifications.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware import validate_hmac_signature
from app.config import get_settings
from app.models.recommendation import RecommendationResult, SimilarIncident
from app.storage.postgres import (
    IncidentDB,
    RecommendationDB,
    get_db_session,
)

router = APIRouter(prefix="/incidents", tags=["incidents"])
logger = structlog.get_logger(__name__)

# ── Request / Response Models ───────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Request body for incident analysis."""

    snow_sys_id: str = Field(..., description="ServiceNow incident sys_id")
    engineer_id: str = Field(..., description="Requesting engineer sys_id")


class AnalyzeResponse(BaseModel):
    """Response from incident analysis."""

    recommendation_id: str = Field(..., description="UUID of the generated recommendation")
    status: str = Field(..., description="Processing status: completed | queued")
    result: Optional[RecommendationResult] = Field(
        default=None,
        description="Recommendation result (present when status is 'completed')",
    )


class SimilarIncidentResponse(BaseModel):
    """Response listing similar historical incidents."""

    snow_sys_id: str = Field(..., description="Source incident sys_id")
    similar: list[SimilarIncident] = Field(default_factory=list)
    count: int = Field(default=0, description="Number of similar incidents returned")


class WebhookResponse(BaseModel):
    """Response for webhook acceptance."""

    status: str = Field(default="accepted", description="Webhook processing status")
    task_id: str = Field(default="", description="Background task ID")


class IncidentListItem(BaseModel):
    """Item in the incident list return model."""
    id: UUID
    snow_sys_id: str
    number: str
    short_description: str
    category: str
    priority: int
    state: str
    cmdb_ci: str
    opened_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    ai_status: str  # "analyzed", "low_confidence", "not_analyzed"
    ai_confidence: Optional[float] = None
    recommendation_id: Optional[UUID] = None


# ── Recommendation Cache TTL ────────────────────────────────────────────────

_RECOMMENDATION_TTL = timedelta(minutes=30)


# ── Helper Functions ────────────────────────────────────────────────────────


async def _get_incident_by_sys_id(
    session: AsyncSession, snow_sys_id: str
) -> Optional[IncidentDB]:
    """Fetch an incident row by ServiceNow sys_id.

    Args:
        session: Active database session.
        snow_sys_id: ServiceNow sys_id.

    Returns:
        The IncidentDB row, or None if not found.
    """
    stmt = select(IncidentDB).where(IncidentDB.snow_sys_id == snow_sys_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def _get_latest_recommendation(
    session: AsyncSession, incident_id: UUID
) -> Optional[RecommendationDB]:
    """Get the most recent recommendation for an incident.

    Args:
        session: Active database session.
        incident_id: Internal incident UUID.

    Returns:
        The latest RecommendationDB row, or None.
    """
    stmt = (
        select(RecommendationDB)
        .where(RecommendationDB.incident_id == incident_id)
        .order_by(RecommendationDB.created_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


def _recommendation_db_to_result(rec: RecommendationDB) -> RecommendationResult:
    """Convert a database recommendation row to the API response model.

    Args:
        rec: The database recommendation row.

    Returns:
        A RecommendationResult Pydantic model.
    """
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


# ── Routes ──────────────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=list[IncidentListItem],
    summary="List all incidents with their latest AI recommendation status",
    description="Returns a list of all incidents in the database combined with their AI analysis status.",
)
async def list_incidents() -> list[IncidentListItem]:
    """Retrieve all incidents with their latest AI recommendation status.

    Uses an outer join to combine IncidentDB with the latest RecommendationDB.
    """
    from sqlalchemy import func
    
    async for session in get_db_session():
        # Subquery to identify the latest recommendation created_at timestamp per incident
        subq = (
            select(
                RecommendationDB.incident_id,
                func.max(RecommendationDB.created_at).label("max_created_at")
            )
            .group_by(RecommendationDB.incident_id)
            .subquery()
        )
        
        # Main query joining incidents with the latest recommendation
        stmt = (
            select(IncidentDB, RecommendationDB)
            .outerjoin(subq, IncidentDB.id == subq.c.incident_id)
            .outerjoin(
                RecommendationDB,
                (IncidentDB.id == RecommendationDB.incident_id) & 
                (RecommendationDB.created_at == subq.c.max_created_at)
            )
            .order_by(IncidentDB.number.asc())
        )
        
        db_result = await session.execute(stmt)
        rows = db_result.all()
        
        items = []
        for inc_db, rec_db in rows:
            ai_status = "not_analyzed"
            ai_confidence = None
            rec_id = None
            
            if rec_db:
                rec_id = rec_db.id
                ai_confidence = rec_db.confidence_score
                if rec_db.confidence_score >= 0.6:
                    ai_status = "analyzed"
                else:
                    ai_status = "low_confidence"
            
            items.append(
                IncidentListItem(
                    id=inc_db.id,
                    snow_sys_id=inc_db.snow_sys_id,
                    number=inc_db.number,
                    short_description=inc_db.short_description,
                    category=inc_db.category,
                    priority=inc_db.priority,
                    state=inc_db.state,
                    cmdb_ci=inc_db.cmdb_ci,
                    opened_at=inc_db.opened_at,
                    resolved_at=inc_db.resolved_at,
                    ai_status=ai_status,
                    ai_confidence=ai_confidence,
                    recommendation_id=rec_id,
                )
            )
        return items
    return []


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze incident via RAG pipeline",
    description="Triggers the full RAG pipeline for the given ServiceNow incident.",
)
async def analyze_incident(body: AnalyzeRequest) -> AnalyzeResponse:
    """Trigger RAG pipeline analysis for an incident.

    Dispatches the analysis as a Celery task and returns the task ID.
    If the incident already has a recent recommendation (<30 min old),
    returns it immediately without re-running the pipeline.

    Args:
        body: The analysis request with snow_sys_id and engineer_id.

    Returns:
        AnalyzeResponse with recommendation_id, status, and optional result.
    """
    log = logger.bind(
        snow_sys_id=body.snow_sys_id,
        engineer_id=body.engineer_id,
    )

    # Check for existing recent recommendation
    async for session in get_db_session():
        incident = await _get_incident_by_sys_id(session, body.snow_sys_id)
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident with sys_id {body.snow_sys_id} not found",
            )
        
        # Exclude resolved/closed incidents from AI analysis
        if incident.state in ("6", "7", "resolved", "closed"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resolved or closed incidents cannot be submitted for AI analysis.",
            )

        rec = await _get_latest_recommendation(session, incident.id)
        if rec and rec.created_at:
            rec_age = datetime.now(timezone.utc) - rec.created_at.replace(
                tzinfo=timezone.utc
            )
            if rec_age < _RECOMMENDATION_TTL:
                log.info(
                    "returning_cached_recommendation",
                    recommendation_id=str(rec.id),
                    age_seconds=rec_age.total_seconds(),
                )
                return AnalyzeResponse(
                    recommendation_id=str(rec.id),
                    status="completed",
                    result=_recommendation_db_to_result(rec),
                )

    # Enqueue async analysis via Celery
    try:
        from app.workers.ingestion_worker import analyze_incident_async

        task = analyze_incident_async.delay(body.snow_sys_id, body.engineer_id)
        log.info("analysis_task_enqueued", task_id=task.id)

        return AnalyzeResponse(
            recommendation_id=task.id,
            status="queued",
            result=None,
        )
    except Exception as exc:
        log.error("analysis_enqueue_failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to enqueue analysis task. Please retry.",
        ) from exc


@router.get(
    "/{snow_sys_id}/recommendation",
    response_model=AnalyzeResponse,
    summary="Get incident recommendation",
    description="Returns cached recommendation if < 30 min old, else triggers fresh analysis.",
)
async def get_recommendation(snow_sys_id: str) -> AnalyzeResponse:
    """Retrieve the recommendation for an incident.

    Returns a cached recommendation if it is less than 30 minutes old.
    Otherwise triggers a fresh analysis via the Celery task queue.

    Args:
        snow_sys_id: The ServiceNow incident sys_id.

    Returns:
        AnalyzeResponse with recommendation data.

    Raises:
        HTTPException: If the incident is not found in the database.
    """
    log = logger.bind(snow_sys_id=snow_sys_id)

    async for session in get_db_session():
        incident = await _get_incident_by_sys_id(session, snow_sys_id)
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {snow_sys_id} not found",
            )

        rec = await _get_latest_recommendation(session, incident.id)
        if rec and rec.created_at:
            rec_age = datetime.now(timezone.utc) - rec.created_at.replace(
                tzinfo=timezone.utc
            )
            if rec_age < _RECOMMENDATION_TTL:
                log.info("returning_cached_recommendation", age_seconds=rec_age.total_seconds())
                return AnalyzeResponse(
                    recommendation_id=str(rec.id),
                    status="completed",
                    result=_recommendation_db_to_result(rec),
                )

        # Stale or missing — trigger fresh analysis. Check if resolved first!
        if incident.state in ("6", "7", "resolved", "closed"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resolved or closed incidents cannot be submitted for AI analysis.",
            )

        log.info("triggering_fresh_analysis")
        try:
            from app.workers.ingestion_worker import analyze_incident_async

            task = analyze_incident_async.delay(snow_sys_id, "system")
            return AnalyzeResponse(
                recommendation_id=task.id,
                status="queued",
                result=None,
            )
        except Exception as exc:
            log.error("fresh_analysis_failed", error=str(exc), exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to enqueue analysis task.",
            ) from exc

    # Fallback — should not reach here
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error retrieving recommendation.",
    )


@router.get(
    "/{snow_sys_id}/similar",
    response_model=SimilarIncidentResponse,
    summary="Find similar incidents",
    description="Query the vector store for historically similar incidents.",
)
async def get_similar_incidents(
    snow_sys_id: str,
    top_n: int = Query(default=5, ge=1, le=50, description="Number of similar incidents to return"),
    min_similarity: float = Query(
        default=0.7, ge=0.0, le=1.0, description="Minimum cosine similarity threshold"
    ),
) -> SimilarIncidentResponse:
    """Find incidents similar to the given one.

    Queries the vector store using the incident's embedded representation
    and returns the top_n most similar historical incidents above
    the min_similarity threshold.

    Args:
        snow_sys_id: The ServiceNow incident sys_id.
        top_n: Maximum number of results (default 5).
        min_similarity: Minimum similarity score (default 0.7).

    Returns:
        SimilarIncidentResponse with ranked similar incidents.

    Raises:
        HTTPException: If the incident is not found.
    """
    log = logger.bind(
        snow_sys_id=snow_sys_id,
        top_n=top_n,
        min_similarity=min_similarity,
    )

    async for session in get_db_session():
        incident = await _get_incident_by_sys_id(session, snow_sys_id)
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {snow_sys_id} not found",
            )

        # Check if there's a recommendation with similar incidents already
        rec = await _get_latest_recommendation(session, incident.id)
        if rec and rec.similar_incidents:
            raw_similar = rec.similar_incidents or []
            filtered: list[SimilarIncident] = []
            for item in raw_similar:
                if isinstance(item, dict):
                    sim = SimilarIncident(**item)
                else:
                    sim = item
                if sim.similarity_score >= min_similarity:
                    filtered.append(sim)
            filtered = sorted(
                filtered, key=lambda s: s.similarity_score, reverse=True
            )[:top_n]

            log.info("returning_similar_incidents", count=len(filtered))
            return SimilarIncidentResponse(
                snow_sys_id=snow_sys_id,
                similar=filtered,
                count=len(filtered),
            )

        # No recommendation exists — return empty result
        log.info("no_similar_incidents_available")
        return SimilarIncidentResponse(
            snow_sys_id=snow_sys_id,
            similar=[],
            count=0,
        )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error retrieving similar incidents.",
    )


@router.post(
    "/{snow_sys_id}/webhook",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="ServiceNow webhook receiver",
    description="Receives webhook notifications from ServiceNow and enqueues analysis tasks.",
)
async def servicenow_webhook(
    snow_sys_id: str,
    request: Request,
) -> WebhookResponse:
    """Receive a ServiceNow webhook for an incident.

    Validates the HMAC signature from the X-ServiceNow-Signature header,
    then enqueues a Celery task for async incident analysis.

    Args:
        snow_sys_id: The ServiceNow incident sys_id from the URL path.
        request: The raw HTTP request for body and header access.

    Returns:
        WebhookResponse with acceptance status and task ID.

    Raises:
        HTTPException: 401 if HMAC validation fails, 503 on task queue errors.
    """
    log = logger.bind(snow_sys_id=snow_sys_id)
    settings = get_settings()

    # Read raw body for HMAC validation
    body = await request.body()
    signature = request.headers.get("X-ServiceNow-Signature", "")

    webhook_secret = settings.servicenow.webhook_secret
    if webhook_secret:
        if not validate_hmac_signature(body, signature, webhook_secret):
            log.warning(
                "webhook_hmac_validation_failed",
                signature_present=bool(signature),
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )
    else:
        log.warning("webhook_secret_not_configured")

    # Enqueue analysis task
    try:
        from app.workers.ingestion_worker import analyze_incident_async

        task = analyze_incident_async.delay(snow_sys_id, "webhook")
        log.info("webhook_task_enqueued", task_id=task.id)

        return WebhookResponse(
            status="accepted",
            task_id=task.id,
        )
    except Exception as exc:
        log.error("webhook_enqueue_failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to enqueue webhook task.",
        ) from exc
