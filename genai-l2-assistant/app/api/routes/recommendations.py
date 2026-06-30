"""Recommendation retrieval API routes.

Provides a single endpoint for fetching the full recommendation result
for a given incident, identified by its internal UUID.

The GET /{id} endpoint accepts EITHER:
  - An internal RecommendationDB UUID, OR
  - A Celery task ID returned by POST /incidents/analyze when status="queued"

This dual-mode lookup allows the frontend to poll a single URL while the
Celery worker is still processing, receiving status=queued until the result
is ready.
"""

import uuid as uuid_mod
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.models.recommendation import RecommendationResult
from app.storage.postgres import (
    IncidentDB,
    RecommendationDB,
    get_db_session,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
logger = structlog.get_logger(__name__)


def _rec_db_to_result(rec: RecommendationDB) -> RecommendationResult:
    """Convert a RecommendationDB row to the Pydantic response model."""
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


def _check_celery_task(task_id: str) -> Optional[dict]:
    """Check the state of a Celery task by ID.

    Returns a dict with 'state' and optionally 'result' keys, or None if
    the task cannot be found (e.g. Redis unavailable or ID not a task).
    """
    try:
        from celery.result import AsyncResult

        from app.workers.celery_app import celery_app

        ar = AsyncResult(task_id, app=celery_app)
        state = ar.state  # PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
        if state in ("PENDING", "STARTED", "RECEIVED", "RETRY"):
            return {"state": state, "result": None}
        if state == "SUCCESS":
            return {"state": state, "result": ar.result}
        if state in ("FAILURE", "REVOKED"):
            return {"state": state, "result": None}
        return None
    except Exception:
        return None


@router.get(
    "/{recommendation_id}",
    response_model=RecommendationResult,
    summary="Get recommendation by ID or Celery task ID",
    description=(
        "Returns the AI recommendation for a given recommendation UUID. "
        "Also accepts a Celery task ID: while the task is still processing, "
        "returns HTTP 202 so the UI knows to keep polling."
    ),
    responses={
        202: {"description": "Analysis still in progress — keep polling"},
        404: {"description": "Recommendation not found"},
    },
)
async def get_recommendation(recommendation_id: str) -> RecommendationResult:
    """Fetch a recommendation by recommendation UUID or Celery task ID.

    The frontend calls this endpoint using the identifier returned by
    POST /incidents/analyze:
      - When status="completed" the identifier is a RecommendationDB UUID.
      - When status="queued"   the identifier is a Celery task ID.

    Args:
        recommendation_id: RecommendationDB UUID **or** Celery task ID string.

    Returns:
        The full RecommendationResult once the analysis is complete.

    Raises:
        HTTPException 202: Analysis is still queued/running — client should retry.
        HTTPException 404: No recommendation found for the given ID.
    """
    log = logger.bind(recommendation_id=recommendation_id)

    # ── Step 1: Try to parse as a UUID ──────────────────────────────────────
    rec_uuid: Optional[uuid_mod.UUID] = None
    try:
        from app.workers.celery_app import task_to_recommendation_map
        if recommendation_id in task_to_recommendation_map:
            recommendation_id = task_to_recommendation_map[recommendation_id]
        rec_uuid = uuid_mod.UUID(recommendation_id)
    except (ValueError, AttributeError):
        # Not a standard UUID — likely a Celery task ID (also UUID-shaped but
        # let's handle both paths gracefully).
        pass

    # ── Step 2: Try database lookup first if we have a valid UUID ──────────────────
    if rec_uuid is not None:
        async for session in get_db_session():
            rec_stmt = (
                select(RecommendationDB)
                .where(RecommendationDB.id == rec_uuid)
                .limit(1)
            )
            rec_result = await session.execute(rec_stmt)
            rec = rec_result.scalar_one_or_none()
            if rec:
                log.info(
                    "recommendation_retrieved",
                    recommendation_id=str(rec.id),
                    confidence=rec.confidence_score,
                )
                return _rec_db_to_result(rec)

    # ── Step 3: Check if this is a live Celery task (only if not found in DB) ────
    celery_info = _check_celery_task(recommendation_id)
    if celery_info is not None:
        state = celery_info["state"]
        if state in ("PENDING", "STARTED", "RECEIVED", "RETRY"):
            log.info("celery_task_still_processing", state=state)
            # Return 202 so the UI polling loop keeps going
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=f"Analysis in progress (state={state}). Please retry shortly.",
            )

        if state == "SUCCESS" and celery_info["result"]:
            # The worker result dict has recommendation_id → look it up in the DB
            result_dict = celery_info["result"]
            actual_rec_id_str = result_dict.get("recommendation_id")
            incident_id_str = result_dict.get("incident_id")

            if actual_rec_id_str:
                try:
                    rec_uuid = uuid_mod.UUID(actual_rec_id_str)
                except (ValueError, AttributeError):
                    pass

            if rec_uuid is not None:
                async for session in get_db_session():
                    rec_stmt = (
                        select(RecommendationDB)
                        .where(RecommendationDB.id == rec_uuid)
                        .limit(1)
                    )
                    rec_result = await session.execute(rec_stmt)
                    rec = rec_result.scalar_one_or_none()
                    if rec:
                        return _rec_db_to_result(rec)

            # Fall back to incident_id lookup if we couldn't parse rec uuid
            if rec_uuid is None and incident_id_str:
                try:
                    inc_uuid = uuid_mod.UUID(incident_id_str)
                    async for session in get_db_session():
                        rec_stmt = (
                            select(RecommendationDB)
                            .where(RecommendationDB.incident_id == inc_uuid)
                            .order_by(RecommendationDB.created_at.desc())
                            .limit(1)
                        )
                        rec_result = await session.execute(rec_stmt)
                        rec = rec_result.scalar_one_or_none()
                        if rec:
                            log.info(
                                "recommendation_found_via_incident_id",
                                recommendation_id=str(rec.id),
                            )
                            return _rec_db_to_result(rec)
                except (ValueError, AttributeError):
                    pass

        if state in ("FAILURE", "REVOKED"):
            log.warning("celery_task_failed_or_revoked", state=state)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis task failed or was revoked. Please trigger a new analysis.",
            )

    # ── Step 4: Final fallback: 404 Not Found ──────────────────────────────────
    log.warning("recommendation_not_found", recommendation_id=recommendation_id)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Recommendation or task {recommendation_id} not found",
    )

