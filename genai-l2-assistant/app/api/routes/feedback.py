"""Feedback submission API routes.

Allows L2 engineers to submit thumbs-up/down feedback on AI recommendations,
which feeds into the nightly feedback processing loop for quality improvement.
"""

import uuid as uuid_mod
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status

from app.models.feedback import FeedbackResponse, FeedbackSubmission
from app.storage.postgres import (
    FeedbackDB,
    RecommendationDB,
    get_db_session,
)
from sqlalchemy import select

router = APIRouter(prefix="/feedback", tags=["feedback"])
logger = structlog.get_logger(__name__)


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback on a recommendation",
    description="Records engineer feedback (rating + optional comment) for a recommendation.",
)
async def submit_feedback(body: FeedbackSubmission) -> FeedbackResponse:
    """Store engineer feedback for a recommendation.

    Validates the referenced recommendation exists, then persists the
    feedback record to the database. The nightly feedback processor
    uses these records to adjust source quality scores.

    Args:
        body: FeedbackSubmission with recommendation_id, rating,
              optional comment, acted_on_steps, and engineer_id.

    Returns:
        FeedbackResponse with the feedback record ID and status.

    Raises:
        HTTPException: 404 if the recommendation doesn't exist,
                       500 on database errors.
    """
    log = logger.bind(
        recommendation_id=str(body.recommendation_id),
        engineer_id=body.engineer_id,
        rating=body.rating,
    )

    async for session in get_db_session():
        # Verify the recommendation exists
        rec_stmt = select(RecommendationDB).where(
            RecommendationDB.id == body.recommendation_id
        )
        rec_result = await session.execute(rec_stmt)
        recommendation = rec_result.scalar_one_or_none()

        if not recommendation:
            log.warning("recommendation_not_found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation {body.recommendation_id} not found",
            )

        # Create the feedback record
        feedback_id = uuid_mod.uuid4()
        feedback = FeedbackDB(
            id=feedback_id,
            recommendation_id=body.recommendation_id,
            incident_id=recommendation.incident_id,
            engineer_id=body.engineer_id,
            rating=body.rating,
            comment=body.comment,
            acted_on_steps=body.acted_on_steps,
        )

        session.add(feedback)
        await session.flush()

        log.info(
            "feedback_recorded",
            feedback_id=str(feedback_id),
            incident_id=str(recommendation.incident_id),
        )

        return FeedbackResponse(
            id=feedback_id,
            status="recorded",
        )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error recording feedback.",
    )
