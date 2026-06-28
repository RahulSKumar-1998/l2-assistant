"""Pydantic models for engineer feedback on AI recommendations."""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FeedbackRecord(BaseModel):
    """Feedback submitted by an engineer on a recommendation.

    Rating scale:
        1 = thumbs down (unhelpful/incorrect)
        5 = thumbs up (helpful/accurate)
    """
    id: UUID = Field(default_factory=uuid4, description="Feedback record UUID")
    recommendation_id: UUID = Field(..., description="Associated recommendation UUID")
    incident_id: UUID = Field(..., description="Associated incident UUID")
    engineer_id: str = Field(..., description="ServiceNow user sys_id")
    rating: int = Field(..., ge=1, le=5, description="Rating: 1=thumbs_down, 5=thumbs_up")
    comment: Optional[str] = Field(default=None, description="Optional engineer comment")
    acted_on_steps: Optional[list[int]] = Field(
        default=None,
        description="Which triage step numbers the engineer followed",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When feedback was submitted",
    )


class FeedbackSubmission(BaseModel):
    """API request body for submitting feedback."""
    recommendation_id: UUID = Field(..., description="Recommendation to provide feedback on")
    rating: int = Field(..., ge=1, le=5, description="Rating: 1=thumbs_down, 5=thumbs_up")
    comment: Optional[str] = Field(default=None, description="Optional comment")
    acted_on_steps: Optional[list[int]] = Field(
        default=None,
        description="Step numbers the engineer acted on",
    )
    engineer_id: str = Field(..., description="ServiceNow user sys_id")


class FeedbackResponse(BaseModel):
    """API response after recording feedback."""
    id: UUID = Field(..., description="Feedback record UUID")
    status: str = Field(default="recorded", description="Processing status")


class FeedbackStats(BaseModel):
    """Aggregated feedback statistics."""
    total_feedback: int = Field(default=0, description="Total feedback records processed")
    positive_count: int = Field(default=0, description="Number of positive (rating=5) records")
    negative_count: int = Field(default=0, description="Number of negative (rating=1) records")
    positive_rate: float = Field(default=0.0, description="Positive feedback rate (0.0-1.0)")
    sources_updated: int = Field(default=0, description="Number of source quality scores updated")


class FeedbackWeight(BaseModel):
    """Quality score for a source document based on feedback signals."""
    source_id: str = Field(..., description="Source document ID")
    source_type: str = Field(..., description="Source type (incident, kb_article, runbook)")
    quality_score: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Quality weight (>1 = boosted, <1 = penalised)",
    )
    positive_signals: int = Field(default=0, description="Total positive feedback signals")
    negative_signals: int = Field(default=0, description="Total negative feedback signals")
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the quality score was last computed",
    )
