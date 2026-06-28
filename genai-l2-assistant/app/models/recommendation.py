"""Pydantic models for AI recommendation results.

Defines the structured output format from the RAG pipeline,
including root cause predictions, triage steps, and resolution drafts.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TriageStep(BaseModel):
    """A single triage/resolution step recommended by the AI."""
    step: int = Field(..., description="Step number (1-indexed)")
    action: str = Field(..., description="Action to take")
    rationale: str = Field(default="", description="Why this step is recommended")
    command: Optional[str] = Field(
        default=None,
        description="Optional CLI command or script to execute",
    )


class SimilarIncident(BaseModel):
    """A historically similar incident found during retrieval."""
    number: str = Field(..., description="Incident number (e.g., INC0039201)")
    sys_id: str = Field(default="", description="ServiceNow sys_id")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    resolution_summary: str = Field(default="", description="Summary of how this was resolved")
    resolution_time_min: Optional[int] = Field(
        default=None,
        description="Time to resolve in minutes",
    )
    category: str = Field(default="", description="Incident category")


class KBReference(BaseModel):
    """A knowledge base article reference found during retrieval."""
    kb_number: str = Field(..., description="KB article number (e.g., KB0012345)")
    title: str = Field(default="", description="Article title")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class SourceReference(BaseModel):
    """Reference to a source document used in the recommendation."""
    source_id: str = Field(..., description="Document ID (incident number or KB number)")
    source_type: str = Field(..., description="incident, kb_article, or runbook")
    relevance_score: float = Field(default=0.0, description="How relevant this source was")


class RecommendationResult(BaseModel):
    """Complete AI recommendation result from the RAG pipeline.

    This is the primary output model that gets returned to the API
    and displayed in the ServiceNow AI sidebar widget.
    """
    id: UUID = Field(default_factory=uuid4, description="Recommendation UUID")
    incident_id: Optional[UUID] = Field(default=None, description="Associated incident UUID")
    snow_sys_id: str = Field(default="", description="ServiceNow incident sys_id")

    # AI-generated content
    root_cause_prediction: str = Field(
        ...,
        description="2-3 sentence root cause explanation",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the prediction (0.0 to 1.0)",
    )
    triage_steps: list[TriageStep] = Field(
        default_factory=list,
        description="Ordered triage/resolution steps",
    )
    resolution_draft: str = Field(
        default="",
        description="Draft resolution note for ServiceNow work notes",
    )
    escalate_to_l3: bool = Field(
        default=False,
        description="Whether L3 escalation is recommended",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for L3 escalation (if recommended)",
    )

    # Retrieved context references
    similar_incidents: list[SimilarIncident] = Field(
        default_factory=list,
        description="Top similar historical incidents",
    )
    kb_references: list[KBReference] = Field(
        default_factory=list,
        description="Relevant KB article references",
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="Source IDs used to generate recommendation",
    )

    # Performance metrics
    retrieval_latency_ms: int = Field(default=0, description="Retrieval latency in milliseconds")
    generation_latency_ms: int = Field(default=0, description="LLM generation latency in milliseconds")

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this recommendation was generated",
    )
