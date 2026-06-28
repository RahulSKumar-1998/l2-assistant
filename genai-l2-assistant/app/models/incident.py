"""Pydantic models for incident data structures.

Defines data models for ServiceNow incidents, processed tickets,
text chunks, and query parameters used across the application.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class IncidentState(str, Enum):
    """ServiceNow incident states."""
    NEW = "1"
    IN_PROGRESS = "2"
    ON_HOLD = "3"
    RESOLVED = "6"
    CLOSED = "7"
    CANCELLED = "8"


class IncidentType(str, Enum):
    """Classified incident types."""
    APPLICATION_ERROR = "application_error"
    INFRASTRUCTURE = "infrastructure"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ACCESS_MANAGEMENT = "access_management"
    DATA_ISSUE = "data_issue"
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    """Type of text chunk."""
    DESCRIPTION = "description"
    WORK_NOTES = "work_notes"
    RESOLUTION = "resolution"
    KB_ARTICLE = "kb_article"


class SourceType(str, Enum):
    """Type of source document."""
    INCIDENT = "incident"
    KB_ARTICLE = "kb_article"
    RUNBOOK = "runbook"


# ── ServiceNow Record Models ────────────────────────────────────────────────


class IncidentRecord(BaseModel):
    """Raw incident record from ServiceNow REST API."""
    sys_id: str = Field(..., description="ServiceNow sys_id")
    number: str = Field(..., description="Incident number (e.g., INC0042871)")
    short_description: str = Field(default="", description="Incident short description")
    description: str = Field(default="", description="Full incident description")
    category: str = Field(default="", description="Incident category")
    subcategory: str = Field(default="", description="Incident subcategory")
    priority: int = Field(default=4, ge=1, le=5, description="Priority 1 (Critical) to 5 (Planning)")
    state: str = Field(default="1", description="Incident state code")
    assignment_group: str = Field(default="", description="Assignment group name")
    assigned_to: str = Field(default="", description="Assigned engineer")
    cmdb_ci: str = Field(default="", description="Impacted CI from CMDB")
    opened_at: Optional[datetime] = Field(default=None, description="When the incident was opened")
    resolved_at: Optional[datetime] = Field(default=None, description="When the incident was resolved")
    work_notes: str = Field(default="", description="Work notes history")
    resolution_notes: Optional[str] = Field(default=None, description="Resolution notes")
    root_cause: Optional[str] = Field(default=None, description="Root cause (custom field u_root_cause)")


class KBArticle(BaseModel):
    """Knowledge base article from ServiceNow."""
    sys_id: str = Field(..., description="ServiceNow sys_id")
    number: str = Field(..., description="KB article number (e.g., KB0012345)")
    short_description: str = Field(default="", description="Article title")
    text: str = Field(default="", description="Article body content (HTML)")
    category: str = Field(default="", description="Article category")
    valid_to: Optional[datetime] = Field(default=None, description="Article validity end date")
    workflow_state: str = Field(default="published", description="Publication state")


class CMDBRecord(BaseModel):
    """CMDB Configuration Item record."""
    sys_id: str = Field(..., description="ServiceNow sys_id")
    name: str = Field(..., description="CI name")
    sys_class_name: str = Field(default="", description="CI class (e.g., cmdb_ci_service)")
    operational_status: str = Field(default="1", description="Operational status code")
    environment: str = Field(default="", description="Environment (production, staging, dev)")
    service_tier: str = Field(default="", description="Service tier (custom field u_service_tier)")
    relationships: list[dict] = Field(
        default_factory=list,
        description="Related CIs (upstream/downstream services)",
    )


# ── Query Parameters ────────────────────────────────────────────────────────


class IncidentQueryParams(BaseModel):
    """Query parameters for listing incidents from ServiceNow."""
    state: Optional[str] = Field(default=None, description="Filter by incident state")
    assignment_group: Optional[str] = Field(default=None, description="Filter by assignment group")
    opened_at_start: Optional[datetime] = Field(default=None, description="Opened after this date")
    opened_at_end: Optional[datetime] = Field(default=None, description="Opened before this date")
    category: Optional[str] = Field(default=None, description="Filter by category")
    limit: int = Field(default=100, ge=1, le=1000, description="Max results per page")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class KBQueryParams(BaseModel):
    """Query parameters for listing KB articles."""
    category: Optional[str] = Field(default=None, description="Filter by category")
    workflow_state: str = Field(default="published", description="Publication state filter")
    limit: int = Field(default=100, ge=1, le=1000, description="Max results per page")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


# ── Processed Data Models ───────────────────────────────────────────────────


class ExtractedEntity(BaseModel):
    """Named entity extracted from incident text."""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label (SERVICE, ERROR_CODE, HOSTNAME, etc.)")
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")


class ProcessedTicket(BaseModel):
    """Ticket after NLP preprocessing."""
    source_id: str = Field(..., description="Incident number (e.g., INC0042871)")
    sys_id: str = Field(..., description="ServiceNow sys_id")
    cleaned_text: str = Field(..., description="Cleaned and normalized text")
    entities: list[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    keywords: list[str] = Field(default_factory=list, description="Top TF-IDF keywords")
    category: str = Field(default="", description="Incident category")
    subcategory: str = Field(default="", description="Incident subcategory")
    incident_type: IncidentType = Field(
        default=IncidentType.UNKNOWN,
        description="Classified incident type",
    )
    summary: str = Field(default="", description="2-sentence extractive summary")
    priority: int = Field(default=4, ge=1, le=5)
    cmdb_ci: str = Field(default="")
    assignment_group: str = Field(default="")
    opened_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    root_cause: Optional[str] = None
    work_notes: str = Field(default="")


class TextChunk(BaseModel):
    """A text chunk ready for embedding and indexing."""
    chunk_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique chunk identifier",
    )
    chunk_text: str = Field(..., description="The text content of this chunk")
    chunk_type: ChunkType = Field(..., description="Type of content in this chunk")
    source_id: str = Field(..., description="Source document ID (incident number or KB number)")
    source_type: SourceType = Field(
        default=SourceType.INCIDENT,
        description="Type of source document",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (category, priority, cmdb_ci, etc.)",
    )
    chunk_index: int = Field(default=0, description="Index of this chunk within the source document")
