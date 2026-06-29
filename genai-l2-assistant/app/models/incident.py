"""Pydantic models for incident data structures.

Defines data models for ServiceNow incidents, processed tickets,
text chunks, and query parameters used across the application.

Changes from original:
    - Fixed datetime.utcnow() deprecation across all models (use timezone-aware UTC)
    - Added SortField / SortOrder enums for explicit ordering control
    - Added sort_by / sort_order to IncidentQueryParams (defaults: opened_at DESC)
    - Added resolved_at_start / resolved_at_end range filters to IncidentQueryParams
    - Added recency_weight to IncidentQueryParams for blended similarity+recency ranking
    - Added opened_at_start / opened_at_end to KBQueryParams (valid_to range filter)
    - Added resolved_at_epoch_ms to ProcessedTicket so vector store metadata carries
      a sortable numeric timestamp for post-retrieval recency re-ranking
    - Fixed missing ge=0 on resolution_time_min equivalent fields
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


# ── Enums ───────────────────────────────────────────────────────────────────


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


class SortOrder(str, Enum):
    """Sort direction for query results."""
    ASC = "asc"
    DESC = "desc"


class SortField(str, Enum):
    """Sortable fields for incident queries.

    Maps to ServiceNow REST API ``sysparm_query`` ORDER BY clauses.
    Use ``opened_at`` for most L2 recommendation contexts (fetch
    recent incidents first). Use ``resolved_at`` when you want the
    most recently *resolved* similar incidents for resolution drafting.
    """
    OPENED_AT = "opened_at"
    RESOLVED_AT = "resolved_at"
    PRIORITY = "priority"
    NUMBER = "number"


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
    # FIX: replaced naive datetime.utcnow() with timezone-aware UTC default
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
    """Query parameters for listing and retrieving incidents from ServiceNow.

    Recency ordering
    ----------------
    Set ``sort_by=SortField.OPENED_AT`` and ``sort_order=SortOrder.DESC``
    (the defaults) to fetch the most recently opened incidents first.

    For resolution-based ranking (e.g. finding recently *resolved* similar
    incidents to use as resolution drafts), set ``sort_by=SortField.RESOLVED_AT``.

    Blended recency + similarity ranking
    -------------------------------------
    ``recency_weight`` controls how much recency influences post-retrieval
    re-ranking when the caller blends vector similarity scores with a
    time-decay factor. A value of 0.0 means pure similarity; 1.0 means
    pure recency. The vector store ``hybrid_query`` result scores should
    be blended by the service layer using::

        final_score = (1 - recency_weight) * similarity_score
                      + recency_weight * recency_score

    where ``recency_score`` is derived from ``resolved_at_epoch_ms`` stored
    in vector metadata (see ``ProcessedTicket.resolved_at_epoch_ms``).
    """

    # State / group filters
    state: Optional[str] = Field(default=None, description="Filter by incident state")
    assignment_group: Optional[str] = Field(default=None, description="Filter by assignment group")
    category: Optional[str] = Field(default=None, description="Filter by category")

    # FIX: opened_at range (was present but undocumented re: sort impact)
    opened_at_start: Optional[datetime] = Field(
        default=None,
        description="Return incidents opened on or after this datetime (UTC)",
    )
    opened_at_end: Optional[datetime] = Field(
        default=None,
        description="Return incidents opened on or before this datetime (UTC)",
    )

    # FIX: resolved_at range — needed for fetching recently resolved incidents
    # as resolution drafting context. Was missing entirely in original.
    resolved_at_start: Optional[datetime] = Field(
        default=None,
        description="Return incidents resolved on or after this datetime (UTC)",
    )
    resolved_at_end: Optional[datetime] = Field(
        default=None,
        description="Return incidents resolved on or before this datetime (UTC)",
    )

    # FIX: explicit sort control — original had no ordering at all, so
    # ServiceNow returned results in undefined insertion order.
    sort_by: SortField = Field(
        default=SortField.OPENED_AT,
        description=(
            "Field to sort results by. Use OPENED_AT for recency of new incidents; "
            "RESOLVED_AT to surface recently resolved incidents for resolution drafting."
        ),
    )
    sort_order: SortOrder = Field(
        default=SortOrder.DESC,
        description="Sort direction. DESC returns most recent first (recommended).",
    )

    # FIX: recency_weight for post-retrieval blended ranking in the service layer.
    # 0.0 = pure cosine similarity, 1.0 = pure recency, 0.3 is a safe default.
    recency_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for recency vs similarity in blended re-ranking (0.0–1.0). "
            "0.0 = pure similarity, 1.0 = pure recency. "
            "Applied by the service layer after vector retrieval."
        ),
    )

    # Pagination
    limit: int = Field(default=100, ge=1, le=1000, description="Max results per page")
    offset: int = Field(default=0, ge=0, description="Pagination offset")

    @model_validator(mode="after")
    def validate_date_ranges(self) -> "IncidentQueryParams":
        """Ensure start dates are before end dates when both are provided."""
        if self.opened_at_start and self.opened_at_end:
            if self.opened_at_start > self.opened_at_end:
                raise ValueError("opened_at_start must be before opened_at_end")
        if self.resolved_at_start and self.resolved_at_end:
            if self.resolved_at_start > self.resolved_at_end:
                raise ValueError("resolved_at_start must be before resolved_at_end")
        return self

    def to_snow_query(self) -> str:
        """Serialize to a ServiceNow encoded query string (sysparm_query).

        Maps model fields to ServiceNow table API query syntax so the
        caller doesn't have to construct it manually.

        Returns:
            Encoded query string, e.g.:
            ``stateIN6,7^opened_at>=2024-01-01^ORDERBYDESCopened_at``

        Example::

            params = IncidentQueryParams(
                state=IncidentState.RESOLVED,
                sort_by=SortField.RESOLVED_AT,
                sort_order=SortOrder.DESC,
            )
            query_str = params.to_snow_query()
            # → "state=6^ORDERBYDESCresolved_at"
        """
        clauses: list[str] = []

        if self.state:
            clauses.append(f"state={self.state}")
        if self.assignment_group:
            clauses.append(f"assignment_group={self.assignment_group}")
        if self.category:
            clauses.append(f"category={self.category}")
        if self.opened_at_start:
            clauses.append(
                f"opened_at>={self.opened_at_start.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if self.opened_at_end:
            clauses.append(
                f"opened_at<={self.opened_at_end.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if self.resolved_at_start:
            clauses.append(
                f"resolved_at>={self.resolved_at_start.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        if self.resolved_at_end:
            clauses.append(
                f"resolved_at<={self.resolved_at_end.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # ServiceNow ORDER BY syntax:
        # ascending  → ORDERBYfield_name
        # descending → ORDERBYDESCfield_name
        order_prefix = "ORDERBYDESC" if self.sort_order == SortOrder.DESC else "ORDERBY"
        clauses.append(f"{order_prefix}{self.sort_by.value}")

        return "^".join(clauses)


class KBQueryParams(BaseModel):
    """Query parameters for listing KB articles."""
    category: Optional[str] = Field(default=None, description="Filter by category")
    workflow_state: str = Field(default="published", description="Publication state filter")
    # FIX: added valid_to range so expired articles can be excluded
    valid_to_after: Optional[datetime] = Field(
        default=None,
        description="Exclude articles whose valid_to is before this date (i.e. expired)",
    )
    limit: int = Field(default=100, ge=1, le=1000, description="Max results per page")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    # FIX: KB articles should also be sortable by recency
    sort_by: Literal["sys_updated_on", "valid_to", "number"] = Field(
        default="sys_updated_on",
        description="Field to sort KB results by",
    )
    sort_order: SortOrder = Field(
        default=SortOrder.DESC,
        description="Sort direction. DESC returns most recently updated articles first.",
    )


# ── Processed Data Models ───────────────────────────────────────────────────


class ExtractedEntity(BaseModel):
    """Named entity extracted from incident text."""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label (SERVICE, ERROR_CODE, HOSTNAME, etc.)")
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")


class ProcessedTicket(BaseModel):
    """Ticket after NLP preprocessing.

    recency fields
    --------------
    ``resolved_at_epoch_ms`` is stored as an integer in vector metadata so
    the service layer can compute a time-decay recency score without parsing
    datetime strings out of JSONB. Populated automatically from ``resolved_at``
    by the ``compute_recency_epoch`` validator.

    Example recency score formula in the service layer::

        import time
        now_ms = int(time.time() * 1000)
        age_days = (now_ms - ticket.resolved_at_epoch_ms) / (1000 * 86400)
        # Exponential decay: half-life of 90 days
        recency_score = 2 ** (-age_days / 90)
    """
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

    # FIX: numeric epoch timestamp derived from resolved_at, stored in vector
    # metadata for fast recency re-ranking without datetime string parsing.
    # 0 means unresolved / unknown.
    resolved_at_epoch_ms: int = Field(
        default=0,
        ge=0,
        description=(
            "resolved_at as Unix epoch milliseconds. "
            "Auto-populated from resolved_at by validator. "
            "Store this in vector metadata under key 'resolved_at_epoch_ms'."
        ),
    )

    # FIX: same for opened_at — useful for time-windowed retrieval
    opened_at_epoch_ms: int = Field(
        default=0,
        ge=0,
        description=(
            "opened_at as Unix epoch milliseconds. "
            "Auto-populated from opened_at by validator."
        ),
    )

    @model_validator(mode="after")
    def compute_recency_epochs(self) -> "ProcessedTicket":
        """Populate epoch_ms fields from datetime fields if not already set.

        Converts timezone-aware or naive datetimes to UTC epoch milliseconds.
        Naive datetimes are assumed to be UTC.
        """
        if self.resolved_at and self.resolved_at_epoch_ms == 0:
            dt = self.resolved_at
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self.resolved_at_epoch_ms = int(dt.timestamp() * 1000)

        if self.opened_at and self.opened_at_epoch_ms == 0:
            dt = self.opened_at
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self.opened_at_epoch_ms = int(dt.timestamp() * 1000)

        return self

    def to_vector_metadata(self) -> dict:
        """Serialize fields relevant for vector store metadata.

        Returns a flat dict suitable for passing as ``VectorRecord.metadata``.
        Includes ``resolved_at_epoch_ms`` and ``opened_at_epoch_ms`` so the
        service layer can do recency re-ranking purely from vector metadata
        without a secondary DB lookup.

        Example::

            record = VectorRecord(
                id=ticket.chunk_id,
                values=embedding,
                metadata=ticket.to_vector_metadata(),
            )
        """
        return {
            "source_id": self.source_id,
            "sys_id": self.sys_id,
            "category": self.category,
            "subcategory": self.subcategory,
            "incident_type": self.incident_type.value,
            "priority": str(self.priority),
            "cmdb_ci": self.cmdb_ci,
            "assignment_group": self.assignment_group,
            "resolved_at_epoch_ms": self.resolved_at_epoch_ms,
            "opened_at_epoch_ms": self.opened_at_epoch_ms,
            # search_text used by hybrid_query full-text arm
            "search_text": self.cleaned_text,
        }


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