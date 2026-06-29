"""PostgreSQL database layer with SQLAlchemy async models.

Defines all database tables and provides async session management
for the L2 Support Assistant.
"""

import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy import JSON, Uuid, String
# Dialect-agnostic type definitions for both PostgreSQL and SQLite
JSONB = JSON
UUID = Uuid
INET = String(45)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings


# ── Base Model ──────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""
    pass


# ── Table Models ────────────────────────────────────────────────────────────


class IncidentDB(Base):
    """Incident records synced from ServiceNow.

    Stores incident metadata, tracks indexing status, and links to
    recommendations and feedback.
    """
    __tablename__ = "incidents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    snow_sys_id: Mapped[str] = mapped_column(
        String(32), unique=True, nullable=False, comment="ServiceNow sys_id"
    )
    number: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="Incident number (e.g., INC0042871)"
    )
    short_description: Mapped[str] = mapped_column(
        Text, default="", comment="Short description"
    )
    description: Mapped[str] = mapped_column(
        Text, default="", comment="Full description"
    )
    category: Mapped[str] = mapped_column(
        String(100), default="", comment="Incident category"
    )
    subcategory: Mapped[str] = mapped_column(
        String(100), default="", comment="Incident subcategory"
    )
    priority: Mapped[int] = mapped_column(
        SmallInteger, default=4, comment="Priority 1-5"
    )
    state: Mapped[str] = mapped_column(
        String(50), default="1", comment="Incident state"
    )
    assignment_group: Mapped[str] = mapped_column(
        String(200), default="", comment="Assignment group name"
    )
    assigned_to: Mapped[str] = mapped_column(
        String(200), default="", comment="Assigned engineer"
    )
    cmdb_ci: Mapped[str] = mapped_column(
        String(200), default="", comment="Impacted CI from CMDB"
    )
    opened_at = mapped_column(
        DateTime(timezone=True), nullable=True, comment="When opened"
    )
    resolved_at = mapped_column(
        DateTime(timezone=True), nullable=True, comment="When resolved"
    )
    resolution_notes = mapped_column(
        Text, nullable=True, comment="Resolution notes"
    )
    root_cause = mapped_column(
        Text, nullable=True, comment="Root cause analysis"
    )
    is_indexed: Mapped[bool] = mapped_column(
        Boolean, default=False, comment="Whether indexed in vector store"
    )
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Record creation time",
    )
    updated_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last update time",
    )

    # Relationships
    recommendations = relationship("RecommendationDB", back_populates="incident")
    feedback_records = relationship("FeedbackDB", back_populates="incident")
    chat_sessions = relationship("ChatSessionDB", back_populates="incident")

    __table_args__ = (
        Index("ix_incidents_snow_sys_id", "snow_sys_id"),
        Index("ix_incidents_is_indexed", "is_indexed"),
        Index("ix_incidents_state", "state"),
        Index("ix_incidents_category", "category"),
    )


class RecommendationDB(Base):
    """AI-generated recommendations for incidents.

    Stores the full RAG pipeline output including root cause predictions,
    triage steps, similar incidents, and performance metrics.
    """
    __tablename__ = "recommendations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    incident_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False
    )
    root_cause_prediction: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Root cause prediction"
    )
    confidence_score: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Confidence 0.0-1.0"
    )
    triage_steps = mapped_column(
        JSONB, default=list, comment="List of triage step objects"
    )
    similar_incidents = mapped_column(
        JSONB, default=list, comment="List of similar incident references"
    )
    kb_references = mapped_column(
        JSONB, default=list, comment="List of KB article references"
    )
    resolution_draft: Mapped[str] = mapped_column(
        Text, default="", comment="Draft resolution note"
    )
    retrieval_latency_ms: Mapped[int] = mapped_column(
        Integer, default=0, comment="Retrieval latency in ms"
    )
    generation_latency_ms: Mapped[int] = mapped_column(
        Integer, default=0, comment="LLM generation latency in ms"
    )
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Generation time",
    )

    # Relationships
    incident = relationship("IncidentDB", back_populates="recommendations")
    feedback_records = relationship("FeedbackDB", back_populates="recommendation")

    __table_args__ = (
        Index("ix_recommendations_incident_id", "incident_id"),
    )


class FeedbackDB(Base):
    """Engineer feedback on AI recommendations.

    Captures thumbs up/down ratings and optional comments to drive
    continuous learning via the feedback processor.
    """
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recommendation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recommendations.id"), nullable=False
    )
    incident_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False
    )
    engineer_id: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="ServiceNow user sys_id"
    )
    rating: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, comment="1=thumbs_down, 5=thumbs_up"
    )
    comment = mapped_column(
        Text, nullable=True, comment="Optional comment"
    )
    acted_on_steps = mapped_column(
        JSONB, nullable=True, comment="Which steps were followed"
    )
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Feedback submission time",
    )

    # Relationships
    recommendation = relationship("RecommendationDB", back_populates="feedback_records")
    incident = relationship("IncidentDB", back_populates="feedback_records")

    __table_args__ = (
        Index("ix_feedback_recommendation_id", "recommendation_id"),
        Index("ix_feedback_engineer_id", "engineer_id"),
    )


class AuditEventDB(Base):
    """Structured audit event log.

    Records all significant system events for compliance and debugging.
    """
    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    event_type: Mapped[str] = mapped_column(
        String(100), nullable=False,
        comment="Event type: ticket_analyzed, recommendation_served, etc.",
    )
    actor_id: Mapped[str] = mapped_column(
        String(200), default="system", comment="User or system identifier"
    )
    actor_role: Mapped[str] = mapped_column(
        String(50), default="system",
        comment="Role: l2_engineer, l3_engineer, admin, system",
    )
    resource_type: Mapped[str] = mapped_column(
        String(50), default="", comment="Resource type (incident, recommendation, etc.)"
    )
    resource_id: Mapped[str] = mapped_column(
        String(200), default="", comment="Resource identifier"
    )
    payload = mapped_column(
        JSONB, default=dict, comment="Event-specific payload data"
    )
    ip_address = mapped_column(
        INET, nullable=True, comment="Client IP address"
    )
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Event timestamp",
    )

    __table_args__ = (
        Index("ix_audit_actor_created", "actor_id", "created_at"),
        Index("ix_audit_event_type", "event_type"),
    )


class ChatSessionDB(Base):
    """Conversational chat sessions tied to incidents.

    Stores the full message history for follow-up conversations
    about AI recommendations.
    """
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    incident_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False
    )
    engineer_id: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="ServiceNow user sys_id"
    )
    messages = mapped_column(
        JSONB, default=list,
        comment="List of {role, content, timestamp} message objects",
    )
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Session start time",
    )
    updated_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last activity time",
    )

    # Relationships
    incident = relationship("IncidentDB", back_populates="chat_sessions")

    __table_args__ = (
        Index("ix_chat_sessions_incident_id", "incident_id"),
    )


class FeedbackWeightDB(Base):
    """Source document quality scores based on aggregated feedback.

    Used by the feedback processor to adjust retrieval ranking
    based on engineer feedback signals.
    """
    __tablename__ = "feedback_weights"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[str] = mapped_column(
        String(200), unique=True, nullable=False,
        comment="Source document ID (incident number or KB number)",
    )
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="Source type: incident, kb_article, runbook",
    )
    quality_score: Mapped[float] = mapped_column(
        Float, default=1.0, comment="Quality weight (>1 boosted, <1 penalised)"
    )
    positive_signals: Mapped[int] = mapped_column(
        Integer, default=0, comment="Total positive feedback signals"
    )
    negative_signals: Mapped[int] = mapped_column(
        Integer, default=0, comment="Total negative feedback signals"
    )
    last_updated = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last score computation time",
    )

    __table_args__ = (
        Index("ix_feedback_weights_source_id", "source_id"),
    )


class ReviewQueueDB(Base):
    """Queue for recommendations flagged for human review.

    High-confidence but negatively-rated recommendations are
    queued here as valuable training signals.
    """
    __tablename__ = "review_queue"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recommendation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recommendations.id"), nullable=False
    )
    reason: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Why this was flagged for review"
    )
    status: Mapped[str] = mapped_column(
        String(50), default="pending",
        comment="Review status: pending, reviewed, dismissed",
    )
    reviewed_by = mapped_column(
        String(200), nullable=True, comment="Reviewer user ID"
    )
    reviewed_at = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Review timestamp"
    )
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="When flagged for review",
    )


# ── Database Session Management ─────────────────────────────────────────────


_engine = None
_session_factory = None


def get_engine():
    """Get or create the async database engine (singleton)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database.postgres_url
        if url.startswith("sqlite"):
            _engine = create_async_engine(
                url,
                echo=settings.is_development,
                connect_args={"timeout": 30},
            )
            from sqlalchemy import event
            @event.listens_for(_engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
        else:
            _engine = create_async_engine(
                url,
                echo=settings.is_development,
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,
            )
    return _engine


def get_session_factory():
    """Get or create the async session factory (singleton)."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to provide an async database session.

    Yields:
        AsyncSession: An async SQLAlchemy session.
    """
    factory = get_session_factory()
    async with factory() as session:
        committed = False
        try:
            yield session
            await session.commit()
            committed = True
        except Exception:
            await session.rollback()
            raise
        finally:
            if not committed:
                try:
                    await session.commit()
                except Exception:
                    await session.rollback()


async def init_db() -> None:
    """Initialize database: create all tables.

    Use for development/testing. In production, use Alembic migrations.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close the database engine connection pool."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
