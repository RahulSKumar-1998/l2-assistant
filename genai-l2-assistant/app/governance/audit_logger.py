"""Structured audit logger for compliance and operational tracing.

Writes audit events to two destinations simultaneously:
    1. Structured log output via ``structlog`` (for real-time monitoring)
    2. PostgreSQL ``audit_events`` table (for compliance queries)

All events are typed using ``AuditEventType`` and carry actor, resource,
and payload information for full traceability.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import structlog
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.postgres import AuditEventDB, get_session_factory

logger = structlog.get_logger(__name__)


# ── Event Types ──────────────────────────────────────────────────────────────


class AuditEventType(str, Enum):
    """Audit event types for significant system actions."""
    # Ingestion events
    TICKET_INGESTED = "ticket_ingested"
    TICKET_INDEXED = "ticket_indexed"
    KB_ARTICLE_INDEXED = "kb_article_indexed"
    BATCH_INGESTION_COMPLETED = "batch_ingestion_completed"

    # Recommendation events
    RECOMMENDATION_GENERATED = "recommendation_generated"
    RECOMMENDATION_SERVED = "recommendation_served"
    RECOMMENDATION_ESCALATED = "recommendation_escalated"

    # Feedback events
    FEEDBACK_SUBMITTED = "feedback_submitted"
    FEEDBACK_PROCESSED = "feedback_processed"

    # ServiceNow interaction events
    WORK_NOTE_POSTED = "work_note_posted"
    RESOLUTION_POSTED = "resolution_posted"

    # Chat events
    CHAT_SESSION_STARTED = "chat_session_started"
    CHAT_MESSAGE_SENT = "chat_message_sent"

    # Admin events
    INDEX_REBUILT = "index_rebuilt"
    CONFIG_CHANGED = "config_changed"
    USER_ROLE_CHANGED = "user_role_changed"

    # Security events
    AUTHORIZATION_DENIED = "authorization_denied"
    PII_DETECTED = "pii_detected"
    PII_ANONYMIZED = "pii_anonymized"


# ── Audit Event Model ────────────────────────────────────────────────────────


class AuditEvent:
    """Internal representation of an audit event before persistence.

    This is intentionally not a Pydantic model to avoid import circularity
    with the DB models. It maps directly to AuditEventDB columns.
    """

    __slots__ = (
        "id", "event_type", "actor_id", "actor_role",
        "resource_type", "resource_id", "payload",
        "ip_address", "created_at",
    )

    def __init__(
        self,
        event_type: AuditEventType,
        actor_id: str = "system",
        actor_role: str = "system",
        resource_type: str = "",
        resource_id: str = "",
        payload: Optional[dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        self.id = uuid.uuid4()
        self.event_type = event_type.value
        self.actor_id = actor_id
        self.actor_role = actor_role
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.payload = payload or {}
        self.ip_address = ip_address
        self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict suitable for DB insertion and logging."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "actor_role": self.actor_role,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "payload": self.payload,
            "ip_address": self.ip_address,
            "created_at": self.created_at,
        }


# ── Audit Logger ─────────────────────────────────────────────────────────────


class AuditLogger:
    """Dual-destination audit logger.

    Writes structured events to both ``structlog`` and the PostgreSQL
    ``audit_events`` table. Database write failures are logged but do
    not propagate — the structured log serves as the fallback record.

    Example:
        >>> audit = AuditLogger()
        >>> await audit.log(
        ...     event_type=AuditEventType.RECOMMENDATION_GENERATED,
        ...     actor_id="user_abc",
        ...     actor_role="l2_engineer",
        ...     resource_type="incident",
        ...     resource_id="INC0042871",
        ...     payload={"confidence": 0.92},
        ... )
    """

    def __init__(self, *, write_to_db: bool = True) -> None:
        """Initialize the audit logger.

        Args:
            write_to_db: Whether to persist events to PostgreSQL.
                Set to False in testing or when DB is unavailable.
        """
        self._write_to_db = write_to_db
        self._log = logger.bind(component="audit_logger")

    async def log(
        self,
        event_type: AuditEventType,
        *,
        actor_id: str = "system",
        actor_role: str = "system",
        resource_type: str = "",
        resource_id: str = "",
        payload: Optional[dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """Record an audit event to both log output and database.

        Args:
            event_type: The type of audit event.
            actor_id: User or system ID performing the action.
            actor_role: Role of the actor (l2_engineer, admin, system, etc.).
            resource_type: Type of resource being acted upon.
            resource_id: Identifier of the resource.
            payload: Additional event-specific data.
            ip_address: Client IP address if available.
        """
        event = AuditEvent(
            event_type=event_type,
            actor_id=actor_id,
            actor_role=actor_role,
            resource_type=resource_type,
            resource_id=resource_id,
            payload=payload,
            ip_address=ip_address,
        )

        # 1. Always write to structured log
        self._log.info(
            "audit_event",
            event_id=str(event.id),
            event_type=event.event_type,
            actor_id=event.actor_id,
            actor_role=event.actor_role,
            resource_type=event.resource_type,
            resource_id=event.resource_id,
            payload=event.payload,
            ip_address=event.ip_address,
        )

        # 2. Persist to database if enabled
        if self._write_to_db:
            await self._persist_to_db(event)

    async def _persist_to_db(self, event: AuditEvent) -> None:
        """Persist an audit event to the PostgreSQL audit_events table.

        Database failures are caught and logged — they never propagate
        to the caller. The structured log entry serves as fallback.

        Args:
            event: The audit event to persist.
        """
        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                async with session.begin():
                    stmt = insert(AuditEventDB).values(
                        id=event.id,
                        event_type=event.event_type,
                        actor_id=event.actor_id,
                        actor_role=event.actor_role,
                        resource_type=event.resource_type,
                        resource_id=event.resource_id,
                        payload=event.payload,
                        ip_address=event.ip_address,
                    )
                    await session.execute(stmt)
        except Exception as exc:
            # DB write failure should never block the caller
            self._log.error(
                "audit_db_write_failed",
                event_id=str(event.id),
                event_type=event.event_type,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def log_authorization_denied(
        self,
        user_id: str,
        role: str,
        permission: str,
        ip_address: Optional[str] = None,
    ) -> None:
        """Convenience method for logging authorization denials.

        Args:
            user_id: The user who was denied.
            role: The user's role.
            permission: The permission that was denied.
            ip_address: Client IP address.
        """
        await self.log(
            event_type=AuditEventType.AUTHORIZATION_DENIED,
            actor_id=user_id,
            actor_role=role,
            resource_type="permission",
            resource_id=permission,
            payload={"denied_permission": permission},
            ip_address=ip_address,
        )

    async def log_pii_detected(
        self,
        source_id: str,
        pii_types: list[str],
        pii_count: int,
        actor_id: str = "system",
    ) -> None:
        """Convenience method for logging PII detection events.

        Args:
            source_id: The document where PII was found.
            pii_types: List of PII types detected.
            pii_count: Total number of PII items.
            actor_id: The actor performing the scan.
        """
        await self.log(
            event_type=AuditEventType.PII_ANONYMIZED,
            actor_id=actor_id,
            resource_type="document",
            resource_id=source_id,
            payload={"pii_types": pii_types, "pii_count": pii_count},
        )


# ── Module-level convenience ─────────────────────────────────────────────────


_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(*, write_to_db: bool = True) -> AuditLogger:
    """Get the singleton audit logger instance.

    Args:
        write_to_db: Whether to persist to the database.

    Returns:
        AuditLogger: The audit logger instance.
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(write_to_db=write_to_db)
    return _audit_logger
