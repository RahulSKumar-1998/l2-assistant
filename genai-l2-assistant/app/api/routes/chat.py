"""Chat API routes.

Provides a conversational interface for engineers to ask follow-up
questions about incidents and AI recommendations via the RAG pipeline.
"""

import uuid as uuid_mod
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.models.chat import ChatMessage, ChatRequest, ChatResponse
from app.storage.postgres import (
    ChatSessionDB,
    IncidentDB,
    get_db_session,
)

router = APIRouter(prefix="/chat", tags=["chat"])
logger = structlog.get_logger(__name__)


async def _get_or_create_session(
    session: Any,
    incident_id: uuid_mod.UUID,
    engineer_id: str,
    session_id: Optional[str],
) -> ChatSessionDB:
    """Retrieve an existing chat session or create a new one.

    Args:
        session: Active database session.
        incident_id: Internal incident UUID.
        engineer_id: ServiceNow engineer sys_id.
        session_id: Optional existing session ID to resume.

    Returns:
        The ChatSessionDB row (existing or newly created).
    """
    if session_id:
        try:
            sid = uuid_mod.UUID(session_id)
            stmt = select(ChatSessionDB).where(ChatSessionDB.id == sid)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing:
                return existing
        except ValueError:
            pass  # Invalid UUID format — create new session

    # Create a new session
    chat_session = ChatSessionDB(
        id=uuid_mod.uuid4(),
        incident_id=incident_id,
        engineer_id=engineer_id,
        messages=[],
    )
    session.add(chat_session)
    await session.flush()
    return chat_session


async def _run_chat_pipeline(
    incident: IncidentDB,
    message: str,
    history: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Execute the RAG chat pipeline to generate a response.

    This function orchestrates the retrieval and generation steps.
    Currently provides a structured placeholder that downstream
    RAG pipeline modules will implement.

    Args:
        incident: The incident database record.
        message: The user's chat message.
        history: Previous messages in the session for context.

    Returns:
        Tuple of (response_text, source_ids).
    """
    # Build context from incident data
    context_parts: list[str] = [
        f"Incident: {incident.number}",
        f"Description: {incident.short_description}",
        f"Category: {incident.category}",
        f"Priority: {incident.priority}",
    ]
    if incident.resolution_notes:
        context_parts.append(f"Resolution: {incident.resolution_notes}")

    # The RAG pipeline will be integrated here.
    # For now, return a structured response indicating the pipeline
    # needs the full RAG modules (retriever, LLM client) to be wired in.
    try:
        # Attempt to import and use the RAG pipeline if available
        from app.core.rag_pipeline import chat as rag_chat  # type: ignore[import-not-found]

        response_text, sources = await rag_chat(
            incident_context="\n".join(context_parts),
            user_message=message,
            chat_history=history,
        )
        return response_text, sources
    except ImportError:
        # RAG pipeline not yet implemented — return informative placeholder
        logger.info(
            "rag_pipeline_not_available",
            incident_number=incident.number,
        )
        return (
            f"I'm the L2 Support Assistant analyzing incident {incident.number}. "
            f"The full RAG pipeline is being integrated. "
            f"Incident details: {incident.short_description or 'N/A'}. "
            f"Category: {incident.category or 'N/A'}. "
            f"Please check back once the RAG pipeline modules are deployed.",
            [],
        )


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description="Send a message about an incident and receive an AI-generated response.",
)
async def chat(body: ChatRequest) -> ChatResponse:
    """Handle a chat message from an engineer.

    Finds or creates a chat session, appends the user message,
    runs the RAG pipeline for a response, and persists the full
    conversation history.

    Args:
        body: ChatRequest with incident_id, message, optional session_id,
              and engineer_id.

    Returns:
        ChatResponse with the AI response, source references, and session ID.

    Raises:
        HTTPException: 404 if the incident is not found, 500 on pipeline errors.
    """
    log = logger.bind(
        incident_id=body.incident_id,
        engineer_id=body.engineer_id,
        session_id=body.session_id,
    )

    async for session in get_db_session():
        # Resolve the incident — support both sys_id and UUID
        incident: Optional[IncidentDB] = None
        try:
            inc_uuid = uuid_mod.UUID(body.incident_id)
            stmt = select(IncidentDB).where(IncidentDB.id == inc_uuid)
            result = await session.execute(stmt)
            incident = result.scalar_one_or_none()
        except ValueError:
            pass

        if not incident:
            stmt = select(IncidentDB).where(
                IncidentDB.snow_sys_id == body.incident_id
            )
            result = await session.execute(stmt)
            incident = result.scalar_one_or_none()

        if not incident:
            log.warning("incident_not_found_for_chat")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {body.incident_id} not found",
            )

        # Get or create chat session
        chat_session = await _get_or_create_session(
            session, incident.id, body.engineer_id, body.session_id
        )

        # Reconstruct history from stored messages
        history: list[dict[str, Any]] = chat_session.messages or []

        # Add user message to history
        user_msg = {
            "role": "user",
            "content": body.message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        history.append(user_msg)

        # Run RAG pipeline
        try:
            response_text, sources = await _run_chat_pipeline(
                incident, body.message, history
            )
        except Exception as exc:
            log.error("chat_pipeline_failed", error=str(exc), exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate chat response.",
            ) from exc

        # Add assistant response to history
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
        }
        history.append(assistant_msg)

        # Persist updated messages
        chat_session.messages = history
        await session.flush()

        log.info(
            "chat_response_generated",
            session_id=str(chat_session.id),
            sources_count=len(sources),
        )

        return ChatResponse(
            response=response_text,
            sources=sources,
            session_id=str(chat_session.id),
        )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error processing chat message.",
    )
