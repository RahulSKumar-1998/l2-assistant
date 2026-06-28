"""Pydantic models for chat/conversational interface."""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the message was sent",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source IDs referenced in the message",
    )


class ChatSession(BaseModel):
    """A chat session tied to a specific incident."""
    id: UUID = Field(default_factory=uuid4, description="Session UUID")
    incident_id: UUID = Field(..., description="Associated incident UUID")
    engineer_id: str = Field(..., description="ServiceNow user sys_id")
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="Ordered list of chat messages",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session start time",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last activity time",
    )


class ChatRequest(BaseModel):
    """API request body for sending a chat message."""
    incident_id: str = Field(..., description="ServiceNow incident sys_id or UUID")
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID (creates new session if not provided)",
    )
    engineer_id: str = Field(..., description="ServiceNow user sys_id")


class ChatResponse(BaseModel):
    """API response for a chat message."""
    response: str = Field(..., description="AI assistant response")
    sources: list[str] = Field(
        default_factory=list,
        description="Source documents referenced in the response",
    )
    session_id: str = Field(..., description="Chat session ID")


# ── LLM Interaction Models ──────────────────────────────────────────────────


class LLMPrompt(BaseModel):
    """Structured prompt for LLM generation."""
    system_prompt: str = Field(..., description="System/role instructions")
    user_message: str = Field(..., description="User-facing message content")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=2000, ge=1, le=16000, description="Maximum response tokens")
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata for tracing (incident_id, etc.)",
    )


class LLMResponse(BaseModel):
    """Response from an LLM generation call."""
    content: str = Field(..., description="Generated text content")
    model: str = Field(..., description="Model identifier used")
    input_tokens: int = Field(default=0, description="Input token count")
    output_tokens: int = Field(default=0, description="Output token count")
    latency_ms: int = Field(default=0, description="Generation latency in milliseconds")
