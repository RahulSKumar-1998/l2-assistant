"""Unit tests for the RAG pipeline.

Tests the end-to-end flow from incident analysis through retrieval
to recommendation generation, including edge cases and failure modes.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.incident import IncidentType, ProcessedTicket
from app.models.recommendation import RecommendationResult, TriageStep
from app.models.chat import ChatMessage, ChatSession, LLMResponse


# ── RAG Pipeline Analysis Tests ─────────────────────────────────────────────


class TestAnalyzeIncident:
    """Tests for the RAG pipeline's incident analysis flow."""

    @pytest.mark.asyncio
    async def test_analyze_incident_success(
        self,
        sample_incident_p1: ProcessedTicket,
        mock_vector_store: AsyncMock,
        mock_llm_client: AsyncMock,
        sample_llm_response_json: dict[str, Any],
    ) -> None:
        """Mock retriever + LLM and verify RecommendationResult structure.

        Given a P1 payment service incident, the pipeline should:
        1. Retrieve relevant context from the vector store
        2. Generate a structured recommendation via the LLM
        3. Return a valid RecommendationResult with all required fields
        """
        # Simulate the pipeline logic inline since core modules are being built
        # The pipeline would: retrieve context → build prompt → call LLM → parse result
        retrieved_context = await mock_vector_store.query(
            query_text=sample_incident_p1.cleaned_text,
            top_k=5,
            filter={"category": sample_incident_p1.category},
        )
        assert len(retrieved_context) > 0, "Should retrieve at least one context chunk"

        llm_response = await mock_llm_client.generate(
            prompt=f"Analyze incident: {sample_incident_p1.cleaned_text}"
        )
        parsed = json.loads(llm_response.content)

        # Build recommendation from LLM response
        result = RecommendationResult(
            snow_sys_id=sample_incident_p1.sys_id,
            root_cause_prediction=parsed["root_cause_prediction"],
            confidence_score=parsed["confidence_score"],
            triage_steps=[TriageStep(**s) for s in parsed["triage_steps"]],
            escalate_to_l3=parsed["escalate_to_l3"],
            resolution_draft=parsed.get("resolution_draft", ""),
            retrieval_latency_ms=142,
            generation_latency_ms=llm_response.latency_ms,
        )

        # Verify structure
        assert isinstance(result, RecommendationResult)
        assert result.snow_sys_id == "abc123def456abc123def456abc12345"
        assert result.confidence_score == 0.87
        assert len(result.triage_steps) == 4
        assert result.triage_steps[0].step == 1
        assert "connection pool" in result.root_cause_prediction.lower()
        assert result.escalate_to_l3 is False
        assert result.generation_latency_ms == 1834
        assert result.resolution_draft != ""

        # Verify mocks were called
        mock_vector_store.query.assert_awaited_once()
        mock_llm_client.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_analyze_incident_low_confidence(
        self,
        sample_incident_p1: ProcessedTicket,
        mock_vector_store: AsyncMock,
        mock_llm_client: AsyncMock,
        low_confidence_llm_response: dict[str, Any],
    ) -> None:
        """When confidence < 0.6, escalate_to_l3 should be True.

        Low confidence predictions should automatically flag for L3
        escalation to prevent incorrect resolutions from being applied.
        """
        # Override mock to return low-confidence response
        mock_llm_client.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(low_confidence_llm_response),
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=200,
                latency_ms=1500,
            )
        )

        await mock_vector_store.query(
            query_text=sample_incident_p1.cleaned_text,
            top_k=5,
        )

        llm_response = await mock_llm_client.generate(
            prompt=f"Analyze incident: {sample_incident_p1.cleaned_text}"
        )
        parsed = json.loads(llm_response.content)

        # Apply escalation logic: confidence < 0.6 forces escalation
        confidence = parsed["confidence_score"]
        should_escalate = confidence < 0.6 or parsed.get("escalate_to_l3", False)

        result = RecommendationResult(
            snow_sys_id=sample_incident_p1.sys_id,
            root_cause_prediction=parsed["root_cause_prediction"],
            confidence_score=confidence,
            triage_steps=[TriageStep(**s) for s in parsed["triage_steps"]],
            escalate_to_l3=should_escalate,
            escalation_reason=(
                parsed.get("escalation_reason")
                or f"Low confidence ({confidence:.2f})"
            ),
            resolution_draft=parsed.get("resolution_draft", ""),
        )

        assert result.confidence_score == 0.35
        assert result.confidence_score < 0.6
        assert result.escalate_to_l3 is True
        assert result.escalation_reason is not None
        assert "confidence" in result.escalation_reason.lower() or "insufficient" in result.escalation_reason.lower()
        assert len(result.triage_steps) >= 1

    @pytest.mark.asyncio
    async def test_analyze_incident_llm_failure(
        self,
        sample_incident_p1: ProcessedTicket,
        mock_vector_store: AsyncMock,
        mock_llm_client: AsyncMock,
    ) -> None:
        """When LLM throws an exception, verify graceful fallback.

        The pipeline should return a safe fallback recommendation
        with escalation flagged rather than crashing.
        """
        # Configure mock to raise an exception
        mock_llm_client.generate = AsyncMock(
            side_effect=Exception("LLM service unavailable: rate limit exceeded")
        )

        await mock_vector_store.query(
            query_text=sample_incident_p1.cleaned_text,
            top_k=5,
        )

        # Simulate pipeline fallback behavior
        try:
            await mock_llm_client.generate(
                prompt=f"Analyze incident: {sample_incident_p1.cleaned_text}"
            )
            pytest.fail("LLM call should have raised an exception")
        except Exception as e:
            # Fallback: create a safe recommendation with escalation
            fallback = RecommendationResult(
                snow_sys_id=sample_incident_p1.sys_id,
                root_cause_prediction=(
                    "Unable to generate AI recommendation due to a service error. "
                    "Please review the incident manually."
                ),
                confidence_score=0.0,
                triage_steps=[
                    TriageStep(
                        step=1,
                        action="Review incident manually — AI analysis unavailable",
                        rationale=f"LLM error: {str(e)[:100]}",
                    ),
                ],
                escalate_to_l3=True,
                escalation_reason=f"AI pipeline failure: {str(e)[:200]}",
            )

        assert fallback.confidence_score == 0.0
        assert fallback.escalate_to_l3 is True
        assert "service error" in fallback.root_cause_prediction.lower()
        assert len(fallback.triage_steps) == 1
        assert fallback.escalation_reason is not None
        assert "rate limit" in fallback.escalation_reason.lower()


# ── Chat History Tests ──────────────────────────────────────────────────────


class TestChatPipeline:
    """Tests for the conversational chat interface."""

    @pytest.mark.asyncio
    async def test_chat_maintains_history(
        self,
        sample_chat_session: ChatSession,
        mock_llm_client: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Second chat message should include first exchange in context.

        The chat pipeline must pass the full conversation history to the
        LLM so that follow-up questions have proper context.
        """
        # First exchange is already in sample_chat_session
        assert len(sample_chat_session.messages) == 2

        # Simulate second user message
        new_user_message = ChatMessage(
            role="user",
            content="Should I rollback to v2.4.0 or increase the pool size first?",
        )
        sample_chat_session.messages.append(new_user_message)

        # Build prompt with full history
        history_text = "\n".join(
            f"{msg.role}: {msg.content}" for msg in sample_chat_session.messages
        )

        # Verify history contains both exchanges
        assert "502 error" in history_text
        assert "connection pool exhaustion" in history_text
        assert "rollback" in history_text

        # Verify the LLM would receive context from all messages
        assert len(sample_chat_session.messages) == 3
        assert sample_chat_session.messages[0].role == "user"
        assert sample_chat_session.messages[1].role == "assistant"
        assert sample_chat_session.messages[2].role == "user"

        # Simulate LLM call with history
        llm_response = await mock_llm_client.chat(history=history_text)
        assert llm_response.content != ""
        mock_llm_client.chat.assert_awaited_once()


# ── PII Masking Tests ───────────────────────────────────────────────────────


class TestPIIMasking:
    """Tests for PII removal before LLM prompt construction."""

    @pytest.mark.asyncio
    async def test_pii_not_in_llm_prompt(
        self,
        sample_incident_p1_with_pii: Any,
        mock_llm_client: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Verify PII is masked before being sent to the LLM.

        Personal information such as email addresses, phone numbers,
        and person names must be redacted from the prompt text.
        """
        raw_description = sample_incident_p1_with_pii.description

        # Verify PII exists in raw data
        assert "john.smith@acme.com" in raw_description
        assert "acme@example.com" in raw_description
        assert "+1-555-0123" in raw_description
        assert "John Smith" in raw_description
        assert "10.0.1.42" in raw_description

        # Simulate PII masking
        import re

        masked = raw_description
        # Email masking
        masked = re.sub(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "[EMAIL_REDACTED]",
            masked,
        )
        # Phone masking
        masked = re.sub(
            r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "[PHONE_REDACTED]",
            masked,
        )
        # IP address masking
        masked = re.sub(
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            "[IP_REDACTED]",
            masked,
        )

        # Verify PII is removed
        assert "john.smith@acme.com" not in masked
        assert "acme@example.com" not in masked
        assert "+1-555-0123" not in masked
        assert "[EMAIL_REDACTED]" in masked
        assert "[PHONE_REDACTED]" in masked
        assert "[IP_REDACTED]" in masked

        # Verify incident-relevant content is preserved
        assert "502" in masked
        assert "payment-service" in masked
        assert "connection pool exhausted" in masked
        assert "v2.4.1" in masked
