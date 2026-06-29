"""Shared pytest fixtures for the GenAI L2 Support Assistant test suite.

Provides reusable fixtures for sample data, mock services, and test
infrastructure across unit, integration, and evaluation tests.
"""

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import requests
import urllib3

# Bypass SSL verification for tiktoken downloads in corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
original_get = requests.get
def patched_get(url, *args, **kwargs):
    if "openaipublic" in str(url):
        kwargs["verify"] = False
    return original_get(url, *args, **kwargs)
requests.get = patched_get

from app.models.incident import (
    ExtractedEntity,
    IncidentRecord,
    IncidentType,
    KBArticle,
    ProcessedTicket,
    TextChunk,
    ChunkType,
    SourceType,
)
from app.models.recommendation import (
    KBReference,
    RecommendationResult,
    SimilarIncident,
    SourceReference,
    TriageStep,
)
from app.models.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSession,
    LLMPrompt,
    LLMResponse,
)
from app.models.feedback import FeedbackRecord, FeedbackSubmission


# ── Fixed UUIDs for deterministic tests ──────────────────────────────────────

INCIDENT_UUID = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
RECOMMENDATION_UUID = UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")
FEEDBACK_UUID = UUID("c3d4e5f6-a7b8-9012-cdef-123456789012")
SESSION_UUID = UUID("d4e5f6a7-b8c9-0123-defa-234567890123")


# ── Sample Incident Data ────────────────────────────────────────────────────


@pytest.fixture
def sample_incident_record() -> IncidentRecord:
    """Raw ServiceNow incident record for a payment service 502 error."""
    return IncidentRecord(
        sys_id="abc123def456abc123def456abc12345",
        number="INC0042871",
        short_description="Payment service returning 502 errors in production",
        description=(
            "Since 2024-01-15 14:30 UTC, the payment-service has been returning "
            "intermittent HTTP 502 Bad Gateway errors. Approximately 15% of "
            "checkout requests are failing. The error started after the v2.4.1 "
            "deployment. Affected endpoints: /api/v2/payments/process and "
            "/api/v2/payments/refund. Upstream health checks on payment-gateway "
            "are passing. Error logs show: 'Connection pool exhausted: max_connections=50 "
            "reached, active=50, idle=0'. Customer: Acme Corp (acme@example.com). "
            "Contact: John Smith, john.smith@acme.com, +1-555-0123. "
            "Server: prod-payment-01.internal.corp.net (10.0.1.42)."
        ),
        category="application",
        subcategory="availability",
        priority=1,
        state="2",
        assignment_group="L2-Application-Support",
        assigned_to="Jane Engineer",
        cmdb_ci="payment-service",
        opened_at=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        resolved_at=None,
        work_notes=(
            "2024-01-15 14:45: Confirmed 502s in Datadog APM. "
            "Error rate at 15.2% on /api/v2/payments/process. "
            "2024-01-15 15:00: Checked pod logs, seeing connection pool exhaustion."
        ),
        resolution_notes=None,
        root_cause=None,
    )


@pytest.fixture
def sample_incident_p1() -> ProcessedTicket:
    """Processed ticket fixture: payment service 502 error, P1."""
    return ProcessedTicket(
        source_id="INC0042871",
        sys_id="abc123def456abc123def456abc12345",
        cleaned_text=(
            "payment service returning 502 errors in production. "
            "since 2024-01-15 14:30 utc, the payment-service has been returning "
            "intermittent http 502 bad gateway errors. approximately 15% of "
            "checkout requests are failing. the error started after the v2.4.1 "
            "deployment. affected endpoints: /api/v2/payments/process and "
            "/api/v2/payments/refund. upstream health checks on payment-gateway "
            "are passing. error logs show: connection pool exhausted: max_connections=50 "
            "reached, active=50, idle=0."
        ),
        entities=[
            ExtractedEntity(
                text="payment-service", label="SERVICE", start=0, end=15
            ),
            ExtractedEntity(text="502", label="ERROR_CODE", start=35, end=38),
            ExtractedEntity(
                text="v2.4.1", label="VERSION", start=180, end=186
            ),
        ],
        keywords=[
            "502",
            "payment-service",
            "connection pool",
            "bad gateway",
            "deployment",
        ],
        category="application",
        subcategory="availability",
        incident_type=IncidentType.APPLICATION_ERROR,
        summary=(
            "Payment service returning HTTP 502 Bad Gateway errors affecting 15% "
            "of checkout requests. Error started after v2.4.1 deployment with "
            "connection pool exhaustion."
        ),
        priority=1,
        cmdb_ci="payment-service",
        assignment_group="L2-Application-Support",
        opened_at=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        resolved_at=None,
        resolution_notes=None,
        root_cause=None,
        work_notes=(
            "2024-01-15 14:45: Confirmed 502s in Datadog APM. "
            "Error rate at 15.2% on /api/v2/payments/process."
        ),
    )


@pytest.fixture
def sample_incident_p1_with_pii(sample_incident_record: IncidentRecord) -> IncidentRecord:
    """Incident record containing PII that should be masked before LLM calls."""
    return sample_incident_record


# ── Sample Recommendation ───────────────────────────────────────────────────


@pytest.fixture
def sample_recommendation() -> RecommendationResult:
    """RecommendationResult fixture with realistic triage data."""
    return RecommendationResult(
        id=RECOMMENDATION_UUID,
        incident_id=INCIDENT_UUID,
        snow_sys_id="abc123def456abc123def456abc12345",
        root_cause_prediction=(
            "The HTTP 502 errors are caused by connection pool exhaustion in the "
            "payment-service following the v2.4.1 deployment. The new version likely "
            "introduced a connection leak or increased connection hold times, causing "
            "all 50 pool slots to be consumed with no idle connections available."
        ),
        confidence_score=0.87,
        triage_steps=[
            TriageStep(
                step=1,
                action="Check connection pool metrics in Datadog",
                rationale="Confirm pool exhaustion and identify leak pattern",
                command="kubectl exec -it deploy/payment-service -- curl localhost:8080/actuator/metrics/hikaricp.connections.active",
            ),
            TriageStep(
                step=2,
                action="Review v2.4.1 changelog for connection handling changes",
                rationale="Identify the specific code change that caused the regression",
                command=None,
            ),
            TriageStep(
                step=3,
                action="Increase max_connections to 100 as immediate mitigation",
                rationale="Provides headroom while root cause is fixed",
                command="kubectl set env deploy/payment-service MAX_POOL_SIZE=100",
            ),
            TriageStep(
                step=4,
                action="Prepare rollback to v2.4.0 if pool increase does not resolve",
                rationale="Revert to last known good version as fallback",
                command="kubectl rollout undo deploy/payment-service --to-revision=3",
            ),
        ],
        resolution_draft=(
            "Root cause: Connection pool exhaustion in payment-service after v2.4.1 "
            "deployment. The new version introduced a connection leak in the payment "
            "gateway client. Mitigation: Increased max_connections from 50 to 100. "
            "Permanent fix: Rolled back to v2.4.0 and filed bug JIRA-4521."
        ),
        escalate_to_l3=False,
        escalation_reason=None,
        similar_incidents=[
            SimilarIncident(
                number="INC0039201",
                sys_id="def456abc123def456abc123def45678",
                similarity_score=0.92,
                resolution_summary="Connection pool tuning resolved 502 errors on auth-service",
                resolution_time_min=45,
                category="application",
            ),
            SimilarIncident(
                number="INC0041055",
                sys_id="ghi789abc123ghi789abc123ghi78901",
                similarity_score=0.85,
                resolution_summary="Rollback of v3.1.0 resolved payment gateway timeouts",
                resolution_time_min=30,
                category="application",
            ),
            SimilarIncident(
                number="INC0037892",
                sys_id="jkl012abc123jkl012abc123jkl01234",
                similarity_score=0.78,
                resolution_summary="Database connection leak fixed by upgrading HikariCP",
                resolution_time_min=120,
                category="application",
            ),
        ],
        kb_references=[
            KBReference(
                kb_number="KB0012345",
                title="Troubleshooting HTTP 502 Errors in Microservices",
                relevance_score=0.91,
            ),
        ],
        sources_used=["INC0039201", "INC0041055", "INC0037892", "KB0012345"],
        retrieval_latency_ms=142,
        generation_latency_ms=1834,
    )


# ── Mock Services ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_query_matches() -> list[dict[str, Any]]:
    """Sample query match results from a vector store."""
    return [
        {
            "id": "chunk-001",
            "score": 0.92,
            "metadata": {
                "source_id": "INC0039201",
                "source_type": "incident",
                "chunk_type": "resolution",
                "category": "application",
                "text": (
                    "Connection pool tuning resolved 502 errors. Increased "
                    "HikariCP max pool size from 20 to 50 and set connection "
                    "timeout to 30s. Error rate dropped to 0% within 5 minutes."
                ),
            },
        },
        {
            "id": "chunk-002",
            "score": 0.88,
            "metadata": {
                "source_id": "INC0041055",
                "source_type": "incident",
                "chunk_type": "resolution",
                "category": "application",
                "text": (
                    "Rollback of v3.1.0 resolved payment gateway timeouts. "
                    "The deployment introduced a breaking change in the gRPC "
                    "client configuration causing connection refused errors."
                ),
            },
        },
        {
            "id": "chunk-003",
            "score": 0.85,
            "metadata": {
                "source_id": "KB0012345",
                "source_type": "kb_article",
                "chunk_type": "kb_article",
                "category": "application",
                "text": (
                    "## Troubleshooting HTTP 502 Errors\n"
                    "1. Check upstream service health\n"
                    "2. Verify connection pool settings\n"
                    "3. Review recent deployments for regressions\n"
                    "4. Check for resource exhaustion (CPU, memory, connections)"
                ),
            },
        },
        {
            "id": "chunk-004",
            "score": 0.78,
            "metadata": {
                "source_id": "INC0037892",
                "source_type": "incident",
                "chunk_type": "resolution",
                "category": "application",
                "text": (
                    "Database connection leak fixed by upgrading HikariCP from "
                    "4.0.3 to 5.0.1. The older version had a known bug where "
                    "connections were not returned to the pool on transaction rollback."
                ),
            },
        },
    ]


@pytest.fixture
def mock_vector_store(sample_query_matches: list[dict[str, Any]]) -> AsyncMock:
    """AsyncMock of VectorStore returning sample query match results."""
    mock = AsyncMock()
    mock.query = AsyncMock(return_value=sample_query_matches)
    mock.upsert = AsyncMock(return_value={"upserted_count": 4})
    mock.delete = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def sample_llm_response_json() -> dict[str, Any]:
    """Fixture JSON response that an LLM would return for recommendation generation."""
    return {
        "root_cause_prediction": (
            "The HTTP 502 errors are caused by connection pool exhaustion in the "
            "payment-service following the v2.4.1 deployment. The new version likely "
            "introduced a connection leak or increased connection hold times."
        ),
        "confidence_score": 0.87,
        "triage_steps": [
            {
                "step": 1,
                "action": "Check connection pool metrics in Datadog",
                "rationale": "Confirm pool exhaustion and identify leak pattern",
                "command": "kubectl exec -it deploy/payment-service -- curl localhost:8080/actuator/metrics/hikaricp.connections.active",
            },
            {
                "step": 2,
                "action": "Review v2.4.1 changelog for connection handling changes",
                "rationale": "Identify the specific code change that caused the regression",
                "command": None,
            },
            {
                "step": 3,
                "action": "Increase max_connections to 100 as immediate mitigation",
                "rationale": "Provides headroom while root cause is fixed",
                "command": "kubectl set env deploy/payment-service MAX_POOL_SIZE=100",
            },
            {
                "step": 4,
                "action": "Prepare rollback to v2.4.0 if pool increase does not resolve",
                "rationale": "Revert to last known good version as fallback",
                "command": "kubectl rollout undo deploy/payment-service --to-revision=3",
            },
        ],
        "escalate_to_l3": False,
        "escalation_reason": None,
        "resolution_draft": (
            "Root cause: Connection pool exhaustion in payment-service after v2.4.1 "
            "deployment. Mitigation: Increased max_connections from 50 to 100."
        ),
    }


@pytest.fixture
def mock_llm_client(sample_llm_response_json: dict[str, Any]) -> AsyncMock:
    """AsyncMock LLM client returning fixture JSON recommendation response."""
    mock = AsyncMock()
    mock.generate = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(sample_llm_response_json),
            model="gpt-4o",
            input_tokens=1250,
            output_tokens=450,
            latency_ms=1834,
        )
    )
    mock.chat = AsyncMock(
        return_value=LLMResponse(
            content="Based on the similar incidents and KB articles, the connection pool exhaustion is the most likely root cause. I recommend checking the HikariCP metrics first.",
            model="gpt-4o",
            input_tokens=800,
            output_tokens=120,
            latency_ms=920,
        )
    )
    return mock


@pytest.fixture
def mock_servicenow_client(sample_incident_record: IncidentRecord) -> AsyncMock:
    """AsyncMock ServiceNow client returning fixture incident data."""
    mock = AsyncMock()
    mock.get_incident = AsyncMock(return_value=sample_incident_record)
    mock.list_incidents = AsyncMock(return_value=[sample_incident_record])
    mock.update_incident = AsyncMock(return_value=sample_incident_record)
    mock.get_kb_article = AsyncMock(
        return_value=KBArticle(
            sys_id="kb123def456abc123def456abc12345",
            number="KB0012345",
            short_description="Troubleshooting HTTP 502 Errors in Microservices",
            text=(
                "<h2>Troubleshooting HTTP 502 Errors</h2>"
                "<p>HTTP 502 Bad Gateway errors indicate that a gateway or proxy "
                "received an invalid response from an upstream server.</p>"
                "<h3>Common Causes</h3>"
                "<ul>"
                "<li>Upstream service is down or unreachable</li>"
                "<li>Connection pool exhaustion</li>"
                "<li>Resource limits exceeded (CPU, memory)</li>"
                "<li>Deployment-related regressions</li>"
                "</ul>"
                "<h3>Resolution Steps</h3>"
                "<ol>"
                "<li>Check upstream service health endpoints</li>"
                "<li>Verify connection pool settings (HikariCP, etc.)</li>"
                "<li>Review recent deployments for regressions</li>"
                "<li>Check for resource exhaustion in monitoring</li>"
                "</ol>"
            ),
            category="application",
            workflow_state="published",
        )
    )
    return mock


# ── Low Confidence Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def low_confidence_llm_response() -> dict[str, Any]:
    """LLM response with low confidence that should trigger L3 escalation."""
    return {
        "root_cause_prediction": (
            "Unable to determine root cause with high confidence. "
            "The symptoms could indicate either a network issue or an "
            "application-level problem."
        ),
        "confidence_score": 0.35,
        "triage_steps": [
            {
                "step": 1,
                "action": "Gather additional diagnostic data",
                "rationale": "Insufficient information to determine root cause",
                "command": None,
            },
        ],
        "escalate_to_l3": True,
        "escalation_reason": "Confidence below threshold; insufficient similar incidents in knowledge base",
        "resolution_draft": "",
    }


# ── Text Chunk Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_text_chunks() -> list[TextChunk]:
    """Sample text chunks for vector store operations."""
    return [
        TextChunk(
            chunk_id="chunk-inc-001",
            chunk_text=(
                "Payment service returning 502 errors. Connection pool exhausted "
                "with max_connections=50 reached."
            ),
            chunk_type=ChunkType.DESCRIPTION,
            source_id="INC0042871",
            source_type=SourceType.INCIDENT,
            metadata={
                "category": "application",
                "priority": 1,
                "cmdb_ci": "payment-service",
            },
            chunk_index=0,
        ),
        TextChunk(
            chunk_id="chunk-inc-002",
            chunk_text=(
                "Confirmed 502s in Datadog APM. Error rate at 15.2% on "
                "/api/v2/payments/process. Checking pod logs."
            ),
            chunk_type=ChunkType.WORK_NOTES,
            source_id="INC0042871",
            source_type=SourceType.INCIDENT,
            metadata={
                "category": "application",
                "priority": 1,
                "cmdb_ci": "payment-service",
            },
            chunk_index=1,
        ),
        TextChunk(
            chunk_id="chunk-kb-001",
            chunk_text=(
                "## Troubleshooting HTTP 502 Errors\n"
                "1. Check upstream service health\n"
                "2. Verify connection pool settings\n"
                "3. Review recent deployments for regressions"
            ),
            chunk_type=ChunkType.KB_ARTICLE,
            source_id="KB0012345",
            source_type=SourceType.KB_ARTICLE,
            metadata={"category": "application"},
            chunk_index=0,
        ),
    ]


# ── Chat Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_chat_session() -> ChatSession:
    """A chat session with one exchange already recorded."""
    return ChatSession(
        id=SESSION_UUID,
        incident_id=INCIDENT_UUID,
        engineer_id="user-sys-id-001",
        messages=[
            ChatMessage(
                role="user",
                content="What is the most likely root cause for this 502 error?",
                timestamp=datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc),
            ),
            ChatMessage(
                role="assistant",
                content=(
                    "Based on similar incidents, the 502 errors are most likely caused "
                    "by connection pool exhaustion. INC0039201 had the same symptoms "
                    "and was resolved by increasing the HikariCP pool size."
                ),
                timestamp=datetime(2024, 1, 15, 15, 0, 5, tzinfo=timezone.utc),
                sources=["INC0039201", "KB0012345"],
            ),
        ],
    )


@pytest.fixture
def sample_chat_request() -> ChatRequest:
    """Chat request for follow-up question."""
    return ChatRequest(
        incident_id="abc123def456abc123def456abc12345",
        message="Should I rollback to v2.4.0 or try increasing the pool size first?",
        session_id=str(SESSION_UUID),
        engineer_id="user-sys-id-001",
    )
