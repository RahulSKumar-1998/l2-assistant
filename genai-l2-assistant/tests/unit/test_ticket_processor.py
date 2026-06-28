"""Unit tests for ticket processing and text chunking.

Tests the NLP preprocessing pipeline including error code extraction,
incident type classification, sentence-boundary-aware chunking, and
KB article section title preservation.
"""

import re

import pytest

from app.models.incident import (
    ExtractedEntity,
    IncidentType,
    ProcessedTicket,
    TextChunk,
    ChunkType,
    SourceType,
)
from app.utils.text_utils import (
    chunk_text_by_sentences,
    clean_text,
    extract_section_title,
    split_sentences,
)


# ── Error Code Extraction ───────────────────────────────────────────────────

# Patterns used by the ticket processor for entity extraction
_ERROR_CODE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("HTTP_STATUS", re.compile(r"\b(?:HTTP\s*)?([45]\d{2})\b")),
    ("ORA_ERROR", re.compile(r"\b(ORA-\d{4,5})\b")),
    ("JAVA_EXCEPTION", re.compile(r"\b(\w+(?:Exception|Error))\b")),
    ("EXIT_CODE", re.compile(r"\bexit\s*(?:code\s*)?(\d+)\b", re.IGNORECASE)),
    ("SIGNAL", re.compile(r"\b(SIG(?:SEGV|KILL|TERM|ABRT|BUS))\b")),
    ("ERRNO", re.compile(r"\b(ECONNREFUSED|ETIMEDOUT|ECONNRESET|ENOENT)\b")),
]


def extract_error_codes(text: str) -> list[ExtractedEntity]:
    """Extract error codes from incident text.

    Args:
        text: Incident description or work notes.

    Returns:
        List of extracted error code entities.
    """
    entities: list[ExtractedEntity] = []
    for label, pattern in _ERROR_CODE_PATTERNS:
        for match in pattern.finditer(text):
            entities.append(
                ExtractedEntity(
                    text=match.group(0),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return entities


# ── Incident Type Classification ────────────────────────────────────────────

_TYPE_KEYWORDS: dict[IncidentType, list[str]] = {
    IncidentType.APPLICATION_ERROR: [
        "application", "app", "502", "500", "503", "exception",
        "stack trace", "null pointer", "out of memory", "deployment",
        "crash", "error rate", "timeout", "connection pool",
    ],
    IncidentType.INFRASTRUCTURE: [
        "server", "vm", "disk", "cpu", "memory", "hardware",
        "reboot", "power", "hypervisor", "storage", "capacity",
    ],
    IncidentType.NETWORK: [
        "network", "dns", "firewall", "load balancer", "routing",
        "packet loss", "latency", "connectivity", "vpn", "ssl",
        "certificate", "tls",
    ],
    IncidentType.ACCESS_MANAGEMENT: [
        "access", "permission", "role", "ldap", "sso", "login",
        "authentication", "authorization", "mfa", "password",
        "account locked", "privilege",
    ],
    IncidentType.SECURITY: [
        "security", "vulnerability", "exploit", "malware",
        "intrusion", "breach", "suspicious", "unauthorized",
    ],
    IncidentType.PERFORMANCE: [
        "slow", "performance", "degradation", "high latency",
        "response time", "throughput", "bottleneck",
    ],
    IncidentType.DATA_ISSUE: [
        "data", "database", "corruption", "replication",
        "consistency", "migration", "etl", "sync",
    ],
}


def classify_incident_type(text: str, category: str = "") -> IncidentType:
    """Classify an incident based on text content and category.

    Args:
        text: Cleaned incident text.
        category: ServiceNow category field.

    Returns:
        Classified IncidentType.
    """
    text_lower = text.lower()

    # Score each type
    scores: dict[IncidentType, int] = {}
    for incident_type, keywords in _TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[incident_type] = score

    if not scores:
        return IncidentType.UNKNOWN

    return max(scores, key=lambda t: scores[t])


# ── KB Chunking with Section Titles ─────────────────────────────────────────


def chunk_kb_article(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict[str, str]]:
    """Chunk a KB article preserving section titles.

    When text crosses a section boundary (markdown ## header),
    the section title is prepended to each chunk for context.

    Args:
        text: KB article text with markdown headers.
        chunk_size: Maximum tokens per chunk.
        overlap: Overlap tokens between chunks.

    Returns:
        List of dicts with 'text' and 'section_title' keys.
    """
    # Split by section headers
    sections = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)

    chunks: list[dict[str, str]] = []
    current_title = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue

        if re.match(r"^#{1,3}\s+", part):
            current_title = part.lstrip("#").strip()
            continue

        # Chunk this section's content
        section_chunks = chunk_text_by_sentences(
            part, chunk_size=chunk_size, overlap=overlap
        )

        for chunk_text in section_chunks:
            chunks.append({
                "text": f"[{current_title}] {chunk_text}" if current_title else chunk_text,
                "section_title": current_title,
            })

    return chunks


# ── Tests ───────────────────────────────────────────────────────────────────


class TestErrorCodeExtraction:
    """Tests for error code extraction from incident text."""

    def test_preprocess_extracts_error_codes(self) -> None:
        """Verify extraction of HTTP status codes and error identifiers.

        The preprocessor should detect HTTP 502, Java exceptions, and
        system error codes embedded in incident descriptions.
        """
        text = (
            "The payment-service is returning HTTP 502 Bad Gateway errors. "
            "Stack trace shows java.lang.NullPointerException in "
            "PaymentProcessor.processRefund(). Also seeing ECONNREFUSED "
            "when connecting to upstream at port 8443. Process exited "
            "with exit code 137 (SIGKILL)."
        )

        entities = extract_error_codes(text)

        # Collect extracted entity texts
        entity_texts = {e.text for e in entities}
        entity_labels = {e.label for e in entities}

        # HTTP 502
        assert "502" in entity_texts
        assert "HTTP_STATUS" in entity_labels

        # Java exception
        assert "NullPointerException" in entity_texts
        assert "JAVA_EXCEPTION" in entity_labels

        # ECONNREFUSED
        assert "ECONNREFUSED" in entity_texts
        assert "ERRNO" in entity_labels

        # Exit code
        assert any("137" in e.text for e in entities if e.label == "EXIT_CODE")

        # SIGKILL
        assert "SIGKILL" in entity_texts
        assert "SIGNAL" in entity_labels

        # Verify positions are valid
        for entity in entities:
            assert entity.start >= 0
            assert entity.end > entity.start
            assert entity.end <= len(text)

    def test_oracle_error_extraction(self) -> None:
        """ORA-XXXXX error codes should be extracted."""
        text = "Database connection failed with ORA-12154 TNS could not resolve."
        entities = extract_error_codes(text)

        assert any(e.text == "ORA-12154" for e in entities)

    def test_multiple_http_codes(self) -> None:
        """Multiple HTTP status codes should all be extracted."""
        text = "Seeing mix of 500 and 503 errors on the health endpoint."
        entities = extract_error_codes(text)

        codes = {e.text for e in entities if e.label == "HTTP_STATUS"}
        assert "500" in codes
        assert "503" in codes


class TestIncidentTypeClassification:
    """Tests for incident type classification."""

    def test_classify_incident_type_application_error(self) -> None:
        """Incident with application error keywords should classify correctly.

        Text mentioning HTTP errors, deployments, and connection pools
        should be classified as APPLICATION_ERROR.
        """
        text = (
            "payment service returning 502 errors in production. "
            "the error started after the v2.4.1 deployment. "
            "connection pool exhausted with max_connections=50."
        )

        result = classify_incident_type(text, category="application")

        assert result == IncidentType.APPLICATION_ERROR

    def test_classify_infrastructure(self) -> None:
        """Infrastructure-related incidents should be classified correctly."""
        text = (
            "Server prod-db-01 running out of disk space. "
            "CPU utilization at 98%. Memory exhausted, VM needs reboot."
        )

        result = classify_incident_type(text)

        assert result == IncidentType.INFRASTRUCTURE

    def test_classify_network(self) -> None:
        """Network-related incidents should be classified correctly."""
        text = (
            "DNS resolution failing for internal services. "
            "Load balancer health checks timeout. Packet loss at 15%."
        )

        result = classify_incident_type(text)

        assert result == IncidentType.NETWORK

    def test_classify_access_management(self) -> None:
        """Access management incidents should be classified correctly."""
        text = (
            "User unable to login to SSO portal. Account locked after "
            "multiple failed MFA attempts. LDAP sync may be delayed."
        )

        result = classify_incident_type(text)

        assert result == IncidentType.ACCESS_MANAGEMENT

    def test_classify_unknown(self) -> None:
        """Text without matching keywords should classify as UNKNOWN."""
        text = "Something happened. Please investigate."

        result = classify_incident_type(text)

        assert result == IncidentType.UNKNOWN


class TestChunkingSentenceBoundaries:
    """Tests for sentence-boundary-aware text chunking."""

    def test_chunking_respects_sentence_boundaries(self) -> None:
        """Chunks should not split in the middle of a sentence.

        When chunking text with a small token budget, sentence boundaries
        should be respected to preserve semantic coherence.
        """
        text = (
            "The payment service started returning 502 errors at 14:30 UTC. "
            "Approximately 15% of checkout requests are failing. "
            "The error correlates with the v2.4.1 deployment. "
            "Connection pool metrics show all 50 connections are active. "
            "No idle connections are available for new requests. "
            "Upstream health checks on payment-gateway are passing. "
            "The issue appears to be isolated to the payment-service pods. "
            "Rolling restart has been attempted but did not resolve the issue."
        )

        # Use a very small chunk size to force multiple chunks
        chunks = chunk_text_by_sentences(text, chunk_size=50, overlap=10)

        assert len(chunks) > 1, "Should produce multiple chunks with small budget"

        # Each chunk should end at a sentence boundary (end with period)
        for chunk in chunks:
            assert chunk.strip().endswith("."), (
                f"Chunk does not end at sentence boundary: '{chunk[-50:]}'"
            )

    def test_chunking_empty_text(self) -> None:
        """Empty text should return empty list."""
        assert chunk_text_by_sentences("") == []
        assert chunk_text_by_sentences("   ") == []

    def test_chunking_single_sentence(self) -> None:
        """Single sentence within budget should return one chunk."""
        text = "The service is healthy."
        chunks = chunk_text_by_sentences(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestKBChunkingSectionTitles:
    """Tests for KB article chunking with section title preservation."""

    def test_kb_chunking_preserves_section_titles(self) -> None:
        """KB article chunks should retain section context via prepended titles.

        When a KB article with markdown headers is chunked, each chunk
        should be prefixed with its parent section title for retrieval context.
        """
        kb_text = """## Troubleshooting HTTP 502 Errors

HTTP 502 Bad Gateway errors indicate that a gateway or proxy received an invalid response from an upstream server. This is one of the most common errors in microservices architectures.

## Common Causes

Connection pool exhaustion is the leading cause. Check HikariCP settings for max pool size. Resource limits (CPU, memory) on upstream pods can also cause 502 errors. Recent deployments may introduce regressions.

## Resolution Steps

First check upstream service health endpoints. Then verify connection pool settings. Review recent deployments for regressions. Check monitoring dashboards for resource exhaustion patterns.
"""

        chunks = chunk_kb_article(kb_text, chunk_size=100, overlap=10)

        assert len(chunks) > 0, "Should produce at least one chunk"

        # Verify section titles are preserved
        section_titles = {c["section_title"] for c in chunks}
        assert "Troubleshooting HTTP 502 Errors" in section_titles
        assert "Common Causes" in section_titles
        assert "Resolution Steps" in section_titles

        # Verify title is prepended to chunk text
        for chunk in chunks:
            if chunk["section_title"]:
                assert chunk["text"].startswith(f"[{chunk['section_title']}]")

    def test_kb_chunking_no_headers(self) -> None:
        """KB article without headers should still produce chunks."""
        text = "This is a plain KB article without any section headers. It contains useful troubleshooting information for engineers."
        chunks = chunk_kb_article(text, chunk_size=200)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["section_title"] == ""
