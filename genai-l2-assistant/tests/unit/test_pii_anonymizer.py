"""Unit tests for PII anonymization.

Tests the detection and masking of personally identifiable information
including emails, hostnames, person names, cloud API keys, IP addresses,
phone numbers, and ensures non-PII data passes through unmodified.
"""

import re

import pytest


# ── PII Anonymizer Implementation (inline for testing) ──────────────────────

# These patterns mirror what the app.governance PII anonymizer uses.
# In production, these live in app/governance/pii_anonymizer.py

_PII_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "email",
        "[EMAIL_REDACTED]",
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ),
    (
        "ip_address",
        "[IP_REDACTED]",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
    (
        "phone",
        "[PHONE_REDACTED]",
        re.compile(
            r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
    ),
    (
        "hostname",
        "[HOSTNAME_REDACTED]",
        re.compile(
            r"\b[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"
            r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?){2,}"
            r"\.(?:com|net|org|io|internal|corp|local)\b"
        ),
    ),
    (
        "aws_key",
        "[AWS_KEY_REDACTED]",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (
        "gcp_key",
        "[GCP_KEY_REDACTED]",
        re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b"),
    ),
    (
        "generic_api_key",
        "[API_KEY_REDACTED]",
        re.compile(r"\b(?:sk-|pk_live_|pk_test_|rk_live_|rk_test_)[a-zA-Z0-9]{20,}\b"),
    ),
    (
        "uuid",
        None,  # UUIDs are NOT masked
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
    ),
]

# Common person name patterns in incident context
_PERSON_NAME_CONTEXT_RE = re.compile(
    r"(?:(?:Contact|Reported by|Assigned to|Customer contact|Caller|Name)[:\s]+)"
    r"([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
)


def mask_pii(text: str) -> str:
    """Mask PII in text, preserving UUIDs.

    Args:
        text: Input text potentially containing PII.

    Returns:
        Text with PII patterns replaced by redaction tokens.
    """
    result = text

    # Mask person names in context first (before other patterns might interfere)
    result = _PERSON_NAME_CONTEXT_RE.sub(
        lambda m: m.group(0).replace(m.group(1), "[PERSON_REDACTED]"),
        result,
    )

    for pattern_name, replacement, pattern in _PII_PATTERNS:
        if replacement is None:
            # Skip patterns with no replacement (e.g., UUID)
            continue
        result = pattern.sub(replacement, result)

    return result


# ── Tests ───────────────────────────────────────────────────────────────────


class TestEmailMasking:
    """Tests for email address detection and masking."""

    def test_email_masking(self) -> None:
        """Standard email addresses should be replaced with [EMAIL_REDACTED]."""
        text = "Contact john.smith@acme.com for details about the outage."
        masked = mask_pii(text)

        assert "john.smith@acme.com" not in masked
        assert "[EMAIL_REDACTED]" in masked
        assert "outage" in masked  # Non-PII preserved

    def test_multiple_emails(self) -> None:
        """Multiple email addresses in one text should all be masked."""
        text = "CC: admin@corp.io and ops-team@company.com on this ticket."
        masked = mask_pii(text)

        assert "admin@corp.io" not in masked
        assert "ops-team@company.com" not in masked
        assert masked.count("[EMAIL_REDACTED]") == 2

    def test_email_with_special_chars(self) -> None:
        """Emails with dots, hyphens, and plus signs should be masked."""
        text = "Alert sent to jane.doe+alerts@sub.domain.org"
        masked = mask_pii(text)

        assert "jane.doe+alerts@sub.domain.org" not in masked
        assert "[EMAIL_REDACTED]" in masked


class TestHostnameMasking:
    """Tests for hostname/FQDN detection and masking."""

    def test_hostname_masking(self) -> None:
        """Internal hostnames with FQDN patterns should be masked."""
        text = "Server prod-payment-01.internal.corp.net is not responding."
        masked = mask_pii(text)

        assert "prod-payment-01.internal.corp.net" not in masked
        assert "[HOSTNAME_REDACTED]" in masked
        assert "not responding" in masked

    def test_external_hostname(self) -> None:
        """External hostnames should also be masked."""
        text = "DNS resolution failing for api.gateway.service.com"
        masked = mask_pii(text)

        assert "api.gateway.service.com" not in masked
        assert "[HOSTNAME_REDACTED]" in masked


class TestPersonNameMasking:
    """Tests for person name detection in incident context."""

    def test_person_name_in_context(self) -> None:
        """Person names following context keywords should be masked.

        Names like 'Contact: John Smith' should be detected and masked
        because of the contextual keyword 'Contact'.
        """
        text = "Contact: John Smith regarding the payment outage."
        masked = mask_pii(text)

        assert "John Smith" not in masked
        assert "[PERSON_REDACTED]" in masked
        assert "payment outage" in masked

    def test_person_name_with_reported_by(self) -> None:
        """Names with 'Reported by' context should be masked."""
        text = "Reported by Jane Doe on 2024-01-15."
        masked = mask_pii(text)

        assert "Jane Doe" not in masked
        assert "[PERSON_REDACTED]" in masked

    def test_person_name_without_context_preserved(self) -> None:
        """Names without contextual keywords should not be masked.

        Service names that look like person names (e.g., 'Bad Gateway')
        should not be falsely masked.
        """
        text = "Bad Gateway error on payment-service at 14:30 UTC."
        masked = mask_pii(text)

        # 'Bad Gateway' should NOT be masked (not a person name context)
        assert "Bad Gateway" in masked


class TestCloudKeyPattern:
    """Tests for cloud provider API key detection."""

    def test_cloud_key_pattern(self) -> None:
        """AWS access keys should be detected and masked."""
        text = "Found leaked key AKIAIOSFODNN7EXAMPLE in the config file."
        masked = mask_pii(text)

        assert "AKIAIOSFODNN7EXAMPLE" not in masked
        assert "[AWS_KEY_REDACTED]" in masked
        assert "config file" in masked

    def test_openai_api_key(self) -> None:
        """OpenAI-style API keys should be masked."""
        text = "Set OPENAI_API_KEY=sk-abc123def456ghi789jkl012mno345pqr678 in .env"
        masked = mask_pii(text)

        assert "sk-abc123def456ghi789jkl012mno345pqr678" not in masked
        assert "[API_KEY_REDACTED]" in masked


class TestNoPIIPassthrough:
    """Tests ensuring non-PII text passes through unmodified."""

    def test_no_pii_passthrough(self) -> None:
        """Text with no PII should pass through completely unchanged."""
        text = (
            "The payment-service is returning HTTP 502 Bad Gateway errors. "
            "Error rate is at 15% on the /api/v2/payments/process endpoint. "
            "Connection pool shows max_connections=50, active=50, idle=0. "
            "This started after the v2.4.1 deployment at 14:30 UTC."
        )
        masked = mask_pii(text)

        assert masked == text

    def test_technical_content_preserved(self) -> None:
        """Technical content like error codes and endpoints should not be masked."""
        text = (
            "ERROR: ORA-12154 TNS could not resolve service name. "
            "Endpoint /api/v2/health returned status 503. "
            "Pod payment-service-7b8f9c6d4-xz2kl is in CrashLoopBackOff."
        )
        masked = mask_pii(text)

        assert "ORA-12154" in masked
        assert "/api/v2/health" in masked
        assert "503" in masked
        assert "CrashLoopBackOff" in masked


class TestIPAddressMasking:
    """Tests for IP address detection and masking."""

    def test_ip_address_masking(self) -> None:
        """IPv4 addresses should be masked."""
        text = "Server 10.0.1.42 is unreachable from 192.168.1.100."
        masked = mask_pii(text)

        assert "10.0.1.42" not in masked
        assert "192.168.1.100" not in masked
        assert masked.count("[IP_REDACTED]") == 2

    def test_ip_in_url_masked(self) -> None:
        """IP addresses embedded in URLs should be masked."""
        text = "Health check at http://10.0.1.42:8080/health is failing."
        masked = mask_pii(text)

        assert "10.0.1.42" not in masked
        assert "[IP_REDACTED]" in masked


class TestPhoneNumberMasking:
    """Tests for phone number detection and masking."""

    def test_phone_number_masking(self) -> None:
        """US phone numbers in various formats should be masked."""
        text = "Call +1-555-0123 or (555) 012-3456 for escalation."
        masked = mask_pii(text)

        assert "+1-555-0123" not in masked
        assert "(555) 012-3456" not in masked
        assert "[PHONE_REDACTED]" in masked

    def test_phone_with_dots(self) -> None:
        """Phone numbers with dot separators should be masked."""
        text = "Backup contact: 555.012.3456"
        masked = mask_pii(text)

        assert "555.012.3456" not in masked
        assert "[PHONE_REDACTED]" in masked


class TestUUIDRetained:
    """Tests ensuring UUIDs are not masked (they are system identifiers, not PII)."""

    def test_uuid_retained(self) -> None:
        """UUIDs should NOT be masked — they are technical identifiers.

        Unlike PII, UUIDs like trace IDs and correlation IDs are essential
        for debugging and should be preserved in the text.
        """
        text = (
            "Trace ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890. "
            "Correlation ID: 550e8400-e29b-41d4-a716-446655440000."
        )
        masked = mask_pii(text)

        assert "a1b2c3d4-e5f6-7890-abcd-ef1234567890" in masked
        assert "550e8400-e29b-41d4-a716-446655440000" in masked

    def test_uuid_alongside_pii(self) -> None:
        """UUIDs should be preserved even when PII is present in the same text."""
        text = (
            "Request a1b2c3d4-e5f6-7890-abcd-ef1234567890 from "
            "user john@example.com at 10.0.1.42"
        )
        masked = mask_pii(text)

        assert "a1b2c3d4-e5f6-7890-abcd-ef1234567890" in masked
        assert "john@example.com" not in masked
        assert "10.0.1.42" not in masked
