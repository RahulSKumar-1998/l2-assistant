"""PII anonymizer for incident and KB article text.

Provides pattern-based and context-aware detection of Personally
Identifiable Information (PII) with reversible anonymization using
deterministic placeholder tokens.

Supported PII types:
    - Email addresses
    - IPv4 / IPv6 addresses
    - Hostnames / FQDNs
    - Phone numbers (US / international)
    - Credit card numbers (with Luhn validation)
    - Cloud provider API keys (AWS, Azure, GCP)
    - Person names from ServiceNow context fields

UUIDs are explicitly retained — they are system identifiers, not PII.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


# ── PII Type Enumeration ────────────────────────────────────────────────────


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    IP_ADDRESS = "ip_address"
    HOSTNAME = "hostname"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    CLOUD_KEY = "cloud_key"
    PERSON_NAME = "person_name"


# ── Result Models ────────────────────────────────────────────────────────────


class PIIMatch(BaseModel):
    """A single PII detection result."""
    original: str = Field(..., description="Original PII text")
    replacement: str = Field(..., description="Anonymized replacement token")
    pii_type: PIIType = Field(..., description="Type of PII detected")
    start: int = Field(..., description="Start character position in original text")
    end: int = Field(..., description="End character position in original text")


class AnonymizedResult(BaseModel):
    """Result of PII anonymization on a text string."""
    masked_text: str = Field(..., description="Text with PII replaced by tokens")
    replacements: list[PIIMatch] = Field(
        default_factory=list,
        description="List of PII matches and their replacements",
    )
    pii_count: int = Field(default=0, description="Total PII items detected")


class SafetyCheckResult(BaseModel):
    """Result of a safety check for indexing."""
    is_safe: bool = Field(..., description="Whether the text is safe to index")
    reasons: list[str] = Field(
        default_factory=list,
        description="Reasons why the text is not safe (empty if safe)",
    )


# ── Compiled Patterns ────────────────────────────────────────────────────────


# UUID pattern — used to EXCLUDE from PII detection
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# Email address
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# IPv4 address (not inside a UUID or version string)
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# IPv6 address (simplified)
_IPV6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    r"|\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b"
    r"|\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b"
)

# Phone numbers (US / international)
_PHONE_RE = re.compile(
    r"(?<!\d)"                           # not preceded by digit
    r"(?:\+?1[\s.-]?)?"                  # optional country code
    r"(?:\(?\d{3}\)?[\s.-]?)"            # area code
    r"\d{3}[\s.-]?\d{4}"                 # subscriber number
    r"(?!\d)"                            # not followed by digit
)

# Credit card number (13-19 digits, optionally grouped with spaces/dashes)
_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ \-]?){13,19}\b"
)

# FQDN hostname (min 2 labels, TLD ≥ 2 chars)
_HOSTNAME_RE = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.){1,}"
    r"[a-zA-Z]{2,}\b"
)

# Common hostname-only TLDs to filter hostnames
_HOSTNAME_TLDS = {
    "com", "org", "net", "io", "dev", "co", "edu", "gov", "mil",
    "int", "info", "biz", "cloud", "app", "internal", "local",
    "example", "test", "corp",
}

# Cloud provider API key patterns
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_AZURE_KEY_RE = re.compile(
    r"\b[A-Za-z0-9/+]{86}==\b"
)
_GCP_KEY_RE = re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b")

# Person name patterns — context-aware extraction from ServiceNow fields
_PERSON_CONTEXT_RE = re.compile(
    r"(?:assigned\s+to|reported\s+by|opened\s+by|resolved\s+by|"
    r"caller|requested\s+by|closed\s+by|updated\s+by|contact)"
    r"\s*[:=]\s*"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    re.IGNORECASE,
)


# ── Luhn Checksum ────────────────────────────────────────────────────────────


def _passes_luhn(number_str: str) -> bool:
    """Validate a number string using the Luhn algorithm.

    Args:
        number_str: Numeric string (spaces/dashes stripped).

    Returns:
        True if the number passes the Luhn checksum.
    """
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13:
        return False

    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# ── PIIAnonymizer ────────────────────────────────────────────────────────────


class PIIAnonymizer:
    """Detects and anonymizes PII in incident and KB article text.

    Uses a two-pass approach:
        1. Pattern-based detection for structured PII (emails, IPs, etc.)
        2. Context-aware regex for person names in ServiceNow fields

    UUIDs are explicitly excluded from detection since they are
    system identifiers, not personally identifiable.

    Example:
        >>> anonymizer = PIIAnonymizer()
        >>> result = anonymizer.anonymize("Contact john.doe@acme.com for help")
        >>> result.masked_text
        'Contact [EMAIL_a1b2c3] for help'
    """

    def __init__(self, *, salt: str = "l2-assistant") -> None:
        """Initialize the PII anonymizer.

        Args:
            salt: Salt for deterministic hash-based placeholder generation.
                  Using a salt ensures the same PII always maps to the same
                  placeholder within a session, enabling consistency.
        """
        self._salt = salt
        self._log = logger.bind(component="pii_anonymizer")

    def _make_placeholder(self, pii_type: PIIType, original: str) -> str:
        """Generate a deterministic placeholder token for a PII match.

        Args:
            pii_type: The type of PII.
            original: The original PII text.

        Returns:
            A placeholder string like ``[EMAIL_a1b2c3]``.
        """
        hash_input = f"{self._salt}:{original}".encode()
        short_hash = hashlib.sha256(hash_input).hexdigest()[:6]
        return f"[{pii_type.value.upper()}_{short_hash}]"

    def _collect_uuid_spans(self, text: str) -> set[tuple[int, int]]:
        """Find all UUID spans to exclude from PII detection.

        Args:
            text: Input text.

        Returns:
            Set of (start, end) tuples for UUID matches.
        """
        return {(m.start(), m.end()) for m in _UUID_RE.finditer(text)}

    def _overlaps_uuid(
        self, start: int, end: int, uuid_spans: set[tuple[int, int]]
    ) -> bool:
        """Check whether a span overlaps with any UUID span.

        Args:
            start: Match start position.
            end: Match end position.
            uuid_spans: Set of UUID (start, end) spans.

        Returns:
            True if the span overlaps any UUID.
        """
        for u_start, u_end in uuid_spans:
            if start < u_end and end > u_start:
                return True
        return False

    def _overlaps_existing(
        self,
        start: int,
        end: int,
        existing: list[PIIMatch],
    ) -> bool:
        """Check whether a span overlaps with already-detected PII.

        Args:
            start: Match start position.
            end: Match end position.
            existing: Already-detected PII matches.

        Returns:
            True if the span overlaps any existing match.
        """
        for m in existing:
            if start < m.end and end > m.start:
                return True
        return False

    def _detect_emails(
        self,
        text: str,
        uuid_spans: set[tuple[int, int]],
        existing: list[PIIMatch],
    ) -> list[PIIMatch]:
        """Detect email addresses in text."""
        matches: list[PIIMatch] = []
        for m in _EMAIL_RE.finditer(text):
            if self._overlaps_uuid(m.start(), m.end(), uuid_spans):
                continue
            if self._overlaps_existing(m.start(), m.end(), existing):
                continue
            matches.append(PIIMatch(
                original=m.group(),
                replacement=self._make_placeholder(PIIType.EMAIL, m.group()),
                pii_type=PIIType.EMAIL,
                start=m.start(),
                end=m.end(),
            ))
        return matches

    def _detect_ip_addresses(
        self,
        text: str,
        uuid_spans: set[tuple[int, int]],
        existing: list[PIIMatch],
    ) -> list[PIIMatch]:
        """Detect IPv4 and IPv6 addresses in text."""
        matches: list[PIIMatch] = []
        for pattern in (_IPV4_RE, _IPV6_RE):
            for m in pattern.finditer(text):
                if self._overlaps_uuid(m.start(), m.end(), uuid_spans):
                    continue
                if self._overlaps_existing(m.start(), m.end(), existing):
                    continue
                matches.append(PIIMatch(
                    original=m.group(),
                    replacement=self._make_placeholder(PIIType.IP_ADDRESS, m.group()),
                    pii_type=PIIType.IP_ADDRESS,
                    start=m.start(),
                    end=m.end(),
                ))
        return matches

    def _detect_phones(
        self,
        text: str,
        uuid_spans: set[tuple[int, int]],
        existing: list[PIIMatch],
    ) -> list[PIIMatch]:
        """Detect phone numbers in text."""
        matches: list[PIIMatch] = []
        for m in _PHONE_RE.finditer(text):
            if self._overlaps_uuid(m.start(), m.end(), uuid_spans):
                continue
            if self._overlaps_existing(m.start(), m.end(), existing):
                continue
            # Skip very short matches that are likely not phone numbers
            digits_only = re.sub(r"\D", "", m.group())
            if len(digits_only) < 10:
                continue
            matches.append(PIIMatch(
                original=m.group(),
                replacement=self._make_placeholder(PIIType.PHONE, m.group()),
                pii_type=PIIType.PHONE,
                start=m.start(),
                end=m.end(),
            ))
        return matches

    def _detect_credit_cards(
        self,
        text: str,
        uuid_spans: set[tuple[int, int]],
        existing: list[PIIMatch],
    ) -> list[PIIMatch]:
        """Detect credit card numbers with Luhn validation."""
        matches: list[PIIMatch] = []
        for m in _CREDIT_CARD_RE.finditer(text):
            if self._overlaps_uuid(m.start(), m.end(), uuid_spans):
                continue
            if self._overlaps_existing(m.start(), m.end(), existing):
                continue
            digits = re.sub(r"\D", "", m.group())
            if len(digits) < 13 or len(digits) > 19:
                continue
            if not _passes_luhn(digits):
                continue
            matches.append(PIIMatch(
                original=m.group(),
                replacement=self._make_placeholder(PIIType.CREDIT_CARD, m.group()),
                pii_type=PIIType.CREDIT_CARD,
                start=m.start(),
                end=m.end(),
            ))
        return matches

    def _detect_hostnames(
        self,
        text: str,
        uuid_spans: set[tuple[int, int]],
        existing: list[PIIMatch],
    ) -> list[PIIMatch]:
        """Detect FQDNs that look like internal hostnames."""
        matches: list[PIIMatch] = []
        for m in _HOSTNAME_RE.finditer(text):
            if self._overlaps_uuid(m.start(), m.end(), uuid_spans):
                continue
            if self._overlaps_existing(m.start(), m.end(), existing):
                continue
            hostname = m.group()
            # Only flag if TLD matches known hostname TLDs
            tld = hostname.rsplit(".", 1)[-1].lower()
            if tld not in _HOSTNAME_TLDS:
                continue
            # Skip common non-PII hostnames
            lower = hostname.lower()
            if lower in ("e.g", "i.e", "etc.com", "example.com"):
                continue
            matches.append(PIIMatch(
                original=hostname,
                replacement=self._make_placeholder(PIIType.HOSTNAME, hostname),
                pii_type=PIIType.HOSTNAME,
                start=m.start(),
                end=m.end(),
            ))
        return matches

    def _detect_cloud_keys(
        self,
        text: str,
        uuid_spans: set[tuple[int, int]],
        existing: list[PIIMatch],
    ) -> list[PIIMatch]:
        """Detect cloud provider API keys (AWS, Azure, GCP)."""
        matches: list[PIIMatch] = []
        for pattern in (_AWS_KEY_RE, _AZURE_KEY_RE, _GCP_KEY_RE):
            for m in pattern.finditer(text):
                if self._overlaps_uuid(m.start(), m.end(), uuid_spans):
                    continue
                if self._overlaps_existing(m.start(), m.end(), existing):
                    continue
                matches.append(PIIMatch(
                    original=m.group(),
                    replacement=self._make_placeholder(PIIType.CLOUD_KEY, m.group()),
                    pii_type=PIIType.CLOUD_KEY,
                    start=m.start(),
                    end=m.end(),
                ))
        return matches

    def _detect_person_names(
        self,
        text: str,
        uuid_spans: set[tuple[int, int]],
        existing: list[PIIMatch],
    ) -> list[PIIMatch]:
        """Detect person names from ServiceNow context fields.

        Uses context patterns like 'assigned to: John Smith' to extract
        names without requiring a full NER model.
        """
        matches: list[PIIMatch] = []
        for m in _PERSON_CONTEXT_RE.finditer(text):
            name = m.group(1).strip()
            if not name:
                continue
            # Locate the name within the full match
            name_start = m.start() + m.group().index(name)
            name_end = name_start + len(name)
            if self._overlaps_uuid(name_start, name_end, uuid_spans):
                continue
            if self._overlaps_existing(name_start, name_end, existing):
                continue
            matches.append(PIIMatch(
                original=name,
                replacement=self._make_placeholder(PIIType.PERSON_NAME, name),
                pii_type=PIIType.PERSON_NAME,
                start=name_start,
                end=name_end,
            ))
        return matches

    def anonymize(self, text: str) -> AnonymizedResult:
        """Anonymize all PII in the given text.

        Performs multi-pass detection and replaces PII with deterministic
        placeholder tokens. UUIDs are preserved.

        Args:
            text: Input text to anonymize.

        Returns:
            AnonymizedResult with masked text and replacement metadata.
        """
        if not text or not text.strip():
            return AnonymizedResult(masked_text=text, replacements=[], pii_count=0)

        uuid_spans = self._collect_uuid_spans(text)
        all_matches: list[PIIMatch] = []

        # Detection order: most specific first to avoid overlap issues
        all_matches.extend(self._detect_emails(text, uuid_spans, all_matches))
        all_matches.extend(self._detect_cloud_keys(text, uuid_spans, all_matches))
        all_matches.extend(self._detect_credit_cards(text, uuid_spans, all_matches))
        all_matches.extend(self._detect_ip_addresses(text, uuid_spans, all_matches))
        all_matches.extend(self._detect_phones(text, uuid_spans, all_matches))
        all_matches.extend(self._detect_hostnames(text, uuid_spans, all_matches))
        all_matches.extend(self._detect_person_names(text, uuid_spans, all_matches))

        if not all_matches:
            return AnonymizedResult(masked_text=text, replacements=[], pii_count=0)

        # Sort matches by start position in reverse to replace from end to start
        all_matches.sort(key=lambda m: m.start, reverse=True)

        masked = text
        for match in all_matches:
            masked = masked[:match.start] + match.replacement + masked[match.end:]

        # Re-sort for output in reading order
        all_matches.sort(key=lambda m: m.start)

        self._log.info(
            "pii_anonymized",
            pii_count=len(all_matches),
            pii_types=[m.pii_type.value for m in all_matches],
        )

        return AnonymizedResult(
            masked_text=masked,
            replacements=all_matches,
            pii_count=len(all_matches),
        )

    def is_safe_to_index(self, text: str) -> tuple[bool, list[str]]:
        """Check whether text is safe to index in the vector store.

        Text is considered unsafe if it contains credit card numbers
        or cloud API keys. Other PII types are expected to be masked
        before indexing.

        Args:
            text: Text to check.

        Returns:
            Tuple of (is_safe, reasons). Reasons is empty if safe.
        """
        reasons: list[str] = []
        uuid_spans = self._collect_uuid_spans(text)

        credit_cards = self._detect_credit_cards(text, uuid_spans, [])
        if credit_cards:
            reasons.append(
                f"Contains {len(credit_cards)} credit card number(s)"
            )

        cloud_keys = self._detect_cloud_keys(text, uuid_spans, [])
        if cloud_keys:
            reasons.append(
                f"Contains {len(cloud_keys)} cloud API key(s)"
            )

        # Check for unmasked emails in high-sensitivity mode
        emails = self._detect_emails(text, uuid_spans, [])
        if len(emails) > 3:
            reasons.append(
                f"Contains {len(emails)} email addresses (possible data dump)"
            )

        is_safe = len(reasons) == 0
        if not is_safe:
            self._log.warning(
                "unsafe_for_indexing",
                reasons=reasons,
                text_length=len(text),
            )

        return is_safe, reasons
