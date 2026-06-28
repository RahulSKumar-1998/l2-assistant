"""NLP preprocessing pipeline for ServiceNow incident tickets.

Provides:
    - ``TicketPreprocessor``: Full NLP pipeline producing ``ProcessedTicket``
      with cleaned text, extracted entities, keywords, and summary.
    - ``TicketChunker``: Sentence-aware chunking into ``TextChunk`` objects
      with metadata for embedding.

Uses regex-based entity extraction (no spaCy model required) and
integrates with PII anonymization for safe indexing.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import structlog

from app.governance.pii_anonymizer import PIIAnonymizer
from app.models.incident import (
    ChunkType,
    ExtractedEntity,
    IncidentRecord,
    IncidentType,
    ProcessedTicket,
    SourceType,
    TextChunk,
)
from app.utils.text_utils import (
    chunk_text_by_sentences,
    clean_text,
    combine_incident_text,
    split_sentences,
    strip_html,
)

logger = structlog.get_logger(__name__)


# ── Entity Extraction Patterns ───────────────────────────────────────────────

# HTTP error codes (4xx, 5xx)
_HTTP_ERROR_RE = re.compile(
    r"\b(?:HTTP\s*)?([45]\d{2})\b"
)

# Service/application names (word-word pattern like 'payment-service')
_SERVICE_NAME_RE = re.compile(
    r"\b([a-z][a-z0-9]*(?:-[a-z0-9]+){1,4})\b"
)

# Hostnames / FQDNs
_HOSTNAME_RE = re.compile(
    r"\b([a-zA-Z][a-zA-Z0-9\-]+(?:\.[a-zA-Z][a-zA-Z0-9\-]+)+)\b"
)

# Error codes (ERR-XXXX, E####, or all-caps ERROR_CODE patterns)
_ERROR_CODE_RE = re.compile(
    r"\b((?:ERR|ERROR|WARN|FATAL)[-_]?\d{3,6})\b", re.IGNORECASE
)

# Kubernetes resources (pod/deployment/service names)
_K8S_RESOURCE_RE = re.compile(
    r"\b((?:pod|deploy(?:ment)?|svc|service|statefulset|daemonset|job|cronjob)"
    r"[/\s]+[a-z][a-z0-9\-]+)\b",
    re.IGNORECASE,
)

# IP addresses
_IP_RE = re.compile(
    r"\b((?:\d{1,3}\.){3}\d{1,3})\b"
)

# Port numbers in context (e.g., ":8080", "port 443")
_PORT_RE = re.compile(
    r"(?::(\d{2,5}))\b|(?:port\s+(\d{2,5}))\b",
    re.IGNORECASE,
)

# Stack trace indicators
_STACK_TRACE_RE = re.compile(
    r"\b((?:java\.\w+\.[\w.]+Exception|"
    r"Traceback|"
    r"at\s+[\w.$]+\([\w.]+:\d+\)|"
    r"File\s+\"[^\"]+\",\s+line\s+\d+))\b"
)

# AWS/Cloud resource ARNs
_ARN_RE = re.compile(
    r"\b(arn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d*:[a-zA-Z0-9\-_/:.]+)\b"
)


# ── Incident Type Classification ─────────────────────────────────────────────

_INCIDENT_TYPE_KEYWORDS: dict[IncidentType, list[str]] = {
    IncidentType.APPLICATION_ERROR: [
        "exception", "stack trace", "traceback", "null pointer", "segfault",
        "crash", "oom", "out of memory", "heap", "application error",
        "500 error", "internal server error", "unhandled exception",
    ],
    IncidentType.INFRASTRUCTURE: [
        "disk full", "disk space", "cpu", "memory", "server", "node",
        "vm", "virtual machine", "hardware", "reboot", "kernel panic",
        "host unreachable", "container", "docker", "kubernetes", "k8s",
    ],
    IncidentType.NETWORK: [
        "dns", "network", "timeout", "connection refused", "502", "503",
        "504", "gateway", "load balancer", "proxy", "latency", "packet loss",
        "routing", "firewall", "vpc", "subnet",
    ],
    IncidentType.SECURITY: [
        "ssl", "tls", "certificate", "auth", "authentication",
        "unauthorized", "401", "403", "forbidden", "vulnerability",
        "cve", "breach", "intrusion", "xss", "injection",
    ],
    IncidentType.PERFORMANCE: [
        "slow", "performance", "latency", "response time", "throughput",
        "bottleneck", "degraded", "lag", "queue depth", "backlog",
    ],
    IncidentType.ACCESS_MANAGEMENT: [
        "access", "permission", "role", "iam", "rbac", "ldap",
        "active directory", "login", "sso", "mfa", "password reset",
    ],
    IncidentType.DATA_ISSUE: [
        "data loss", "data corruption", "replication", "backup",
        "restore", "database", "migration", "etl", "pipeline",
        "inconsistency", "missing data",
    ],
}


# ── Stop words for keyword extraction ────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "need",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "about", "into", "through", "during", "before", "after",
    "and", "but", "or", "not", "no", "if", "so", "as", "all", "each",
    "which", "when", "where", "how", "what", "who", "whom",
    "there", "here", "also", "very", "just", "than", "then", "now",
    "only", "any", "some", "other", "new", "more", "out", "over",
    "such", "same", "well", "however", "since", "until", "while",
}


# ── TicketPreprocessor ───────────────────────────────────────────────────────


class TicketPreprocessor:
    """NLP preprocessing pipeline for ServiceNow incident tickets.

    Transforms raw ``IncidentRecord`` objects into ``ProcessedTicket``
    models ready for embedding and indexing. Uses regex-based entity
    extraction (no spaCy model load required).

    Pipeline stages:
        1. Combine and clean text fields
        2. PII anonymization
        3. Entity extraction (services, errors, hostnames, etc.)
        4. Keyword extraction (frequency-based)
        5. Incident type classification
        6. Extractive summary generation

    Example:
        >>> preprocessor = TicketPreprocessor()
        >>> processed = preprocessor.preprocess(incident_record)
        >>> processed.entities  # [ExtractedEntity(...), ...]
        >>> processed.incident_type  # IncidentType.NETWORK
    """

    def __init__(
        self,
        *,
        anonymize_pii: bool = True,
        max_keywords: int = 15,
    ) -> None:
        """Initialize the ticket preprocessor.

        Args:
            anonymize_pii: Whether to mask PII in processed text.
            max_keywords: Maximum number of keywords to extract.
        """
        self._anonymize_pii = anonymize_pii
        self._max_keywords = max_keywords
        self._pii_anonymizer = PIIAnonymizer() if anonymize_pii else None
        self._log = logger.bind(component="ticket_preprocessor")

    def preprocess(self, incident: IncidentRecord) -> ProcessedTicket:
        """Run the full preprocessing pipeline on an incident.

        Args:
            incident: Raw incident record from ServiceNow.

        Returns:
            ProcessedTicket with cleaned text, entities, keywords, and summary.
        """
        self._log.info("preprocessing_ticket", number=incident.number)

        # 1. Combine text fields
        combined_text = combine_incident_text(
            short_description=incident.short_description,
            description=incident.description,
            work_notes=incident.work_notes,
            resolution_notes=incident.resolution_notes or "",
        )

        # 2. Clean text (strip HTML, normalize whitespace)
        cleaned = clean_text(combined_text)

        # 3. PII anonymization
        if self._pii_anonymizer:
            anon_result = self._pii_anonymizer.anonymize(cleaned)
            cleaned = anon_result.masked_text
            if anon_result.pii_count > 0:
                self._log.info(
                    "pii_masked_in_ticket",
                    number=incident.number,
                    pii_count=anon_result.pii_count,
                )

        # 4. Entity extraction
        entities = self._extract_entities(combined_text)

        # 5. Keyword extraction
        keywords = self._extract_keywords(cleaned)

        # 6. Incident type classification
        incident_type = self.classify_incident_type(
            cleaned,
            category=incident.category,
        )

        # 7. Summary generation
        summary = self._generate_summary(combined_text)

        return ProcessedTicket(
            source_id=incident.number,
            sys_id=incident.sys_id,
            cleaned_text=cleaned,
            entities=entities,
            keywords=keywords,
            category=incident.category,
            subcategory=incident.subcategory,
            incident_type=incident_type,
            summary=summary,
            priority=incident.priority,
            cmdb_ci=incident.cmdb_ci,
            assignment_group=incident.assignment_group,
            opened_at=incident.opened_at,
            resolved_at=incident.resolved_at,
            resolution_notes=incident.resolution_notes,
            root_cause=incident.root_cause,
            work_notes=incident.work_notes,
        )

    def mask_pii(self, text: str) -> str:
        """Anonymize PII in the given text.

        Args:
            text: Input text.

        Returns:
            Text with PII replaced by placeholder tokens.
        """
        if self._pii_anonymizer is None:
            return text
        result = self._pii_anonymizer.anonymize(text)
        return result.masked_text

    def classify_incident_type(
        self,
        text: str,
        category: str = "",
    ) -> IncidentType:
        """Classify the incident type using keyword matching.

        Scores each incident type by counting keyword matches in
        the text. Falls back to category mapping if no keywords match.

        Args:
            text: Cleaned incident text.
            category: ServiceNow category field.

        Returns:
            Best-matching IncidentType.
        """
        text_lower = text.lower()
        scores: dict[IncidentType, int] = {}

        for inc_type, keywords in _INCIDENT_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[inc_type] = score

        if scores:
            return max(scores, key=scores.get)  # type: ignore[arg-type]

        # Fallback to category mapping
        category_map: dict[str, IncidentType] = {
            "network": IncidentType.NETWORK,
            "software": IncidentType.APPLICATION_ERROR,
            "hardware": IncidentType.INFRASTRUCTURE,
            "security": IncidentType.SECURITY,
            "database": IncidentType.DATA_ISSUE,
            "access": IncidentType.ACCESS_MANAGEMENT,
        }
        return category_map.get(category.lower(), IncidentType.UNKNOWN)

    def _extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Extract named entities using regex patterns.

        Extracts service names, error codes, hostnames, IPs, ports,
        Kubernetes resources, and stack trace indicators.

        Args:
            text: Raw combined text (not cleaned/lowered).

        Returns:
            List of ExtractedEntity objects.
        """
        entities: list[ExtractedEntity] = []
        seen_spans: set[tuple[int, int]] = set()

        def _add_matches(
            pattern: re.Pattern[str],
            label: str,
            source_text: str,
            group: int = 0,
        ) -> None:
            for m in pattern.finditer(source_text):
                start, end = m.start(group), m.end(group)
                if (start, end) in seen_spans:
                    continue
                value = m.group(group).strip()
                if not value or len(value) < 2:
                    continue
                seen_spans.add((start, end))
                entities.append(ExtractedEntity(
                    text=value,
                    label=label,
                    start=start,
                    end=end,
                ))

        _add_matches(_HTTP_ERROR_RE, "HTTP_ERROR", text, group=0)
        _add_matches(_ERROR_CODE_RE, "ERROR_CODE", text, group=1)
        _add_matches(_SERVICE_NAME_RE, "SERVICE", text, group=1)
        _add_matches(_HOSTNAME_RE, "HOSTNAME", text, group=1)
        _add_matches(_IP_RE, "IP_ADDRESS", text, group=1)
        _add_matches(_K8S_RESOURCE_RE, "K8S_RESOURCE", text, group=1)
        _add_matches(_STACK_TRACE_RE, "STACK_TRACE", text, group=1)
        _add_matches(_ARN_RE, "AWS_ARN", text, group=1)

        # Deduplicate by text+label
        unique: dict[tuple[str, str], ExtractedEntity] = {}
        for entity in entities:
            key = (entity.text.lower(), entity.label)
            if key not in unique:
                unique[key] = entity
        entities = list(unique.values())

        self._log.debug(
            "entities_extracted",
            count=len(entities),
            labels=[e.label for e in entities],
        )
        return entities

    def _extract_keywords(
        self,
        cleaned_text: str,
    ) -> list[str]:
        """Extract keywords using term frequency.

        Simple approach: tokenize, filter stop words and short tokens,
        return top-N by frequency.

        Args:
            cleaned_text: Cleaned and lowered text.

        Returns:
            Top keywords sorted by frequency.
        """
        # Tokenize on non-alphanumeric boundaries
        tokens = re.findall(r"\b[a-z][a-z0-9_\-]{2,}\b", cleaned_text.lower())
        # Filter stop words
        filtered = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
        # Count frequencies
        freq = Counter(filtered)
        top = freq.most_common(self._max_keywords)
        return [word for word, _ in top]

    def _generate_summary(self, text: str) -> str:
        """Generate a 2-sentence extractive summary.

        Picks the first two most informative sentences based on
        length and keyword density.

        Args:
            text: Original combined text.

        Returns:
            2-sentence summary string.
        """
        clean = strip_html(text)
        sentences = split_sentences(clean)
        if not sentences:
            return text[:200] if text else ""

        if len(sentences) <= 2:
            return " ".join(sentences)

        # Score sentences by length (prefer medium-length) and position
        scored = []
        for i, sent in enumerate(sentences):
            word_count = len(sent.split())
            # Prefer sentences with 8-40 words, early in the document
            length_score = min(word_count, 40) / 40
            position_score = 1.0 / (i + 1)
            score = length_score * 0.6 + position_score * 0.4
            scored.append((score, i, sent))

        # Sort by score, take top 2, then re-sort by original position
        scored.sort(key=lambda x: x[0], reverse=True)
        top_two = sorted(scored[:2], key=lambda x: x[1])
        return " ".join(s for _, _, s in top_two)


# ── TicketChunker ────────────────────────────────────────────────────────────


class TicketChunker:
    """Sentence-aware text chunker for incident tickets.

    Splits processed ticket text into ``TextChunk`` objects suitable
    for embedding and vector storage. Chunks respect sentence boundaries
    and include metadata for retrieval.

    Example:
        >>> chunker = TicketChunker(chunk_size=512, overlap=50)
        >>> chunks = chunker.chunk(processed_ticket)
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> None:
        """Initialize the ticket chunker.

        Args:
            chunk_size: Maximum tokens per chunk.
            overlap: Overlap tokens between consecutive chunks.
        """
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._log = logger.bind(component="ticket_chunker")

    def chunk(self, ticket: ProcessedTicket) -> list[TextChunk]:
        """Chunk a processed ticket into TextChunk objects.

        Creates separate chunks for description, work notes, and
        resolution (if present), each with appropriate metadata.

        Args:
            ticket: Processed ticket to chunk.

        Returns:
            List of TextChunk objects with metadata.
        """
        chunks: list[TextChunk] = []
        metadata = self._build_metadata(ticket)
        chunk_index = 0

        # Chunk the main cleaned text (description + work notes)
        desc_chunks = chunk_text_by_sentences(
            ticket.cleaned_text,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )
        for text in desc_chunks:
            chunks.append(TextChunk(
                chunk_text=text,
                chunk_type=ChunkType.DESCRIPTION,
                source_id=ticket.source_id,
                source_type=SourceType.INCIDENT,
                metadata=metadata,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

        # Chunk resolution notes separately if present
        if ticket.resolution_notes:
            res_text = clean_text(ticket.resolution_notes)
            res_chunks = chunk_text_by_sentences(
                res_text,
                chunk_size=self._chunk_size,
                overlap=self._overlap,
            )
            res_metadata = {**metadata, "has_resolution": True}
            for text in res_chunks:
                chunks.append(TextChunk(
                    chunk_text=text,
                    chunk_type=ChunkType.RESOLUTION,
                    source_id=ticket.source_id,
                    source_type=SourceType.INCIDENT,
                    metadata=res_metadata,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

        self._log.info(
            "ticket_chunked",
            source_id=ticket.source_id,
            total_chunks=len(chunks),
        )
        return chunks

    @staticmethod
    def _build_metadata(ticket: ProcessedTicket) -> dict:
        """Build metadata dict from ProcessedTicket fields.

        Args:
            ticket: The processed ticket.

        Returns:
            Metadata dict for vector store.
        """
        metadata: dict = {
            "source_id": ticket.source_id,
            "source_type": "incident",
            "category": ticket.category,
            "subcategory": ticket.subcategory,
            "incident_type": ticket.incident_type.value,
            "priority": ticket.priority,
            "cmdb_ci": ticket.cmdb_ci,
            "assignment_group": ticket.assignment_group,
        }
        if ticket.opened_at:
            metadata["opened_at"] = ticket.opened_at.isoformat()
        if ticket.resolved_at:
            metadata["resolved_at"] = ticket.resolved_at.isoformat()
            metadata["is_resolved"] = True
        else:
            metadata["is_resolved"] = False
        if ticket.keywords:
            metadata["keywords"] = ",".join(ticket.keywords[:10])
        return metadata
