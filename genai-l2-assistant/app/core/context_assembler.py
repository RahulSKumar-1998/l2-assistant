"""Token-aware context window builder for RAG pipeline.

Assembles retrieved chunks into a structured context string that
fits within the LLM's context budget, prioritising resolution chunks,
high-similarity content, and always including KB articles.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import structlog
from pydantic import BaseModel, Field

from app.core.retriever import RetrievedChunk
from app.models.incident import ChunkType, ProcessedTicket, SourceType
from app.models.recommendation import SimilarIncident
from app.utils.text_utils import count_tokens, truncate_to_tokens

logger = structlog.get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_CONTEXT_TOKENS: int = 6000
_INCIDENT_SUMMARY_BUDGET: int = 800
_SIMILAR_INCIDENTS_BUDGET: int = 2000
_KB_BUDGET: int = 2000
_OVERFLOW_RESERVE: int = 200  # headroom for formatting/separators


# ── Models ───────────────────────────────────────────────────────────────────


class AssembledContext(BaseModel):
    """The assembled context window ready for LLM consumption.

    Contains formatted sections for the current incident, similar
    historical incidents, and relevant KB articles, plus metadata
    about token usage and sources referenced.
    """
    incident_summary: str = Field(
        default="",
        description="Formatted summary of the current incident",
    )
    similar_incidents_context: str = Field(
        default="",
        description="Formatted context from similar historical incidents",
    )
    kb_context: str = Field(
        default="",
        description="Formatted context from KB articles and runbooks",
    )
    total_tokens: int = Field(
        default=0,
        description="Total token count of the assembled context",
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="Source IDs included in the context",
    )

    @property
    def full_context(self) -> str:
        """Combine all context sections into a single string.

        Returns:
            Full context string with section headers.
        """
        sections: list[str] = []
        if self.incident_summary:
            sections.append(self.incident_summary)
        if self.similar_incidents_context:
            sections.append(self.similar_incidents_context)
        if self.kb_context:
            sections.append(self.kb_context)
        return "\n\n".join(sections)


# ── Context Assembler ────────────────────────────────────────────────────────


class ContextAssembler:
    """Token-aware context window builder.

    Assembles retrieved chunks and incident metadata into a structured
    context string that respects the token budget. Applies the following
    priority order:

    1. **Current incident summary** — always included.
    2. **Resolution chunks** — highest value for troubleshooting.
    3. **KB articles** — always included for reference.
    4. **High-similarity chunks** — filled greedily by score.
    5. **Remaining chunks** — until budget is exhausted.

    Deduplicates sources and tracks all source IDs used.

    Args:
        max_context_tokens: Maximum total tokens for the context window.
    """

    def __init__(
        self,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> None:
        self._max_tokens = max_context_tokens
        logger.info("context_assembler_initialised", max_tokens=max_context_tokens)

    # ── Formatting helpers ───────────────────────────────────────────────

    @staticmethod
    def format_incident_context(ticket: ProcessedTicket) -> str:
        """Format the current incident into a structured context block.

        Args:
            ticket: The preprocessed incident ticket.

        Returns:
            Formatted incident summary string.
        """
        parts = [
            "=== CURRENT INCIDENT ===",
            f"Incident: {ticket.source_id}",
            f"Category: {ticket.category} / {ticket.subcategory}",
            f"Priority: P{ticket.priority}",
            f"Assignment Group: {ticket.assignment_group}",
            f"CMDB CI: {ticket.cmdb_ci}",
        ]

        if ticket.opened_at:
            parts.append(f"Opened: {ticket.opened_at.strftime('%Y-%m-%d %H:%M UTC')}")

        parts.append(f"\nDescription:\n{ticket.cleaned_text}")

        if ticket.entities:
            entity_strs = [f"  - {e.label}: {e.text}" for e in ticket.entities[:10]]
            parts.append(f"\nExtracted Entities:\n" + "\n".join(entity_strs))

        if ticket.keywords:
            parts.append(f"Keywords: {', '.join(ticket.keywords[:10])}")

        if ticket.summary:
            parts.append(f"\nSummary: {ticket.summary}")

        return "\n".join(parts)

    @staticmethod
    def format_similar_incident(
        incident: SimilarIncident,
        index: int,
    ) -> str:
        """Format a single similar incident for context inclusion.

        Args:
            incident: The similar incident to format.
            index: 1-based index number.

        Returns:
            Formatted similar incident string.
        """
        parts = [
            f"--- Similar Incident #{index} ---",
            f"Number: {incident.number}",
            f"Similarity: {incident.similarity_score:.0%}",
            f"Category: {incident.category}",
        ]

        if incident.resolution_time_min is not None:
            hours = incident.resolution_time_min // 60
            mins = incident.resolution_time_min % 60
            parts.append(f"Resolution Time: {hours}h {mins}m")

        if incident.resolution_summary:
            parts.append(f"Resolution: {incident.resolution_summary}")

        return "\n".join(parts)

    @staticmethod
    def format_kb_context(
        chunks: list[RetrievedChunk],
    ) -> str:
        """Format KB article chunks into a context block.

        Args:
            chunks: List of KB article chunks.

        Returns:
            Formatted KB articles context string.
        """
        if not chunks:
            return ""

        parts = ["=== KNOWLEDGE BASE ARTICLES ==="]
        seen_sources: set[str] = set()

        for chunk in chunks:
            if chunk.source_id in seen_sources:
                continue
            seen_sources.add(chunk.source_id)

            title = chunk.metadata.get("title", chunk.source_id)
            parts.append(f"\n--- {title} ({chunk.source_id}) ---")
            parts.append(chunk.chunk_text)

        return "\n".join(parts)

    # ── Core Assembly ────────────────────────────────────────────────────

    async def assemble(
        self,
        ticket: ProcessedTicket,
        chunks: list[RetrievedChunk],
        similar_incidents: list[SimilarIncident],
    ) -> AssembledContext:
        """Assemble the context window from incident, chunks, and similar incidents.

        Follows a greedy budget-filling strategy:

        1. Format and include the current incident summary.
        2. Always include KB article chunks.
        3. Prioritise resolution chunks from similar incidents.
        4. Fill remaining budget with highest-scoring chunks.

        Args:
            ticket: The preprocessed current incident.
            chunks: Retrieved and reranked chunks.
            similar_incidents: List of similar historical incidents.

        Returns:
            AssembledContext with formatted sections and metadata.
        """
        start = time.monotonic()
        sources_used: list[str] = []
        remaining_budget = self._max_tokens - _OVERFLOW_RESERVE

        # ── 1. Current incident summary ─────────────────────────────────
        incident_text = self.format_incident_context(ticket)
        incident_tokens = count_tokens(incident_text)

        if incident_tokens > _INCIDENT_SUMMARY_BUDGET:
            incident_text = truncate_to_tokens(incident_text, _INCIDENT_SUMMARY_BUDGET)
            incident_tokens = _INCIDENT_SUMMARY_BUDGET

        remaining_budget -= incident_tokens
        sources_used.append(ticket.source_id)

        # ── 2. Separate chunks by type ──────────────────────────────────
        kb_chunks: list[RetrievedChunk] = []
        resolution_chunks: list[RetrievedChunk] = []
        other_chunks: list[RetrievedChunk] = []

        for chunk in chunks:
            if chunk.source_type == SourceType.KB_ARTICLE or chunk.chunk_type == ChunkType.KB_ARTICLE:
                kb_chunks.append(chunk)
            elif chunk.chunk_type == ChunkType.RESOLUTION:
                resolution_chunks.append(chunk)
            else:
                other_chunks.append(chunk)

        # ── 3. KB articles — always include ─────────────────────────────
        kb_text = ""
        if kb_chunks:
            raw_kb = self.format_kb_context(kb_chunks)
            kb_tokens = count_tokens(raw_kb)
            kb_budget = min(kb_tokens, _KB_BUDGET, remaining_budget)

            if kb_budget > 0:
                kb_text = truncate_to_tokens(raw_kb, kb_budget)
                remaining_budget -= count_tokens(kb_text)
                for c in kb_chunks:
                    if c.source_id not in sources_used:
                        sources_used.append(c.source_id)

        # ── 4. Similar incidents with resolution focus ──────────────────
        similar_parts: list[str] = ["=== SIMILAR HISTORICAL INCIDENTS ==="]
        similar_budget = min(_SIMILAR_INCIDENTS_BUDGET, remaining_budget)
        similar_tokens_used = count_tokens(similar_parts[0])

        for idx, incident in enumerate(similar_incidents, 1):
            formatted = self.format_similar_incident(incident, idx)
            formatted_tokens = count_tokens(formatted)

            if similar_tokens_used + formatted_tokens > similar_budget:
                break

            similar_parts.append(formatted)
            similar_tokens_used += formatted_tokens
            if incident.number not in sources_used:
                sources_used.append(incident.number)

        # ── 5. Inject resolution chunks into similar incidents section ──
        for chunk in resolution_chunks:
            chunk_text_formatted = f"\n[Resolution from {chunk.source_id}]:\n{chunk.chunk_text}"
            chunk_tokens = count_tokens(chunk_text_formatted)

            if similar_tokens_used + chunk_tokens > similar_budget:
                # Try truncating
                available = similar_budget - similar_tokens_used
                if available > 50:
                    chunk_text_formatted = truncate_to_tokens(chunk_text_formatted, available)
                    similar_parts.append(chunk_text_formatted)
                    similar_tokens_used += available
                break

            similar_parts.append(chunk_text_formatted)
            similar_tokens_used += chunk_tokens
            if chunk.source_id not in sources_used:
                sources_used.append(chunk.source_id)

        similar_text = "\n".join(similar_parts) if len(similar_parts) > 1 else ""
        remaining_budget -= similar_tokens_used if similar_text else 0

        # ── 6. Fill remaining budget greedily with other chunks ─────────
        extra_parts: list[str] = []
        for chunk in other_chunks:
            if remaining_budget <= 50:
                break

            chunk_formatted = f"[{chunk.chunk_type.value.upper()} from {chunk.source_id}]:\n{chunk.chunk_text}"
            chunk_tokens = count_tokens(chunk_formatted)

            if chunk_tokens > remaining_budget:
                chunk_formatted = truncate_to_tokens(chunk_formatted, remaining_budget)
                chunk_tokens = remaining_budget

            extra_parts.append(chunk_formatted)
            remaining_budget -= chunk_tokens
            if chunk.source_id not in sources_used:
                sources_used.append(chunk.source_id)

        # Append extra chunks to similar incidents section
        if extra_parts and similar_text:
            similar_text += "\n\n" + "\n\n".join(extra_parts)
        elif extra_parts:
            similar_text = "\n\n".join(extra_parts)

        # ── Build final context ─────────────────────────────────────────
        total_tokens = (
            count_tokens(incident_text)
            + count_tokens(similar_text)
            + count_tokens(kb_text)
        )

        context = AssembledContext(
            incident_summary=incident_text,
            similar_incidents_context=similar_text,
            kb_context=kb_text,
            total_tokens=total_tokens,
            sources_used=sources_used,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "context_assembled",
            total_tokens=total_tokens,
            max_tokens=self._max_tokens,
            sources_count=len(sources_used),
            latency_ms=elapsed_ms,
        )

        return context
