"""Main RAG orchestration pipeline.

Coordinates the full incident analysis flow:
  load → preprocess → retrieve → rerank → assemble → generate → parse → store → audit

Also provides conversational follow-up (chat) and resolution draft generation.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import structlog
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.context_assembler import AssembledContext, ContextAssembler
from app.core.embedder import Embedder
from app.core.llm_client import LLMClient, LLMError
from app.core.reranker import Reranker
from app.core.retriever import (
    HybridRetriever,
    RetrievalFilters,
    RetrievalQuery,
    RetrievedChunk,
)
from app.models.chat import ChatMessage, LLMPrompt, LLMResponse
from app.models.incident import ProcessedTicket, SourceType
from app.models.recommendation import (
    KBReference,
    RecommendationResult,
    SimilarIncident,
    SourceReference,
    TriageStep,
)

logger = structlog.get_logger(__name__)

# ── Prompt Templates ─────────────────────────────────────────────────────────

RECOMMENDATION_PROMPT_TEMPLATE = """You are an expert L2 Support Engineer AI assistant specializing in IT incident resolution.

Your task is to analyze the current incident using the provided context (similar historical incidents and knowledge base articles) and produce a structured recommendation.

INSTRUCTIONS:
1. Analyze the incident description and context carefully.
2. Identify the most likely root cause based on similar past incidents and KB articles.
3. Provide actionable triage steps with rationale and commands where applicable.
4. Draft a resolution note suitable for ServiceNow work notes.
5. Determine if L3 escalation is needed (escalate for: security incidents, data corruption, infrastructure failures requiring vendor involvement, or issues beyond L2 scope).
6. Assign a confidence score between 0.0 and 1.0 based on how well the context matches.

You MUST respond with valid JSON in exactly this format:
{{
    "root_cause_prediction": "2-3 sentence explanation of the likely root cause",
    "confidence_score": 0.85,
    "triage_steps": [
        {{
            "step": 1,
            "action": "Description of the action to take",
            "rationale": "Why this step is recommended",
            "command": "optional CLI command or null"
        }}
    ],
    "resolution_draft": "Draft resolution note for ServiceNow work notes",
    "escalate_to_l3": false,
    "escalation_reason": null,
    "sources_used": ["INC0039201", "KB0012345"]
}}

IMPORTANT:
- Be specific and actionable in your recommendations.
- Reference specific incident numbers and KB articles from the context.
- Include CLI commands or scripts when they would help the engineer.
- Set escalate_to_l3 to true ONLY when the issue genuinely requires L3.
- The confidence_score should reflect how much supporting evidence exists in the context.
"""

CHAT_PROMPT_TEMPLATE = """You are an expert L2 Support Engineer AI assistant. You are in a follow-up conversation about an incident that you previously analyzed.

PREVIOUS RECOMMENDATION:
{recommendation_summary}

CONVERSATION HISTORY:
{conversation_history}

INSTRUCTIONS:
- Answer the engineer's follow-up question based on your previous analysis.
- If the question is about a specific triage step, provide more detail.
- If asked to clarify, reference the original context and similar incidents.
- Be concise but thorough.
- If you don't have enough context to answer, say so clearly.
"""

RESOLUTION_DRAFT_TEMPLATE = """Based on the following incident analysis, write a concise resolution note suitable for ServiceNow work notes:

Incident: {incident_number}
Root Cause: {root_cause}
Steps Taken: {steps_summary}

Write a professional resolution note in 3-5 sentences. Start with "Resolution:" and include:
1. What was found (root cause)
2. What was done (key actions)
3. Current status and any follow-up needed
"""

# ── RAG Pipeline ─────────────────────────────────────────────────────────────


class RAGPipeline:
    """Main RAG orchestration engine for incident analysis.

    Coordinates the full pipeline from incident ingestion through
    retrieval, context assembly, LLM generation, and result storage.

    Args:
        embedder: Embedding model wrapper.
        retriever: Hybrid BM25 + dense retriever.
        reranker: Cross-encoder reranker.
        context_assembler: Token-aware context builder.
        llm_client: Multi-provider LLM client.
        db_session_factory: Async database session factory.
    """

    def __init__(
        self,
        embedder: Embedder,
        retriever: HybridRetriever,
        reranker: Reranker,
        context_assembler: ContextAssembler,
        llm_client: LLMClient,
        db_session_factory: object | None = None,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker
        self._context_assembler = context_assembler
        self._llm = llm_client
        self._db_session_factory = db_session_factory

        logger.info("rag_pipeline_initialised")

    # ── JSON Parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_llm_json(content: str) -> dict:
        """Parse JSON from LLM output, handling markdown code blocks.

        The LLM may wrap JSON in ```json ... ``` blocks, so we strip
        those before parsing.

        Args:
            content: Raw LLM output string.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If the output cannot be parsed as JSON.
        """
        # Strip markdown code fences
        cleaned = content.strip()
        if cleaned.startswith("```"):
            # Remove first line (```json) and last line (```)
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from the text
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse LLM output as JSON: {cleaned[:200]}...")

    def _build_recommendation(
        self,
        parsed: dict,
        ticket: ProcessedTicket,
        context: AssembledContext,
        similar_incidents: list[SimilarIncident],
        kb_chunks: list[RetrievedChunk],
        retrieval_latency_ms: int,
        generation_latency_ms: int,
    ) -> RecommendationResult:
        """Build a RecommendationResult from parsed LLM JSON output.

        Args:
            parsed: Parsed JSON dictionary from LLM.
            ticket: The processed incident ticket.
            context: The assembled context.
            similar_incidents: List of similar incidents found.
            kb_chunks: KB article chunks used.
            retrieval_latency_ms: Retrieval phase latency.
            generation_latency_ms: LLM generation latency.

        Returns:
            Complete RecommendationResult.
        """
        # Parse triage steps
        triage_steps: list[TriageStep] = []
        for step_data in parsed.get("triage_steps", []):
            triage_steps.append(
                TriageStep(
                    step=step_data.get("step", len(triage_steps) + 1),
                    action=step_data.get("action", ""),
                    rationale=step_data.get("rationale", ""),
                    command=step_data.get("command"),
                )
            )

        # Build KB references from chunks
        kb_refs: list[KBReference] = []
        seen_kb: set[str] = set()
        for chunk in kb_chunks:
            if chunk.source_id not in seen_kb:
                seen_kb.add(chunk.source_id)
                kb_refs.append(
                    KBReference(
                        kb_number=chunk.source_id,
                        title=chunk.metadata.get("title", ""),
                        relevance_score=min(chunk.score, 1.0),
                    )
                )

        return RecommendationResult(
            snow_sys_id=ticket.sys_id,
            root_cause_prediction=parsed.get("root_cause_prediction", "Unable to determine root cause."),
            confidence_score=max(0.0, min(1.0, parsed.get("confidence_score", 0.5))),
            triage_steps=triage_steps,
            resolution_draft=parsed.get("resolution_draft", ""),
            escalate_to_l3=parsed.get("escalate_to_l3", False),
            escalation_reason=parsed.get("escalation_reason"),
            similar_incidents=similar_incidents,
            kb_references=kb_refs,
            sources_used=parsed.get("sources_used", context.sources_used),
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
        )

    # ── Storage Helpers ──────────────────────────────────────────────────

    async def _store_recommendation(
        self,
        recommendation: RecommendationResult,
        incident_db_id: uuid.UUID | None = None,
    ) -> None:
        """Persist a recommendation to the database.

        Args:
            recommendation: The recommendation to store.
            incident_db_id: Database UUID of the incident.
        """
        if self._db_session_factory is None:
            logger.debug("store_recommendation_skipped_no_db")
            return

        try:
            from app.storage.postgres import RecommendationDB, get_session_factory

            factory = self._db_session_factory or get_session_factory()
            async with factory() as session:  # type: ignore[operator]
                rec_db = RecommendationDB(
                    id=recommendation.id,
                    incident_id=incident_db_id or recommendation.incident_id,
                    root_cause_prediction=recommendation.root_cause_prediction,
                    confidence_score=recommendation.confidence_score,
                    triage_steps=[s.model_dump() for s in recommendation.triage_steps],
                    similar_incidents=[s.model_dump() for s in recommendation.similar_incidents],
                    kb_references=[r.model_dump() for r in recommendation.kb_references],
                    resolution_draft=recommendation.resolution_draft,
                    retrieval_latency_ms=recommendation.retrieval_latency_ms,
                    generation_latency_ms=recommendation.generation_latency_ms,
                )
                session.add(rec_db)
                await session.commit()

            logger.info(
                "recommendation_stored",
                recommendation_id=str(recommendation.id),
            )
        except Exception as exc:
            logger.error("recommendation_store_error", error=str(exc))

    async def _audit_event(
        self,
        event_type: str,
        resource_type: str,
        resource_id: str,
        payload: dict,
    ) -> None:
        """Record an audit event in the database.

        Args:
            event_type: Type of event (e.g., ``ticket_analyzed``).
            resource_type: Resource type (e.g., ``incident``).
            resource_id: Resource identifier.
            payload: Event-specific payload data.
        """
        if self._db_session_factory is None:
            logger.debug("audit_event_skipped_no_db")
            return

        try:
            from app.storage.postgres import AuditEventDB

            factory = self._db_session_factory
            async with factory() as session:  # type: ignore[operator]
                event = AuditEventDB(
                    event_type=event_type,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    payload=payload,
                )
                session.add(event)
                await session.commit()

            logger.debug("audit_event_recorded", event_type=event_type)
        except Exception as exc:
            logger.error("audit_event_error", error=str(exc))

    # ── Main Pipeline ────────────────────────────────────────────────────

    async def analyze_incident(
        self,
        ticket: ProcessedTicket,
        incident_db_id: uuid.UUID | None = None,
        filters: RetrievalFilters | None = None,
    ) -> RecommendationResult:
        """Run the full RAG pipeline on a preprocessed incident.

        Pipeline stages:
          1. **Retrieve**: Hybrid BM25 + dense retrieval with filters.
          2. **Rerank**: Cross-encoder/heuristic reranking.
          3. **Similar Incidents**: Find similar historical incidents.
          4. **Assemble**: Build token-aware context window.
          5. **Generate**: LLM produces structured JSON recommendation.
          6. **Parse**: Extract RecommendationResult from JSON.
          7. **Store**: Persist recommendation to database.
          8. **Audit**: Log the analysis event.

        Args:
            ticket: Preprocessed incident ticket.
            incident_db_id: Optional database UUID for the incident.
            filters: Optional retrieval filters to apply.

        Returns:
            RecommendationResult with root cause, triage steps, and metadata.
        """
        pipeline_start = time.monotonic()
        log = logger.bind(
            incident_number=ticket.source_id,
            sys_id=ticket.sys_id,
        )
        log.info("rag_pipeline_start")

        # ── 1. Build query text ──────────────────────────────────────────
        query_text = f"{ticket.cleaned_text}"
        if ticket.keywords:
            query_text += f" | Keywords: {', '.join(ticket.keywords[:5])}"

        retrieval_filters = filters or RetrievalFilters(
            categories=[ticket.category] if ticket.category else None,
            cmdb_cis=[ticket.cmdb_ci] if ticket.cmdb_ci else None,
        )

        retrieval_query = RetrievalQuery(
            query_text=query_text,
            filters=retrieval_filters,
            top_k=20,
        )

        # ── 2. Retrieve ─────────────────────────────────────────────────
        retrieval_start = time.monotonic()
        chunks = await self._retriever.retrieve(retrieval_query)
        retrieval_latency_ms = int((time.monotonic() - retrieval_start) * 1000)
        log.info("retrieval_complete", chunks=len(chunks), latency_ms=retrieval_latency_ms)

        # ── 3. Rerank ───────────────────────────────────────────────────
        chunks = await self._reranker.rerank(
            query=query_text,
            chunks=chunks,
            top_k=15,
        )

        # ── 4. Similar incidents ─────────────────────────────────────────
        similar_incidents = await self._retriever.get_similar_incidents(
            query_text=query_text,
            top_k=5,
        )

        # ── 5. Identify KB chunks for context ───────────────────────────
        kb_chunks = [
            c for c in chunks
            if c.source_type == SourceType.KB_ARTICLE
        ]

        # ── 6. Assemble context ─────────────────────────────────────────
        context = await self._context_assembler.assemble(
            ticket=ticket,
            chunks=chunks,
            similar_incidents=similar_incidents,
        )

        # ── 7. Generate ─────────────────────────────────────────────────
        prompt = LLMPrompt(
            system_prompt=RECOMMENDATION_PROMPT_TEMPLATE,
            user_message=context.full_context,
            temperature=0.1,
            max_tokens=2000,
            metadata={
                "incident_number": ticket.source_id,
                "pipeline_stage": "recommendation",
            },
        )

        generation_start = time.monotonic()
        try:
            llm_response = await self._llm.generate(prompt)
        except LLMError as exc:
            log.error("llm_generation_failed", error=str(exc))
            return self._fallback_recommendation(
                ticket=ticket,
                similar_incidents=similar_incidents,
                retrieval_latency_ms=retrieval_latency_ms,
                error=str(exc),
            )
        generation_latency_ms = int((time.monotonic() - generation_start) * 1000)

        # ── 8. Parse LLM output ─────────────────────────────────────────
        try:
            parsed = self._parse_llm_json(llm_response.content)
        except ValueError as exc:
            log.error("llm_json_parse_error", error=str(exc))
            return self._fallback_recommendation(
                ticket=ticket,
                similar_incidents=similar_incidents,
                retrieval_latency_ms=retrieval_latency_ms,
                error=f"JSON parse error: {exc}",
            )

        recommendation = self._build_recommendation(
            parsed=parsed,
            ticket=ticket,
            context=context,
            similar_incidents=similar_incidents,
            kb_chunks=kb_chunks,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
        )
        recommendation.incident_id = incident_db_id

        # ── 9. Store recommendation ──────────────────────────────────────
        await self._store_recommendation(recommendation, incident_db_id)

        # ── 10. Audit ────────────────────────────────────────────────────
        total_latency_ms = int((time.monotonic() - pipeline_start) * 1000)
        await self._audit_event(
            event_type="ticket_analyzed",
            resource_type="incident",
            resource_id=ticket.source_id,
            payload={
                "recommendation_id": str(recommendation.id),
                "confidence_score": recommendation.confidence_score,
                "escalate_to_l3": recommendation.escalate_to_l3,
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "total_latency_ms": total_latency_ms,
                "sources_used": recommendation.sources_used,
            },
        )

        log.info(
            "rag_pipeline_complete",
            confidence=recommendation.confidence_score,
            escalate=recommendation.escalate_to_l3,
            triage_steps=len(recommendation.triage_steps),
            total_latency_ms=total_latency_ms,
        )

        return recommendation

    # ── Fallback Recommendation ──────────────────────────────────────────

    @staticmethod
    def _fallback_recommendation(
        ticket: ProcessedTicket,
        similar_incidents: list[SimilarIncident],
        retrieval_latency_ms: int,
        error: str,
    ) -> RecommendationResult:
        """Create a fallback recommendation when LLM generation fails.

        Provides basic guidance based on similar incidents and suggests
        L3 escalation.

        Args:
            ticket: The processed incident ticket.
            similar_incidents: Any similar incidents found.
            retrieval_latency_ms: Retrieval latency for metrics.
            error: Description of the error that occurred.

        Returns:
            A low-confidence fallback RecommendationResult.
        """
        resolution_hint = ""
        if similar_incidents:
            top = similar_incidents[0]
            resolution_hint = (
                f" A similar incident ({top.number}) was resolved with: "
                f"{top.resolution_summary}"
            )

        return RecommendationResult(
            snow_sys_id=ticket.sys_id,
            root_cause_prediction=(
                f"Automated analysis encountered an error ({error}). "
                f"Manual investigation required.{resolution_hint}"
            ),
            confidence_score=0.1,
            triage_steps=[
                TriageStep(
                    step=1,
                    action="Review the incident description and work notes manually.",
                    rationale="Automated analysis was unable to complete.",
                ),
                TriageStep(
                    step=2,
                    action="Check similar incidents listed below for resolution guidance.",
                    rationale="Historical patterns may provide resolution hints.",
                ),
            ],
            resolution_draft="",
            escalate_to_l3=True,
            escalation_reason=f"Automated analysis failed: {error}",
            similar_incidents=similar_incidents,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=0,
        )

    # ── Chat / Follow-up ─────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        recommendation: RecommendationResult,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """Handle a conversational follow-up about a previous recommendation.

        Loads the existing recommendation context and conversation history,
        then generates a response.

        Args:
            message: The engineer's follow-up message.
            recommendation: The existing recommendation to reference.
            history: Optional list of previous chat messages.

        Returns:
            The AI assistant's response text.
        """
        log = logger.bind(
            recommendation_id=str(recommendation.id),
            snow_sys_id=recommendation.snow_sys_id,
        )
        log.info("chat_follow_up_start")

        # Format recommendation summary
        recommendation_summary = (
            f"Root Cause: {recommendation.root_cause_prediction}\n"
            f"Confidence: {recommendation.confidence_score:.0%}\n"
            f"Escalate to L3: {recommendation.escalate_to_l3}\n"
        )
        if recommendation.triage_steps:
            steps_text = "\n".join(
                f"  Step {s.step}: {s.action}" for s in recommendation.triage_steps
            )
            recommendation_summary += f"Triage Steps:\n{steps_text}\n"
        if recommendation.resolution_draft:
            recommendation_summary += f"Resolution Draft: {recommendation.resolution_draft}\n"

        # Format conversation history
        conversation_history = ""
        if history:
            for msg in history:
                role_label = "Engineer" if msg.role == "user" else "AI Assistant"
                conversation_history += f"{role_label}: {msg.content}\n"

        # Build prompt
        system_prompt = CHAT_PROMPT_TEMPLATE.format(
            recommendation_summary=recommendation_summary,
            conversation_history=conversation_history,
        )

        prompt = LLMPrompt(
            system_prompt=system_prompt,
            user_message=message,
            temperature=0.3,
            max_tokens=1000,
            metadata={
                "recommendation_id": str(recommendation.id),
                "pipeline_stage": "chat",
            },
        )

        try:
            response = await self._llm.generate(prompt)
            log.info("chat_follow_up_complete", latency_ms=response.latency_ms)
            return response.content
        except LLMError as exc:
            log.error("chat_follow_up_error", error=str(exc))
            return (
                "I apologize, but I'm unable to process your follow-up question "
                "at the moment. Please try again or refer to the original "
                "recommendation for guidance."
            )

    # ── Resolution Draft Generation ──────────────────────────────────────

    async def generate_resolution_draft(
        self,
        incident_number: str,
        root_cause: str,
        steps_summary: str,
    ) -> str:
        """Generate a formatted resolution note from a brief summary.

        Produces a professional resolution note suitable for ServiceNow
        work notes / resolution fields.

        Args:
            incident_number: The incident number (e.g., ``INC0042871``).
            root_cause: Brief root cause description.
            steps_summary: Summary of steps taken.

        Returns:
            Formatted resolution note text.
        """
        log = logger.bind(incident_number=incident_number)
        log.info("resolution_draft_start")

        prompt = LLMPrompt(
            system_prompt=(
                "You are an expert IT support engineer. Write professional "
                "ServiceNow resolution notes."
            ),
            user_message=RESOLUTION_DRAFT_TEMPLATE.format(
                incident_number=incident_number,
                root_cause=root_cause,
                steps_summary=steps_summary,
            ),
            temperature=0.2,
            max_tokens=500,
            metadata={
                "incident_number": incident_number,
                "pipeline_stage": "resolution_draft",
            },
        )

        try:
            response = await self._llm.generate(prompt)
            log.info("resolution_draft_complete", latency_ms=response.latency_ms)
            return response.content
        except LLMError as exc:
            log.error("resolution_draft_error", error=str(exc))
            return (
                f"Resolution: Root cause identified as: {root_cause}. "
                f"Actions taken: {steps_summary}. "
                f"Incident resolved per standard procedures."
            )
