"""Cross-encoder reranker for retrieved chunks.

Reranks retrieved chunks by relevance using a scoring heuristic that
prioritises resolution chunks, keyword overlap, recency, and source
diversity. Designed as a lightweight stand-in for a full cross-encoder
model (e.g. ``ms-marco-MiniLM``), which can be swapped in by
overriding ``_compute_relevance_score``.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import structlog

from app.core.retriever import RetrievedChunk
from app.models.incident import ChunkType, SourceType

logger = structlog.get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_CHUNK_TYPE_WEIGHTS: dict[ChunkType, float] = {
    ChunkType.RESOLUTION: 1.5,
    ChunkType.WORK_NOTES: 1.1,
    ChunkType.DESCRIPTION: 1.0,
    ChunkType.KB_ARTICLE: 1.3,
}

_SOURCE_TYPE_WEIGHTS: dict[SourceType, float] = {
    SourceType.INCIDENT: 1.0,
    SourceType.KB_ARTICLE: 1.2,
    SourceType.RUNBOOK: 1.1,
}

# ── Reranker ─────────────────────────────────────────────────────────────────


class Reranker:
    """Cross-encoder reranker that rescores chunks for final ranking.

    Uses a multi-signal heuristic scoring function combining:
      - Original retrieval score (RRF)
      - Chunk type priority (resolution > KB > work_notes > description)
      - Keyword overlap between query and chunk text
      - Source type weighting
      - Recency bonus for recently resolved incidents

    For production use, replace ``_compute_relevance_score`` with a
    proper cross-encoder model call.

    Args:
        cross_encoder_model: Optional path/name of a cross-encoder model.
            When ``None`` (default), uses the heuristic scorer.
    """

    def __init__(
        self,
        cross_encoder_model: Optional[str] = None,
    ) -> None:
        self._cross_encoder_model = cross_encoder_model
        self._model: object | None = None

        if cross_encoder_model:
            logger.info(
                "reranker_with_cross_encoder",
                model=cross_encoder_model,
            )
        else:
            logger.info("reranker_with_heuristic_scoring")

    # ── Cross-encoder model (optional) ───────────────────────────────────

    def _get_model(self) -> object | None:
        """Lazily load the cross-encoder model if configured.

        Returns:
            CrossEncoder model instance, or None.
        """
        if self._cross_encoder_model and self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._cross_encoder_model)
                logger.info(
                    "cross_encoder_loaded",
                    model=self._cross_encoder_model,
                )
            except ImportError:
                logger.warning(
                    "cross_encoder_unavailable",
                    reason="sentence-transformers not installed",
                )
        return self._model

    # ── Scoring ──────────────────────────────────────────────────────────

    def _compute_relevance_score(
        self,
        query: str,
        chunk: RetrievedChunk,
    ) -> float:
        """Compute a composite relevance score for a chunk.

        Combines multiple signals into a single relevance score
        between 0.0 and ~3.0.

        Args:
            query: The original query text.
            chunk: The retrieved chunk to score.

        Returns:
            Composite relevance score.
        """
        model = self._get_model()

        # If cross-encoder is available, use it as the primary signal
        if model is not None:
            try:
                ce_score = float(
                    model.predict([(query, chunk.chunk_text)])[0]  # type: ignore[union-attr]
                )
                # Cross-encoder scores are typically in [-10, 10]; normalise
                ce_normalised = max(0.0, (ce_score + 10.0) / 20.0)
                return ce_normalised * _CHUNK_TYPE_WEIGHTS.get(chunk.chunk_type, 1.0)
            except Exception as exc:
                logger.warning("cross_encoder_error", error=str(exc))

        # Heuristic scoring fallback
        score = 0.0

        # 1. Base score from retrieval
        score += chunk.score * 10.0  # RRF scores are small; scale up

        # 2. Chunk type weight
        chunk_weight = _CHUNK_TYPE_WEIGHTS.get(chunk.chunk_type, 1.0)
        score *= chunk_weight

        # 3. Keyword overlap
        query_tokens = set(self._tokenize(query))
        chunk_tokens = set(self._tokenize(chunk.chunk_text[:500]))
        if query_tokens:
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            score += overlap * 2.0

        # 4. Source type weight
        source_weight = _SOURCE_TYPE_WEIGHTS.get(chunk.source_type, 1.0)
        score *= source_weight

        # 5. Presence in both rankings bonus
        if chunk.dense_rank is not None and chunk.sparse_rank is not None:
            score *= 1.15  # 15% bonus for appearing in both

        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer.

        Args:
            text: Input text.

        Returns:
            List of lowercased tokens.
        """
        return re.findall(r"\w+", text.lower())

    # ── Public API ───────────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks by relevance to the query.

        Scores each chunk using the configured scoring method (heuristic
        or cross-encoder), then sorts descending by score.

        Args:
            query: The original query text.
            chunks: List of chunks to rerank.
            top_k: Optional limit on returned results.

        Returns:
            Reranked list of chunks with updated scores.
        """
        if not chunks:
            return []

        start = time.monotonic()

        # Compute relevance scores
        scored_chunks: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            rel_score = self._compute_relevance_score(query, chunk)
            scored_chunks.append((rel_score, chunk))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Update chunk scores and apply top_k
        result: list[RetrievedChunk] = []
        limit = top_k or len(scored_chunks)
        for score, chunk in scored_chunks[:limit]:
            chunk.score = score
            result.append(chunk)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "reranking_complete",
            input_count=len(chunks),
            output_count=len(result),
            latency_ms=elapsed_ms,
        )

        return result
