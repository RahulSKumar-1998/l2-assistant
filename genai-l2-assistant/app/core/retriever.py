"""Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion.

Combines sparse (BM25) and dense (vector) retrieval for improved recall,
with configurable filters, deduplication, and similar-incident lookup.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from datetime import datetime
from enum import Enum
from typing import Optional

import structlog
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.core.embedder import Embedder
from app.models.incident import ChunkType, SourceType

logger = structlog.get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_RRF_K: int = 60
_DEFAULT_TOP_K: int = 20
_SIMILARITY_THRESHOLD: float = 0.5


# ── Pydantic Models ─────────────────────────────────────────────────────────


class RetrievalFilters(BaseModel):
    """Filters applied during retrieval to narrow search scope.

    All filter fields are optional; only non-None values are applied.
    """
    source_types: Optional[list[SourceType]] = Field(
        default=None,
        description="Restrict to these source types (incident, kb_article, runbook)",
    )
    categories: Optional[list[str]] = Field(
        default=None,
        description="Restrict to these incident/KB categories",
    )
    cmdb_cis: Optional[list[str]] = Field(
        default=None,
        description="Restrict to these CMDB CI names",
    )
    min_resolution_date: Optional[datetime] = Field(
        default=None,
        description="Only include sources resolved after this date",
    )
    chunk_types: Optional[list[ChunkType]] = Field(
        default=None,
        description="Restrict to these chunk types (description, resolution, etc.)",
    )


class RetrievalQuery(BaseModel):
    """A retrieval query with text, optional embedding, and filters."""
    query_text: str = Field(..., description="Natural-language query text")
    query_embedding: Optional[list[float]] = Field(
        default=None,
        description="Pre-computed query embedding (computed if absent)",
    )
    filters: RetrievalFilters = Field(
        default_factory=RetrievalFilters,
        description="Retrieval filters",
    )
    top_k: int = Field(
        default=_DEFAULT_TOP_K,
        ge=1,
        le=100,
        description="Number of results to return",
    )


class RetrievedChunk(BaseModel):
    """A single chunk returned from retrieval with fusion score."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_text: str = Field(..., description="The text content of this chunk")
    chunk_type: ChunkType = Field(..., description="Type of content")
    source_id: str = Field(..., description="Source document ID")
    source_type: SourceType = Field(
        default=SourceType.INCIDENT,
        description="Type of source document",
    )
    score: float = Field(default=0.0, description="Fused relevance score")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    dense_rank: Optional[int] = Field(default=None, description="Rank in dense retrieval")
    sparse_rank: Optional[int] = Field(default=None, description="Rank in BM25 retrieval")


# ── Hybrid Retriever ─────────────────────────────────────────────────────────


class HybridRetriever:
    """Hybrid retriever combining BM25 sparse and dense vector retrieval.

    Uses Reciprocal Rank Fusion (RRF) to merge rankings from both
    retrievers, with configurable filters and deduplication.

    Args:
        embedder: Embedder instance for computing query embeddings.
        vector_store: Optional async vector store client (e.g. Pinecone, pgvector).
        redis_client: Optional async Redis client for BM25 index persistence.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: object | None = None,
        redis_client: object | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._redis = redis_client

        # BM25 index — built from stored corpus
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: list[dict] = []  # list of {chunk_id, chunk_text, ...}
        self._bm25_lock = asyncio.Lock()

        logger.info("hybrid_retriever_initialised")

    # ── BM25 Index Management ────────────────────────────────────────────

    async def rebuild_bm25_index(
        self,
        corpus: list[dict],
    ) -> int:
        """Rebuild the in-memory BM25 index from a list of chunk dicts.

        Each dict should have at minimum: ``chunk_id``, ``chunk_text``,
        ``chunk_type``, ``source_id``, ``source_type``.

        Intended to be called during application startup or by a nightly
        scheduled job.

        Args:
            corpus: List of chunk dictionaries to index.

        Returns:
            Number of documents indexed.
        """
        async with self._bm25_lock:
            self._bm25_corpus = corpus
            tokenised = [
                doc.get("chunk_text", "").lower().split() for doc in corpus
            ]
            self._bm25 = BM25Okapi(tokenised) if tokenised else None

            logger.info(
                "bm25_index_rebuilt",
                corpus_size=len(corpus),
            )
            return len(corpus)

    # ── Dense retrieval ──────────────────────────────────────────────────

    async def _dense_retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks via vector similarity search.

        Falls back to an empty list if no vector store is configured.

        Args:
            query: The retrieval query with (optionally pre-computed) embedding.

        Returns:
            List of retrieved chunks ordered by dense similarity.
        """
        if self._vector_store is None:
            logger.debug("dense_retrieve_skipped_no_vector_store")
            return []

        # Compute embedding if not provided
        embedding = query.query_embedding
        if embedding is None:
            embedding = await self._embedder.embed_text(query.query_text)

        try:
            # Build metadata filter dict for vector store query
            filter_dict = self._build_vector_filter(query.filters)

            # Call vector store — interface adapts to Pinecone / pgvector
            results = await self._query_vector_store(
                embedding=embedding,
                top_k=query.top_k * 2,  # over-fetch for RRF fusion
                filter_dict=filter_dict,
            )
            return results
        except Exception as exc:
            logger.error("dense_retrieval_error", error=str(exc))
            return []

    def _build_vector_filter(self, filters: RetrievalFilters) -> dict:
        """Convert RetrievalFilters into a vector-store filter dict.

        Args:
            filters: The retrieval filters.

        Returns:
            Dictionary suitable for Pinecone/pgvector metadata filtering.
        """
        filter_dict: dict = {}
        if filters.source_types:
            filter_dict["source_type"] = {
                "$in": [st.value for st in filters.source_types]
            }
        if filters.categories:
            filter_dict["category"] = {"$in": filters.categories}
        if filters.cmdb_cis:
            filter_dict["cmdb_ci"] = {"$in": filters.cmdb_cis}
        if filters.chunk_types:
            filter_dict["chunk_type"] = {
                "$in": [ct.value for ct in filters.chunk_types]
            }
        if filters.min_resolution_date:
            filter_dict["resolved_at"] = {
                "$gte": filters.min_resolution_date.isoformat()
            }
        return filter_dict

    async def _query_vector_store(
        self,
        embedding: list[float],
        top_k: int,
        filter_dict: dict,
    ) -> list[RetrievedChunk]:
        """Execute vector similarity query against the configured store.

        Handles both Pinecone-style and generic dict-result interfaces.

        Args:
            embedding: Query embedding vector.
            top_k: Number of results to request.
            filter_dict: Metadata filter dictionary.

        Returns:
            List of retrieved chunks with dense scores.
        """
        results: list[RetrievedChunk] = []

        try:
            query_method = getattr(self._vector_store, "query", None)
            if query_method is None:
                return []

            # Support the project's async VectorStore abstraction.
            if inspect.iscoroutinefunction(query_method):
                response = await query_method(
                    vector=embedding,
                    top_k=top_k,
                    filter_metadata=filter_dict or None,
                    include_metadata=True,
                )

                for match in response:
                    metadata = getattr(match, "metadata", {}) or {}
                    chunk = RetrievedChunk(
                        chunk_id=getattr(match, "id", ""),
                        chunk_text=metadata.get("chunk_text") or metadata.get("text") or "",
                        chunk_type=ChunkType(metadata.get("chunk_type", "description")),
                        source_id=metadata.get("source_id", ""),
                        source_type=SourceType(metadata.get("source_type", "incident")),
                        score=float(getattr(match, "score", 0.0)),
                        metadata=metadata,
                    )
                    results.append(chunk)
                return results

            # Fallback for older synchronous Pinecone-style clients.
            query_kwargs: dict = {
                "vector": embedding,
                "top_k": top_k,
                "include_metadata": True,
            }
            if filter_dict:
                query_kwargs["filter"] = filter_dict

            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: query_method(**query_kwargs),
            )

            for match in response.get("matches", []):
                metadata = match.get("metadata", {})
                chunk = RetrievedChunk(
                    chunk_id=match.get("id", ""),
                    chunk_text=metadata.get("chunk_text") or metadata.get("text") or "",
                    chunk_type=ChunkType(metadata.get("chunk_type", "description")),
                    source_id=metadata.get("source_id", ""),
                    source_type=SourceType(metadata.get("source_type", "incident")),
                    score=match.get("score", 0.0),
                    metadata=metadata,
                )
                results.append(chunk)

        except Exception as exc:
            logger.error("vector_store_query_error", error=str(exc))

        return results

    # ── Sparse (BM25) retrieval ──────────────────────────────────────────

    async def _sparse_retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using BM25 sparse keyword matching.

        Args:
            query: The retrieval query.

        Returns:
            List of retrieved chunks ordered by BM25 score.
        """
        if self._bm25 is None or not self._bm25_corpus:
            logger.debug("bm25_retrieve_skipped_no_index")
            return []

        tokenised_query = query.query_text.lower().split()
        scores = self._bm25.get_scores(tokenised_query)

        # Pair scores with corpus docs and sort descending
        scored_docs = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for idx, bm25_score in scored_docs[: query.top_k * 2]:
            if bm25_score <= 0:
                continue

            doc = self._bm25_corpus[idx]

            # Apply filters
            if not self._passes_filters(doc, query.filters):
                continue

            chunk = RetrievedChunk(
                chunk_id=doc.get("chunk_id", f"bm25_{idx}"),
                chunk_text=doc.get("chunk_text", ""),
                chunk_type=ChunkType(doc.get("chunk_type", "description")),
                source_id=doc.get("source_id", ""),
                source_type=SourceType(doc.get("source_type", "incident")),
                score=float(bm25_score),
                metadata=doc.get("metadata", {}),
            )
            results.append(chunk)

        return results

    @staticmethod
    def _passes_filters(doc: dict, filters: RetrievalFilters) -> bool:
        """Check whether a BM25 document passes the retrieval filters.

        Args:
            doc: BM25 corpus document dict.
            filters: Retrieval filters to apply.

        Returns:
            True if the document passes all active filters.
        """
        if filters.source_types:
            doc_st = doc.get("source_type", "incident")
            if doc_st not in [st.value for st in filters.source_types]:
                return False

        if filters.categories:
            doc_cat = doc.get("metadata", {}).get("category", "")
            if doc_cat and doc_cat not in filters.categories:
                return False

        if filters.cmdb_cis:
            doc_ci = doc.get("metadata", {}).get("cmdb_ci", "")
            if doc_ci and doc_ci not in filters.cmdb_cis:
                return False

        if filters.chunk_types:
            doc_ct = doc.get("chunk_type", "description")
            if doc_ct not in [ct.value for ct in filters.chunk_types]:
                return False

        if filters.min_resolution_date:
            resolved_str = doc.get("metadata", {}).get("resolved_at")
            if resolved_str:
                try:
                    resolved = datetime.fromisoformat(resolved_str)
                    if resolved < filters.min_resolution_date:
                        return False
                except (ValueError, TypeError):
                    pass

        return True

    # ── Reciprocal Rank Fusion ───────────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        k: int = _RRF_K,
        top_k: int = _DEFAULT_TOP_K,
    ) -> list[RetrievedChunk]:
        """Merge dense and sparse results using Reciprocal Rank Fusion.

        RRF score = Σ 1 / (k + rank_i) for each ranking list.

        Deduplicates by ``(source_id, chunk_type)`` keeping the highest
        scoring variant.

        Args:
            dense_results: Results from dense vector retrieval.
            sparse_results: Results from BM25 sparse retrieval.
            k: RRF smoothing constant (default: 60).
            top_k: Maximum number of results to return.

        Returns:
            Fused and deduplicated list of chunks sorted by RRF score.
        """
        # Map chunk_id -> (chunk, rrf_score, dense_rank, sparse_rank)
        chunk_map: dict[str, dict] = {}

        # Score dense results
        for rank, chunk in enumerate(dense_results):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk": chunk,
                    "score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                }
            chunk_map[cid]["score"] += 1.0 / (k + rank + 1)
            chunk_map[cid]["dense_rank"] = rank + 1

        # Score sparse results
        for rank, chunk in enumerate(sparse_results):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk": chunk,
                    "score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                }
            chunk_map[cid]["score"] += 1.0 / (k + rank + 1)
            chunk_map[cid]["sparse_rank"] = rank + 1

        # Deduplicate by (source_id, chunk_type) — keep highest score
        dedup_map: dict[str, dict] = {}
        for entry in chunk_map.values():
            chunk = entry["chunk"]
            dedup_key = f"{chunk.source_id}::{chunk.chunk_type.value}"
            if dedup_key not in dedup_map or entry["score"] > dedup_map[dedup_key]["score"]:
                dedup_map[dedup_key] = entry

        # Sort by RRF score descending
        sorted_entries = sorted(
            dedup_map.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for entry in sorted_entries[:top_k]:
            chunk = entry["chunk"]
            chunk.score = entry["score"]
            chunk.dense_rank = entry["dense_rank"]
            chunk.sparse_rank = entry["sparse_rank"]
            results.append(chunk)

        return results

    # ── Public API ───────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievedChunk]:
        """Execute hybrid retrieval: dense + BM25 with RRF fusion.

        Runs dense and sparse retrieval in parallel, fuses results via
        Reciprocal Rank Fusion, and returns deduplicated top-k chunks.

        Args:
            query: Retrieval query with text, optional embedding, and filters.

        Returns:
            List of retrieved chunks sorted by RRF score.
        """
        start = time.monotonic()

        # Ensure embedding is computed
        if query.query_embedding is None:
            query.query_embedding = await self._embedder.embed_text(query.query_text)

        # Run dense and sparse in parallel
        dense_results, sparse_results = await asyncio.gather(
            self._dense_retrieve(query),
            self._sparse_retrieve(query),
        )

        # Fuse results with RRF
        fused = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=_RRF_K,
            top_k=query.top_k,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "hybrid_retrieval_complete",
            query_length=len(query.query_text),
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            fused_count=len(fused),
            latency_ms=elapsed_ms,
        )

        return fused

    async def get_similar_incidents(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = _SIMILARITY_THRESHOLD,
    ) -> list["SimilarIncidentResult"]:
        """Find incidents similar to the given text.

        Retrieves only incident-type chunks and maps them to
        ``SimilarIncidentResult`` objects with similarity scores.

        Args:
            query_text: The incident text to find matches for.
            top_k: Maximum number of similar incidents.
            min_score: Minimum RRF score threshold.

        Returns:
            List of similar incident results.
        """
        from app.models.recommendation import SimilarIncident

        retrieval_query = RetrievalQuery(
            query_text=query_text,
            filters=RetrievalFilters(
                source_types=[SourceType.INCIDENT],
                chunk_types=[ChunkType.RESOLUTION, ChunkType.DESCRIPTION],
            ),
            top_k=top_k * 3,  # over-fetch to allow filtering
        )

        chunks = await self.retrieve(retrieval_query)

        # Group by source_id and pick the best-scoring chunk per incident
        incident_map: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            if chunk.source_id not in incident_map:
                incident_map[chunk.source_id] = chunk
            elif chunk.score > incident_map[chunk.source_id].score:
                incident_map[chunk.source_id] = chunk

        results: list[SimilarIncident] = []
        for source_id, chunk in sorted(
            incident_map.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )[:top_k]:
            # Normalise RRF score to 0-1 range for similarity_score field
            max_possible_rrf = 2.0 / (_RRF_K + 1)  # rank 1 in both
            normalised_score = min(chunk.score / max_possible_rrf, 1.0)

            if normalised_score < min_score:
                continue

            metadata = chunk.metadata
            results.append(
                SimilarIncident(
                    number=source_id,
                    sys_id=metadata.get("sys_id", ""),
                    similarity_score=round(normalised_score, 4),
                    resolution_summary=metadata.get("resolution_notes", ""),
                    resolution_time_min=metadata.get("resolution_time_min"),
                    category=metadata.get("category", ""),
                )
            )

        logger.info(
            "similar_incidents_found",
            query_length=len(query_text),
            count=len(results),
        )
        return results


# ── Helper type for similar incident results ─────────────────────────────────
# Re-export SimilarIncident from models for convenience
SimilarIncidentResult = "SimilarIncident"
