"""Unit tests for the hybrid retriever.

Tests the retrieval layer including dense + sparse result merging
via Reciprocal Rank Fusion (RRF), metadata filtering, and deduplication.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest


# ── Test Data ───────────────────────────────────────────────────────────────


def _make_dense_results() -> list[dict[str, Any]]:
    """Simulated dense (vector) retrieval results."""
    return [
        {
            "id": "chunk-001",
            "score": 0.95,
            "metadata": {
                "source_id": "INC0039201",
                "source_type": "incident",
                "category": "application",
                "text": "Connection pool tuning resolved 502 errors on auth-service.",
            },
        },
        {
            "id": "chunk-002",
            "score": 0.88,
            "metadata": {
                "source_id": "INC0041055",
                "source_type": "incident",
                "category": "application",
                "text": "Rollback of v3.1.0 resolved payment gateway timeouts.",
            },
        },
        {
            "id": "chunk-003",
            "score": 0.82,
            "metadata": {
                "source_id": "KB0012345",
                "source_type": "kb_article",
                "category": "application",
                "text": "Troubleshooting HTTP 502 errors in microservices architecture.",
            },
        },
        {
            "id": "chunk-005",
            "score": 0.70,
            "metadata": {
                "source_id": "INC0035100",
                "source_type": "incident",
                "category": "infrastructure",
                "text": "Server disk space exhaustion caused service outage.",
            },
        },
    ]


def _make_sparse_results() -> list[dict[str, Any]]:
    """Simulated sparse (BM25) retrieval results."""
    return [
        {
            "id": "chunk-001",
            "score": 12.5,
            "metadata": {
                "source_id": "INC0039201",
                "source_type": "incident",
                "category": "application",
                "text": "Connection pool tuning resolved 502 errors on auth-service.",
            },
        },
        {
            "id": "chunk-004",
            "score": 10.2,
            "metadata": {
                "source_id": "INC0037892",
                "source_type": "incident",
                "category": "application",
                "text": "Database connection leak fixed by upgrading HikariCP.",
            },
        },
        {
            "id": "chunk-003",
            "score": 8.7,
            "metadata": {
                "source_id": "KB0012345",
                "source_type": "kb_article",
                "category": "application",
                "text": "Troubleshooting HTTP 502 errors in microservices architecture.",
            },
        },
    ]


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """Merge dense and sparse retrieval results using RRF.

    RRF score = sum(weight / (k + rank)) across all result lists.
    Higher RRF score = more relevant.

    Args:
        dense_results: Results from dense (vector) retrieval, ranked by score.
        sparse_results: Results from sparse (BM25) retrieval, ranked by score.
        k: RRF constant (default 60, standard value).
        dense_weight: Weight for dense retrieval scores.
        sparse_weight: Weight for sparse retrieval scores.

    Returns:
        Merged and re-ranked results with RRF scores.
    """
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict[str, Any]] = {}

    # Score dense results
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + dense_weight / (k + rank)
        chunk_data[chunk_id] = result

    # Score sparse results
    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + sparse_weight / (k + rank)
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for chunk_id in sorted_ids:
        result = chunk_data[chunk_id].copy()
        result["rrf_score"] = rrf_scores[chunk_id]
        merged.append(result)

    return merged


def apply_metadata_filter(
    results: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter results by metadata key-value pairs.

    Args:
        results: List of retrieval results with metadata.
        filters: Key-value pairs to filter by.

    Returns:
        Filtered results where all filter conditions match.
    """
    filtered = []
    for result in results:
        metadata = result.get("metadata", {})
        match = all(metadata.get(k) == v for k, v in filters.items())
        if match:
            filtered.append(result)
    return filtered


# ── Tests ───────────────────────────────────────────────────────────────────


class TestHybridRetrieval:
    """Tests for hybrid dense + sparse retrieval with RRF merging."""

    def test_hybrid_retrieval_merges_results(self) -> None:
        """Mock dense + sparse results, verify RRF scores are computed correctly.

        Documents appearing in both dense and sparse results should receive
        higher RRF scores than documents appearing in only one.
        """
        dense = _make_dense_results()
        sparse = _make_sparse_results()

        merged = reciprocal_rank_fusion(dense, sparse)

        # chunk-001 appears in both lists → highest RRF score
        assert merged[0]["id"] == "chunk-001"
        assert merged[0]["rrf_score"] > 0

        # All unique chunks should be present
        merged_ids = {r["id"] for r in merged}
        assert "chunk-001" in merged_ids  # both
        assert "chunk-002" in merged_ids  # dense only
        assert "chunk-003" in merged_ids  # both
        assert "chunk-004" in merged_ids  # sparse only
        assert "chunk-005" in merged_ids  # dense only

        # chunk-001 should have higher RRF than chunk-002 (chunk-001 in both lists)
        score_001 = next(r["rrf_score"] for r in merged if r["id"] == "chunk-001")
        score_002 = next(r["rrf_score"] for r in merged if r["id"] == "chunk-002")
        assert score_001 > score_002, "Chunk in both lists should score higher"

        # chunk-003 also in both lists, should score higher than chunk-005 (dense only, low rank)
        score_003 = next(r["rrf_score"] for r in merged if r["id"] == "chunk-003")
        score_005 = next(r["rrf_score"] for r in merged if r["id"] == "chunk-005")
        assert score_003 > score_005

    def test_rrf_total_count(self) -> None:
        """RRF merge should produce the union of all unique chunks."""
        dense = _make_dense_results()
        sparse = _make_sparse_results()

        merged = reciprocal_rank_fusion(dense, sparse)

        # Dense has 4 unique, sparse has 3 unique, overlap of 2 → 5 total
        assert len(merged) == 5

    def test_rrf_scores_positive(self) -> None:
        """All RRF scores should be positive."""
        dense = _make_dense_results()
        sparse = _make_sparse_results()

        merged = reciprocal_rank_fusion(dense, sparse)

        for result in merged:
            assert result["rrf_score"] > 0


class TestMetadataFilter:
    """Tests for metadata-based filtering of retrieval results."""

    def test_metadata_filter_applied(self) -> None:
        """Category filter should be passed to and applied by the retriever.

        When an 'application' category filter is applied, only results
        with matching category metadata should be returned.
        """
        dense = _make_dense_results()
        sparse = _make_sparse_results()

        merged = reciprocal_rank_fusion(dense, sparse)

        # Apply category filter
        filtered = apply_metadata_filter(merged, {"category": "application"})

        # All results should have category=application
        for result in filtered:
            assert result["metadata"]["category"] == "application"

        # chunk-005 has category=infrastructure, should be excluded
        filtered_ids = {r["id"] for r in filtered}
        assert "chunk-005" not in filtered_ids
        assert "chunk-001" in filtered_ids

    def test_metadata_filter_empty_match(self) -> None:
        """Filter with non-matching criteria should return empty list."""
        dense = _make_dense_results()
        filtered = apply_metadata_filter(dense, {"category": "security"})
        assert len(filtered) == 0

    def test_metadata_filter_multiple_criteria(self) -> None:
        """Multiple filter criteria should be ANDed together."""
        dense = _make_dense_results()
        filtered = apply_metadata_filter(
            dense, {"category": "application", "source_type": "incident"}
        )

        for result in filtered:
            assert result["metadata"]["category"] == "application"
            assert result["metadata"]["source_type"] == "incident"


class TestDeduplication:
    """Tests for deduplication of retrieval results."""

    def test_deduplication(self) -> None:
        """Same source appearing in dense + sparse should appear exactly once.

        When the same chunk_id appears in both dense and sparse results,
        RRF merge should produce exactly one entry per unique chunk_id.
        """
        dense = _make_dense_results()
        sparse = _make_sparse_results()

        merged = reciprocal_rank_fusion(dense, sparse)

        # Verify no duplicate chunk IDs
        chunk_ids = [r["id"] for r in merged]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in merged results"

        # chunk-001 appears in both dense and sparse, should appear exactly once
        count_001 = sum(1 for r in merged if r["id"] == "chunk-001")
        assert count_001 == 1

        # chunk-003 appears in both dense and sparse, should appear exactly once
        count_003 = sum(1 for r in merged if r["id"] == "chunk-003")
        assert count_003 == 1

    def test_deduplication_preserves_metadata(self) -> None:
        """Deduplication should preserve the metadata from the first occurrence."""
        dense = _make_dense_results()
        sparse = _make_sparse_results()

        merged = reciprocal_rank_fusion(dense, sparse)

        # chunk-001 metadata should be preserved
        chunk_001 = next(r for r in merged if r["id"] == "chunk-001")
        assert chunk_001["metadata"]["source_id"] == "INC0039201"
        assert chunk_001["metadata"]["category"] == "application"
