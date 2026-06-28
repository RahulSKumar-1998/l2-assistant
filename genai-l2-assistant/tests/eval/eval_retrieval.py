"""Retrieval evaluation suite for the RAG pipeline.

Computes IR metrics (Precision@k, Recall@k, MRR, NDCG@k) against
a ground-truth eval dataset. Supports per-category breakdown and
aggregate reporting.
"""

import json
import math
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Evaluation Data Models ──────────────────────────────────────────────────


class EvalCase(BaseModel):
    """A single retrieval evaluation case.

    Contains a query, its ground-truth relevant document IDs, and
    optional metadata for category-based breakdown.
    """
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="Natural language query text")
    relevant_doc_ids: list[str] = Field(
        ...,
        description="Ground-truth relevant document IDs",
    )
    category: str = Field(default="", description="Incident category for breakdown")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (priority, cmdb_ci, etc.)",
    )


class MetricResult(BaseModel):
    """Computed metric values for a single evaluation run."""
    precision_at_k: float = Field(default=0.0, description="Precision@k")
    recall_at_k: float = Field(default=0.0, description="Recall@k")
    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank")
    ndcg_at_k: float = Field(default=0.0, description="NDCG@k")
    k: int = Field(default=5, description="k value used for metrics")


class CategoryBreakdown(BaseModel):
    """Metrics broken down by incident category."""
    category: str = Field(..., description="Category name")
    num_queries: int = Field(default=0, description="Number of queries in category")
    metrics: MetricResult = Field(
        default_factory=MetricResult,
        description="Computed metrics for this category",
    )


class EvalReport(BaseModel):
    """Complete evaluation report with aggregate and per-category metrics."""
    eval_type: str = Field(default="retrieval", description="Evaluation type")
    total_queries: int = Field(default=0, description="Total evaluation queries")
    aggregate_metrics: MetricResult = Field(
        default_factory=MetricResult,
        description="Aggregate metrics across all queries",
    )
    category_breakdown: list[CategoryBreakdown] = Field(
        default_factory=list,
        description="Per-category metric breakdown",
    )
    quality_gate_passed: bool = Field(
        default=False,
        description="Whether quality gate thresholds are met",
    )
    quality_gate_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Details of quality gate evaluation",
    )


# ── Metric Computation Functions ────────────────────────────────────────────


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Precision@k value (0.0 to 1.0).
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall@k value (0.0 to 1.0).
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Reciprocal Rank.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.

    Returns:
        Reciprocal rank (1/rank of first relevant result, or 0.0).
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Discounted Cumulative Gain at k.

    Uses binary relevance: 1 if relevant, 0 otherwise.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        DCG@k value.
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        NDCG@k value (0.0 to 1.0).
    """
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)

    # Ideal DCG: all relevant docs at the top
    ideal_relevant = min(len(relevant_ids), k)
    ideal_ids = list(relevant_ids)[:ideal_relevant]
    idcg = dcg_at_k(ideal_ids, relevant_ids, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ── Retrieval Evaluator ─────────────────────────────────────────────────────


class RetrievalEvaluator:
    """Evaluates retrieval quality against ground-truth eval cases.

    Computes Precision@k, Recall@k, MRR, and NDCG@k both in aggregate
    and broken down by incident category.

    Usage:
        evaluator = RetrievalEvaluator(k=5)
        eval_cases = evaluator.load_eval_cases("tests/eval/data/retrieval_eval.jsonl")
        report = await evaluator.evaluate(eval_cases, retriever)
    """

    def __init__(self, k: int = 5) -> None:
        """Initialize evaluator.

        Args:
            k: Number of top results for metric computation.
        """
        self.k = k

    def load_eval_cases(self, path: str | Path) -> list[EvalCase]:
        """Load evaluation cases from a JSONL file.

        Args:
            path: Path to JSONL file with eval cases.

        Returns:
            List of EvalCase objects.
        """
        cases: list[EvalCase] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(EvalCase.model_validate_json(line))
        return cases

    def compute_metrics(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> MetricResult:
        """Compute all retrieval metrics for a single query.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of ground-truth relevant document IDs.

        Returns:
            MetricResult with all computed metrics.
        """
        return MetricResult(
            precision_at_k=precision_at_k(retrieved_ids, relevant_ids, self.k),
            recall_at_k=recall_at_k(retrieved_ids, relevant_ids, self.k),
            mrr=reciprocal_rank(retrieved_ids, relevant_ids),
            ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, self.k),
            k=self.k,
        )

    async def evaluate(
        self,
        eval_cases: list[EvalCase],
        retriever: Any,
    ) -> EvalReport:
        """Run evaluation across all cases and produce a report.

        Args:
            eval_cases: List of evaluation cases with ground truth.
            retriever: Retriever instance with an async `retrieve(query)` method
                       that returns a list of dicts with 'metadata.source_id'.

        Returns:
            EvalReport with aggregate and per-category metrics.
        """
        all_metrics: list[MetricResult] = []
        category_metrics: dict[str, list[MetricResult]] = {}

        for case in eval_cases:
            # Retrieve results
            results = await retriever.retrieve(case.query)
            retrieved_ids = [
                r.get("metadata", {}).get("source_id", r.get("id", ""))
                for r in results
            ]
            relevant_set = set(case.relevant_doc_ids)

            # Compute metrics
            metrics = self.compute_metrics(retrieved_ids, relevant_set)
            all_metrics.append(metrics)

            # Group by category
            cat = case.category or "uncategorized"
            if cat not in category_metrics:
                category_metrics[cat] = []
            category_metrics[cat].append(metrics)

        # Aggregate metrics
        aggregate = self._aggregate_metrics(all_metrics)

        # Per-category breakdown
        breakdown = []
        for cat, cat_metrics_list in sorted(category_metrics.items()):
            cat_agg = self._aggregate_metrics(cat_metrics_list)
            breakdown.append(
                CategoryBreakdown(
                    category=cat,
                    num_queries=len(cat_metrics_list),
                    metrics=cat_agg,
                )
            )

        # Quality gate: MRR >= 0.5
        quality_gate_passed = aggregate.mrr >= 0.5

        return EvalReport(
            eval_type="retrieval",
            total_queries=len(eval_cases),
            aggregate_metrics=aggregate,
            category_breakdown=breakdown,
            quality_gate_passed=quality_gate_passed,
            quality_gate_details={
                "mrr_threshold": 0.5,
                "mrr_actual": aggregate.mrr,
                "passed": quality_gate_passed,
            },
        )

    @staticmethod
    def _aggregate_metrics(metrics_list: list[MetricResult]) -> MetricResult:
        """Average metrics across multiple queries.

        Args:
            metrics_list: List of per-query MetricResult objects.

        Returns:
            Averaged MetricResult.
        """
        if not metrics_list:
            return MetricResult()

        n = len(metrics_list)
        return MetricResult(
            precision_at_k=sum(m.precision_at_k for m in metrics_list) / n,
            recall_at_k=sum(m.recall_at_k for m in metrics_list) / n,
            mrr=sum(m.mrr for m in metrics_list) / n,
            ndcg_at_k=sum(m.ndcg_at_k for m in metrics_list) / n,
            k=metrics_list[0].k,
        )
