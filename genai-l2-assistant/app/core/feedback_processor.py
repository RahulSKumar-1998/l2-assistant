"""Feedback analysis and quality score computation.

Processes engineer feedback on AI recommendations to compute per-source
quality scores, track positive/negative signals, and flag high-confidence
wrong predictions for human review.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import structlog
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.feedback import FeedbackRecord, FeedbackStats, FeedbackWeight

logger = structlog.get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_POSITIVE_THRESHOLD: int = 4  # rating >= 4 is positive
_NEGATIVE_THRESHOLD: int = 2  # rating <= 2 is negative
_HIGH_CONFIDENCE_THRESHOLD: float = 0.8  # flag if confidence was high but feedback negative
_REVIEW_FLAG_RATING_MAX: int = 2  # flag for review if rating <= this


# ── Models ───────────────────────────────────────────────────────────────────


class SourceSignals(BaseModel):
    """Accumulated positive/negative signals for a single source."""
    source_id: str = Field(..., description="Source document ID")
    source_type: str = Field(default="incident", description="Source type")
    positive: int = Field(default=0, description="Positive signal count")
    negative: int = Field(default=0, description="Negative signal count")
    quality_score: float = Field(default=1.0, description="Computed quality score")


class ReviewFlag(BaseModel):
    """A recommendation flagged for human review."""
    recommendation_id: uuid.UUID = Field(..., description="Recommendation UUID")
    incident_id: Optional[uuid.UUID] = Field(default=None, description="Incident UUID")
    reason: str = Field(..., description="Why this was flagged")
    confidence_score: float = Field(default=0.0, description="Original confidence")
    feedback_rating: int = Field(default=1, description="Engineer rating")


# ── Feedback Processor ───────────────────────────────────────────────────────


class FeedbackProcessor:
    """Processes engineer feedback to compute source quality scores.

    Tracks positive and negative signals per source document and
    computes quality scores using the formula:

        quality_score = positive / (positive + negative + 1)

    High-confidence recommendations that receive negative feedback
    are flagged for human review as valuable training signals.

    Args:
        db_session_factory: Async database session factory.
    """

    def __init__(
        self,
        db_session_factory: object | None = None,
    ) -> None:
        self._db_session_factory = db_session_factory
        logger.info("feedback_processor_initialised")

    # ── Quality Score Computation ────────────────────────────────────────

    @staticmethod
    def compute_quality_score(positive: int, negative: int) -> float:
        """Compute quality score from positive/negative signal counts.

        Uses the formula: score = positive / (positive + negative + 1)

        This produces scores in (0, 1) where:
          - 0.0 = all negative signals
          - 0.5 = balanced signals
          - approaching 1.0 = mostly positive signals
          - 0.0 with no signals (laplace smoothing via +1)

        Args:
            positive: Number of positive feedback signals.
            negative: Number of negative feedback signals.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        return positive / (positive + negative + 1)

    # ── Batch Processing ─────────────────────────────────────────────────

    async def process_feedback_batch(
        self,
        feedback_records: list[FeedbackRecord],
        recommendation_sources: dict[uuid.UUID, list[str]] | None = None,
    ) -> FeedbackStats:
        """Process a batch of feedback records and update source quality scores.

        For each feedback record:
          1. Classify as positive (rating >= 4) or negative (rating <= 2).
          2. Propagate the signal to all source documents used in the
             recommendation.
          3. Recompute the quality score for each affected source.
          4. Store updated weights in the ``feedback_weights`` table.
          5. Flag high-confidence-wrong predictions for review.

        Args:
            feedback_records: List of feedback records to process.
            recommendation_sources: Optional mapping from recommendation UUID
                to list of source IDs used. If not provided, sources are loaded
                from the database.

        Returns:
            FeedbackStats with processing summary.
        """
        start = time.monotonic()
        log = logger.bind(batch_size=len(feedback_records))
        log.info("feedback_batch_processing_start")

        # Accumulate signals per source
        source_signals: dict[str, SourceSignals] = defaultdict(
            lambda: SourceSignals(source_id="", source_type="incident")
        )

        positive_count = 0
        negative_count = 0
        review_flags: list[ReviewFlag] = []

        for record in feedback_records:
            is_positive = record.rating >= _POSITIVE_THRESHOLD
            is_negative = record.rating <= _NEGATIVE_THRESHOLD

            if is_positive:
                positive_count += 1
            if is_negative:
                negative_count += 1

            # Get sources for this recommendation
            sources = await self._get_recommendation_sources(
                record.recommendation_id,
                recommendation_sources,
            )

            # Propagate signal to each source
            for source_id in sources:
                if source_id not in source_signals:
                    source_signals[source_id] = SourceSignals(
                        source_id=source_id,
                        source_type=self._infer_source_type(source_id),
                    )

                signals = source_signals[source_id]
                if is_positive:
                    signals.positive += 1
                elif is_negative:
                    signals.negative += 1

            # Check for high-confidence wrong predictions
            if is_negative:
                flag = await self._check_for_review_flag(record)
                if flag:
                    review_flags.append(flag)

        # Compute quality scores
        sources_updated = 0
        for source_id, signals in source_signals.items():
            signals.quality_score = self.compute_quality_score(
                signals.positive,
                signals.negative,
            )
            sources_updated += 1

        # Persist to database
        await self._store_feedback_weights(source_signals)

        # Flag items for review
        for flag in review_flags:
            await self.flag_for_review(flag)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        total_feedback = len(feedback_records)
        positive_rate = positive_count / total_feedback if total_feedback > 0 else 0.0

        stats = FeedbackStats(
            total_feedback=total_feedback,
            positive_count=positive_count,
            negative_count=negative_count,
            positive_rate=round(positive_rate, 4),
            sources_updated=sources_updated,
        )

        log.info(
            "feedback_batch_processing_complete",
            positive=positive_count,
            negative=negative_count,
            sources_updated=sources_updated,
            review_flags=len(review_flags),
            latency_ms=elapsed_ms,
        )

        return stats

    # ── Source Resolution ────────────────────────────────────────────────

    async def _get_recommendation_sources(
        self,
        recommendation_id: uuid.UUID,
        recommendation_sources: dict[uuid.UUID, list[str]] | None,
    ) -> list[str]:
        """Get the source IDs used in a recommendation.

        First checks the provided mapping, then falls back to
        database lookup.

        Args:
            recommendation_id: The recommendation UUID.
            recommendation_sources: Pre-loaded source mapping.

        Returns:
            List of source IDs.
        """
        # Check pre-loaded mapping
        if recommendation_sources and recommendation_id in recommendation_sources:
            return recommendation_sources[recommendation_id]

        # Fall back to database lookup
        if self._db_session_factory is not None:
            try:
                from sqlalchemy import select

                from app.storage.postgres import RecommendationDB

                factory = self._db_session_factory
                async with factory() as session:  # type: ignore[operator]
                    result = await session.execute(
                        select(RecommendationDB).where(
                            RecommendationDB.id == recommendation_id
                        )
                    )
                    rec = result.scalar_one_or_none()
                    if rec and rec.similar_incidents:
                        sources = []
                        for si in rec.similar_incidents:
                            if isinstance(si, dict) and "number" in si:
                                sources.append(si["number"])
                        return sources

            except Exception as exc:
                logger.warning(
                    "recommendation_sources_lookup_error",
                    recommendation_id=str(recommendation_id),
                    error=str(exc),
                )

        return []

    @staticmethod
    def _infer_source_type(source_id: str) -> str:
        """Infer source type from the source ID format.

        Args:
            source_id: Source document ID.

        Returns:
            Inferred source type string.
        """
        source_upper = source_id.upper()
        if source_upper.startswith("KB"):
            return "kb_article"
        elif source_upper.startswith("RB") or source_upper.startswith("RUN"):
            return "runbook"
        return "incident"

    # ── Review Flagging ──────────────────────────────────────────────────

    async def _check_for_review_flag(
        self,
        record: FeedbackRecord,
    ) -> Optional[ReviewFlag]:
        """Check if a negative feedback record should be flagged for review.

        Flags recommendations where the AI had high confidence
        (>= 0.8) but the engineer rated it poorly (<= 2).

        Args:
            record: The feedback record to evaluate.

        Returns:
            ReviewFlag if the record should be flagged, None otherwise.
        """
        if record.rating > _REVIEW_FLAG_RATING_MAX:
            return None

        # Look up the recommendation's confidence score
        confidence = await self._get_recommendation_confidence(
            record.recommendation_id
        )

        if confidence is not None and confidence >= _HIGH_CONFIDENCE_THRESHOLD:
            return ReviewFlag(
                recommendation_id=record.recommendation_id,
                incident_id=record.incident_id,
                reason=(
                    f"High-confidence prediction ({confidence:.0%}) "
                    f"received negative feedback (rating={record.rating}). "
                    f"Comment: {record.comment or 'No comment provided'}"
                ),
                confidence_score=confidence,
                feedback_rating=record.rating,
            )

        return None

    async def _get_recommendation_confidence(
        self,
        recommendation_id: uuid.UUID,
    ) -> Optional[float]:
        """Get the confidence score of a recommendation from the database.

        Args:
            recommendation_id: The recommendation UUID.

        Returns:
            Confidence score, or None if not found.
        """
        if self._db_session_factory is None:
            return None

        try:
            from sqlalchemy import select

            from app.storage.postgres import RecommendationDB

            factory = self._db_session_factory
            async with factory() as session:  # type: ignore[operator]
                result = await session.execute(
                    select(RecommendationDB.confidence_score).where(
                        RecommendationDB.id == recommendation_id
                    )
                )
                row = result.scalar_one_or_none()
                return float(row) if row is not None else None

        except Exception as exc:
            logger.warning(
                "confidence_lookup_error",
                recommendation_id=str(recommendation_id),
                error=str(exc),
            )
            return None

    async def flag_for_review(self, flag: ReviewFlag) -> None:
        """Store a review flag in the review queue.

        Args:
            flag: The review flag to persist.
        """
        log = logger.bind(
            recommendation_id=str(flag.recommendation_id),
            confidence=flag.confidence_score,
            rating=flag.feedback_rating,
        )

        if self._db_session_factory is None:
            log.info("review_flag_logged_no_db", reason=flag.reason)
            return

        try:
            from app.storage.postgres import ReviewQueueDB

            factory = self._db_session_factory
            async with factory() as session:  # type: ignore[operator]
                review = ReviewQueueDB(
                    recommendation_id=flag.recommendation_id,
                    reason=flag.reason,
                    status="pending",
                )
                session.add(review)
                await session.commit()

            log.info("review_flag_stored", reason=flag.reason)

        except Exception as exc:
            log.error("review_flag_store_error", error=str(exc))

    # ── Weight Storage ───────────────────────────────────────────────────

    async def _store_feedback_weights(
        self,
        source_signals: dict[str, SourceSignals],
    ) -> None:
        """Persist updated feedback weights to the database.

        Uses upsert semantics: creates new records or updates existing
        ones based on source_id.

        Args:
            source_signals: Dictionary of source signals to persist.
        """
        if self._db_session_factory is None or not source_signals:
            return

        try:
            from sqlalchemy import select

            from app.storage.postgres import FeedbackWeightDB

            factory = self._db_session_factory
            async with factory() as session:  # type: ignore[operator]
                for source_id, signals in source_signals.items():
                    # Try to find existing weight record
                    result = await session.execute(
                        select(FeedbackWeightDB).where(
                            FeedbackWeightDB.source_id == source_id
                        )
                    )
                    existing = result.scalar_one_or_none()

                    if existing:
                        # Update existing: merge signals
                        existing.positive_signals += signals.positive
                        existing.negative_signals += signals.negative
                        existing.quality_score = self.compute_quality_score(
                            existing.positive_signals,
                            existing.negative_signals,
                        )
                    else:
                        # Create new
                        weight = FeedbackWeightDB(
                            source_id=source_id,
                            source_type=signals.source_type,
                            quality_score=signals.quality_score,
                            positive_signals=signals.positive,
                            negative_signals=signals.negative,
                        )
                        session.add(weight)

                await session.commit()

            logger.info(
                "feedback_weights_stored",
                sources_count=len(source_signals),
            )

        except Exception as exc:
            logger.error("feedback_weights_store_error", error=str(exc))
