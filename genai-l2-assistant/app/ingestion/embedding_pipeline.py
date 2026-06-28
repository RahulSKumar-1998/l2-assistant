"""Embedding pipeline for vectorizing and indexing text chunks.

Orchestrates the end-to-end flow from text chunks to vector store:
    1. Generate embeddings via OpenAI embedding API
    2. Upsert vectors to the configured vector store
    3. Track progress and report results

Supports both batch processing (for bulk ingestion) and single-chunk
real-time indexing (for webhook-driven updates).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.incident import TextChunk
from app.storage.vector_store import UpsertResult, VectorRecord, VectorStore

logger = structlog.get_logger(__name__)


# ── Result Models ────────────────────────────────────────────────────────────


class EmbeddingResult(BaseModel):
    """Result of embedding a single text chunk."""
    chunk_id: str = Field(..., description="Chunk identifier")
    success: bool = Field(default=True, description="Whether embedding succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BatchResult(BaseModel):
    """Result of a batch embedding + indexing operation."""
    total_chunks: int = Field(default=0, description="Total chunks submitted")
    embedded_count: int = Field(default=0, description="Successfully embedded")
    upserted_count: int = Field(default=0, description="Successfully upserted to vector store")
    failed_count: int = Field(default=0, description="Failed embeddings")
    errors: list[str] = Field(default_factory=list, description="Error messages")
    elapsed_seconds: float = Field(default=0.0, description="Total processing time")
    chunks_per_second: float = Field(default=0.0, description="Throughput metric")


class PipelineProgress(BaseModel):
    """Progress tracking for long-running batch operations."""
    total: int = Field(default=0, description="Total items to process")
    completed: int = Field(default=0, description="Items completed so far")
    failed: int = Field(default=0, description="Items failed so far")
    percent: float = Field(default=0.0, description="Completion percentage")
    status: str = Field(default="pending", description="Current status")


# ── Embedding Pipeline ───────────────────────────────────────────────────────


class EmbeddingPipeline:
    """Orchestrates text chunk embedding and vector store indexing.

    Uses the OpenAI embeddings API for vector generation and upserts
    to the configured vector store (Pinecone or pgvector).

    Supports:
        - Batch processing with configurable concurrency
        - Single-chunk real-time indexing
        - Progress tracking via callback
        - Comprehensive error handling and reporting

    Example:
        >>> pipeline = EmbeddingPipeline(vector_store=store)
        >>> result = await pipeline.run_batch(chunks, namespace="incidents")
        >>> result.upserted_count
        42
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        embedding_model: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
        batch_size: int = 50,
        max_concurrent: int = 5,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the embedding pipeline.

        Args:
            vector_store: Vector store backend for upserts.
            embedding_model: OpenAI embedding model name.
                Defaults to settings.
            embedding_dimensions: Vector dimensions.
                Defaults to settings.
            batch_size: Number of texts per embedding API call.
            max_concurrent: Maximum concurrent embedding API calls.
            openai_api_key: OpenAI API key. Defaults to settings.
        """
        self._store = vector_store
        self._batch_size = batch_size
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._progress = PipelineProgress()
        self._log = logger.bind(component="embedding_pipeline")

        settings = get_settings()
        self._model = embedding_model or settings.embedding.model
        self._dimensions = embedding_dimensions or settings.embedding.dimensions
        self._api_key = (
            openai_api_key
            or settings.llm.openai_api_key
            or settings.llm.azure_openai_api_key
        )

    @property
    def progress(self) -> PipelineProgress:
        """Current pipeline progress."""
        return self._progress

    async def run_batch(
        self,
        chunks: list[TextChunk],
        *,
        namespace: str = "",
        progress_callback: Optional[Any] = None,
    ) -> BatchResult:
        """Run batch embedding and upsert for multiple text chunks.

        Processes chunks in batches, generates embeddings via the
        OpenAI API, and upserts the resulting vectors to the store.

        Args:
            chunks: List of TextChunk objects to embed and index.
            namespace: Vector store namespace/partition.
            progress_callback: Optional async callable(PipelineProgress)
                invoked after each batch for progress reporting.

        Returns:
            BatchResult with counts, errors, and timing.
        """
        if not chunks:
            return BatchResult()

        start_time = time.monotonic()
        self._progress = PipelineProgress(
            total=len(chunks),
            status="running",
        )

        self._log.info(
            "batch_embedding_started",
            total_chunks=len(chunks),
            batch_size=self._batch_size,
            namespace=namespace,
        )

        all_records: list[VectorRecord] = []
        errors: list[str] = []
        embedded_count = 0

        # Process in batches
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            batch_num = i // self._batch_size + 1

            try:
                records = await self._embed_batch(batch)
                all_records.extend(records)
                embedded_count += len(records)

                self._progress.completed = embedded_count
                self._progress.percent = (
                    embedded_count / self._progress.total * 100
                )

                self._log.debug(
                    "batch_embedded",
                    batch_num=batch_num,
                    count=len(records),
                    total_progress=f"{self._progress.percent:.1f}%",
                )

                if progress_callback:
                    await progress_callback(self._progress)

            except Exception as exc:
                error_msg = f"Batch {batch_num} embedding failed: {exc}"
                errors.append(error_msg)
                self._progress.failed += len(batch)
                self._log.error(
                    "batch_embedding_failed",
                    batch_num=batch_num,
                    error=str(exc),
                )

        # Upsert all successfully embedded records
        upsert_result = UpsertResult()
        if all_records:
            try:
                upsert_result = await self._store.upsert(
                    all_records,
                    namespace=namespace,
                    batch_size=self._batch_size,
                )
                if upsert_result.errors:
                    errors.extend(upsert_result.errors)
            except Exception as exc:
                error_msg = f"Vector store upsert failed: {exc}"
                errors.append(error_msg)
                self._log.error("upsert_failed", error=str(exc))

        elapsed = time.monotonic() - start_time
        throughput = len(chunks) / elapsed if elapsed > 0 else 0.0

        self._progress.status = "completed" if not errors else "completed_with_errors"
        self._progress.percent = 100.0

        result = BatchResult(
            total_chunks=len(chunks),
            embedded_count=embedded_count,
            upserted_count=upsert_result.upserted_count,
            failed_count=self._progress.failed,
            errors=errors,
            elapsed_seconds=round(elapsed, 2),
            chunks_per_second=round(throughput, 2),
        )

        self._log.info(
            "batch_embedding_completed",
            total=result.total_chunks,
            embedded=result.embedded_count,
            upserted=result.upserted_count,
            failed=result.failed_count,
            elapsed_seconds=result.elapsed_seconds,
            throughput=result.chunks_per_second,
        )
        return result

    async def run_single(
        self,
        chunk: TextChunk,
        *,
        namespace: str = "",
    ) -> EmbeddingResult:
        """Embed and index a single text chunk in real-time.

        Used for webhook-driven updates where low latency matters
        more than throughput.

        Args:
            chunk: Single TextChunk to embed and index.
            namespace: Vector store namespace.

        Returns:
            EmbeddingResult indicating success or failure.
        """
        self._log.info(
            "single_embedding_started",
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
        )

        try:
            records = await self._embed_batch([chunk])
            if not records:
                return EmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    success=False,
                    error="Embedding returned empty result",
                )

            await self._store.upsert(records, namespace=namespace)

            self._log.info(
                "single_embedding_completed",
                chunk_id=chunk.chunk_id,
            )
            return EmbeddingResult(chunk_id=chunk.chunk_id, success=True)

        except Exception as exc:
            error_msg = f"Single embed failed: {exc}"
            self._log.error(
                "single_embedding_failed",
                chunk_id=chunk.chunk_id,
                error=str(exc),
            )
            return EmbeddingResult(
                chunk_id=chunk.chunk_id,
                success=False,
                error=error_msg,
            )

    async def _embed_batch(self, chunks: list[TextChunk]) -> list[VectorRecord]:
        """Generate embeddings for a batch of text chunks.

        Uses the OpenAI embeddings API with concurrency limiting.

        Args:
            chunks: Batch of TextChunk objects.

        Returns:
            List of VectorRecord objects ready for upsert.
        """
        async with self._semaphore:
            texts = [chunk.chunk_text for chunk in chunks]
            embeddings = await self._call_embedding_api(texts)

            records: list[VectorRecord] = []
            for chunk, embedding in zip(chunks, embeddings):
                records.append(VectorRecord(
                    id=chunk.chunk_id,
                    values=embedding,
                    metadata={
                        **chunk.metadata,
                        "chunk_text": chunk.chunk_text[:1000],  # Store truncated text
                        "chunk_type": chunk.chunk_type.value,
                        "source_id": chunk.source_id,
                        "source_type": chunk.source_type.value,
                        "chunk_index": chunk.chunk_index,
                    },
                ))
            return records

    async def _call_embedding_api(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Call the OpenAI embeddings API.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self._api_key)

            response = await client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
            )

            # Sort by index to ensure order matches input
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]

        except ImportError as exc:
            raise RuntimeError(
                "openai package required: pip install openai"
            ) from exc
        except Exception as exc:
            self._log.error(
                "embedding_api_error",
                error=str(exc),
                text_count=len(texts),
            )
            raise RuntimeError(f"Embedding API call failed: {exc}") from exc

    async def delete_by_source(
        self,
        source_id: str,
        *,
        namespace: str = "",
    ) -> int:
        """Delete all vectors for a given source document.

        Useful when re-indexing a ticket or KB article.

        Args:
            source_id: Source document ID (incident number or KB number).
            namespace: Vector store namespace.

        Returns:
            Number of vectors deleted.
        """
        self._log.info("deleting_vectors_for_source", source_id=source_id)
        try:
            # Query for all chunk IDs with this source_id
            # For Pinecone, we'd use metadata filter; for pgvector, SQL
            # This is a best-effort approach
            matches = await self._store.query(
                vector=[0.0] * self._dimensions,
                top_k=1000,
                namespace=namespace,
                filter_metadata={"source_id": source_id},
                include_metadata=False,
            )
            if not matches:
                return 0

            ids = [m.id for m in matches]
            deleted = await self._store.delete(ids, namespace=namespace)
            self._log.info(
                "source_vectors_deleted",
                source_id=source_id,
                count=deleted,
            )
            return deleted
        except Exception as exc:
            self._log.error(
                "delete_by_source_failed",
                source_id=source_id,
                error=str(exc),
            )
            raise
