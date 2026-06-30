"""Vector store abstraction layer with Pinecone and pgvector backends.

Provides a unified interface for vector similarity search with two
production-ready implementations:

    - **PineconeVectorStore**: Managed vector database via pinecone-client v3
    - **PGVectorStore**: Self-hosted PostgreSQL + pgvector via SQLAlchemy async

A factory function ``get_vector_store()`` selects the backend based
on application configuration.

Changes from original:
    - Fixed silent $in empty-list skip bug (now returns [] immediately)
    - Added HNSW vector index on embedding column for cosine similarity
    - Added GIN index on metadata JSONB column
    - Moved import json out of inner upsert loop
    - Combined describe_index into a single SQL round-trip
    - Added hybrid_query() using full-text search + RRF for ServiceNow L2
"""

from __future__ import annotations

import abc
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field
import structlog

from app.config import VectorStoreProvider, get_settings

logger = structlog.get_logger(__name__)


# ── Data Models ──────────────────────────────────────────────────────────────


class VectorRecord(BaseModel):
    """A record to upsert into the vector store."""
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique vector ID",
    )
    values: list[float] = Field(..., description="Embedding vector")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata payload (filterable key-values)",
    )


class QueryMatch(BaseModel):
    """A single match returned from a vector similarity query."""
    id: str = Field(..., description="Matched vector ID")
    score: float = Field(..., description="Similarity score (0.0 to 1.0)")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata associated with the matched vector",
    )


class UpsertResult(BaseModel):
    """Result of a batch upsert operation."""
    upserted_count: int = Field(default=0, description="Number of vectors upserted")
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages for failed upserts",
    )


class IndexStats(BaseModel):
    """Descriptive statistics for the vector index."""
    total_vector_count: int = Field(default=0, description="Total vectors in the index")
    dimension: int = Field(default=0, description="Vector dimensionality")
    index_fullness: float = Field(default=0.0, description="Index fullness (0.0 to 1.0)")
    namespaces: dict[str, int] = Field(
        default_factory=dict,
        description="Vector count per namespace",
    )


# ── Abstract Base ────────────────────────────────────────────────────────────


class VectorStore(abc.ABC):
    """Abstract base class for vector store implementations.

    All methods are async. Implementations must handle their own
    connection management and error recovery.
    """

    @abc.abstractmethod
    async def upsert(
        self,
        records: list[VectorRecord],
        *,
        namespace: str = "",
        batch_size: int = 100,
    ) -> UpsertResult:
        """Insert or update vectors in the store.

        Args:
            records: List of vector records to upsert.
            namespace: Logical namespace/partition.
            batch_size: Number of records per batch request.

        Returns:
            UpsertResult with counts and any errors.
        """
        ...

    @abc.abstractmethod
    async def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        namespace: str = "",
        filter_metadata: Optional[dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> list[QueryMatch]:
        """Query the store for nearest-neighbor vectors.

        Args:
            vector: Query embedding vector.
            top_k: Maximum number of results to return.
            namespace: Logical namespace to search within.
            filter_metadata: Metadata filter conditions.
            include_metadata: Whether to include metadata in results.

        Returns:
            List of QueryMatch results sorted by similarity score.
        """
        ...

    @abc.abstractmethod
    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str = "",
    ) -> int:
        """Delete vectors by their IDs.

        Args:
            ids: List of vector IDs to delete.
            namespace: Logical namespace.

        Returns:
            Number of vectors deleted.
        """
        ...

    @abc.abstractmethod
    async def describe_index(self) -> IndexStats:
        """Get descriptive statistics about the index.

        Returns:
            IndexStats with counts and dimensions.
        """
        ...


# ── Pinecone Implementation ─────────────────────────────────────────────────


class PineconeVectorStore(VectorStore):
    """Pinecone vector store using pinecone-client v3.

    Manages connection to a Pinecone serverless or pod-based index.
    Supports namespaces, metadata filtering, and batch operations.

    Example:
        >>> store = PineconeVectorStore(api_key="...", index_name="my-index")
        >>> await store.upsert([VectorRecord(id="1", values=[0.1, ...], metadata={...})])
        >>> matches = await store.query(vector=[0.1, ...], top_k=5)
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str = "us-east-1-aws",
    ) -> None:
        """Initialize the Pinecone vector store.

        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index.
            environment: Pinecone environment/region.
        """
        self._api_key = api_key
        self._index_name = index_name
        self._environment = environment
        self._index: Any = None
        self._log = logger.bind(
            component="pinecone_vector_store",
            index_name=index_name,
        )

    def _get_index(self) -> Any:
        """Lazily initialize and return the Pinecone index client.

        Returns:
            Pinecone Index client.

        Raises:
            ImportError: If pinecone-client is not installed.
        """
        if self._index is None:
            try:
                from pinecone import Pinecone
            except ImportError as exc:
                raise ImportError(
                    "pinecone-client v3 is required: pip install pinecone-client"
                ) from exc

            pc = Pinecone(api_key=self._api_key)
            self._index = pc.Index(self._index_name)
            self._log.info("pinecone_index_connected", index=self._index_name)
        return self._index

    async def upsert(
        self,
        records: list[VectorRecord],
        *,
        namespace: str = "",
        batch_size: int = 100,
    ) -> UpsertResult:
        """Upsert vectors to Pinecone in batches.

        Args:
            records: List of vector records.
            namespace: Pinecone namespace.
            batch_size: Records per batch.

        Returns:
            UpsertResult with count and errors.
        """
        import asyncio

        index = self._get_index()
        total_upserted = 0
        errors: list[str] = []

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            vectors = [
                {
                    "id": r.id,
                    "values": r.values,
                    "metadata": r.metadata,
                }
                for r in batch
            ]
            try:
                # Pinecone client v3 is synchronous — run in executor
                result = await asyncio.to_thread(
                    index.upsert,
                    vectors=vectors,
                    namespace=namespace,
                )
                upserted = getattr(result, "upserted_count", len(batch))
                total_upserted += upserted
                self._log.debug(
                    "pinecone_batch_upserted",
                    batch_index=i // batch_size,
                    count=upserted,
                )
            except Exception as exc:
                error_msg = f"Batch {i // batch_size} failed: {exc}"
                errors.append(error_msg)
                self._log.error(
                    "pinecone_upsert_error",
                    batch_index=i // batch_size,
                    error=str(exc),
                )

        self._log.info(
            "pinecone_upsert_completed",
            total_upserted=total_upserted,
            error_count=len(errors),
        )
        return UpsertResult(upserted_count=total_upserted, errors=errors)

    async def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        namespace: str = "",
        filter_metadata: Optional[dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> list[QueryMatch]:
        """Query Pinecone for similar vectors.

        Args:
            vector: Query embedding.
            top_k: Max results.
            namespace: Pinecone namespace.
            filter_metadata: Metadata filter dict (Pinecone filter syntax).
            include_metadata: Whether to return metadata.

        Returns:
            List of QueryMatch sorted by score descending.
        """
        import asyncio

        index = self._get_index()
        try:
            result = await asyncio.to_thread(
                index.query,
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_metadata,
                include_metadata=include_metadata,
            )
            matches = [
                QueryMatch(
                    id=m["id"],
                    score=float(m.get("score", 0.0)),
                    metadata=m.get("metadata", {}),
                )
                for m in result.get("matches", [])
            ]
            self._log.debug(
                "pinecone_query_completed",
                match_count=len(matches),
                top_k=top_k,
            )
            return matches
        except Exception as exc:
            self._log.error("pinecone_query_error", error=str(exc))
            raise

    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str = "",
    ) -> int:
        """Delete vectors by ID from Pinecone.

        Args:
            ids: Vector IDs to delete.
            namespace: Pinecone namespace.

        Returns:
            Number of IDs submitted for deletion.
        """
        import asyncio

        index = self._get_index()
        try:
            await asyncio.to_thread(
                index.delete,
                ids=ids,
                namespace=namespace,
            )
            self._log.info("pinecone_deleted", count=len(ids))
            return len(ids)
        except Exception as exc:
            self._log.error("pinecone_delete_error", error=str(exc))
            raise

    async def describe_index(self) -> IndexStats:
        """Get Pinecone index statistics.

        Returns:
            IndexStats with vector counts and dimensions.
        """
        import asyncio

        index = self._get_index()
        try:
            stats = await asyncio.to_thread(index.describe_index_stats)
            namespaces = {}
            for ns_name, ns_info in (stats.get("namespaces", {}) or {}).items():
                namespaces[ns_name or "(default)"] = ns_info.get("vector_count", 0)

            return IndexStats(
                total_vector_count=stats.get("total_vector_count", 0),
                dimension=stats.get("dimension", 0),
                index_fullness=stats.get("index_fullness", 0.0),
                namespaces=namespaces,
            )
        except Exception as exc:
            self._log.error("pinecone_describe_error", error=str(exc))
            raise


# ── PGVector Implementation ─────────────────────────────────────────────────


class PGVectorStore(VectorStore):
    """PostgreSQL + pgvector vector store using SQLAlchemy async.

    Uses a dedicated ``vector_embeddings`` table with the pgvector
    extension for cosine similarity search. Supports metadata JSONB
    filtering and batch operations.

    Indexes created on initialization:
        - HNSW index on ``embedding`` column for fast cosine ANN search
        - GIN index on ``metadata`` JSONB column for fast metadata filtering
        - B-tree index on ``namespace`` column

    Example:
        >>> store = PGVectorStore(connection_url="postgresql+asyncpg://...")
        >>> await store.initialize()
        >>> await store.upsert([VectorRecord(id="1", values=[0.1, ...], metadata={...})])
        >>> matches = await store.query(vector=[0.1, ...], top_k=5)
        >>> matches = await store.hybrid_query(text="disk quota exceeded", vector=[0.1, ...], top_k=5)
    """

    def __init__(
        self,
        connection_url: Optional[str] = None,
        table_name: str = "vector_embeddings",
        dimensions: int = 3072,
    ) -> None:
        """Initialize the PGVector store.

        Args:
            connection_url: PostgreSQL async connection URL.
                Defaults to the app database URL.
            table_name: Table name for embeddings.
            dimensions: Vector dimensionality.
        """
        self._connection_url = connection_url
        self._table_name = table_name
        self._dimensions = dimensions
        self._engine: Any = None
        self._session_factory: Any = None
        self._initialized = False
        self._log = logger.bind(
            component="pgvector_store",
            table=table_name,
        )

    async def _ensure_initialized(self) -> None:
        """Lazily create engine, session factory, and table if needed.

        Creates the following on first call:
            - pgvector extension
            - Embeddings table
            - HNSW ANN index on embedding column (pgvector >= 0.5 required)
            - GIN index on metadata JSONB column
            - B-tree index on namespace column
        """
        if self._initialized:
            return

        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlalchemy import text

        url = self._connection_url
        if url is None:
            settings = get_settings()
            url = settings.database.postgres_url

        self._engine = create_async_engine(
            url,
            pool_size=10,
            max_overflow=5,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self._dimensions}),
                    metadata JSONB DEFAULT '{{}}',
                    namespace TEXT DEFAULT '',
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """))

            # B-tree index for namespace equality filters
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS ix_{self._table_name}_namespace
                ON {self._table_name} (namespace)
            """))

            # FIX: HNSW index for fast cosine ANN search (pgvector >= 0.5).
            # Replaces full table scan on every query call.
            # m=16, ef_construction=64 are good defaults; tune for recall vs speed.
            # If on pgvector < 0.5, replace with:
            #   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            # and only create after loading initial data (needs ~3000+ rows).
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS ix_{self._table_name}_embedding_hnsw
                ON {self._table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))

            # FIX: GIN index for JSONB metadata filtering (assignment_group,
            # category, priority, etc.). Avoids seq scans on metadata column.
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS ix_{self._table_name}_metadata_gin
                ON {self._table_name} USING GIN (metadata)
            """))

        self._initialized = True
        self._log.info("pgvector_initialized", dimensions=self._dimensions)

    async def upsert(
        self,
        records: list[VectorRecord],
        *,
        namespace: str = "",
        batch_size: int = 100,
    ) -> UpsertResult:
        """Upsert vectors into pgvector table.

        Uses PostgreSQL ``ON CONFLICT ... DO UPDATE`` for upsert semantics.

        Args:
            records: List of vector records.
            namespace: Logical namespace stored as a column.
            batch_size: Records per SQL batch.

        Returns:
            UpsertResult with count and errors.
        """
        from sqlalchemy import text

        await self._ensure_initialized()
        total_upserted = 0
        errors: list[str] = []

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        for record in batch:
                            vec_str = "[" + ",".join(str(v) for v in record.values) + "]"
                            # FIX: moved json import out of inner loop
                            meta_json = json.dumps(record.metadata)
                            await session.execute(
                                text(f"""
                                    INSERT INTO {self._table_name}
                                        (id, embedding, metadata, namespace)
                                    VALUES (:id, :embedding, :metadata::jsonb, :namespace)
                                    ON CONFLICT (id) DO UPDATE SET
                                        embedding = EXCLUDED.embedding,
                                        metadata = EXCLUDED.metadata,
                                        namespace = EXCLUDED.namespace
                                """),
                                {
                                    "id": record.id,
                                    "embedding": vec_str,
                                    "metadata": meta_json,
                                    "namespace": namespace,
                                },
                            )
                        total_upserted += len(batch)
                self._log.debug(
                    "pgvector_batch_upserted",
                    batch_index=i // batch_size,
                    count=len(batch),
                )
            except Exception as exc:
                error_msg = f"PGVector batch {i // batch_size} failed: {exc}"
                errors.append(error_msg)
                self._log.error(
                    "pgvector_upsert_error",
                    batch_index=i // batch_size,
                    error=str(exc),
                )

        self._log.info(
            "pgvector_upsert_completed",
            total_upserted=total_upserted,
            error_count=len(errors),
        )
        return UpsertResult(upserted_count=total_upserted, errors=errors)

    async def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        namespace: str = "",
        filter_metadata: Optional[dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> list[QueryMatch]:
        """Query pgvector for nearest neighbors using cosine distance.

        Args:
            vector: Query embedding.
            top_k: Max results.
            namespace: Namespace filter.
            filter_metadata: JSONB metadata filter conditions.
                Supports simple equality checks: ``{"key": "value"}``,
                ``$in`` list checks, and ``$gte`` range checks.
            include_metadata: Whether to return metadata.

        Returns:
            List of QueryMatch sorted by similarity (highest first).
        """
        from sqlalchemy import text

        await self._ensure_initialized()

        # FIX: short-circuit on empty $in list instead of silently skipping
        if filter_metadata:
            for value in filter_metadata.values():
                if isinstance(value, dict) and "$in" in value:
                    if not value["$in"]:
                        return []

        vec_str = "[" + ",".join(str(v) for v in vector) + "]"

        where_clauses = []
        params: dict[str, Any] = {
            "query_vec": vec_str,
            "top_k": top_k,
        }

        if namespace:
            where_clauses.append("namespace = :namespace")
            params["namespace"] = namespace

        if filter_metadata:
            for idx, (key, value) in enumerate(filter_metadata.items()):
                if isinstance(value, dict):
                    if "$in" in value:
                        in_values = value["$in"]
                        # Already short-circuited above if empty
                        placeholders = []
                        for item_idx, item in enumerate(in_values):
                            param_name = f"meta_val_{idx}_{item_idx}"
                            placeholders.append(f":{param_name}")
                            params[param_name] = str(item)
                        where_clauses.append(
                            f"metadata->>'{key}' IN ({', '.join(placeholders)})"
                        )
                        continue

                    if "$gte" in value:
                        param_name = f"meta_val_{idx}"
                        if key == "resolved_at":
                            where_clauses.append(
                                f"NULLIF(metadata->>'{key}', '')::timestamptz >= :{param_name}::timestamptz"
                            )
                        else:
                            where_clauses.append(f"metadata->>'{key}' >= :{param_name}")
                        params[param_name] = str(value["$gte"])
                        continue

                param_name = f"meta_val_{idx}"
                where_clauses.append(f"metadata->>'{key}' = :{param_name}")
                params[param_name] = str(value)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        metadata_col = ", metadata" if include_metadata else ""

        sql = f"""
            SELECT id, 1 - (embedding <=> :query_vec::vector) AS score
                   {metadata_col}
            FROM {self._table_name}
            {where_sql}
            ORDER BY embedding <=> :query_vec::vector
            LIMIT :top_k
        """

        try:
            async with self._session_factory() as session:
                result = await session.execute(text(sql), params)
                rows = result.fetchall()

            matches = []
            for row in rows:
                meta = {}
                if include_metadata and len(row) > 2:
                    meta = row[2] if isinstance(row[2], dict) else {}
                matches.append(QueryMatch(
                    id=row[0],
                    score=float(row[1]),
                    metadata=meta,
                ))

            self._log.debug(
                "pgvector_query_completed",
                match_count=len(matches),
                top_k=top_k,
            )
            return matches
        except Exception as exc:
            self._log.error("pgvector_query_error", error=str(exc))
            raise

    async def hybrid_query(
        self,
        text_query: str,
        vector: list[float],
        *,
        top_k: int = 10,
        namespace: str = "",
        filter_metadata: Optional[dict[str, Any]] = None,
        vector_weight: float = 0.6,
        text_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> list[QueryMatch]:
        """Hybrid semantic + keyword search using Reciprocal Rank Fusion (RRF).

        Combines pgvector cosine similarity (ANN) with PostgreSQL full-text
        search (ts_vector / ts_rank) and merges results via RRF. Particularly
        effective for ServiceNow L2 recommendations where exact terms like
        error codes, CI names, or KB article IDs carry strong signal that
        pure vector search may dilute.

        The ``metadata`` column must contain a ``search_text`` key populated
        with the full ticket/article text for full-text ranking to work::

            metadata = {
                "search_text": "disk quota exceeded /var/log full",
                "assignment_group": "storage-ops",
                "category": "infrastructure",
                ...
            }

        RRF score formula:  1 / (rrf_k + rank)
        Final score:        vector_weight * rrf_vector + text_weight * rrf_text

        Args:
            text_query: Raw keyword/phrase query string (e.g. ticket short desc).
            vector: Semantic embedding of the query.
            top_k: Number of final results to return.
            namespace: Namespace filter.
            filter_metadata: Metadata filter conditions (same syntax as query()).
            vector_weight: Weight applied to vector RRF score (default 0.6).
            text_weight: Weight applied to full-text RRF score (default 0.4).
            rrf_k: RRF smoothing constant (default 60, per the original paper).

        Returns:
            List of QueryMatch sorted by combined RRF score descending.
        """
        from sqlalchemy import text as sa_text

        await self._ensure_initialized()

        # Short-circuit on empty $in filter
        if filter_metadata:
            for value in filter_metadata.values():
                if isinstance(value, dict) and "$in" in value:
                    if not value["$in"]:
                        return []

        vec_str = "[" + ",".join(str(v) for v in vector) + "]"

        # Build shared WHERE clause (reused in both CTEs)
        where_clauses = []
        params: dict[str, Any] = {
            "query_vec": vec_str,
            "text_query": text_query,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "vector_weight": vector_weight,
            "text_weight": text_weight,
            # Fetch more candidates from each arm before merging
            "candidate_k": top_k * 4,
        }

        if namespace:
            where_clauses.append("namespace = :namespace")
            params["namespace"] = namespace

        if filter_metadata:
            for idx, (key, value) in enumerate(filter_metadata.items()):
                if isinstance(value, dict):
                    if "$in" in value:
                        placeholders = []
                        for item_idx, item in enumerate(value["$in"]):
                            param_name = f"meta_val_{idx}_{item_idx}"
                            placeholders.append(f":{param_name}")
                            params[param_name] = str(item)
                        where_clauses.append(
                            f"metadata->>'{key}' IN ({', '.join(placeholders)})"
                        )
                        continue
                    if "$gte" in value:
                        param_name = f"meta_val_{idx}"
                        if key == "resolved_at":
                            where_clauses.append(
                                f"NULLIF(metadata->>'{key}', '')::timestamptz >= :{param_name}::timestamptz"
                            )
                        else:
                            where_clauses.append(f"metadata->>'{key}' >= :{param_name}")
                        params[param_name] = str(value["$gte"])
                        continue
                param_name = f"meta_val_{idx}"
                where_clauses.append(f"metadata->>'{key}' = :{param_name}")
                params[param_name] = str(value)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        sql = f"""
            WITH vector_ranked AS (
                -- Semantic ANN arm: rank by cosine distance
                SELECT
                    id,
                    metadata,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> :query_vec::vector) AS rank
                FROM {self._table_name}
                {where_sql}
                ORDER BY embedding <=> :query_vec::vector
                LIMIT :candidate_k
            ),
            text_ranked AS (
                -- Full-text arm: rank by ts_rank on metadata->>'search_text'
                SELECT
                    id,
                    metadata,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(
                            to_tsvector('english', COALESCE(metadata->>'search_text', '')),
                            plainto_tsquery('english', :text_query)
                        ) DESC
                    ) AS rank
                FROM {self._table_name}
                {where_sql}
                ORDER BY ts_rank(
                    to_tsvector('english', COALESCE(metadata->>'search_text', '')),
                    plainto_tsquery('english', :text_query)
                ) DESC
                LIMIT :candidate_k
            ),
            rrf_merged AS (
                -- Reciprocal Rank Fusion: merge both result sets
                SELECT
                    COALESCE(v.id, t.id) AS id,
                    COALESCE(v.metadata, t.metadata) AS metadata,
                    (
                        COALESCE(:vector_weight * (1.0 / (:rrf_k + v.rank)), 0) +
                        COALESCE(:text_weight  * (1.0 / (:rrf_k + t.rank)), 0)
                    ) AS rrf_score
                FROM vector_ranked v
                FULL OUTER JOIN text_ranked t ON v.id = t.id
            )
            SELECT id, rrf_score, metadata
            FROM rrf_merged
            ORDER BY rrf_score DESC
            LIMIT :top_k
        """

        try:
            async with self._session_factory() as session:
                result = await session.execute(sa_text(sql), params)
                rows = result.fetchall()

            matches = [
                QueryMatch(
                    id=row[0],
                    score=float(row[1]),
                    metadata=row[2] if isinstance(row[2], dict) else {},
                )
                for row in rows
            ]

            self._log.debug(
                "pgvector_hybrid_query_completed",
                match_count=len(matches),
                top_k=top_k,
            )
            return matches
        except Exception as exc:
            self._log.error("pgvector_hybrid_query_error", error=str(exc))
            raise

    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str = "",
    ) -> int:
        """Delete vectors by ID from pgvector table.

        Args:
            ids: Vector IDs to delete.
            namespace: Namespace filter (optional additional constraint).

        Returns:
            Number of rows deleted.
        """
        from sqlalchemy import text

        await self._ensure_initialized()

        try:
            where = "WHERE id = ANY(:ids)"
            params: dict[str, Any] = {"ids": ids}
            if namespace:
                where += " AND namespace = :namespace"
                params["namespace"] = namespace

            async with self._session_factory() as session:
                async with session.begin():
                    result = await session.execute(
                        text(f"DELETE FROM {self._table_name} {where}"),
                        params,
                    )
                    deleted = result.rowcount

            self._log.info("pgvector_deleted", count=deleted)
            return deleted
        except Exception as exc:
            self._log.error("pgvector_delete_error", error=str(exc))
            raise

    async def describe_index(self) -> IndexStats:
        """Get pgvector table statistics.

        FIX: Combined into a single SQL round-trip (was two separate queries).

        Returns:
            IndexStats with vector count and dimension info.
        """
        from sqlalchemy import text

        await self._ensure_initialized()

        try:
            async with self._session_factory() as session:
                # FIX: Single query replaces two round-trips.
                # Total count is derived by summing namespace counts.
                result = await session.execute(
                    text(
                        f"SELECT namespace, COUNT(*) AS cnt "
                        f"FROM {self._table_name} "
                        f"GROUP BY namespace"
                    )
                )
                rows = result.fetchall()

            namespaces = {(row[0] or "(default)"): int(row[1]) for row in rows}
            total = sum(namespaces.values())

            return IndexStats(
                total_vector_count=total,
                dimension=self._dimensions,
                index_fullness=0.0,  # Not applicable for pgvector
                namespaces=namespaces,
            )
        except Exception as exc:
            self._log.error("pgvector_describe_error", error=str(exc))
            raise


# ── SQLite Implementation ───────────────────────────────────────────────────


class SQLiteVectorStore(VectorStore):
    """SQLite-backed vector store for local development.

    Stores vectors as JSON array strings and performs cosine similarity
    calculation in Python. Fully compatible with SQLite.
    """

    def __init__(
        self,
        connection_url: Optional[str] = None,
        table_name: str = "vector_embeddings",
        dimensions: int = 3072,
    ) -> None:
        """Initialize the SQLite vector store."""
        self._connection_url = connection_url
        self._table_name = table_name
        self._dimensions = dimensions
        self._engine: Any = None
        self._session_factory: Any = None
        self._initialized = False
        self._log = logger.bind(
            component="sqlite_vector_store",
            table=table_name,
        )

    async def _ensure_initialized(self) -> None:
        """Lazily create engine, session factory, and table if needed."""
        if self._initialized:
            return

        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )
        from sqlalchemy import text

        url = self._connection_url
        if url is None:
            settings = get_settings()
            url = settings.database.postgres_url

        if url.startswith("sqlite"):
            self._engine = create_async_engine(
                url,
                connect_args={"timeout": 30},
            )
            from sqlalchemy import event
            @event.listens_for(self._engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
        else:
            self._engine = create_async_engine(url)

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async with self._engine.begin() as conn:
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    embedding TEXT,
                    metadata TEXT DEFAULT '{{}}',
                    namespace TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS ix_{self._table_name}_namespace
                ON {self._table_name} (namespace)
            """))
        self._initialized = True

    async def upsert(
        self,
        records: list[VectorRecord],
        *,
        namespace: str = "",
        batch_size: int = 100,
    ) -> UpsertResult:
        await self._ensure_initialized()
        from sqlalchemy import text

        upserted = 0
        errors = []

        try:
            async with self._session_factory() as session:
                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]
                    async with session.begin():
                        for record in batch:
                            stmt = text(f"""
                                INSERT OR REPLACE INTO {self._table_name}
                                (id, embedding, metadata, namespace)
                                VALUES (:id, :embedding, :metadata, :namespace)
                            """)
                            await session.execute(
                                stmt,
                                {
                                    "id": record.id,
                                    "embedding": json.dumps(record.values),
                                    "metadata": json.dumps(record.metadata),
                                    "namespace": namespace,
                                },
                            )
                            upserted += 1
            return UpsertResult(upserted_count=upserted, errors=errors)
        except Exception as exc:
            self._log.error("sqlite_vector_store_upsert_failed", error=str(exc))
            return UpsertResult(upserted_count=upserted, errors=[str(exc)])

    async def query(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        namespace: str = "",
        filter_metadata: Optional[dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> list[QueryMatch]:
        await self._ensure_initialized()
        from sqlalchemy import text
        import numpy as np

        query_vec = np.array(vector, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        try:
            async with self._session_factory() as session:
                stmt = text(f"""
                    SELECT id, embedding, metadata FROM {self._table_name}
                    WHERE namespace = :namespace
                """)
                result = await session.execute(stmt, {"namespace": namespace})
                rows = result.fetchall()

            matches = []
            for row in rows:
                row_id = row[0]
                emb_json = row[1]
                meta_json = row[2]

                rec_metadata = json.loads(meta_json) if meta_json else {}

                if filter_metadata:
                    match_filter = True
                    for k, v in filter_metadata.items():
                        val = rec_metadata.get(k)
                        if isinstance(v, dict):
                            if "$in" in v:
                                if val not in v["$in"]:
                                    match_filter = False
                                    break
                            elif "$gte" in v:
                                if val is None or val < v["$gte"]:
                                    match_filter = False
                                    break
                            else:
                                if val != v:
                                    match_filter = False
                                    break
                        else:
                            if val != v:
                                match_filter = False
                                break
                    if not match_filter:
                        continue

                rec_vector = np.array(json.loads(emb_json), dtype=np.float32)
                rec_norm = np.linalg.norm(rec_vector)

                if query_norm > 0 and rec_norm > 0:
                    score = float(np.dot(query_vec, rec_vector) / (query_norm * rec_norm))
                    score = (score + 1.0) / 2.0
                else:
                    score = 0.0

                matches.append(
                    QueryMatch(
                        id=row_id,
                        score=score,
                        metadata=rec_metadata if include_metadata else {},
                    )
                )

            matches.sort(key=lambda x: x.score, reverse=True)
            return matches[:top_k]

        except Exception as exc:
            self._log.error("sqlite_vector_store_query_failed", error=str(exc))
            raise

    async def delete(
        self,
        ids: list[str],
        *,
        namespace: str = "",
    ) -> int:
        await self._ensure_initialized()
        from sqlalchemy import text

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    deleted_count = 0
                    chunk_size = 100
                    for i in range(0, len(ids), chunk_size):
                        chunk = ids[i : i + chunk_size]
                        placeholders = ", ".join(f":id_{idx}" for idx in range(len(chunk)))
                        stmt = text(f"""
                            DELETE FROM {self._table_name}
                            WHERE namespace = :namespace AND id IN ({placeholders})
                        """)
                        params = {"namespace": namespace}
                        for idx, id_val in enumerate(chunk):
                            params[f"id_{idx}"] = id_val
                        res = await session.execute(stmt, params)
                        deleted_count += res.rowcount
            return deleted_count
        except Exception as exc:
            self._log.error("sqlite_vector_store_delete_failed", error=str(exc))
            raise

    async def describe_index(self) -> IndexStats:
        await self._ensure_initialized()
        from sqlalchemy import text

        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    text(f"SELECT COUNT(*) FROM {self._table_name}")
                )
                total = result.scalar() or 0

                result = await session.execute(
                    text(f"SELECT namespace, COUNT(*) FROM {self._table_name} GROUP BY namespace")
                )
                namespaces = {row[0]: row[1] for row in result.fetchall()}

            return IndexStats(
                total_vector_count=total,
                dimension=self._dimensions,
                index_fullness=0.0,
                namespaces=namespaces,
            )
        except Exception as exc:
            self._log.error("sqlite_vector_store_describe_failed", error=str(exc))
            raise


# ── Factory Function ─────────────────────────────────────────────────────────


def get_vector_store() -> VectorStore:
    """Create a vector store instance based on application configuration.

    Reads ``VECTOR_STORE_PROVIDER`` from settings to determine the backend.

    Returns:
        VectorStore: Configured vector store implementation.

    Raises:
        ValueError: If the configured provider is not supported.
        ValueError: If required configuration is missing.
    """
    settings = get_settings()
    provider = settings.vector_store.provider
    url = settings.database.postgres_url

    if provider == VectorStoreProvider.PINECONE:
        api_key = settings.vector_store.pinecone_api_key
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY is required when using Pinecone provider"
            )
        return PineconeVectorStore(
            api_key=api_key,
            index_name=settings.vector_store.pinecone_index_name,
            environment=settings.vector_store.pinecone_environment,
        )

    if provider == VectorStoreProvider.PGVECTOR:
        if url.startswith("sqlite"):
            return SQLiteVectorStore(
                connection_url=url,
                dimensions=settings.embedding.dimensions,
            )
        return PGVectorStore(
            connection_url=url,
            dimensions=settings.embedding.dimensions,
        )

    raise ValueError(f"Unsupported vector store provider: {provider}")