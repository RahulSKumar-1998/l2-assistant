"""Focused tests for the worker/RAG integration seams.

These tests cover the newly wired end-to-end helper points without needing
external infrastructure such as Postgres, ServiceNow, or a hosted vector DB.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.retriever import HybridRetriever, RetrievalQuery
from app.ingestion import embedding_pipeline as embedding_pipeline_module
from app.ingestion import pipeline as ingestion_pipeline
from app.storage import vector_store as vector_store_module
from app.storage.vector_store import QueryMatch


@pytest.mark.asyncio
async def test_hybrid_retriever_supports_async_vector_store() -> None:
    """Dense retrieval should work with the project's async VectorStore API."""
    embedder = SimpleNamespace(embed_text=AsyncMock(return_value=[0.1, 0.2, 0.3]))

    query_mock = AsyncMock(
        return_value=[
            QueryMatch(
                id="chunk-1",
                score=0.92,
                metadata={
                    "text": "Historical resolution text",
                    "chunk_type": "resolution",
                    "source_id": "INC0001",
                    "source_type": "incident",
                    "category": "software",
                },
            )
        ]
    )
    vector_store = SimpleNamespace(query=query_mock)

    retriever = HybridRetriever(embedder=embedder, vector_store=vector_store)
    results = await retriever.retrieve(
        RetrievalQuery(query_text="connection pool exhaustion", top_k=3)
    )

    assert len(results) == 1
    assert results[0].chunk_text == "Historical resolution text"
    query_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_and_index_uses_default_retrieval_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Incident indexing should land in the default namespace used by retrieval."""
    captured: dict[str, object] = {}

    class FakeEmbeddingPipeline:
        def __init__(self, vector_store: object) -> None:
            captured["vector_store"] = vector_store

        async def run_batch(self, chunks: list[object], namespace: str = "") -> object:
            captured["namespace"] = namespace
            captured["chunk_count"] = len(chunks)
            return SimpleNamespace(total_chunks=len(chunks), upserted_count=len(chunks), failed_count=0)

    monkeypatch.setattr(embedding_pipeline_module, "EmbeddingPipeline", FakeEmbeddingPipeline)
    monkeypatch.setattr(vector_store_module, "get_vector_store", lambda: object())

    incident = SimpleNamespace(
        snow_sys_id="abc123",
        number="INC0012345",
        short_description="Orders API returning 502",
        description="Connections are timing out after the latest deployment.",
        category="software",
        subcategory="application",
        priority=1,
        state="6",
        assignment_group="L2 App Support",
        assigned_to="engineer",
        cmdb_ci="orders-api",
        opened_at=datetime(2024, 6, 1, 8, 0, 0),
        resolved_at=datetime(2024, 6, 1, 9, 30, 0),
        resolution_notes="Increased pool size and rolled back the leaking build.",
        root_cause="Connection leak introduced in deployment v2.4.1",
        is_indexed=False,
    )

    result = await ingestion_pipeline.process_and_index(incident)

    assert result["upserted_count"] == captured["chunk_count"]
    assert captured["namespace"] == ""
    assert int(captured["chunk_count"]) > 0
    assert incident.is_indexed is True



