"""Unit tests for the Embedder class.

Verifies embedding model wrapper initialization and document embedding logic
using langchain_openai.OpenAIEmbeddings.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.embedder import Embedder, EmbeddingProvider
from app.config import get_settings


@pytest.mark.asyncio
async def test_embedder_initialization_openai():
    """Verify Embedder correctly instantiates OpenAIEmbeddings with config settings."""
    settings = get_settings()
    settings.llm.openai_api_key = "test_key"
    settings.llm.openai_api_base = "https://genailab.tcs.in"
    settings.embedding.model = "azure/genailab-maas-text-embedding-3-large"

    with patch("langchain_openai.OpenAIEmbeddings") as mock_openai_embeddings:
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        # Trigger client creation
        _ = embedder._get_openai_client()

        mock_openai_embeddings.assert_called_once()
        kwargs = mock_openai_embeddings.call_args[1]
        assert kwargs["model"] == "azure/genailab-maas-text-embedding-3-large"
        assert kwargs["openai_api_key"] == "test_key"
        assert kwargs["openai_api_base"] == "https://genailab.tcs.in"
        assert kwargs["http_client"] is not None
        assert kwargs["http_async_client"] is not None


@pytest.mark.asyncio
async def test_embedder_initialization_azure():
    """Verify Embedder correctly instantiates AzureOpenAIEmbeddings when Azure is configured."""
    settings = get_settings()
    settings.llm.azure_openai_api_key = "azure_key"
    settings.llm.azure_endpoint = "https://test-azure.openai.azure.com"
    settings.llm.azure_api_version = "2024-02-01"
    settings.embedding.model = "azure/genailab-maas-text-embedding-3-large"

    with patch("langchain_openai.AzureOpenAIEmbeddings") as mock_azure_embeddings:
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        # Trigger client creation
        _ = embedder._get_openai_client()

        mock_azure_embeddings.assert_called_once()
        kwargs = mock_azure_embeddings.call_args[1]
        assert kwargs["azure_deployment"] == "azure/genailab-maas-text-embedding-3-large"
        assert kwargs["api_key"] == "azure_key"
        assert kwargs["azure_endpoint"] == "https://test-azure.openai.azure.com"
        assert kwargs["api_version"] == "2024-02-01"

    # Reset Azure config
    settings.llm.azure_endpoint = None
    settings.llm.azure_openai_api_key = None


@pytest.mark.asyncio
async def test_embedder_embed_texts_success():
    """Verify _embed_openai correctly calls aembed_documents and normalises the vectors."""
    settings = get_settings()
    settings.llm.openai_api_key = "test_key"
    settings.embedding.model = "azure/genailab-maas-text-embedding-3-large"

    mock_embeddings = [[0.1, 0.2, 0.3]]
    mock_instance = MagicMock()
    mock_instance.aembed_documents = AsyncMock(return_value=mock_embeddings)

    with patch("langchain_openai.OpenAIEmbeddings", return_value=mock_instance):
        embedder = Embedder(provider=EmbeddingProvider.OPENAI)
        vectors = await embedder._embed_openai(["Hello world"])

        assert vectors == mock_embeddings
        mock_instance.aembed_documents.assert_called_once_with(["Hello world"])
