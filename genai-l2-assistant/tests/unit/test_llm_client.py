"""Unit tests for the LLMClient class.

Tests provider initialization, prompt formatting, error translation,
and generation/streaming logic with langchain_openai.ChatOpenAI.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.llm_client import LLMClient, LLMError, LLMRateLimitError, LLMTimeoutError
from app.models.chat import LLMPrompt
from app.config import LLMProvider, get_settings


@pytest.mark.asyncio
async def test_llm_client_initialization_openai():
    """Verify LLMClient correctly instantiates ChatOpenAI with config settings."""
    settings = get_settings()
    settings.llm.provider = LLMProvider.OPENAI
    settings.llm.model_name = "azure_ai/genailab-maas-DeepSeek-V3-0324"
    settings.llm.openai_api_key = "test_key"
    settings.llm.openai_api_base = "https://genailab.tcs.in"

    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
        client = LLMClient()
        # Trigger client creation
        _ = client._get_openai_client()

        mock_chat_openai.assert_called_once()
        kwargs = mock_chat_openai.call_args[1]
        assert kwargs["model"] == "azure_ai/genailab-maas-DeepSeek-V3-0324"
        assert kwargs["api_key"] == "test_key"
        assert kwargs["base_url"] == "https://genailab.tcs.in"
        assert kwargs["http_client"] is not None
        assert kwargs["http_async_client"] is not None


@pytest.mark.asyncio
async def test_llm_client_initialization_azure():
    """Verify LLMClient correctly instantiates AzureChatOpenAI with azure endpoint."""
    settings = get_settings()
    settings.llm.provider = LLMProvider.OPENAI
    settings.llm.azure_openai_api_key = "azure_key"
    settings.llm.azure_endpoint = "https://test-azure.openai.azure.com"
    settings.llm.azure_api_version = "2024-02-01"

    with patch("langchain_openai.AzureChatOpenAI") as mock_azure_chat_openai:
        client = LLMClient()
        # Trigger client creation
        _ = client._get_openai_client()

        mock_azure_chat_openai.assert_called_once()
        kwargs = mock_azure_chat_openai.call_args[1]
        assert kwargs["azure_deployment"] == settings.llm.model_name
        assert kwargs["api_key"] == "azure_key"
        assert kwargs["azure_endpoint"] == "https://test-azure.openai.azure.com"
        assert kwargs["api_version"] == "2024-02-01"

    # Reset Azure config to prevent affecting other tests
    settings.llm.azure_endpoint = None
    settings.llm.azure_openai_api_key = None


@pytest.mark.asyncio
async def test_llm_client_generate_success():
    """Verify generate invokes ChatOpenAI ainvoke and returns LLMResponse."""
    settings = get_settings()
    settings.llm.provider = LLMProvider.OPENAI
    settings.llm.openai_api_key = "test_key"

    prompt = LLMPrompt(
        system_prompt="You are a helper.",
        user_message="Hello",
        temperature=0.7,
        max_tokens=100
    )

    mock_ai_message = AIMessage(
        content="Hi there!",
        response_metadata={"model_name": "azure_ai/genailab-maas-DeepSeek-V3-0324", "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    )

    mock_instance = MagicMock()
    mock_instance.ainvoke = AsyncMock(return_value=mock_ai_message)

    with patch("langchain_openai.ChatOpenAI", return_value=mock_instance):
        client = LLMClient()
        response = await client.generate(prompt)

        assert response.content == "Hi there!"
        assert response.model == "azure_ai/genailab-maas-DeepSeek-V3-0324"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.latency_ms >= 0

        # Verify ainvoke call structure
        mock_instance.ainvoke.assert_called_once()
        args, kwargs = mock_instance.ainvoke.call_args
        messages = args[0]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "You are a helper."
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "Hello"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_llm_client_generate_streaming_success():
    """Verify generate_streaming yields correct text chunks from ChatOpenAI astream."""
    settings = get_settings()
    settings.llm.provider = LLMProvider.OPENAI
    settings.llm.openai_api_key = "test_key"

    prompt = LLMPrompt(
        system_prompt="You are a helper.",
        user_message="Hello",
        temperature=0.7,
        max_tokens=100
    )

    async def mock_astream(*args, **kwargs):
        yield AIMessage(content="Hi")
        yield AIMessage(content=" ")
        yield AIMessage(content="there!")

    mock_instance = MagicMock()
    mock_instance.astream = mock_astream

    with patch("langchain_openai.ChatOpenAI", return_value=mock_instance):
        client = LLMClient()
        chunks = []
        async for chunk in client.generate_streaming(prompt):
            chunks.append(chunk)

        assert chunks == ["Hi", " ", "there!"]
