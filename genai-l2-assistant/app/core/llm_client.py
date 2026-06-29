"""LLM abstraction layer with multi-provider support and observability.

Supports OpenAI (direct + Azure) and Anthropic Claude with:
  - Async generation and streaming
  - Tenacity retry with exponential backoff
  - LangSmith tracing integration
  - Latency tracking
"""

from __future__ import annotations

import os
import time
from typing import AsyncIterator, Optional

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import LLMProvider, get_settings
from app.models.chat import LLMPrompt, LLMResponse

logger = structlog.get_logger(__name__)


# ── Custom Exceptions ────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM API rate limit is hit."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when LLM API authentication fails."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM API call times out."""
    pass


# ── Retry Configuration ─────────────────────────────────────────────────────

_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)),
    reraise=True,
)


# ── LLM Client ──────────────────────────────────────────────────────────────


class LLMClient:
    """Multi-provider LLM client with retry, tracing, and streaming.

    Supports:
      - **OpenAI** direct API (``gpt-4o``, ``gpt-4-turbo``, etc.)
      - **Azure OpenAI** (set ``AZURE_OPENAI_ENDPOINT``)
      - **Anthropic** Claude (``claude-3-5-sonnet-20241022``, etc.)

    Integrates with LangSmith for tracing when ``LANGSMITH_API_KEY``
    is set in the environment.

    Args:
        provider: Override the configured LLM provider.
        model_name: Override the configured model name.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._provider = provider or settings.llm.provider
        self._model_name = model_name or settings.llm.model_name
        self._settings = settings

        # Lazily initialised clients
        self._openai_client: object | None = None
        self._anthropic_client: object | None = None

        # LangSmith tracing
        self._tracing_enabled = self._setup_tracing()

        logger.info(
            "llm_client_initialised",
            provider=self._provider.value,
            model=self._model_name,
            tracing=self._tracing_enabled,
        )

    # ── LangSmith Tracing Setup ──────────────────────────────────────────

    def _setup_tracing(self) -> bool:
        """Configure LangSmith tracing if API key is available.

        Returns:
            True if tracing is enabled.
        """
        settings = self._settings
        api_key = settings.observability.langsmith_api_key

        if api_key:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
            os.environ.setdefault(
                "LANGCHAIN_PROJECT",
                settings.observability.langsmith_project,
            )
            logger.info(
                "langsmith_tracing_enabled",
                project=settings.observability.langsmith_project,
            )
            return True

        return False

    # ── Client Initialisation ────────────────────────────────────────────

    def _get_openai_client(self) -> object:
        """Get or create ChatOpenAI or AzureChatOpenAI client.

        Returns:
            A ChatOpenAI or AzureChatOpenAI client.

        Raises:
            LLMAuthenticationError: If no API key is configured.
        """
        if self._openai_client is None:
            import httpx

            settings = self._settings
            # Disabling SSL verification for custom gateway/proxy support (e.g. TCS GenAI Lab)
            http_client = httpx.Client(verify=False)
            http_async_client = httpx.AsyncClient(verify=False)

            if settings.llm.is_azure:
                from langchain_openai import AzureChatOpenAI
                self._openai_client = AzureChatOpenAI(
                    azure_deployment=self._model_name,
                    api_key=settings.llm.azure_openai_api_key,
                    azure_endpoint=settings.llm.azure_endpoint,
                    api_version=settings.llm.azure_api_version,
                    http_client=http_client,
                    http_async_client=http_async_client,
                )
            elif settings.llm.openai_api_key:
                from langchain_openai import ChatOpenAI
                self._openai_client = ChatOpenAI(
                    model=self._model_name,
                    api_key=settings.llm.openai_api_key,
                    base_url=settings.llm.openai_api_base,
                    http_client=http_client,
                    http_async_client=http_async_client,
                )
            else:
                raise LLMAuthenticationError(
                    "No OpenAI API key configured. Set OPENAI_API_KEY."
                )
        return self._openai_client

    def _get_anthropic_client(self) -> object:
        """Get or create async Anthropic client.

        Returns:
            An ``AsyncAnthropic`` client.

        Raises:
            LLMAuthenticationError: If no Anthropic API key is configured.
        """
        if self._anthropic_client is None:
            from anthropic import AsyncAnthropic

            settings = self._settings
            if not settings.llm.anthropic_api_key:
                raise LLMAuthenticationError(
                    "No Anthropic API key configured. Set ANTHROPIC_API_KEY."
                )
            self._anthropic_client = AsyncAnthropic(
                api_key=settings.llm.anthropic_api_key,
            )
        return self._anthropic_client

    # ── OpenAI Generation ────────────────────────────────────────────────

    async def _generate_openai(self, prompt: LLMPrompt) -> LLMResponse:
        """Generate text using langchain_openai ChatOpenAI client.

        Args:
            prompt: Structured prompt with system/user messages.

        Returns:
            LLMResponse with generated content and usage metrics.

        Raises:
            LLMRateLimitError: On rate limit responses.
            LLMTimeoutError: On request timeouts.
            LLMError: On other API errors.
        """
        from openai import APITimeoutError, RateLimitError
        from langchain_core.messages import SystemMessage, HumanMessage

        client = self._get_openai_client()
        start = time.monotonic()

        messages = [
            SystemMessage(content=prompt.system_prompt),
            HumanMessage(content=prompt.user_message),
        ]

        kwargs = {}
        if prompt.temperature is not None:
            kwargs["temperature"] = prompt.temperature
        if prompt.max_tokens is not None:
            kwargs["max_tokens"] = prompt.max_tokens

        try:
            # client is a ChatOpenAI / AzureChatOpenAI instance
            response = await client.ainvoke(messages, **kwargs)  # type: ignore[attr-defined]

            latency_ms = int((time.monotonic() - start) * 1000)

            # Get token usage from response_metadata / usage_metadata
            usage = getattr(response, "usage_metadata", None)
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            else:
                token_usage = response.response_metadata.get("token_usage", {})
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)

            model_name = response.response_metadata.get("model_name") or self._model_name

            return LLMResponse(
                content=response.content or "",
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )

        except RateLimitError as exc:
            raise LLMRateLimitError(f"OpenAI rate limit: {exc}") from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"OpenAI timeout: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"OpenAI error: {exc}") from exc

    async def _stream_openai(self, prompt: LLMPrompt) -> AsyncIterator[str]:
        """Stream text generation from OpenAI.

        Args:
            prompt: Structured prompt.

        Yields:
            Text chunks as they arrive.
        """
        from openai import APITimeoutError, RateLimitError
        from langchain_core.messages import SystemMessage, HumanMessage

        client = self._get_openai_client()

        messages = [
            SystemMessage(content=prompt.system_prompt),
            HumanMessage(content=prompt.user_message),
        ]

        kwargs = {}
        if prompt.temperature is not None:
            kwargs["temperature"] = prompt.temperature
        if prompt.max_tokens is not None:
            kwargs["max_tokens"] = prompt.max_tokens

        try:
            # client is a ChatOpenAI / AzureChatOpenAI instance
            stream = client.astream(messages, **kwargs)  # type: ignore[attr-defined]

            async for chunk in stream:
                if chunk.content:
                    yield chunk.content

        except RateLimitError as exc:
            raise LLMRateLimitError(f"OpenAI rate limit: {exc}") from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"OpenAI timeout: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"OpenAI streaming error: {exc}") from exc

    # ── Anthropic Generation ─────────────────────────────────────────────

    async def _generate_anthropic(self, prompt: LLMPrompt) -> LLMResponse:
        """Generate text using Anthropic Messages API.

        Args:
            prompt: Structured prompt with system/user messages.

        Returns:
            LLMResponse with generated content and usage metrics.

        Raises:
            LLMRateLimitError: On rate limit errors.
            LLMTimeoutError: On timeouts.
            LLMError: On other API errors.
        """
        from anthropic import APITimeoutError, RateLimitError

        client = self._get_anthropic_client()
        start = time.monotonic()

        try:
            response = await client.messages.create(  # type: ignore[union-attr]
                model=self._model_name,
                max_tokens=prompt.max_tokens,
                system=prompt.system_prompt,
                messages=[
                    {"role": "user", "content": prompt.user_message},
                ],
                temperature=prompt.temperature,
            )

            latency_ms = int((time.monotonic() - start) * 1000)
            content = ""
            if response.content:
                content = response.content[0].text

            return LLMResponse(
                content=content,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
            )

        except RateLimitError as exc:
            raise LLMRateLimitError(f"Anthropic rate limit: {exc}") from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"Anthropic timeout: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Anthropic error: {exc}") from exc

    async def _stream_anthropic(self, prompt: LLMPrompt) -> AsyncIterator[str]:
        """Stream text generation from Anthropic.

        Args:
            prompt: Structured prompt.

        Yields:
            Text chunks as they arrive.
        """
        from anthropic import APITimeoutError, RateLimitError

        client = self._get_anthropic_client()

        try:
            async with client.messages.stream(  # type: ignore[union-attr]
                model=self._model_name,
                max_tokens=prompt.max_tokens,
                system=prompt.system_prompt,
                messages=[
                    {"role": "user", "content": prompt.user_message},
                ],
                temperature=prompt.temperature,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except RateLimitError as exc:
            raise LLMRateLimitError(f"Anthropic rate limit: {exc}") from exc
        except APITimeoutError as exc:
            raise LLMTimeoutError(f"Anthropic timeout: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"Anthropic streaming error: {exc}") from exc

    # ── Public API ───────────────────────────────────────────────────────

    @_RETRY_DECORATOR
    async def generate(self, prompt: LLMPrompt) -> LLMResponse:
        """Generate text from the configured LLM provider.

        Routes to OpenAI or Anthropic based on the configured provider.
        Applies tenacity retry on rate-limit and timeout errors
        (max 3 attempts, exponential backoff 1–10s).

        Args:
            prompt: Structured prompt with system/user messages and params.

        Returns:
            LLMResponse with generated content, model info, token usage,
            and latency.

        Raises:
            LLMAuthenticationError: If API keys are not configured.
            LLMRateLimitError: After retry exhaustion on rate limits.
            LLMTimeoutError: After retry exhaustion on timeouts.
            LLMError: On other generation failures.
        """
        log = logger.bind(
            provider=self._provider.value,
            model=self._model_name,
            metadata=prompt.metadata,
        )
        log.info("llm_generate_start")

        if self._provider == LLMProvider.OPENAI:
            response = await self._generate_openai(prompt)
        elif self._provider == LLMProvider.ANTHROPIC:
            response = await self._generate_anthropic(prompt)
        else:
            raise LLMError(f"Unsupported provider: {self._provider}")

        log.info(
            "llm_generate_complete",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )
        return response

    async def generate_streaming(self, prompt: LLMPrompt) -> AsyncIterator[str]:
        """Stream text generation from the configured LLM provider.

        Yields text chunks as they arrive from the LLM API. Useful for
        real-time display in the UI.

        Note: Streaming does not use tenacity retry — the caller should
        handle reconnection logic for streams.

        Args:
            prompt: Structured prompt with system/user messages and params.

        Yields:
            Text chunks as they are generated.

        Raises:
            LLMAuthenticationError: If API keys are not configured.
            LLMError: On generation failures.
        """
        log = logger.bind(
            provider=self._provider.value,
            model=self._model_name,
        )
        log.info("llm_stream_start")

        start = time.monotonic()
        token_count = 0

        if self._provider == LLMProvider.OPENAI:
            async for chunk in self._stream_openai(prompt):
                token_count += 1
                yield chunk
        elif self._provider == LLMProvider.ANTHROPIC:
            async for chunk in self._stream_anthropic(prompt):
                token_count += 1
                yield chunk
        else:
            raise LLMError(f"Unsupported provider: {self._provider}")

        latency_ms = int((time.monotonic() - start) * 1000)
        log.info(
            "llm_stream_complete",
            chunks_yielded=token_count,
            latency_ms=latency_ms,
        )
