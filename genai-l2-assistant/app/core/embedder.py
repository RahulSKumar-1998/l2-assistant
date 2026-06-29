"""Embedding model wrapper with caching, batching, and rate limiting.

Supports OpenAI (text-embedding-3-large) and HuggingFace (BAAI/bge-large-en-v1.5)
with Redis-based caching, L2 normalization, parallel batch embedding, and
token bucket rate limiting for OpenAI API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from enum import Enum
from typing import Optional

import numpy as np
import structlog
import tiktoken

from app.config import get_settings

logger = structlog.get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_CACHE_TTL_SECONDS: int = 86_400  # 24 hours
_CACHE_KEY_PREFIX: str = "emb:"
_OPENAI_MAX_TOKENS_PER_MIN: int = 1_000_000
_OPENAI_BATCH_CONCURRENCY: int = 10


class EmbeddingProvider(str, Enum):
    """Supported embedding model providers."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


# ── Token Bucket Rate Limiter ────────────────────────────────────────────────


class TokenBucketRateLimiter:
    """Async token-bucket rate limiter for OpenAI embedding API.

    Ensures we stay within the per-minute token quota by tracking
    consumption and sleeping when the bucket is depleted.

    Args:
        tokens_per_minute: Maximum tokens allowed per minute window.
    """

    def __init__(self, tokens_per_minute: int = _OPENAI_MAX_TOKENS_PER_MIN) -> None:
        self._capacity: int = tokens_per_minute
        self._tokens: float = float(tokens_per_minute)
        self._last_refill: float = time.monotonic()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def acquire(self, token_count: int) -> None:
        """Wait until enough tokens are available, then consume them.

        Args:
            token_count: Number of tokens to consume.
        """
        async with self._lock:
            await self._refill()
            while self._tokens < token_count:
                deficit = token_count - self._tokens
                wait_seconds = deficit / (self._capacity / 60.0)
                logger.debug(
                    "rate_limiter_waiting",
                    wait_seconds=round(wait_seconds, 2),
                    token_count=token_count,
                )
                await asyncio.sleep(wait_seconds)
                await self._refill()
            self._tokens -= token_count

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        refill_amount = elapsed * (self._capacity / 60.0)
        self._tokens = min(self._capacity, self._tokens + refill_amount)
        self._last_refill = now


# ── Embedder ─────────────────────────────────────────────────────────────────


class Embedder:
    """Embedding model wrapper with caching, batching, and rate limiting.

    Supports two providers:
      - **OpenAI**: ``text-embedding-3-large`` (3072 dims) via the OpenAI API.
      - **HuggingFace**: ``BAAI/bge-large-en-v1.5`` (1024 dims) via
        ``sentence-transformers`` running locally.

    All vectors are L2-normalised before being returned.

    Args:
        provider: Which embedding backend to use.
        redis_client: Optional async Redis client for embedding caching.
    """

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        redis_client: object | None = None,
    ) -> None:
        settings = get_settings()
        self._provider = provider or self._detect_provider(settings)
        self._redis = redis_client
        self._model_name = settings.embedding.model
        self._dimensions = settings.embedding.dimensions
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._rate_limiter = TokenBucketRateLimiter(_OPENAI_MAX_TOKENS_PER_MIN)

        # Lazily initialised clients
        self._openai_client: object | None = None
        self._hf_model: object | None = None

        logger.info(
            "embedder_initialised",
            provider=self._provider.value,
            model=self._model_name,
            dimensions=self._dimensions,
        )

    # ── Provider detection ───────────────────────────────────────────────

    @staticmethod
    def _detect_provider(settings: object) -> EmbeddingProvider:
        """Auto-detect provider from configured model name.

        Args:
            settings: Application settings instance.

        Returns:
            The detected embedding provider.
        """
        model = settings.embedding.model  # type: ignore[attr-defined]
        if "bge" in model.lower() or "huggingface" in model.lower():
            return EmbeddingProvider.HUGGINGFACE
        return EmbeddingProvider.OPENAI

    # ── Lazy client initialisation ───────────────────────────────────────

    def _get_openai_client(self) -> object:
        """Get or create OpenAIEmbeddings / AzureOpenAIEmbeddings client.

        Returns:
            An OpenAIEmbeddings or AzureOpenAIEmbeddings client instance.

        Raises:
            ValueError: If no OpenAI API key is configured.
        """
        if self._openai_client is None:
            import httpx

            settings = get_settings()
            http_client = httpx.Client(verify=False)
            http_async_client = httpx.AsyncClient(verify=False)

            if settings.llm.is_azure:
                from langchain_openai import AzureOpenAIEmbeddings
                self._openai_client = AzureOpenAIEmbeddings(
                    azure_deployment=self._model_name,
                    api_key=settings.llm.azure_openai_api_key,
                    azure_endpoint=settings.llm.azure_endpoint,
                    api_version=settings.llm.azure_api_version,
                    http_client=http_client,
                    http_async_client=http_async_client,
                )
            elif settings.llm.openai_api_key:
                from langchain_openai import OpenAIEmbeddings
                self._openai_client = OpenAIEmbeddings(
                    model=self._model_name,
                    openai_api_key=settings.llm.openai_api_key,
                    openai_api_base=settings.llm.openai_api_base,
                    http_client=http_client,
                    http_async_client=http_async_client,
                )
            else:
                raise ValueError(
                    "No OpenAI API key configured. Set OPENAI_API_KEY."
                )
        return self._openai_client

    def _get_hf_model(self) -> object:
        """Get or create HuggingFace SentenceTransformer model.

        Returns:
            A ``SentenceTransformer`` model instance.
        """
        if self._hf_model is None:
            from sentence_transformers import SentenceTransformer

            self._hf_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            logger.info("huggingface_model_loaded", model="BAAI/bge-large-en-v1.5")
        return self._hf_model

    # ── Caching ──────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a Redis cache key from text content.

        Uses the first 16 hex characters of the SHA-256 digest.

        Args:
            text: Text to hash.

        Returns:
            Cache key string.
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{_CACHE_KEY_PREFIX}{digest}"

    async def _get_cached_embedding(self, text: str) -> Optional[list[float]]:
        """Retrieve a cached embedding vector from Redis.

        Args:
            text: The original text that was embedded.

        Returns:
            Cached embedding vector, or ``None`` on cache miss.
        """
        if self._redis is None:
            return None
        try:
            key = self._cache_key(text)
            cached = await self._redis.get(key)
            if cached is not None:
                logger.debug("embedding_cache_hit", key=key)
                return json.loads(cached)
        except Exception as exc:
            logger.warning("embedding_cache_read_error", error=str(exc))
        return None

    async def _set_cached_embedding(self, text: str, vector: list[float]) -> None:
        """Store an embedding vector in Redis with TTL.

        Args:
            text: The original text that was embedded.
            vector: The embedding vector to cache.
        """
        if self._redis is None:
            return
        try:
            key = self._cache_key(text)
            await self._redis.set(key, json.dumps(vector), ex=_CACHE_TTL_SECONDS)
            logger.debug("embedding_cache_set", key=key, ttl=_CACHE_TTL_SECONDS)
        except Exception as exc:
            logger.warning("embedding_cache_write_error", error=str(exc))

    # ── Token counting ───────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text using tiktoken cl100k_base.

        Args:
            text: Input text.

        Returns:
            Token count.
        """
        return len(self._encoding.encode(text))

    # ── L2 normalisation ─────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(vector: list[float]) -> list[float]:
        """L2-normalise a vector to unit length.

        Args:
            vector: Input vector.

        Returns:
            Unit-normalised vector.
        """
        arr = np.array(vector, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            return vector
        return (arr / norm).tolist()

    # ── Core embedding methods ───────────────────────────────────────────

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the OpenAI Embeddings API.

        Applies rate limiting before making the API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        total_tokens = sum(self.count_tokens(t) for t in texts)
        await self._rate_limiter.acquire(total_tokens)

        client = self._get_openai_client()
        # client is an OpenAIEmbeddings or AzureOpenAIEmbeddings instance
        vectors = await client.aembed_documents(texts)  # type: ignore[attr-defined]
        logger.debug(
            "openai_embeddings_created",
            count=len(texts),
            tokens_used=total_tokens,
        )
        return vectors

    async def _embed_huggingface(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local HuggingFace SentenceTransformer.

        Runs the model in a thread-pool executor to avoid blocking
        the async event loop.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        model = self._get_hf_model()
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, normalize_embeddings=False).tolist(),  # type: ignore[union-attr]
        )
        logger.debug("huggingface_embeddings_created", count=len(texts))
        return embeddings

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        """Route to the appropriate provider and embed texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of L2-normalised embedding vectors.
        """
        if self._provider == EmbeddingProvider.OPENAI:
            vectors = await self._embed_openai(texts)
        else:
            vectors = await self._embed_huggingface(texts)

        # L2 normalize all vectors
        return [self._l2_normalize(v) for v in vectors]

    # ── Public API ───────────────────────────────────────────────────────

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text with Redis caching.

        Checks the Redis cache first; on miss, computes the embedding
        via the configured provider and caches the result for 24 hours.

        Args:
            text: Text to embed.

        Returns:
            L2-normalised embedding vector.
        """
        # Check cache first
        cached = await self._get_cached_embedding(text)
        if cached is not None:
            return cached

        vectors = await self._embed_raw([text])
        vector = vectors[0]

        # Cache for future lookups
        await self._set_cached_embedding(text, vector)
        return vector

    async def embed_batch(
        self,
        texts: list[str],
        max_concurrency: int = _OPENAI_BATCH_CONCURRENCY,
    ) -> list[list[float]]:
        """Embed multiple texts in parallel with concurrency control.

        Uses ``asyncio.Semaphore`` to limit concurrent API calls and
        ``asyncio.gather`` for parallel execution.  Cached embeddings
        are returned immediately without consuming an API call.

        Args:
            texts: List of texts to embed.
            max_concurrency: Maximum concurrent embedding tasks.

        Returns:
            List of L2-normalised embedding vectors, one per input text.
        """
        if not texts:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Optional[list[float]]] = [None] * len(texts)

        async def _embed_single(idx: int, text: str) -> None:
            """Embed a single text respecting the concurrency semaphore."""
            # Check cache first
            cached = await self._get_cached_embedding(text)
            if cached is not None:
                results[idx] = cached
                return

            async with semaphore:
                vectors = await self._embed_raw([text])
                vector = vectors[0]
                results[idx] = vector
                await self._set_cached_embedding(text, vector)

        tasks = [_embed_single(i, t) for i, t in enumerate(texts)]
        await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(
            "batch_embedding_complete",
            total=len(texts),
            provider=self._provider.value,
        )
        return results  # type: ignore[return-value]
