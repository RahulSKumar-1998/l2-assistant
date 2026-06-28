"""Redis cache client with async connection pooling and TTL support.

Provides a simple get/set/delete interface over Redis with JSON
serialization, configurable TTL, and connection pool management.
Used for caching embeddings, ServiceNow API responses, and
intermediate computation results.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)


class CacheError(Exception):
    """Raised when a cache operation fails."""
    pass


class RedisCache:
    """Async Redis cache client with connection pooling.

    Supports JSON-serializable values with configurable TTL.
    All operations are non-blocking and use connection pooling
    via ``redis.asyncio``.

    Example:
        >>> cache = RedisCache()
        >>> await cache.connect()
        >>> await cache.set("incident:INC001", {"status": "open"}, ttl=300)
        >>> data = await cache.get("incident:INC001")
        >>> await cache.close()
    """

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        default_ttl: int = 3600,
        key_prefix: str = "l2asst:",
        max_connections: int = 20,
    ) -> None:
        """Initialize the Redis cache client.

        Args:
            url: Redis connection URL. Defaults to settings.
            default_ttl: Default TTL in seconds for cached values.
            key_prefix: Prefix for all cache keys to avoid collisions.
            max_connections: Maximum connections in the pool.
        """
        self._url = url
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix
        self._max_connections = max_connections
        self._client: Any = None
        self._log = logger.bind(component="redis_cache")

    def _prefixed_key(self, key: str) -> str:
        """Apply the key prefix.

        Args:
            key: Raw cache key.

        Returns:
            Prefixed key string.
        """
        return f"{self._key_prefix}{key}"

    async def connect(self) -> None:
        """Establish the Redis connection pool.

        Creates an async Redis client with connection pooling.
        Safe to call multiple times — subsequent calls are no-ops.

        Raises:
            CacheError: If the connection cannot be established.
        """
        if self._client is not None:
            return

        try:
            import redis.asyncio as aioredis

            url = self._url
            if url is None:
                settings = get_settings()
                url = settings.database.redis_url

            self._client = aioredis.from_url(
                url,
                max_connections=self._max_connections,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # Verify connectivity
            await self._client.ping()
            self._log.info("redis_connected", url=url.split("@")[-1])
        except ImportError as exc:
            raise CacheError(
                "redis package required: pip install redis"
            ) from exc
        except Exception as exc:
            self._client = None
            self._log.error("redis_connection_failed", error=str(exc))
            raise CacheError(f"Failed to connect to Redis: {exc}") from exc

    async def close(self) -> None:
        """Close the Redis connection pool.

        Safe to call multiple times or when not connected.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._log.info("redis_disconnected")

    async def _ensure_connected(self) -> None:
        """Ensure the client is connected, attempting connection if needed.

        Raises:
            CacheError: If unable to connect.
        """
        if self._client is None:
            await self.connect()

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        Args:
            key: Cache key (prefix is applied automatically).

        Returns:
            Deserialized value, or None if key does not exist.

        Raises:
            CacheError: If the read operation fails.
        """
        await self._ensure_connected()
        prefixed = self._prefixed_key(key)
        try:
            raw = await self._client.get(prefixed)
            if raw is None:
                self._log.debug("cache_miss", key=key)
                return None

            value = json.loads(raw)
            self._log.debug("cache_hit", key=key)
            return value
        except json.JSONDecodeError:
            # Return raw string if not JSON
            self._log.debug("cache_hit_raw", key=key)
            return raw
        except Exception as exc:
            self._log.error("cache_get_error", key=key, error=str(exc))
            raise CacheError(f"Cache get failed for key '{key}': {exc}") from exc

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in the cache with optional TTL.

        Args:
            key: Cache key (prefix is applied automatically).
            value: Value to cache (must be JSON-serializable).
            ttl: Time-to-live in seconds. Uses default_ttl if None.

        Returns:
            True if the value was set successfully.

        Raises:
            CacheError: If the write operation fails.
        """
        await self._ensure_connected()
        prefixed = self._prefixed_key(key)
        effective_ttl = ttl if ttl is not None else self._default_ttl

        try:
            serialized = json.dumps(value, default=str)
            await self._client.setex(prefixed, effective_ttl, serialized)
            self._log.debug("cache_set", key=key, ttl=effective_ttl)
            return True
        except Exception as exc:
            self._log.error("cache_set_error", key=key, error=str(exc))
            raise CacheError(f"Cache set failed for key '{key}': {exc}") from exc

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.

        Args:
            key: Cache key (prefix is applied automatically).

        Returns:
            True if the key was deleted, False if it didn't exist.

        Raises:
            CacheError: If the delete operation fails.
        """
        await self._ensure_connected()
        prefixed = self._prefixed_key(key)
        try:
            result = await self._client.delete(prefixed)
            deleted = result > 0
            self._log.debug("cache_delete", key=key, deleted=deleted)
            return deleted
        except Exception as exc:
            self._log.error("cache_delete_error", key=key, error=str(exc))
            raise CacheError(f"Cache delete failed for key '{key}': {exc}") from exc

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key (prefix is applied automatically).

        Returns:
            True if the key exists.
        """
        await self._ensure_connected()
        prefixed = self._prefixed_key(key)
        try:
            return bool(await self._client.exists(prefixed))
        except Exception as exc:
            self._log.error("cache_exists_error", key=key, error=str(exc))
            return False

    async def set_many(
        self,
        mapping: dict[str, Any],
        *,
        ttl: Optional[int] = None,
    ) -> int:
        """Set multiple values in the cache atomically.

        Args:
            mapping: Dict of key-value pairs to cache.
            ttl: TTL in seconds for all keys.

        Returns:
            Number of keys successfully set.

        Raises:
            CacheError: If the operation fails.
        """
        await self._ensure_connected()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        count = 0

        try:
            pipe = self._client.pipeline(transaction=True)
            for key, value in mapping.items():
                prefixed = self._prefixed_key(key)
                serialized = json.dumps(value, default=str)
                pipe.setex(prefixed, effective_ttl, serialized)
                count += 1
            await pipe.execute()
            self._log.debug("cache_set_many", count=count, ttl=effective_ttl)
            return count
        except Exception as exc:
            self._log.error("cache_set_many_error", error=str(exc))
            raise CacheError(f"Cache set_many failed: {exc}") from exc

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from the cache.

        Args:
            keys: List of cache keys.

        Returns:
            Dict of key-value pairs for found keys.
        """
        await self._ensure_connected()
        prefixed_keys = [self._prefixed_key(k) for k in keys]

        try:
            values = await self._client.mget(prefixed_keys)
            result: dict[str, Any] = {}
            for key, raw in zip(keys, values):
                if raw is not None:
                    try:
                        result[key] = json.loads(raw)
                    except json.JSONDecodeError:
                        result[key] = raw
            self._log.debug(
                "cache_get_many",
                requested=len(keys),
                found=len(result),
            )
            return result
        except Exception as exc:
            self._log.error("cache_get_many_error", error=str(exc))
            raise CacheError(f"Cache get_many failed: {exc}") from exc

    async def clear_prefix(self, prefix: str) -> int:
        """Delete all keys matching a prefix pattern.

        Args:
            prefix: Key prefix to match (applied after the global prefix).

        Returns:
            Number of keys deleted.
        """
        await self._ensure_connected()
        pattern = self._prefixed_key(f"{prefix}*")
        count = 0

        try:
            async for key in self._client.scan_iter(match=pattern, count=100):
                await self._client.delete(key)
                count += 1
            self._log.info("cache_clear_prefix", prefix=prefix, deleted=count)
            return count
        except Exception as exc:
            self._log.error("cache_clear_prefix_error", error=str(exc))
            raise CacheError(f"Cache clear_prefix failed: {exc}") from exc


# ── Module-level convenience ─────────────────────────────────────────────────


_cache: Optional[RedisCache] = None


async def get_cache(
    *,
    url: Optional[str] = None,
    default_ttl: int = 3600,
) -> RedisCache:
    """Get or create the singleton Redis cache instance.

    Args:
        url: Optional Redis URL override.
        default_ttl: Default TTL in seconds.

    Returns:
        Connected RedisCache instance.
    """
    global _cache
    if _cache is None:
        _cache = RedisCache(url=url, default_ttl=default_ttl)
        await _cache.connect()
    return _cache


async def close_cache() -> None:
    """Close the singleton cache connection."""
    global _cache
    if _cache is not None:
        await _cache.close()
        _cache = None
