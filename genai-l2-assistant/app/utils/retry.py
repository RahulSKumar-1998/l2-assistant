"""Retry utilities with exponential backoff.

Provides a decorator for retrying async functions with configurable
backoff, jitter, and exception filtering.
"""

import asyncio
import functools
import logging
import random
from typing import Any, Callable, Optional, Type

import structlog

logger = structlog.get_logger(__name__)


def async_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
) -> Callable:
    """Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first try).
        min_wait: Minimum wait time between retries in seconds.
        max_wait: Maximum wait time between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        jitter: Whether to add random jitter to wait times.
        retryable_exceptions: Tuple of exception types to retry on.
        on_retry: Optional callback called before each retry with
            (attempt, exception, wait_time) args.

    Returns:
        Decorated function with retry logic.

    Example:
        @async_retry(max_attempts=3, retryable_exceptions=(httpx.HTTPStatusError,))
        async def call_api():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc

                    if attempt >= max_attempts:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(exc),
                        )
                        raise

                    # Calculate wait time with exponential backoff
                    wait_time = min(
                        max_wait,
                        min_wait * (exponential_base ** (attempt - 1)),
                    )

                    # Add jitter to avoid thundering herd
                    if jitter:
                        wait_time = wait_time * (0.5 + random.random())

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        wait_seconds=round(wait_time, 2),
                        error=str(exc),
                    )

                    if on_retry:
                        on_retry(attempt, exc, wait_time)

                    await asyncio.sleep(wait_time)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def sync_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Synchronous retry decorator with exponential backoff.

    Same as async_retry but for synchronous functions.

    Args:
        max_attempts: Maximum number of attempts.
        min_wait: Minimum wait seconds.
        max_wait: Maximum wait seconds.
        exponential_base: Backoff base.
        retryable_exceptions: Exceptions to retry on.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time

            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc

                    if attempt >= max_attempts:
                        raise

                    wait_time = min(
                        max_wait,
                        min_wait * (exponential_base ** (attempt - 1)),
                    )
                    wait_time = wait_time * (0.5 + random.random())

                    time.sleep(wait_time)

            if last_exception:
                raise last_exception

        return wrapper

    return decorator
