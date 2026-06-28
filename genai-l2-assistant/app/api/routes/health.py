"""Health check endpoint.

Returns overall application health status including database
and Redis connectivity checks for Kubernetes readiness probes.
"""

from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter

from app import __version__

router = APIRouter(tags=["health"])
logger = structlog.get_logger(__name__)


async def _check_db() -> bool:
    """Ping PostgreSQL by executing a lightweight query.

    Returns:
        True if the database responds, False otherwise.
    """
    try:
        from sqlalchemy import text

        from app.storage.postgres import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("health_db_ping_failed", exc_info=True)
        return False


async def _check_redis() -> bool:
    """Ping Redis to verify connectivity.

    Returns:
        True if Redis responds to PING, False otherwise.
    """
    try:
        import redis.asyncio as aioredis

        from app.config import get_settings

        settings = get_settings()
        client = aioredis.from_url(
            settings.database.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        try:
            result = await client.ping()
            return bool(result)
        finally:
            await client.aclose()
    except Exception:
        logger.warning("health_redis_ping_failed", exc_info=True)
        return False


@router.get(
    "/health",
    summary="Application health check",
    description="Returns application health status including DB and Redis connectivity.",
    response_model=None,
)
async def health_check() -> dict[str, Any]:
    """Check application health.

    Returns:
        Dictionary with status, version, and connectivity checks.
    """
    try:
        db_ok = await _check_db()
    except Exception:
        db_ok = False

    try:
        redis_ok = await _check_redis()
    except Exception:
        redis_ok = False

    if db_ok and redis_ok:
        status = "healthy"
    elif not db_ok and not redis_ok:
        status = "degraded — db and redis unavailable"
    elif not db_ok:
        status = "degraded — db unavailable"
    else:
        status = "degraded — redis unavailable"

    return {
        "status": status,
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "db_ping": db_ok,
            "redis_ping": redis_ok,
        },
    }
