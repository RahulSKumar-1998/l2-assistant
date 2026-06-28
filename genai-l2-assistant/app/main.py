"""FastAPI application entry point.

Creates and configures the FastAPI app with:
- All API routers under /api/v1
- Prometheus metrics instrumentation
- CORS middleware
- Custom RBAC and request-logging middleware
- Health endpoint at /health
- Structured logging via structlog
- Startup/shutdown lifecycle events
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.middleware import RBACMiddleware, RequestLoggingMiddleware
from app.api.routes.chat import router as chat_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.health import router as health_router
from app.api.routes.incidents import router as incidents_router
from app.api.routes.recommendations import router as recommendations_router
from app.config import AppEnvironment, LogFormat, get_settings


# ── Structured Logging Setup ───────────────────────────────────────────────


def _configure_logging() -> None:
    """Configure structlog for the application.

    In production (LOG_FORMAT=json), outputs JSON lines for log aggregation.
    In development (LOG_FORMAT=human), outputs coloured, human-readable logs.
    """
    settings = get_settings()
    is_json = settings.log_format == LogFormat.JSON

    # Shared processors applied to every log event
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if is_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to route through structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if settings.is_development else logging.INFO)

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events.

    Startup:
        - Configure structured logging
        - Verify database connectivity
        - Verify vector store connectivity (Redis ping)
        - Log application startup info

    Shutdown:
        - Close database connection pool
    """
    _configure_logging()
    logger = structlog.get_logger("app.main")

    settings = get_settings()

    # ── Startup ──────────────────────────────────────────────────────────
    logger.info(
        "application_starting",
        version=__version__,
        environment=settings.app_env.value,
        log_format=settings.log_format.value,
    )

    # Verify database connection
    db_ok = False
    try:
        from app.storage.postgres import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
        logger.info("database_connection_verified")
    except Exception:
        logger.error("database_connection_failed", exc_info=True)

    # Verify Redis / vector store ping
    vector_store_ok = False
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            settings.database.redis_url,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        try:
            await client.ping()
            vector_store_ok = True
            logger.info("redis_connection_verified")
        finally:
            await client.aclose()
    except Exception:
        logger.warning("redis_connection_failed", exc_info=True)

    logger.info(
        "application_started",
        version=__version__,
        environment=settings.app_env.value,
        db_connected=db_ok,
        redis_connected=vector_store_ok,
        cors_origins=settings.cors_origins,
    )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("application_shutting_down")

    try:
        from app.storage.postgres import close_db

        await close_db()
        logger.info("database_connections_closed")
    except Exception:
        logger.error("database_close_failed", exc_info=True)

    logger.info("application_stopped")


# ── FastAPI App ─────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        Fully configured FastAPI app ready to serve.
    """
    settings = get_settings()

    app = FastAPI(
        title="GenAI L2 Support Assistant",
        description=(
            "RAG-powered incident resolution assistant for L2 support engineers. "
            "Analyzes ServiceNow incidents, retrieves similar historical cases, "
            "and generates triage recommendations."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS Middleware ──────────────────────────────────────────────────
    # In development, allow all origins so the file:// simulator works
    cors_origins = (
        ["*"] if settings.is_development else settings.cors_origins
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False if settings.is_development else True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Response-Time-Ms"],
    )

    # ── Custom Middleware (applied bottom-up, so logging wraps RBAC) ────
    app.add_middleware(RBACMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ── Prometheus Instrumentation ───────────────────────────────────────
    if settings.observability.prometheus_enabled:
        try:
            from prometheus_fastapi_instrumentator import Instrumentator

            Instrumentator(
                should_group_status_codes=True,
                should_ignore_untemplated=True,
                should_respect_env_var=False,
                excluded_handlers=["/health", "/metrics"],
                env_var_name="ENABLE_METRICS",
            ).instrument(app).expose(app, endpoint="/metrics")
        except ImportError:
            # prometheus-fastapi-instrumentator not installed
            pass

    # ── Health Route (at root, not under /api/v1) ────────────────────────
    app.include_router(health_router)

    # ── API v1 Routes ────────────────────────────────────────────────────
    app.include_router(incidents_router, prefix="/api/v1")
    app.include_router(recommendations_router, prefix="/api/v1")
    app.include_router(feedback_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")

    return app


# Module-level app instance for uvicorn: `uvicorn app.main:app`
app = create_app()
