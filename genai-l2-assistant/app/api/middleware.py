"""API middleware for authentication, logging, and webhook validation.

Provides:
- RBACMiddleware: validates engineer identity headers on every request.
- RequestLoggingMiddleware: structured request/response logging with latency.
- HMAC validation utility for ServiceNow webhook payloads.
"""

import hashlib
import hmac
import time
from typing import Optional

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import AppEnvironment, get_settings

logger = structlog.get_logger(__name__)

# Paths that bypass RBAC checks (health, metrics, docs)
_PUBLIC_PATHS: set[str] = {
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
}


# ── RBAC Middleware ─────────────────────────────────────────────────────────


class RBACMiddleware(BaseHTTPMiddleware):
    """Validate X-Engineer-Id and X-Engineer-Role headers.

    In development mode, headers are accepted as-is from the request
    (useful for local testing). In staging/production, headers are
    expected to be set by an upstream API gateway or auth proxy.

    Public paths (health, metrics, docs) bypass validation entirely.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request through RBAC validation.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            Response from downstream handler, or 401/403 on auth failure.
        """
        # Skip public endpoints
        if request.url.path in _PUBLIC_PATHS or request.url.path.startswith("/docs"):
            return await call_next(request)

        settings = get_settings()

        engineer_id: Optional[str] = request.headers.get("X-Engineer-Id")
        engineer_role: Optional[str] = request.headers.get("X-Engineer-Role")

        # In development, accept whatever headers are present (default to dev user)
        if settings.app_env == AppEnvironment.DEVELOPMENT:
            if not engineer_id:
                engineer_id = "dev-engineer-001"
            if not engineer_role:
                engineer_role = "l2_engineer"
            # Store on request state for downstream handlers
            request.state.engineer_id = engineer_id
            request.state.engineer_role = engineer_role
            return await call_next(request)

        # In staging/production, headers are mandatory
        if not engineer_id:
            logger.warning(
                "rbac_missing_engineer_id",
                path=request.url.path,
                method=request.method,
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing X-Engineer-Id header"},
            )

        if not engineer_role:
            logger.warning(
                "rbac_missing_engineer_role",
                path=request.url.path,
                engineer_id=engineer_id,
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing X-Engineer-Role header"},
            )

        # Validate role is a known value
        valid_roles = {"l2_engineer", "l3_engineer", "admin", "system"}
        if engineer_role not in valid_roles:
            logger.warning(
                "rbac_invalid_role",
                engineer_id=engineer_id,
                role=engineer_role,
            )
            return JSONResponse(
                status_code=403,
                content={"detail": f"Invalid role: {engineer_role}"},
            )

        request.state.engineer_id = engineer_id
        request.state.engineer_role = engineer_role

        logger.debug(
            "rbac_validated",
            engineer_id=engineer_id,
            role=engineer_role,
            path=request.url.path,
        )

        return await call_next(request)


# ── Request Logging Middleware ──────────────────────────────────────────────


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status code, and latency.

    Uses structlog for consistent structured logging across the application.
    Skips noisy endpoints like /health and /metrics to keep logs clean.
    """

    _SKIP_PATHS: set[str] = {"/health", "/metrics"}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Log request details and response timing.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The downstream response, unchanged.
        """
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        start_time = time.perf_counter()
        request_id = request.headers.get("X-Request-Id", "")

        # Bind contextual fields for this request
        log = logger.bind(
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else "",
            request_id=request_id,
            client_ip=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            log.error(
                "request_failed",
                latency_ms=latency_ms,
                status_code=500,
                exc_info=True,
            )
            raise

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        log_method = log.info if response.status_code < 400 else log.warning
        log_method(
            "request_completed",
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        # Add latency header for observability
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        return response


# ── HMAC Validation Utility ─────────────────────────────────────────────────


def validate_hmac_signature(
    payload: bytes,
    signature: str,
    secret: str,
    algorithm: str = "sha256",
) -> bool:
    """Validate HMAC signature for webhook payloads.

    ServiceNow sends a webhook signature in the X-ServiceNow-Signature
    header computed as HMAC-SHA256 of the raw request body.

    Args:
        payload: Raw request body bytes.
        signature: The HMAC signature from the request header.
        secret: The shared HMAC secret key.
        algorithm: Hash algorithm (default: sha256).

    Returns:
        True if the computed HMAC matches the provided signature.
    """
    if not signature or not secret:
        return False

    mac = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=getattr(hashlib, algorithm),
    )
    expected = mac.hexdigest()

    return hmac.compare_digest(expected, signature)
