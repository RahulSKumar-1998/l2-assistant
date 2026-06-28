"""Async ServiceNow REST API client for incident, KB, and CMDB operations.

Supports two authentication modes:
    - **OAuth 2.0** (client_credentials grant) — preferred for production
    - **Basic Auth** — fallback for development instances

Includes automatic token refresh on 401, and exponential backoff with
jitter on 429 (rate-limit) and 503 (service unavailable) via tenacity.

All methods return typed Pydantic models from ``app.models.incident``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import ServiceNowSettings, get_settings
from app.models.incident import (
    CMDBRecord,
    IncidentQueryParams,
    IncidentRecord,
    KBArticle,
    KBQueryParams,
)

logger = structlog.get_logger(__name__)


# ── Exceptions ───────────────────────────────────────────────────────────────


class ServiceNowError(Exception):
    """Base exception for ServiceNow client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class ServiceNowAuthError(ServiceNowError):
    """Authentication or authorization failure."""
    pass


class ServiceNowRateLimitError(ServiceNowError):
    """Rate-limited by ServiceNow (HTTP 429)."""
    pass


class ServiceNowTransientError(ServiceNowError):
    """Transient server error (HTTP 503, 502, etc.)."""
    pass


class ServiceNowNotFoundError(ServiceNowError):
    """Resource not found (HTTP 404)."""
    pass


# ── Retry predicate ──────────────────────────────────────────────────────────


_RETRYABLE = (ServiceNowRateLimitError, ServiceNowTransientError, httpx.ConnectError)


# ── ServiceNow REST Client ──────────────────────────────────────────────────


class ServiceNowClient:
    """Async ServiceNow REST API client.

    Handles authentication, token lifecycle, retries, and pagination
    against the ServiceNow Table API and Knowledge Management API.

    Example:
        >>> async with ServiceNowClient() as client:
        ...     incident = await client.get_incident("abc123sys_id")
        ...     incidents = await client.list_incidents(
        ...         IncidentQueryParams(state="2", limit=50)
        ...     )
    """

    # ServiceNow REST API paths
    _TABLE_API = "/api/now/table"
    _OAUTH_TOKEN_URL = "/oauth_token.do"

    def __init__(
        self,
        settings: Optional[ServiceNowSettings] = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the ServiceNow client.

        Args:
            settings: ServiceNow connection settings.
                Defaults to ``get_settings().servicenow``.
            timeout: HTTP request timeout in seconds.
        """
        self._settings = settings or get_settings().servicenow
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._log = logger.bind(
            component="servicenow_client",
            instance=self._settings.instance_url,
        )

    @property
    def base_url(self) -> str:
        """ServiceNow instance base URL."""
        return self._settings.instance_url.rstrip("/")

    async def __aenter__(self) -> ServiceNowClient:
        """Async context manager entry — create httpx client."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit — close httpx client."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create the underlying httpx.AsyncClient.

        Returns:
            The httpx async client.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=True,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            self._log.info("servicenow_client_closed")

    # ── Authentication ───────────────────────────────────────────────────

    async def _get_oauth_token(self) -> str:
        """Obtain an OAuth 2.0 access token using client_credentials grant.

        Returns:
            Access token string.

        Raises:
            ServiceNowAuthError: If token acquisition fails.
        """
        client = await self._ensure_client()
        try:
            response = await client.post(
                self._OAUTH_TOKEN_URL,
                data={
                    "grant_type": "password",
                    "client_id": self._settings.client_id,
                    "client_secret": self._settings.client_secret,
                    "username": self._settings.username,
                    "password": self._settings.password,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data["access_token"]
            expires_in = int(data.get("expires_in", 1800))
            # Refresh 60 seconds early to avoid clock-skew issues
            self._token_expiry = datetime.now(timezone.utc).replace(
                tzinfo=None
            ).__class__.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + expires_in - 60,
                tz=timezone.utc,
            )

            self._log.info("oauth_token_acquired", expires_in=expires_in)
            return self._access_token

        except httpx.HTTPStatusError as exc:
            raise ServiceNowAuthError(
                f"OAuth token request failed: {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from exc
        except Exception as exc:
            raise ServiceNowAuthError(
                f"OAuth token request failed: {exc}"
            ) from exc

    def _is_token_expired(self) -> bool:
        """Check if the current OAuth token is expired or absent."""
        if self._access_token is None or self._token_expiry is None:
            return True
        return datetime.now(timezone.utc) >= self._token_expiry

    async def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers based on configured auth mode.

        Returns:
            Dict with Authorization header.
        """
        if self._settings.use_oauth:
            if self._is_token_expired():
                await self._get_oauth_token()
            return {"Authorization": f"Bearer {self._access_token}"}
        else:
            # Basic auth
            import base64
            creds = base64.b64encode(
                f"{self._settings.username}:{self._settings.password}".encode()
            ).decode()
            return {"Authorization": f"Basic {creds}"}

    # ── HTTP Layer ───────────────────────────────────────────────────────

    def _classify_error(self, exc: httpx.HTTPStatusError) -> ServiceNowError:
        """Map HTTP status codes to typed exceptions.

        Args:
            exc: The httpx status error.

        Returns:
            Appropriately typed ServiceNowError subclass.
        """
        code = exc.response.status_code
        body = exc.response.text[:500]

        if code == 401:
            return ServiceNowAuthError(
                f"Authentication failed: {body}", status_code=code
            )
        if code == 404:
            return ServiceNowNotFoundError(
                f"Resource not found: {body}", status_code=code
            )
        if code == 429:
            return ServiceNowRateLimitError(
                f"Rate limited: {body}", status_code=code
            )
        if code in (502, 503, 504):
            return ServiceNowTransientError(
                f"Transient error ({code}): {body}", status_code=code
            )
        return ServiceNowError(
            f"ServiceNow API error ({code}): {body}", status_code=code
        )

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=1, max=30, jitter=5),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        _retry_on_401: bool = True,
    ) -> dict[str, Any]:
        """Execute an authenticated HTTP request to ServiceNow.

        Handles 401 by refreshing the token and retrying once.
        Translates HTTP errors to typed exceptions.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH).
            path: API path (appended to base_url).
            params: Query parameters.
            json_body: JSON request body.
            _retry_on_401: Whether to retry on 401 (prevents infinite loop).

        Returns:
            Parsed JSON response body.

        Raises:
            ServiceNowError: On API errors after retries.
        """
        client = await self._ensure_client()
        auth_headers = await self._get_auth_headers()

        try:
            response = await client.request(
                method,
                path,
                params=params,
                json=json_body,
                headers=auth_headers,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as exc:
            # Auto-retry on 401 with token refresh
            if exc.response.status_code == 401 and _retry_on_401:
                self._log.warning("token_expired_refreshing")
                self._access_token = None
                self._token_expiry = None
                return await self._request(
                    method, path,
                    params=params,
                    json_body=json_body,
                    _retry_on_401=False,
                )
            raise self._classify_error(exc) from exc

    # ── Incident Operations ──────────────────────────────────────────────

    async def get_incident(self, sys_id: str) -> IncidentRecord:
        """Fetch a single incident by sys_id.

        Args:
            sys_id: ServiceNow incident sys_id.

        Returns:
            IncidentRecord with full incident data.

        Raises:
            ServiceNowNotFoundError: If the incident doesn't exist.
        """
        self._log.info("fetching_incident", sys_id=sys_id)
        data = await self._request(
            "GET",
            f"{self._TABLE_API}/incident/{sys_id}",
            params={
                "sysparm_display_value": "true",
                "sysparm_exclude_reference_link": "true",
            },
        )
        record = data.get("result", {})
        return self._parse_incident(record)

    async def list_incidents(
        self,
        query_params: Optional[IncidentQueryParams] = None,
    ) -> list[IncidentRecord]:
        """List incidents from ServiceNow with optional filters.

        Args:
            query_params: Filter and pagination parameters.

        Returns:
            List of IncidentRecord objects.
        """
        params = query_params or IncidentQueryParams()
        self._log.info(
            "listing_incidents",
            limit=params.limit,
            offset=params.offset,
        )

        # Build ServiceNow encoded query
        query_parts: list[str] = []
        if params.state:
            query_parts.append(f"state={params.state}")
        if params.assignment_group:
            query_parts.append(f"assignment_group.name={params.assignment_group}")
        if params.category:
            query_parts.append(f"category={params.category}")
        if params.opened_at_start:
            dt_str = params.opened_at_start.strftime("%Y-%m-%d %H:%M:%S")
            query_parts.append(f"opened_at>={dt_str}")
        if params.opened_at_end:
            dt_str = params.opened_at_end.strftime("%Y-%m-%d %H:%M:%S")
            query_parts.append(f"opened_at<={dt_str}")

        api_params: dict[str, Any] = {
            "sysparm_limit": params.limit,
            "sysparm_offset": params.offset,
            "sysparm_display_value": "true",
            "sysparm_exclude_reference_link": "true",
        }
        if query_parts:
            api_params["sysparm_query"] = "^".join(query_parts)

        data = await self._request(
            "GET",
            f"{self._TABLE_API}/incident",
            params=api_params,
        )
        records = data.get("result", [])
        return [self._parse_incident(r) for r in records]

    # ── KB Article Operations ────────────────────────────────────────────

    async def list_kb_articles(
        self,
        query_params: Optional[KBQueryParams] = None,
    ) -> list[KBArticle]:
        """List knowledge base articles from ServiceNow.

        Args:
            query_params: Filter and pagination parameters.

        Returns:
            List of KBArticle objects.
        """
        params = query_params or KBQueryParams()
        self._log.info("listing_kb_articles", limit=params.limit)

        query_parts: list[str] = []
        if params.workflow_state:
            query_parts.append(f"workflow_state={params.workflow_state}")
        if params.category:
            query_parts.append(f"kb_category={params.category}")

        api_params: dict[str, Any] = {
            "sysparm_limit": params.limit,
            "sysparm_offset": params.offset,
            "sysparm_display_value": "true",
        }
        if query_parts:
            api_params["sysparm_query"] = "^".join(query_parts)

        data = await self._request(
            "GET",
            f"{self._TABLE_API}/kb_knowledge",
            params=api_params,
        )
        records = data.get("result", [])
        return [self._parse_kb_article(r) for r in records]

    # ── CMDB Operations ──────────────────────────────────────────────────

    async def get_cmdb_ci(self, sys_id: str) -> CMDBRecord:
        """Fetch a CMDB Configuration Item with relationships.

        Args:
            sys_id: CMDB CI sys_id.

        Returns:
            CMDBRecord with CI data and relationships.

        Raises:
            ServiceNowNotFoundError: If the CI doesn't exist.
        """
        self._log.info("fetching_cmdb_ci", sys_id=sys_id)

        # Fetch the CI
        data = await self._request(
            "GET",
            f"{self._TABLE_API}/cmdb_ci/{sys_id}",
            params={
                "sysparm_display_value": "true",
                "sysparm_exclude_reference_link": "true",
            },
        )
        ci_record = data.get("result", {})

        # Fetch relationships
        rel_data = await self._request(
            "GET",
            f"{self._TABLE_API}/cmdb_rel_ci",
            params={
                "sysparm_query": f"parent={sys_id}^ORchild={sys_id}",
                "sysparm_display_value": "true",
                "sysparm_limit": 50,
            },
        )
        relationships = [
            {
                "type": r.get("type", {}).get("display_value", "") if isinstance(r.get("type"), dict) else r.get("type", ""),
                "parent": r.get("parent", {}).get("display_value", "") if isinstance(r.get("parent"), dict) else r.get("parent", ""),
                "child": r.get("child", {}).get("display_value", "") if isinstance(r.get("child"), dict) else r.get("child", ""),
            }
            for r in rel_data.get("result", [])
        ]

        return CMDBRecord(
            sys_id=ci_record.get("sys_id", sys_id),
            name=ci_record.get("name", ""),
            sys_class_name=ci_record.get("sys_class_name", ""),
            operational_status=ci_record.get("operational_status", "1"),
            environment=ci_record.get("u_environment", ci_record.get("environment", "")),
            service_tier=ci_record.get("u_service_tier", ""),
            relationships=relationships,
        )

    # ── Incident Update Operations ───────────────────────────────────────

    async def update_incident_work_note(
        self,
        sys_id: str,
        work_note: str,
    ) -> IncidentRecord:
        """Add a work note to an incident.

        Args:
            sys_id: Incident sys_id.
            work_note: Work note text to append.

        Returns:
            Updated IncidentRecord.
        """
        self._log.info("updating_work_note", sys_id=sys_id)
        data = await self._request(
            "PATCH",
            f"{self._TABLE_API}/incident/{sys_id}",
            json_body={"work_notes": work_note},
        )
        record = data.get("result", {})
        return self._parse_incident(record)

    async def update_incident_resolution(
        self,
        sys_id: str,
        resolution_notes: str,
        resolution_code: str = "Solved (Permanently)",
    ) -> IncidentRecord:
        """Resolve an incident with resolution notes.

        Args:
            sys_id: Incident sys_id.
            resolution_notes: Resolution description.
            resolution_code: Resolution code (default: Solved Permanently).

        Returns:
            Updated IncidentRecord.
        """
        self._log.info("resolving_incident", sys_id=sys_id)
        data = await self._request(
            "PATCH",
            f"{self._TABLE_API}/incident/{sys_id}",
            json_body={
                "state": "6",  # Resolved
                "close_notes": resolution_notes,
                "close_code": resolution_code,
            },
        )
        record = data.get("result", {})
        return self._parse_incident(record)

    # ── Parsing Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        """Parse a ServiceNow datetime string.

        Args:
            value: Date string or None.

        Returns:
            Parsed datetime or None.
        """
        if not value:
            return None
        try:
            # ServiceNow format: 2024-01-15 08:30:00
            return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            try:
                return datetime.fromisoformat(str(value))
            except (ValueError, TypeError):
                return None

    @staticmethod
    def _parse_incident(record: dict[str, Any]) -> IncidentRecord:
        """Parse a ServiceNow incident JSON record.

        Args:
            record: Raw JSON dict from ServiceNow.

        Returns:
            Typed IncidentRecord.
        """
        return IncidentRecord(
            sys_id=record.get("sys_id", ""),
            number=record.get("number", ""),
            short_description=record.get("short_description", ""),
            description=record.get("description", ""),
            category=record.get("category", ""),
            subcategory=record.get("subcategory", ""),
            priority=int(record.get("priority", 4) or 4),
            state=record.get("state", "1"),
            assignment_group=record.get("assignment_group", ""),
            assigned_to=record.get("assigned_to", ""),
            cmdb_ci=record.get("cmdb_ci", ""),
            opened_at=ServiceNowClient._parse_datetime(record.get("opened_at")),
            resolved_at=ServiceNowClient._parse_datetime(record.get("resolved_at")),
            work_notes=record.get("work_notes", ""),
            resolution_notes=record.get("close_notes") or record.get("resolution_notes"),
            root_cause=record.get("u_root_cause"),
        )

    @staticmethod
    def _parse_kb_article(record: dict[str, Any]) -> KBArticle:
        """Parse a ServiceNow KB article JSON record.

        Args:
            record: Raw JSON dict from ServiceNow.

        Returns:
            Typed KBArticle.
        """
        return KBArticle(
            sys_id=record.get("sys_id", ""),
            number=record.get("number", ""),
            short_description=record.get("short_description", ""),
            text=record.get("text", ""),
            category=record.get("kb_category", record.get("category", "")),
            valid_to=ServiceNowClient._parse_datetime(record.get("valid_to")),
            workflow_state=record.get("workflow_state", "published"),
        )
