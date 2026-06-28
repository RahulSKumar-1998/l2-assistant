"""Integration tests for the ServiceNow REST API client.

These tests require a live ServiceNow instance and valid credentials.
They are skipped by default in CI and local development.

Run with: pytest tests/integration/test_servicenow_client.py -v --run-integration
"""

import os

import pytest

from app.models.incident import IncidentQueryParams, IncidentRecord, KBArticle


# Skip all tests in this module unless --run-integration flag is passed
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION_TESTS", "").lower() != "true",
        reason=(
            "Integration tests disabled. Set RUN_INTEGRATION_TESTS=true "
            "and configure SNOW_* environment variables to run."
        ),
    ),
]


@pytest.fixture
def snow_instance_url() -> str:
    """Get ServiceNow instance URL from environment."""
    url = os.environ.get("SNOW_INSTANCE_URL", "")
    if not url:
        pytest.skip("SNOW_INSTANCE_URL not configured")
    return url


@pytest.fixture
def snow_credentials() -> dict[str, str]:
    """Get ServiceNow credentials from environment."""
    username = os.environ.get("SNOW_USERNAME", "")
    password = os.environ.get("SNOW_PASSWORD", "")
    if not username or not password:
        pytest.skip("SNOW_USERNAME and SNOW_PASSWORD not configured")
    return {"username": username, "password": password}


class TestServiceNowConnection:
    """Tests for basic ServiceNow API connectivity."""

    @pytest.mark.asyncio
    async def test_health_check(
        self,
        snow_instance_url: str,
        snow_credentials: dict[str, str],
    ) -> None:
        """Verify connectivity to the ServiceNow instance.

        Should successfully authenticate and receive a valid response
        from the instance health/stats endpoint.
        """
        # TODO: Instantiate ServiceNowClient and call health check
        # client = ServiceNowClient(
        #     instance_url=snow_instance_url,
        #     username=snow_credentials["username"],
        #     password=snow_credentials["password"],
        # )
        # result = await client.health_check()
        # assert result["status"] == "ok"
        pytest.skip("ServiceNow client not yet implemented")


class TestIncidentRetrieval:
    """Tests for incident CRUD operations against ServiceNow."""

    @pytest.mark.asyncio
    async def test_list_incidents(
        self,
        snow_instance_url: str,
        snow_credentials: dict[str, str],
    ) -> None:
        """Verify incident listing with pagination and filters.

        Should return a list of IncidentRecord objects matching the
        query parameters.
        """
        # TODO: Test actual incident listing
        # client = ServiceNowClient(...)
        # params = IncidentQueryParams(limit=5, state="2")
        # incidents = await client.list_incidents(params)
        # assert isinstance(incidents, list)
        # assert len(incidents) <= 5
        # for inc in incidents:
        #     assert isinstance(inc, IncidentRecord)
        #     assert inc.number.startswith("INC")
        pytest.skip("ServiceNow client not yet implemented")

    @pytest.mark.asyncio
    async def test_get_incident_by_sys_id(
        self,
        snow_instance_url: str,
        snow_credentials: dict[str, str],
    ) -> None:
        """Verify single incident retrieval by sys_id.

        Should return a fully populated IncidentRecord.
        """
        # TODO: Test incident retrieval by sys_id
        pytest.skip("ServiceNow client not yet implemented")


class TestKBArticleRetrieval:
    """Tests for Knowledge Base article retrieval."""

    @pytest.mark.asyncio
    async def test_list_kb_articles(
        self,
        snow_instance_url: str,
        snow_credentials: dict[str, str],
    ) -> None:
        """Verify KB article listing with filters.

        Should return published KB articles matching query criteria.
        """
        # TODO: Test KB article listing
        pytest.skip("ServiceNow client not yet implemented")

    @pytest.mark.asyncio
    async def test_get_kb_article_content(
        self,
        snow_instance_url: str,
        snow_credentials: dict[str, str],
    ) -> None:
        """Verify full KB article content retrieval.

        Should return article with HTML body text that can be
        processed by the text cleaning pipeline.
        """
        # TODO: Test KB article content retrieval
        pytest.skip("ServiceNow client not yet implemented")


class TestWebhookValidation:
    """Tests for ServiceNow webhook HMAC signature validation."""

    @pytest.mark.asyncio
    async def test_valid_webhook_signature(self) -> None:
        """Valid HMAC signature should be accepted.

        The webhook handler should validate the X-ServiceNow-Signature
        header against the configured webhook secret.
        """
        # TODO: Test webhook signature validation
        pytest.skip("Webhook handler not yet implemented")

    @pytest.mark.asyncio
    async def test_invalid_webhook_signature_rejected(self) -> None:
        """Invalid HMAC signature should be rejected with 401.

        Webhook payloads with incorrect or missing signatures should
        be rejected to prevent unauthorized data injection.
        """
        # TODO: Test invalid signature rejection
        pytest.skip("Webhook handler not yet implemented")
