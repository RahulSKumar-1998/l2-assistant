"""Mock ServiceNow client for development and testing.

Returns realistic fixture data without requiring a live ServiceNow
instance. Mirrors the ``ServiceNowClient`` interface exactly so it
can be used as a drop-in replacement via dependency injection.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import structlog

from app.models.incident import (
    CMDBRecord,
    IncidentQueryParams,
    IncidentRecord,
    KBArticle,
    KBQueryParams,
)

logger = structlog.get_logger(__name__)


# ── Fixture Data ─────────────────────────────────────────────────────────────


_FIXTURE_INCIDENTS: list[dict[str, Any]] = [
    {
        "sys_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
        "number": "INC0042871",
        "short_description": "Production API gateway returning 502 errors intermittently",
        "description": (
            "The API gateway (kong-prod-01) has been returning HTTP 502 errors "
            "for approximately 15% of requests since 06:30 UTC. Affected services "
            "include payment-service and user-auth-service. Load balancer health "
            "checks are passing but upstream connections are timing out. "
            "Assigned to: Jane Smith. Reported by: Bob Johnson. "
            "Contact: ops-team@acme-corp.internal"
        ),
        "category": "Network",
        "subcategory": "DNS",
        "priority": 1,
        "state": "2",
        "assignment_group": "L2 Cloud Infrastructure",
        "assigned_to": "Jane Smith",
        "cmdb_ci": "kong-prod-01",
        "opened_at": datetime(2024, 6, 15, 6, 30, 0),
        "resolved_at": None,
        "work_notes": (
            "2024-06-15 07:00: Checked Kong proxy logs — upstream_connect_timeout "
            "errors increasing. Backend pods are all healthy per k8s.\n"
            "2024-06-15 07:30: Traced issue to connection pool exhaustion on "
            "payment-service sidecar. Envoy proxy maxing at 1024 connections."
        ),
        "resolution_notes": None,
        "root_cause": None,
    },
    {
        "sys_id": "b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5",
        "number": "INC0042872",
        "short_description": "Elasticsearch cluster red status — primary shards unassigned",
        "description": (
            "Production Elasticsearch cluster 'es-logs-prod' entered RED status "
            "at 14:20 UTC. 3 primary shards unassigned on index 'app-logs-2024.06'. "
            "Data node es-data-07 went offline after OOM kill. Disk usage at 89%. "
            "Impact: Log aggregation pipeline stalled, Kibana dashboards unavailable. "
            "Assigned to: Mike Chen. Reported by: Sarah Williams."
        ),
        "category": "Software",
        "subcategory": "Database",
        "priority": 2,
        "state": "6",
        "assignment_group": "L2 Data Platform",
        "assigned_to": "Mike Chen",
        "cmdb_ci": "es-logs-prod",
        "opened_at": datetime(2024, 6, 14, 14, 20, 0),
        "resolved_at": datetime(2024, 6, 14, 16, 45, 0),
        "work_notes": (
            "2024-06-14 14:30: Confirmed es-data-07 OOM killed. JVM heap set to 16GB "
            "but field data cache consuming 12GB.\n"
            "2024-06-14 15:00: Cleared field data cache, restarted node.\n"
            "2024-06-14 15:30: Shards reassigning. Added index.fielddata.cache.size: 40%."
        ),
        "resolution_notes": (
            "Root cause: Unbounded fielddata cache on high-cardinality field "
            "'request_id'. Fixed by setting fielddata cache limit to 40% and "
            "adding circuit breaker. Also increased heap to 24GB and added "
            "monitoring alert for fielddata usage > 60%."
        ),
        "root_cause": "Unbounded fielddata cache on high-cardinality text field",
    },
    {
        "sys_id": "c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6",
        "number": "INC0042873",
        "short_description": "SSL certificate expired on customer-facing portal",
        "description": (
            "The SSL/TLS certificate for portal.acme-corp.com expired at midnight. "
            "All customer-facing HTTPS traffic is failing with ERR_CERT_DATE_INVALID. "
            "Certificate was issued by DigiCert, auto-renewal failed silently. "
            "Assigned to: Alex Rodriguez. Reported by: Support Desk."
        ),
        "category": "Security",
        "subcategory": "Certificate",
        "priority": 1,
        "state": "6",
        "assignment_group": "L2 Security Operations",
        "assigned_to": "Alex Rodriguez",
        "cmdb_ci": "portal.acme-corp.com",
        "opened_at": datetime(2024, 6, 13, 0, 5, 0),
        "resolved_at": datetime(2024, 6, 13, 1, 30, 0),
        "work_notes": (
            "2024-06-13 00:10: Confirmed cert expired. Checking ACME/certbot renewal.\n"
            "2024-06-13 00:25: Auto-renewal failed — DNS-01 challenge failing due to "
            "stale API key for Route53.\n"
            "2024-06-13 00:45: Rotated Route53 API key, triggered manual renewal.\n"
            "2024-06-13 01:15: New cert deployed via Kubernetes cert-manager."
        ),
        "resolution_notes": (
            "SSL cert auto-renewal was failing silently because the Route53 API "
            "key used by cert-manager had been rotated but not updated in the "
            "Kubernetes secret. Renewed key, triggered manual cert renewal, "
            "and added monitoring for cert expiry < 14 days."
        ),
        "root_cause": "Stale Route53 API key preventing DNS-01 ACME challenge",
    },
    {
        "sys_id": "d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1",
        "number": "INC0042874",
        "short_description": "Kafka consumer lag increasing on order-processing topic",
        "description": (
            "Consumer group 'order-processor-prod' lag has been steadily increasing "
            "on topic 'orders.placed' since 10:00 UTC. Current lag: ~450k messages. "
            "Consumer instances are running but throughput has dropped by 70%. "
            "No recent deployments. Assigned to: Priya Patel."
        ),
        "category": "Software",
        "subcategory": "Middleware",
        "priority": 2,
        "state": "2",
        "assignment_group": "L2 Application Support",
        "assigned_to": "Priya Patel",
        "cmdb_ci": "kafka-prod-cluster",
        "opened_at": datetime(2024, 6, 15, 10, 0, 0),
        "resolved_at": None,
        "work_notes": (
            "2024-06-15 10:20: Confirmed lag on partitions 0-5. Consumer JVM GC "
            "pressure high — G1 old gen collections every 2 seconds.\n"
            "2024-06-15 10:45: Heap dump shows large HashMap in deserialization layer. "
            "Suspect schema change in producer caused excessive object creation."
        ),
        "resolution_notes": None,
        "root_cause": None,
    },
    {
        "sys_id": "e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "number": "INC0042875",
        "short_description": "Jenkins CI/CD pipeline failing on all branches",
        "description": (
            "All Jenkins pipelines are failing at the 'docker build' stage with "
            "'no space left on device' errors. The Jenkins master node disk is "
            "at 98% capacity. Old build artifacts and Docker images have not been "
            "cleaned up. Assigned to: Tom Walker."
        ),
        "category": "Software",
        "subcategory": "Build/Deploy",
        "priority": 3,
        "state": "6",
        "assignment_group": "L2 DevOps",
        "assigned_to": "Tom Walker",
        "cmdb_ci": "jenkins-master-01",
        "opened_at": datetime(2024, 6, 12, 9, 0, 0),
        "resolved_at": datetime(2024, 6, 12, 10, 30, 0),
        "work_notes": (
            "2024-06-12 09:15: Disk at 98%. /var/jenkins_home/workspace is 180GB.\n"
            "2024-06-12 09:30: docker system prune freed 45GB.\n"
            "2024-06-12 09:45: Cleaned old workspace directories (>30 days)."
        ),
        "resolution_notes": (
            "Cleared 75GB of old Docker images and Jenkins workspaces. "
            "Set up cron job for weekly docker system prune and workspace cleanup. "
            "Added disk usage monitoring with alert at 80%."
        ),
        "root_cause": "No automated cleanup of Docker images and Jenkins workspaces",
    },
]


_FIXTURE_KB_ARTICLES: list[dict[str, Any]] = [
    {
        "sys_id": "kb_001_sys_id_placeholder",
        "number": "KB0012345",
        "short_description": "Troubleshooting API Gateway 502 Errors",
        "text": (
            "<h2>Overview</h2>"
            "<p>This article covers common causes of HTTP 502 errors from "
            "API gateways (Kong, NGINX, AWS ALB) and resolution steps.</p>"
            "<h2>Common Causes</h2>"
            "<ul>"
            "<li>Upstream service timeout</li>"
            "<li>Connection pool exhaustion</li>"
            "<li>DNS resolution failures</li>"
            "<li>Backend health check misconfiguration</li>"
            "</ul>"
            "<h2>Resolution Steps</h2>"
            "<p>1. Check upstream service health and logs. "
            "2. Verify connection pool settings (max connections, idle timeout). "
            "3. Test DNS resolution from the gateway node. "
            "4. Review load balancer target group health check configuration.</p>"
            "<h2>Prevention</h2>"
            "<p>Set up monitoring for upstream connect latency and connection pool "
            "utilization. Configure circuit breakers with appropriate thresholds.</p>"
        ),
        "category": "Network",
        "workflow_state": "published",
    },
    {
        "sys_id": "kb_002_sys_id_placeholder",
        "number": "KB0012346",
        "short_description": "Elasticsearch Cluster Recovery Procedures",
        "text": (
            "<h2>Overview</h2>"
            "<p>Procedures for recovering Elasticsearch clusters from RED or "
            "YELLOW status, including shard reallocation and data recovery.</p>"
            "<h2>RED Status Recovery</h2>"
            "<p>1. Identify unassigned shards: GET _cluster/health?level=shards. "
            "2. Check allocation explain: POST _cluster/allocation/explain. "
            "3. If disk-based, clear old indices or add storage. "
            "4. Force allocate if needed: POST _cluster/reroute.</p>"
            "<h2>YELLOW Status Recovery</h2>"
            "<p>YELLOW means replicas are unassigned. Usually resolves when nodes "
            "rejoin. Check with GET _cat/allocation for disk watermark issues.</p>"
            "<h2>Prevention</h2>"
            "<p>Monitor disk watermarks (low: 85%, high: 90%, flood: 95%). "
            "Set up ILM policies for automatic rollover and deletion.</p>"
        ),
        "category": "Software",
        "workflow_state": "published",
    },
    {
        "sys_id": "kb_003_sys_id_placeholder",
        "number": "KB0012347",
        "short_description": "SSL/TLS Certificate Management and Renewal",
        "text": (
            "<h2>Overview</h2>"
            "<p>Guide for managing SSL/TLS certificates across the infrastructure, "
            "including automated renewal with cert-manager and manual procedures.</p>"
            "<h2>Automated Renewal (cert-manager)</h2>"
            "<p>Our Kubernetes clusters use cert-manager with DNS-01 challenges via "
            "Route53. Ensure the IAM credentials in the cert-manager secret are valid. "
            "Check cert-manager logs: kubectl logs -n cert-manager deployment/cert-manager.</p>"
            "<h2>Manual Renewal</h2>"
            "<p>1. Generate CSR. 2. Submit to DigiCert portal. 3. Complete DCV. "
            "4. Download and deploy certificate bundle.</p>"
            "<h2>Monitoring</h2>"
            "<p>Prometheus blackbox_exporter checks certificate validity. "
            "Alert if expiry < 14 days. Dashboard: Grafana > SSL Certificates.</p>"
        ),
        "category": "Security",
        "workflow_state": "published",
    },
]


_FIXTURE_CMDB_CIS: dict[str, dict[str, Any]] = {
    "kong-prod-01": {
        "sys_id": "cmdb_kong_sys_id",
        "name": "kong-prod-01",
        "sys_class_name": "cmdb_ci_service",
        "operational_status": "1",
        "environment": "production",
        "service_tier": "tier-1",
        "relationships": [
            {"type": "Depends on", "parent": "kong-prod-01", "child": "payment-service"},
            {"type": "Depends on", "parent": "kong-prod-01", "child": "user-auth-service"},
            {"type": "Hosted on", "parent": "kong-prod-01", "child": "k8s-prod-cluster"},
        ],
    },
    "es-logs-prod": {
        "sys_id": "cmdb_es_sys_id",
        "name": "es-logs-prod",
        "sys_class_name": "cmdb_ci_service",
        "operational_status": "1",
        "environment": "production",
        "service_tier": "tier-2",
        "relationships": [
            {"type": "Depends on", "parent": "es-logs-prod", "child": "logstash-prod"},
            {"type": "Used by", "parent": "kibana-prod", "child": "es-logs-prod"},
        ],
    },
    "kafka-prod-cluster": {
        "sys_id": "cmdb_kafka_sys_id",
        "name": "kafka-prod-cluster",
        "sys_class_name": "cmdb_ci_cluster",
        "operational_status": "1",
        "environment": "production",
        "service_tier": "tier-1",
        "relationships": [
            {"type": "Used by", "parent": "order-processor-prod", "child": "kafka-prod-cluster"},
            {"type": "Depends on", "parent": "kafka-prod-cluster", "child": "zookeeper-prod"},
        ],
    },
}


# ── Mock Client ──────────────────────────────────────────────────────────────


class MockServiceNowClient:
    """Mock ServiceNow client that returns realistic fixture data.

    Drop-in replacement for ``ServiceNowClient`` in development
    and testing. Uses pre-defined fixture data and supports the
    same async context manager protocol.

    Example:
        >>> async with MockServiceNowClient() as client:
        ...     incident = await client.get_incident("a1b2c3d4...")
        ...     articles = await client.list_kb_articles()
    """

    def __init__(self) -> None:
        """Initialize the mock client."""
        self._log = logger.bind(component="mock_servicenow_client")
        self._incidents = {inc["sys_id"]: inc for inc in _FIXTURE_INCIDENTS}
        self._log.info("mock_client_initialized", incident_count=len(self._incidents))

    async def __aenter__(self) -> MockServiceNowClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        pass

    async def close(self) -> None:
        """No-op close for API compatibility."""
        self._log.debug("mock_client_closed")

    async def get_incident(self, sys_id: str) -> IncidentRecord:
        """Get a mock incident by sys_id.

        Args:
            sys_id: Incident sys_id.

        Returns:
            IncidentRecord from fixtures.

        Raises:
            KeyError: If sys_id not found in fixtures.
        """
        self._log.info("mock_get_incident", sys_id=sys_id)
        record = self._incidents.get(sys_id)
        if record is None:
            # Return the first fixture as a fallback
            record = _FIXTURE_INCIDENTS[0]
            self._log.warning("mock_incident_not_found_using_fallback", sys_id=sys_id)

        return IncidentRecord(**record)

    async def list_incidents(
        self,
        query_params: Optional[IncidentQueryParams] = None,
    ) -> list[IncidentRecord]:
        """List mock incidents with optional filtering.

        Args:
            query_params: Filter and pagination parameters.

        Returns:
            Filtered list of IncidentRecord objects.
        """
        params = query_params or IncidentQueryParams()
        self._log.info("mock_list_incidents", limit=params.limit)

        results = list(_FIXTURE_INCIDENTS)

        # Apply filters
        if params.state:
            results = [r for r in results if r["state"] == params.state]
        if params.category:
            results = [r for r in results if r["category"].lower() == params.category.lower()]
        if params.assignment_group:
            results = [
                r for r in results
                if params.assignment_group.lower() in r["assignment_group"].lower()
            ]

        # Apply pagination
        start = params.offset
        end = start + params.limit
        results = results[start:end]

        return [IncidentRecord(**r) for r in results]

    async def list_kb_articles(
        self,
        query_params: Optional[KBQueryParams] = None,
    ) -> list[KBArticle]:
        """List mock KB articles.

        Args:
            query_params: Filter and pagination parameters.

        Returns:
            List of KBArticle objects from fixtures.
        """
        params = query_params or KBQueryParams()
        self._log.info("mock_list_kb_articles", limit=params.limit)

        results = list(_FIXTURE_KB_ARTICLES)

        if params.category:
            results = [r for r in results if r["category"].lower() == params.category.lower()]

        start = params.offset
        end = start + params.limit
        results = results[start:end]

        return [
            KBArticle(
                sys_id=r["sys_id"],
                number=r["number"],
                short_description=r["short_description"],
                text=r["text"],
                category=r["category"],
                workflow_state=r["workflow_state"],
            )
            for r in results
        ]

    async def get_cmdb_ci(self, sys_id: str) -> CMDBRecord:
        """Get a mock CMDB CI record.

        Args:
            sys_id: CI sys_id or name.

        Returns:
            CMDBRecord from fixtures.
        """
        self._log.info("mock_get_cmdb_ci", sys_id=sys_id)

        # Try looking up by name (since fixture incidents reference CI by name)
        ci = _FIXTURE_CMDB_CIS.get(sys_id)
        if ci is None:
            # Try looking up by sys_id value
            for name, data in _FIXTURE_CMDB_CIS.items():
                if data["sys_id"] == sys_id:
                    ci = data
                    break

        if ci is None:
            # Return a generic CI
            self._log.warning("mock_cmdb_ci_not_found", sys_id=sys_id)
            return CMDBRecord(
                sys_id=sys_id,
                name=sys_id,
                sys_class_name="cmdb_ci_server",
                operational_status="1",
                environment="production",
                service_tier="tier-3",
                relationships=[],
            )

        return CMDBRecord(**ci)

    async def update_incident_work_note(
        self,
        sys_id: str,
        work_note: str,
    ) -> IncidentRecord:
        """Mock adding a work note to an incident.

        Args:
            sys_id: Incident sys_id.
            work_note: Work note text.

        Returns:
            Updated IncidentRecord (simulated).
        """
        self._log.info("mock_update_work_note", sys_id=sys_id, note_length=len(work_note))
        incident = await self.get_incident(sys_id)
        # Simulate appending work note
        updated_notes = f"{incident.work_notes}\n{work_note}" if incident.work_notes else work_note
        return incident.model_copy(update={"work_notes": updated_notes})

    async def update_incident_resolution(
        self,
        sys_id: str,
        resolution_notes: str,
        resolution_code: str = "Solved (Permanently)",
    ) -> IncidentRecord:
        """Mock resolving an incident.

        Args:
            sys_id: Incident sys_id.
            resolution_notes: Resolution description.
            resolution_code: Resolution code.

        Returns:
            Updated IncidentRecord with resolved state.
        """
        self._log.info("mock_resolve_incident", sys_id=sys_id)
        incident = await self.get_incident(sys_id)
        return incident.model_copy(
            update={
                "state": "6",
                "resolution_notes": resolution_notes,
                "resolved_at": datetime.now(timezone.utc),
            }
        )
