"""Seed synthetic test data for local development.

Generates 10 realistic incidents and 5 KB articles, then inserts them
into the PostgreSQL database for local development and testing.

Usage:
    python scripts/seed_test_data.py
    python scripts/seed_test_data.py --skip-db  # Print data only, no DB writes
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from uuid import uuid4, uuid5, NAMESPACE_DNS

import structlog

logger = structlog.get_logger(__name__)


# ── Synthetic Incident Data ─────────────────────────────────────────────────

SYNTHETIC_INCIDENTS: list[dict] = [
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050001").hex,
        "number": "INC0050001",
        "short_description": "Payment gateway returning 502 errors in production",
        "description": "The payment-service has been returning intermittent HTTP 502 Bad Gateway errors since the v2.4.1 deployment. Approximately 15% of checkout requests are failing. Connection pool metrics show exhaustion: max_connections=50, active=50, idle=0.",
        "category": "application",
        "subcategory": "availability",
        "priority": 1,
        "state": "1", # New
        "assignment_group": "L2-Application-Support",
        "cmdb_ci": "payment-service",
        "resolution_notes": "Increased HikariCP max pool size from 50 to 100. Root cause was a connection leak in v2.4.1 PaymentGatewayClient. Hotfix deployed in v2.4.2.",
        "root_cause": "Connection leak in PaymentGatewayClient introduced in v2.4.1",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050002").hex,
        "number": "INC0050002",
        "short_description": "Order processing service OOM crash in production",
        "description": "The order-processing service pods are being OOM-killed repeatedly. Container memory limit is 2Gi but usage spikes to 3.5Gi during batch processing. Started after enabling the new bulk-order feature flag.",
        "category": "application",
        "subcategory": "performance",
        "priority": 1,
        "state": "6", # Resolved
        "assignment_group": "L2-Application-Support",
        "cmdb_ci": "order-service",
        "resolution_notes": "Disabled bulk-order feature flag as immediate mitigation. Fixed memory leak in BulkOrderProcessor by adding batch size limits. Increased pod memory to 4Gi.",
        "root_cause": "Unbounded batch processing in BulkOrderProcessor loading entire result set into memory",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050003").hex,
        "number": "INC0050003",
        "short_description": "DNS resolution failures for internal services",
        "description": "CoreDNS pods are responding slowly, causing DNS lookup timeouts across the Kubernetes cluster. Service discovery is affected for all microservices. The ndots:5 configuration is causing excessive DNS queries.",
        "category": "network",
        "subcategory": "dns",
        "priority": 1,
        "state": "2", # In Progress
        "assignment_group": "L2-Network-Support",
        "cmdb_ci": "coredns",
        "resolution_notes": "Reduced ndots from 5 to 2 in pod DNS config. Added CoreDNS cache plugin with 30s TTL. Scaled CoreDNS from 2 to 4 replicas.",
        "root_cause": "Excessive DNS query amplification due to ndots:5 configuration combined with undersized CoreDNS deployment",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050004").hex,
        "number": "INC0050004",
        "short_description": "SSO login failures after AD sync timeout",
        "description": "Users unable to authenticate via SSO portal. Active Directory synchronization job timed out at 02:00 UTC, leaving user attributes stale. LDAP connection pool showing connection refused errors.",
        "category": "access_management",
        "subcategory": "authentication",
        "priority": 1,
        "state": "6", # Resolved
        "assignment_group": "L2-IAM-Support",
        "cmdb_ci": "sso-portal",
        "resolution_notes": "Restarted AD sync job with increased timeout (30min → 60min). Flushed LDAP connection pool. Root cause: AD sync job exceeded timeout due to 50K new user records from HR migration.",
        "root_cause": "AD sync timeout caused by bulk HR migration of 50K user records",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050005").hex,
        "number": "INC0050005",
        "short_description": "Database disk space at 95% on prod-db-01",
        "description": "PostgreSQL production database server disk utilization at 95%. WAL files accumulating due to failed replication to standby. Autovacuum is blocked by long-running analytical queries.",
        "category": "infrastructure",
        "subcategory": "storage",
        "priority": 1,
        "state": "6", # Resolved
        "assignment_group": "L2-DBA-Support",
        "cmdb_ci": "prod-db-01",
        "resolution_notes": "Terminated long-running queries blocking autovacuum. Cleared orphaned WAL files (120GB recovered). Fixed replication by resetting standby from latest backup. Added disk space alerting at 80%.",
        "root_cause": "Failed replication causing WAL accumulation, compounded by long-running queries blocking autovacuum",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050006").hex,
        "number": "INC0050006",
        "short_description": "API gateway SSL certificate expired blocking HTTPS traffic",
        "description": "All HTTPS requests to the API gateway are failing with ERR_CERT_DATE_INVALID. The wildcard certificate *.prod.company.com expired at midnight UTC. Certificate auto-renewal via cert-manager failed silently.",
        "category": "network",
        "subcategory": "security",
        "priority": 1,
        "state": "6", # Resolved
        "assignment_group": "L2-Network-Support",
        "cmdb_ci": "api-gateway",
        "resolution_notes": "Manually renewed certificate via cert-manager CLI. Fixed cert-manager ACME solver configuration that was preventing auto-renewal. Added certificate expiry monitoring to Prometheus.",
        "root_cause": "cert-manager ACME DNS01 solver misconfigured after cluster migration, preventing automatic certificate renewal",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050007").hex,
        "number": "INC0050007",
        "short_description": "Notification service pods in CrashLoopBackOff",
        "description": "All notification-service pods are in CrashLoopBackOff state. Exit code 137 indicates OOM kill. The service was recently updated to v3.1.0 which includes a new templating engine for email notifications.",
        "category": "application",
        "subcategory": "availability",
        "priority": 2,
        "state": "1", # New
        "assignment_group": "L2-Application-Support",
        "cmdb_ci": "notification-service",
        "resolution_notes": "Rolled back to v3.0.5. Root cause: new Handlebars templating engine had O(n²) memory complexity for batch notifications. Fixed in v3.1.1 with streaming template rendering.",
        "root_cause": "O(n²) memory complexity in new Handlebars templating engine for batch email rendering",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050008").hex,
        "number": "INC0050008",
        "short_description": "Kafka consumer lag growing on inventory-events topic",
        "description": "Consumer group inventory-consumers showing growing lag on inventory-events topic. Current lag: 500K messages. Consumer instances are processing at 50% of normal throughput. Started after partition rebalance.",
        "category": "application",
        "subcategory": "performance",
        "priority": 2,
        "state": "6", # Resolved
        "assignment_group": "L2-Application-Support",
        "cmdb_ci": "inventory-service",
        "resolution_notes": "Increased consumer instances from 3 to 6. Fixed partition assignment strategy from range to cooperative-sticky. Consumer lag recovered in 45 minutes.",
        "root_cause": "Suboptimal partition assignment after rebalance causing uneven load distribution across consumers",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050009").hex,
        "number": "INC0050009",
        "short_description": "VPN tunnel flapping between DC1 and DC2",
        "description": "IPsec VPN tunnel between datacenter DC1 and DC2 is flapping every 5-10 minutes. Causing intermittent connectivity loss for cross-DC service communication. MTU mismatch suspected after ISP change.",
        "category": "network",
        "subcategory": "connectivity",
        "priority": 2,
        "state": "6", # Resolved
        "assignment_group": "L2-Network-Support",
        "cmdb_ci": "vpn-concentrator-01",
        "resolution_notes": "Reduced IPsec tunnel MTU from 1500 to 1400 to accommodate ISP overhead. Enabled PMTUD (Path MTU Discovery). Tunnel stable after configuration change.",
        "root_cause": "MTU mismatch after ISP circuit change, causing IPsec packet fragmentation and reassembly failures",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "INC0050010").hex,
        "number": "INC0050010",
        "short_description": "Service account permissions revoked during IAM cleanup",
        "description": "Automated IAM cleanup job revoked permissions for svc-payment-api service account. The account was tagged as 'inactive' because it authenticates via OAuth tokens rather than password-based login. Payment processing halted.",
        "category": "access_management",
        "subcategory": "authorization",
        "priority": 1,
        "state": "2", # In Progress
        "assignment_group": "L2-IAM-Support",
        "cmdb_ci": "iam-service",
        "resolution_notes": "Restored service account permissions immediately. Updated IAM cleanup job to exclude service accounts with 'svc-' prefix and accounts marked as 'service_type=oauth'. Added pre-execution dry-run check.",
        "root_cause": "IAM cleanup job incorrectly classified OAuth-authenticated service accounts as inactive",
    },
]

# ── Synthetic KB Articles ───────────────────────────────────────────────────

SYNTHETIC_KB_ARTICLES: list[dict] = [
    {
        "sys_id": uuid5(NAMESPACE_DNS, "KB0020001").hex,
        "number": "KB0020001",
        "short_description": "Troubleshooting HTTP 502 Bad Gateway Errors in Microservices",
        "text": "<h2>Troubleshooting HTTP 502 Bad Gateway Errors</h2><h3>Overview</h3><p>HTTP 502 errors indicate that a gateway received an invalid response from an upstream server.</p><h3>Common Causes</h3><ul><li>Connection pool exhaustion (HikariCP, Apache HttpClient)</li><li>Upstream service crashes or restarts</li><li>Network timeouts or firewall changes</li><li>Resource limits exceeded (CPU, memory)</li><li>Deployment regressions</li></ul><h3>Diagnostic Steps</h3><ol><li>Check upstream service health endpoints</li><li>Review connection pool metrics (active, idle, max)</li><li>Check recent deployments in the release log</li><li>Review pod restart counts in Kubernetes</li><li>Examine error rate trends in Datadog/Grafana</li></ol><h3>Resolution Patterns</h3><p>1. Connection pool tuning: Increase max pool size and set appropriate timeouts.<br>2. Rollback: If correlated with a deployment, rollback to last known good version.<br>3. Resource scaling: Increase replica count or resource limits.</p>",
        "category": "application",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "KB0020002").hex,
        "number": "KB0020002",
        "short_description": "Kubernetes Pod Troubleshooting: CrashLoopBackOff and OOMKilled",
        "text": "<h2>Kubernetes Pod Troubleshooting</h2><h3>CrashLoopBackOff</h3><p>A pod enters CrashLoopBackOff when the container repeatedly crashes after starting.</p><h3>Common Exit Codes</h3><ul><li>Exit 1: Application error</li><li>Exit 137: OOMKilled (SIGKILL from kernel)</li><li>Exit 139: Segmentation fault (SIGSEGV)</li><li>Exit 143: Graceful termination (SIGTERM)</li></ul><h3>Diagnostic Commands</h3><pre>kubectl describe pod &lt;pod-name&gt;\nkubectl logs &lt;pod-name&gt; --previous\nkubectl top pod &lt;pod-name&gt;\nkubectl get events --field-selector involvedObject.name=&lt;pod-name&gt;</pre><h3>OOMKilled Resolution</h3><p>1. Increase memory limits in the deployment spec.<br>2. Profile the application for memory leaks.<br>3. Check for unbounded caches or data structures.<br>4. Consider horizontal scaling instead of vertical.</p>",
        "category": "application",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "KB0020003").hex,
        "number": "KB0020003",
        "short_description": "DNS Troubleshooting in Kubernetes Clusters",
        "text": "<h2>DNS Troubleshooting in Kubernetes</h2><h3>Overview</h3><p>DNS issues in Kubernetes can affect all inter-service communication.</p><h3>Common Issues</h3><ul><li>CoreDNS pod resource exhaustion</li><li>Excessive ndots causing query amplification</li><li>DNS cache misconfigurations</li><li>Network policies blocking DNS traffic (port 53)</li></ul><h3>Diagnostic Steps</h3><ol><li>Test DNS resolution from within a pod: <code>nslookup kubernetes.default</code></li><li>Check CoreDNS pod status and logs</li><li>Verify CoreDNS ConfigMap settings</li><li>Check network policies for port 53 rules</li><li>Review ndots configuration in pod spec</li></ol><h3>Best Practices</h3><p>Set ndots to 2 or lower. Enable CoreDNS caching with appropriate TTL. Use FQDN where possible to avoid search domain lookups.</p>",
        "category": "network",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "KB0020004").hex,
        "number": "KB0020004",
        "short_description": "PostgreSQL Disk Space Management and WAL Cleanup",
        "text": "<h2>PostgreSQL Disk Space Management</h2><h3>WAL File Accumulation</h3><p>Write-Ahead Log (WAL) files can accumulate when replication is broken or archiving fails.</p><h3>Causes</h3><ul><li>Failed streaming replication to standby</li><li>WAL archiving command failures</li><li>Replication slots retaining old WAL</li><li>Autovacuum blocked by long transactions</li></ul><h3>Recovery Steps</h3><ol><li>Check replication status: <code>SELECT * FROM pg_stat_replication;</code></li><li>Check replication slot lag: <code>SELECT * FROM pg_replication_slots;</code></li><li>Drop unused replication slots</li><li>Clear orphaned WAL if safe: <code>pg_archivecleanup</code></li><li>Terminate blocking queries: <code>SELECT pg_terminate_backend(pid);</code></li></ol><h3>Prevention</h3><p>Monitor disk usage with alerts at 80%. Set max_wal_size appropriately. Ensure replication health monitoring is in place.</p>",
        "category": "infrastructure",
    },
    {
        "sys_id": uuid5(NAMESPACE_DNS, "KB0020005").hex,
        "number": "KB0020005",
        "short_description": "SSO and LDAP Authentication Troubleshooting Guide",
        "text": "<h2>SSO and LDAP Authentication Troubleshooting</h2><h3>Overview</h3><p>Single Sign-On failures can have multiple root causes including AD sync issues, LDAP connection problems, and token validation errors.</p><h3>Common Failure Modes</h3><ul><li>AD sync job timeout or failure</li><li>LDAP connection pool exhaustion</li><li>Expired or invalid SAML/OIDC tokens</li><li>Clock skew between IdP and SP</li><li>Certificate expiry on IdP</li></ul><h3>Diagnostic Steps</h3><ol><li>Check AD sync job status and logs</li><li>Test LDAP connectivity: <code>ldapsearch -x -H ldap://server -b 'dc=corp'</code></li><li>Verify SAML certificate validity</li><li>Check NTP sync on all SSO components</li><li>Review IdP audit logs for error codes</li></ol><h3>Quick Fixes</h3><p>1. Restart LDAP connection pool.<br>2. Force AD sync re-run with increased timeout.<br>3. Clear SSO session cache.<br>4. Verify certificate chain of trust.</p>",
        "category": "access_management",
    },
]


async def insert_into_database(
    incidents: list[dict],
    kb_articles: list[dict],
) -> dict[str, int]:
    """Insert synthetic data into the PostgreSQL database.

    Args:
        incidents: List of incident records to insert.
        kb_articles: List of KB article records to insert.

    Returns:
        Dict with counts of inserted records.
    """
    from app.config import get_settings
    from app.storage.postgres import (
        IncidentDB,
        get_session_factory,
        init_db,
    )
    from sqlalchemy import delete
    from app.storage.postgres import RecommendationDB, FeedbackDB, ChatSessionDB, AuditEventDB, FeedbackWeightDB, ReviewQueueDB

    # Initialize database tables
    await init_db()

    factory = get_session_factory()
    inserted_incidents = 0
    inserted_articles = 0

    async with factory() as session:
        # Clear existing synthetic records to prevent duplicates and clean start
        await session.execute(delete(FeedbackDB))
        await session.execute(delete(ChatSessionDB))
        await session.execute(delete(ReviewQueueDB))
        await session.execute(delete(RecommendationDB))
        await session.execute(delete(FeedbackWeightDB))
        await session.execute(delete(AuditEventDB))
        await session.execute(delete(IncidentDB))

        # Insert incidents
        now = datetime.now(timezone.utc)
        for i, inc in enumerate(incidents):
            is_resolved = inc["state"] in ("6", "7")
            db_incident = IncidentDB(
                snow_sys_id=inc["sys_id"],
                number=inc["number"],
                short_description=inc["short_description"],
                description=inc["description"],
                category=inc["category"],
                subcategory=inc.get("subcategory", ""),
                priority=inc["priority"],
                state=inc["state"],
                assignment_group=inc["assignment_group"],
                cmdb_ci=inc["cmdb_ci"],
                resolution_notes=inc.get("resolution_notes") if is_resolved else None,
                root_cause=inc.get("root_cause") if is_resolved else None,
                opened_at=now - timedelta(days=30 - i),
                resolved_at=now - timedelta(days=29 - i) if is_resolved else None,
                is_indexed=False,
            )
            session.add(db_incident)
            inserted_incidents += 1

        await session.commit()

    logger.info(
        "seed_complete",
        inserted_incidents=inserted_incidents,
        inserted_articles=inserted_articles,
    )

    return {
        "incidents": inserted_incidents,
        "kb_articles": inserted_articles,
    }


def print_seed_data() -> None:
    """Print synthetic data to stdout for review."""
    print("=" * 80)
    print("SYNTHETIC INCIDENTS (10)")
    print("=" * 80)
    for inc in SYNTHETIC_INCIDENTS:
        print(f"\n--- {inc['number']} ---")
        print(f"  Title:    {inc['short_description']}")
        print(f"  Category: {inc['category']}/{inc.get('subcategory', '')}")
        print(f"  Priority: P{inc['priority']}")
        print(f"  CMDB CI:  {inc['cmdb_ci']}")
        print(f"  Root Cause: {inc.get('root_cause', 'N/A')}")

    print("\n" + "=" * 80)
    print("SYNTHETIC KB ARTICLES (5)")
    print("=" * 80)
    for kb in SYNTHETIC_KB_ARTICLES:
        print(f"\n--- {kb['number']} ---")
        print(f"  Title:    {kb['short_description']}")
        print(f"  Category: {kb['category']}")


async def main_async(skip_db: bool = False) -> None:
    """Async main entry point.

    Args:
        skip_db: If True, only print data without database insertion.
    """
    print_seed_data()

    if skip_db:
        logger.info("skip_db_mode", message="Skipping database insertion")
        return

    try:
        result = await insert_into_database(SYNTHETIC_INCIDENTS, SYNTHETIC_KB_ARTICLES)
        logger.info("seed_success", **result)
    except Exception as e:
        logger.error("seed_failed", error=str(e))
        logger.info(
            "hint",
            message="Run 'docker-compose up postgres' first, or use --skip-db to preview data",
        )
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Seed synthetic test data for local development.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        default=False,
        help="Print data without inserting into database",
    )

    args = parser.parse_args()
    asyncio.run(main_async(skip_db=args.skip_db))


if __name__ == "__main__":
    main()
