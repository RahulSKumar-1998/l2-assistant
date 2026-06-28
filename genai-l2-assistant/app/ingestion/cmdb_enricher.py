"""CMDB enricher for incident records.

Enriches incident data with Configuration Management Database (CMDB)
information including CI metadata and upstream/downstream service
dependency graphs. This context improves root-cause analysis and
blast-radius estimation.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import structlog

from app.models.incident import CMDBRecord, ProcessedTicket

logger = structlog.get_logger(__name__)


# ── Protocol for ServiceNow client ──────────────────────────────────────────


class ServiceNowClientProtocol(Protocol):
    """Protocol for ServiceNow client dependency injection."""

    async def get_cmdb_ci(self, sys_id: str) -> CMDBRecord:
        """Fetch a CMDB CI by sys_id or name."""
        ...


# ── Enrichment Result Model ─────────────────────────────────────────────────


class CMDBEnrichmentResult:
    """Result of CMDB enrichment for an incident.

    Attributes:
        ci: The primary CI record.
        upstream_services: Services that this CI depends on.
        downstream_services: Services that depend on this CI.
        environment: Deployment environment (production, staging, dev).
        service_tier: Service tier (tier-1, tier-2, tier-3).
        blast_radius: Estimated blast radius description.
    """

    __slots__ = (
        "ci", "upstream_services", "downstream_services",
        "environment", "service_tier", "blast_radius",
    )

    def __init__(
        self,
        ci: CMDBRecord,
        upstream_services: list[str],
        downstream_services: list[str],
        environment: str,
        service_tier: str,
        blast_radius: str,
    ) -> None:
        self.ci = ci
        self.upstream_services = upstream_services
        self.downstream_services = downstream_services
        self.environment = environment
        self.service_tier = service_tier
        self.blast_radius = blast_radius

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for metadata enrichment.

        Returns:
            Dict representation of the enrichment result.
        """
        return {
            "ci_name": self.ci.name,
            "ci_class": self.ci.sys_class_name,
            "environment": self.environment,
            "service_tier": self.service_tier,
            "upstream_services": self.upstream_services,
            "downstream_services": self.downstream_services,
            "blast_radius": self.blast_radius,
            "operational_status": self.ci.operational_status,
        }

    def to_context_string(self) -> str:
        """Generate a human-readable context string for LLM prompts.

        Returns:
            Formatted string describing the CI and its dependencies.
        """
        lines = [
            f"Configuration Item: {self.ci.name}",
            f"Type: {self.ci.sys_class_name}",
            f"Environment: {self.environment}",
            f"Service Tier: {self.service_tier}",
        ]

        if self.upstream_services:
            lines.append(
                f"Upstream Dependencies: {', '.join(self.upstream_services)}"
            )
        if self.downstream_services:
            lines.append(
                f"Downstream Dependents: {', '.join(self.downstream_services)}"
            )
        if self.blast_radius:
            lines.append(f"Blast Radius: {self.blast_radius}")

        return "\n".join(lines)


# ── CMDBEnricher ─────────────────────────────────────────────────────────────


class CMDBEnricher:
    """Enriches incidents with CMDB CI relationship data.

    Queries the ServiceNow CMDB for the Configuration Item associated
    with an incident and resolves its upstream and downstream service
    dependencies. This context is used to:

    - Improve root-cause analysis by understanding dependency chains
    - Estimate blast radius for severity assessment
    - Enrich vector metadata for better retrieval filtering

    Example:
        >>> from app.ingestion.mock_client import MockServiceNowClient
        >>> enricher = CMDBEnricher(MockServiceNowClient())
        >>> result = await enricher.enrich("kong-prod-01")
        >>> result.upstream_services
        ['payment-service', 'user-auth-service']
    """

    def __init__(
        self,
        snow_client: ServiceNowClientProtocol,
        *,
        max_depth: int = 2,
    ) -> None:
        """Initialize the CMDB enricher.

        Args:
            snow_client: ServiceNow client (real or mock).
            max_depth: Maximum depth for relationship traversal.
                Default is 2 (direct + one level of transitive deps).
        """
        self._client = snow_client
        self._max_depth = max_depth
        self._log = logger.bind(component="cmdb_enricher")

    async def enrich(self, cmdb_ci: str) -> Optional[CMDBEnrichmentResult]:
        """Enrich an incident with CMDB CI data.

        Fetches the CI record and extracts upstream/downstream service
        relationships from the CMDB relationship table.

        Args:
            cmdb_ci: CMDB CI sys_id or name.

        Returns:
            CMDBEnrichmentResult with dependency data, or None if
            the CI cannot be resolved.
        """
        if not cmdb_ci or not cmdb_ci.strip():
            self._log.debug("no_cmdb_ci_to_enrich")
            return None

        try:
            ci = await self._client.get_cmdb_ci(cmdb_ci)
        except Exception as exc:
            self._log.warning(
                "cmdb_ci_fetch_failed",
                cmdb_ci=cmdb_ci,
                error=str(exc),
            )
            return None

        # Parse relationships
        upstream, downstream = self._parse_relationships(ci)

        # Determine blast radius
        blast_radius = self._estimate_blast_radius(ci, downstream)

        result = CMDBEnrichmentResult(
            ci=ci,
            upstream_services=upstream,
            downstream_services=downstream,
            environment=ci.environment,
            service_tier=ci.service_tier,
            blast_radius=blast_radius,
        )

        self._log.info(
            "cmdb_enrichment_complete",
            ci_name=ci.name,
            upstream_count=len(upstream),
            downstream_count=len(downstream),
            service_tier=ci.service_tier,
        )
        return result

    async def enrich_ticket(
        self,
        ticket: ProcessedTicket,
    ) -> tuple[ProcessedTicket, Optional[CMDBEnrichmentResult]]:
        """Enrich a processed ticket with CMDB data.

        Updates the ticket's metadata with CMDB information and returns
        both the updated ticket and the enrichment result.

        Args:
            ticket: ProcessedTicket to enrich.

        Returns:
            Tuple of (updated ticket, enrichment result).
            Enrichment result is None if CI cannot be resolved.
        """
        enrichment = await self.enrich(ticket.cmdb_ci)
        if enrichment is None:
            return ticket, None

        # We don't modify the ProcessedTicket itself since it's a Pydantic model,
        # but the enrichment data is used by the embedding pipeline to enrich
        # chunk metadata.
        self._log.info(
            "ticket_enriched_with_cmdb",
            source_id=ticket.source_id,
            ci_name=enrichment.ci.name,
        )
        return ticket, enrichment

    def _parse_relationships(
        self,
        ci: CMDBRecord,
    ) -> tuple[list[str], list[str]]:
        """Parse upstream and downstream relationships from CI record.

        Upstream = services that this CI depends on (it's the parent).
        Downstream = services that depend on this CI (it's the child).

        Args:
            ci: CMDB CI record with relationships.

        Returns:
            Tuple of (upstream_services, downstream_services).
        """
        upstream: list[str] = []
        downstream: list[str] = []

        for rel in ci.relationships:
            rel_type = rel.get("type", "").lower()
            parent = rel.get("parent", "")
            child = rel.get("child", "")

            if rel_type in ("depends on", "depends on::used by"):
                if parent == ci.name:
                    # This CI depends on the child
                    if child and child != ci.name:
                        upstream.append(child)
                else:
                    # Something else depends on this CI
                    if parent and parent != ci.name:
                        downstream.append(parent)

            elif rel_type in ("used by", "consumed by"):
                if child == ci.name:
                    # This CI is used by the parent
                    if parent and parent != ci.name:
                        downstream.append(parent)
                else:
                    # This CI uses the child
                    if child and child != ci.name:
                        upstream.append(child)

            elif rel_type in ("hosted on", "runs on"):
                if parent == ci.name:
                    # This CI is hosted on the child (infrastructure dep)
                    if child and child != ci.name:
                        upstream.append(child)

            else:
                # Generic relationship — add based on position
                if parent == ci.name and child and child != ci.name:
                    upstream.append(child)
                elif child == ci.name and parent and parent != ci.name:
                    downstream.append(parent)

        # Deduplicate
        upstream = list(dict.fromkeys(upstream))
        downstream = list(dict.fromkeys(downstream))

        return upstream, downstream

    def _estimate_blast_radius(
        self,
        ci: CMDBRecord,
        downstream_services: list[str],
    ) -> str:
        """Estimate the blast radius based on CI tier and dependencies.

        Args:
            ci: The CI record.
            downstream_services: List of dependent service names.

        Returns:
            Human-readable blast radius description.
        """
        tier = ci.service_tier.lower()
        dep_count = len(downstream_services)

        if tier in ("tier-1", "tier1", "critical"):
            severity = "CRITICAL"
        elif tier in ("tier-2", "tier2", "high"):
            severity = "HIGH"
        elif tier in ("tier-3", "tier3", "medium"):
            severity = "MEDIUM"
        else:
            severity = "LOW"

        if dep_count == 0:
            scope = "No known downstream dependents"
        elif dep_count <= 2:
            scope = f"Limited impact ({dep_count} dependent service(s))"
        elif dep_count <= 5:
            scope = f"Moderate impact ({dep_count} dependent services)"
        else:
            scope = f"Wide impact ({dep_count} dependent services)"

        return f"{severity} — {scope}"
