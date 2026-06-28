"""Role-Based Access Control (RBAC) for the L2 Support Assistant.

Defines engineer roles, permissions, and provides both sync and async
permission-check utilities. Designed for use in FastAPI dependency
injection and service-layer authorization.

Roles:
    - l2_engineer: Standard L2 support — view & interact with incidents
    - l3_engineer: Escalation-capable — full incident lifecycle + KB management
    - admin: Full system access including configuration and audit
    - system: Internal service account role
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


# ── Enumerations ─────────────────────────────────────────────────────────────


class Role(str, Enum):
    """Engineer roles within the support organization."""
    L2_ENGINEER = "l2_engineer"
    L3_ENGINEER = "l3_engineer"
    ADMIN = "admin"
    SYSTEM = "system"


class Permission(str, Enum):
    """Granular permissions for application operations."""
    # Incident operations
    INCIDENT_VIEW = "incident:view"
    INCIDENT_ANALYZE = "incident:analyze"
    INCIDENT_UPDATE = "incident:update"
    INCIDENT_RESOLVE = "incident:resolve"
    INCIDENT_ESCALATE = "incident:escalate"

    # Recommendation operations
    RECOMMENDATION_VIEW = "recommendation:view"
    RECOMMENDATION_GENERATE = "recommendation:generate"
    RECOMMENDATION_FEEDBACK = "recommendation:feedback"

    # KB article operations
    KB_VIEW = "kb:view"
    KB_MANAGE = "kb:manage"

    # Admin operations
    ADMIN_CONFIG = "admin:config"
    ADMIN_USER_MANAGE = "admin:user_manage"
    ADMIN_INDEX_MANAGE = "admin:index_manage"

    # Audit operations
    AUDIT_VIEW = "audit:view"
    AUDIT_EXPORT = "audit:export"

    # Chat operations
    CHAT_USE = "chat:use"
    CHAT_VIEW_HISTORY = "chat:view_history"


# ── Permission Matrix ────────────────────────────────────────────────────────

# Maps each role to its allowed permissions.
_ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.L2_ENGINEER: {
        Permission.INCIDENT_VIEW,
        Permission.INCIDENT_ANALYZE,
        Permission.RECOMMENDATION_VIEW,
        Permission.RECOMMENDATION_GENERATE,
        Permission.RECOMMENDATION_FEEDBACK,
        Permission.KB_VIEW,
        Permission.CHAT_USE,
        Permission.CHAT_VIEW_HISTORY,
    },
    Role.L3_ENGINEER: {
        Permission.INCIDENT_VIEW,
        Permission.INCIDENT_ANALYZE,
        Permission.INCIDENT_UPDATE,
        Permission.INCIDENT_RESOLVE,
        Permission.INCIDENT_ESCALATE,
        Permission.RECOMMENDATION_VIEW,
        Permission.RECOMMENDATION_GENERATE,
        Permission.RECOMMENDATION_FEEDBACK,
        Permission.KB_VIEW,
        Permission.KB_MANAGE,
        Permission.CHAT_USE,
        Permission.CHAT_VIEW_HISTORY,
        Permission.AUDIT_VIEW,
    },
    Role.ADMIN: {perm for perm in Permission},  # All permissions
    Role.SYSTEM: {perm for perm in Permission},  # Internal service — all permissions
}


# ── User Context Model ───────────────────────────────────────────────────────


class UserContext(BaseModel):
    """Authenticated user context for authorization decisions.

    Populated from the JWT token or API key during authentication.
    """
    user_id: str = Field(..., description="ServiceNow user sys_id or username")
    role: Role = Field(..., description="User's assigned role")
    display_name: str = Field(default="", description="Human-readable display name")
    assignment_group: Optional[str] = Field(
        default=None,
        description="User's assignment group (for scoped access)",
    )


# ── Authorization Errors ─────────────────────────────────────────────────────


class AuthorizationError(Exception):
    """Raised when a user lacks the required permission."""

    def __init__(
        self,
        user_id: str,
        role: Role,
        required_permission: Permission,
        message: Optional[str] = None,
    ) -> None:
        self.user_id = user_id
        self.role = role
        self.required_permission = required_permission
        detail = (
            message
            or f"User '{user_id}' with role '{role.value}' lacks "
               f"permission '{required_permission.value}'"
        )
        super().__init__(detail)


# ── RBAC Service ─────────────────────────────────────────────────────────────


class RBACService:
    """Role-Based Access Control service.

    Provides permission checks against the predefined role-permission
    matrix. Thread-safe and stateless.

    Example:
        >>> rbac = RBACService()
        >>> user = UserContext(user_id="u123", role=Role.L2_ENGINEER)
        >>> rbac.check_permission(user, Permission.INCIDENT_VIEW)  # OK
        >>> rbac.check_permission(user, Permission.ADMIN_CONFIG)   # raises
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="rbac")

    def get_permissions(self, role: Role) -> set[Permission]:
        """Get all permissions for a given role.

        Args:
            role: The role to look up.

        Returns:
            Set of permissions granted to the role.
        """
        return _ROLE_PERMISSIONS.get(role, set())

    def has_permission(self, user: UserContext, permission: Permission) -> bool:
        """Check if a user has a specific permission.

        Args:
            user: The authenticated user context.
            permission: The permission to check.

        Returns:
            True if the user's role grants the permission.
        """
        role_perms = self.get_permissions(user.role)
        return permission in role_perms

    def check_permission(
        self,
        user: UserContext,
        permission: Permission,
    ) -> None:
        """Assert that a user has a specific permission; raise if not.

        Args:
            user: The authenticated user context.
            permission: The required permission.

        Raises:
            AuthorizationError: If the user lacks the permission.
        """
        if not self.has_permission(user, permission):
            self._log.warning(
                "authorization_denied",
                user_id=user.user_id,
                role=user.role.value,
                permission=permission.value,
            )
            raise AuthorizationError(
                user_id=user.user_id,
                role=user.role,
                required_permission=permission,
            )
        self._log.debug(
            "authorization_granted",
            user_id=user.user_id,
            role=user.role.value,
            permission=permission.value,
        )

    def check_any_permission(
        self,
        user: UserContext,
        permissions: list[Permission],
    ) -> None:
        """Assert that a user has at least one of the given permissions.

        Args:
            user: The authenticated user context.
            permissions: List of permissions — user needs at least one.

        Raises:
            AuthorizationError: If the user lacks all listed permissions.
        """
        for perm in permissions:
            if self.has_permission(user, perm):
                self._log.debug(
                    "authorization_granted",
                    user_id=user.user_id,
                    role=user.role.value,
                    permission=perm.value,
                    check_type="any_of",
                )
                return

        self._log.warning(
            "authorization_denied",
            user_id=user.user_id,
            role=user.role.value,
            permissions=[p.value for p in permissions],
            check_type="any_of",
        )
        raise AuthorizationError(
            user_id=user.user_id,
            role=user.role,
            required_permission=permissions[0],
            message=(
                f"User '{user.user_id}' with role '{user.role.value}' "
                f"lacks any of: {[p.value for p in permissions]}"
            ),
        )

    def filter_by_role(
        self,
        user: UserContext,
        items: list[dict],
        required_permission: Permission,
    ) -> list[dict]:
        """Filter a list of items based on the user's permission.

        Returns the full list if the user has the required permission,
        otherwise returns an empty list. Useful for API response filtering.

        Args:
            user: The authenticated user context.
            items: List of dict items to potentially filter.
            required_permission: Permission required to see the items.

        Returns:
            The items list if authorized, otherwise empty list.
        """
        if self.has_permission(user, required_permission):
            return items
        return []


# ── Module-level convenience ─────────────────────────────────────────────────

_rbac_service: Optional[RBACService] = None


def get_rbac_service() -> RBACService:
    """Get the singleton RBAC service instance.

    Returns:
        RBACService: The RBAC service.
    """
    global _rbac_service
    if _rbac_service is None:
        _rbac_service = RBACService()
    return _rbac_service
