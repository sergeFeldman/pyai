"""Policy MCP client related classes."""

from typing import Optional

import models as mdl

from .mcp_client import McpStorageClient, McpStorageClientConfig


class PolicyMcpClientConfig(McpStorageClientConfig):
    """Configuration model for PolicyMcpClient."""


class PolicyMcpClient(McpStorageClient[PolicyMcpClientConfig, mdl.PolicyRequest, mdl.Policy]):
    """Configurable client class responsible for retrieving policy records.

    read_by_key() supports direct lookup by policy_id via the inherited get_obj()
    path when called with a PolicyRequest carrying policy_id. The workflow uses
    get_obj() overridden here to look up by customer_id instead, since the
    orchestrator resolves policy from the claim's customer_id.
    """

    _config_data_type = PolicyMcpClientConfig
    _primary_key_field = "policy_id"

    def get_obj(self, request: mdl.PolicyRequest) -> Optional[mdl.Policy]:
        """Retrieve the policy record for the given customer ID.

        Args:
            request: PolicyRequest carrying the customer_id to look up.

        Returns:
            Optional[mdl.Policy]: Policy record if found, None otherwise.
        """
        for policy in self._storage.read():
            if policy.customer_id == request.customer_id:
                return policy
        return None
