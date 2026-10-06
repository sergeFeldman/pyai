"""Policy agent related classes."""

from typing import Optional

import mcp_clients as mcp
import models as mdl

from .base_agent import McpStorageAgent


class PolicyAgentConfig(mdl.WorkflowBaseModel):
    """Configuration model for PolicyAgent."""

    policy_mcp_client_config: mcp.PolicyMcpClientConfig


class PolicyAgent(
    McpStorageAgent[
        PolicyAgentConfig,
        mcp.PolicyMcpClient,
        mdl.PolicyRequest,
        mdl.Policy,
    ]
):
    """Configurable agent class responsible for retrieving policy records by customer ID."""

    _config_data_type = PolicyAgentConfig

    def __init__(self, config: PolicyAgentConfig):
        """Initialize the configurable policy agent.

        Args:
            config (PolicyAgentConfig): Validated policy-agent configuration.
        """
        super().__init__(config, mcp.PolicyMcpClient(config.policy_mcp_client_config))

    def get_policy(self, customer_id: str) -> Optional[mdl.Policy]:
        """Retrieve the active policy for the given customer ID.

        Args:
            customer_id: Customer identifier.

        Returns:
            Optional[mdl.Policy]: Policy record, if found.
        """
        return self.get_obj(mdl.PolicyRequest(customer_id=customer_id))
