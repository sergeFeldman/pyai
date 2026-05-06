"""Customer agent related classes."""

from typing import Optional

import mcp_clients as mcp
import models as mdl

from .base_agent import McpEnabledAgent


class CustomerAgentConfig(mdl.WorkflowBaseModel):
    """Configuration model for CustomerAgent."""

    customer_mcp_client_config: mcp.CustomerMcpClientConfig


class CustomerAgent(
    McpEnabledAgent[
        CustomerAgentConfig,
        mcp.CustomerMcpClient,
        mdl.CustomerRequest,
        mdl.CustomerContext,
    ]
):
    """Configurable agent class responsible for customer context retrieval.

    The agent delegates customer-context record retrieval to the configured MCP
    client and exposes customer-specific workflow methods.
    """

    _config_data_type = CustomerAgentConfig

    def __init__(self, config: CustomerAgentConfig):
        """Initialize the configurable customer agent.

        Args:
            config (CustomerAgentConfig): Validated customer-agent configuration.
        """
        super().__init__(config, mcp.CustomerMcpClient(config.customer_mcp_client_config))

    def get_context(self, request: mdl.CustomerRequest) -> Optional[mdl.CustomerContext]:
        """Retrieve the customer context record for the requested customer.

        Args:
            request (mdl.CustomerRequest): Customer context lookup request object.

        Returns:
            Optional[mdl.CustomerContext]: Matching customer context record, if found.
        """
        return self.get_obj(request)
