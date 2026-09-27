"""Claim agent related classes."""

from typing import Optional

import mcp_clients as mcp
import models as mdl

from .base_agent import McpStorageAgent


class ClaimAgentConfig(mdl.WorkflowBaseModel):
    """Configuration model for ClaimAgent."""

    claim_mcp_client_config: mcp.ClaimMcpClientConfig


class ClaimAgent(
    McpStorageAgent[
        ClaimAgentConfig,
        mcp.ClaimMcpClient,
        mdl.ClaimRequest,
        mdl.Claim,
    ]
):
    """Configurable agent class responsible for claim-related workflow behavior.

    The agent delegates claim-record retrieval to the configured MCP client
    and exposes claim-specific workflow methods.
    """

    _config_data_type = ClaimAgentConfig

    def __init__(self, config: ClaimAgentConfig):
        """Initialize the configurable claim agent.

        Args:
            config (ClaimAgentConfig): Validated claim-agent configuration.
        """
        # MCP clients are not factory-managed: agents are cached by AgentFactory,
        # so the same agent instance always holds the same client instance.
        super().__init__(config, mcp.ClaimMcpClient(config.claim_mcp_client_config))

    def get_status(self, request: mdl.ClaimRequest) -> Optional[mdl.ClaimStatus]:
        """Retrieve the status of the requested claim.

        Args:
            request (mdl.ClaimRequest): Claim lookup request object.

        Returns:
            Optional[mdl.ClaimStatus]: Claim status value, if the claim is found.
        """
        claim = self.get_obj(request)
        return claim.status if claim else None

    def get_status_message(self, request: mdl.ClaimRequest) -> str:
        """Build a user-facing claim-status message.

        Args:
            request (mdl.ClaimRequest): Claim lookup request object.

        Returns:
            str: User-facing claim-status message.
        """
        status = self.get_status(request)
        status_msg = f"is currently {status.value}." if status else "was not found."
        return f"Claim {request.claim_id} {status_msg}"


