"""Policy rule agent related classes."""

from typing import Optional

import mcp_clients as mcp
import models as mdl

from .base_agent import McpEnabledAgent


class PolicyRuleAgentConfig(mdl.WorkflowBaseModel):
    """Configuration model for PolicyRuleAgent."""

    policy_rule_mcp_client_config: mcp.PolicyRuleMcpClientConfig


class PolicyRuleAgent(
    McpEnabledAgent[
        PolicyRuleAgentConfig,
        mcp.PolicyRuleMcpClient,
        mdl.PolicyRuleRequest,
        mdl.PolicyRule,
    ]
):
    """Configurable agent class responsible for policy rule retrieval.

    The agent delegates policy rule lookup to the configured MCP client
    and exposes policy rule-specific workflow methods.
    """

    _config_data_type = PolicyRuleAgentConfig

    def __init__(self, config: PolicyRuleAgentConfig):
        """Initialize the configurable policy rule agent.

        Args:
            config (PolicyRuleAgentConfig): Validated policy rule agent configuration.
        """
        super().__init__(
            config,
            mcp.PolicyRuleMcpClient(config.policy_rule_mcp_client_config),
        )

    def get_rule(self, request: mdl.PolicyRuleRequest) -> Optional[mdl.PolicyRule]:
        """Retrieve the policy rule matching the provided request.

        Args:
            request (mdl.PolicyRuleRequest): Policy rule lookup request object.

        Returns:
            Optional[mdl.PolicyRule]: Matching policy rule, if found.
        """
        return self.get_obj(request)
