"""Claim appeal eligibility agent related classes."""

import mcp_clients as mcp
import models as mdl

from .base_agent import McpEnabledAgent


class ClaimAppealAgentConfig(mdl.WorkflowBaseModel):
    """Configuration model for ClaimAppealAgent."""

    claim_appeal_rule_mcp_client_config: mcp.ClaimAppealRuleMcpClientConfig


class ClaimAppealAgent(
    McpEnabledAgent[
        ClaimAppealAgentConfig,
        mcp.ClaimAppealRuleMcpClient,
        mdl.ClaimAppealRuleRequest,
        mdl.ClaimAppealRule,
    ]
):
    """Configurable agent class responsible for claim appeal eligibility checks.

    Evaluates all disqualification rules against the claim and customer context.
    A claim is eligible for appeal only if no disqualification rule matches.
    """

    _config_data_type = ClaimAppealAgentConfig

    def __init__(self, config: ClaimAppealAgentConfig):
        """Initialize the configurable claim appeal agent.

        Args:
            config (ClaimAppealAgentConfig): Validated claim appeal agent configuration.
        """
        # MCP clients are not factory-managed: agents are cached by AgentFactory,
        # so the same agent instance always holds the same client instance.
        super().__init__(config, mcp.ClaimAppealRuleMcpClient(config.claim_appeal_rule_mcp_client_config))

    def check_eligibility(self, claim: mdl.Claim,
                          customer: mdl.Customer) -> mdl.ClaimAppealResult:
        """Check whether the claim is eligible for appeal.

        Evaluates all disqualification rules. The first matching rule disqualifies
        the claim. If no rule matches, the claim is eligible.

        Args:
            claim (mdl.Claim): Claim to evaluate.
            customer (mdl.Customer): Customer context for the claim.

        Returns:
            mdl.ClaimAppealResult: Eligibility result with reason.
        """
        for rule in self._mcp_client.rules:
            subject = customer if rule.subject == "customer" else claim
            value = getattr(subject, rule.field)
            if rule.matches(value):
                return mdl.ClaimAppealResult(claim.claim_id, False, rule.reason)
        return mdl.ClaimAppealResult(claim.claim_id, True, "Claim is eligible for appeal.")

    def get_eligibility_message(self, claim: mdl.Claim, customer: mdl.Customer) -> str:
        """Build a user-facing appeal eligibility message for the given claim and customer.

        Args:
            claim (mdl.Claim): Claim to evaluate.
            customer (mdl.Customer): Customer context for the claim.

        Returns:
            str: User-facing appeal eligibility message.
        """
        result = self.check_eligibility(claim, customer)
        eligible_str = "eligible" if result.eligible else "not eligible"
        message = f"Claim {result.claim_id} is {eligible_str} for appeal."
        if not result.eligible:
            message += f" {result.reason}"
        return message
