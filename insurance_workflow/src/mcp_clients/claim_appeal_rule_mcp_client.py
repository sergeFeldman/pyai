"""Claim appeal rule MCP client related classes."""

from typing import cast

import rules as rls

from .mcp_client import McpRuleClient


class ClaimAppealRuleMcpClient(McpRuleClient):
    """Rule client responsible for retrieving active claim appeal disqualification rules."""

    @property
    def rules(self) -> list[rls.DecisionRule]:
        """Active claim appeal disqualification rules resolved from the rule registry.

        Returns:
            list[rls.DecisionRule]: Latest effective appeal disqualification rules.
        """
        return cast(list[rls.DecisionRule], self._registry.get_active("claim_appeal"))
