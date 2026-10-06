"""Policy coverage rule MCP client related classes."""

import rules as rls

from .mcp_client import McpRuleClient


class PolicyCoverageRuleMcpClient(McpRuleClient):
    """Rule client responsible for retrieving active policy coverage disqualification rules."""

    @property
    def rules(self) -> list[rls.Rule]:
        """Active policy coverage rules resolved from the rule registry in DAG order."""
        return self._registry.get_effective("policy_coverage")
