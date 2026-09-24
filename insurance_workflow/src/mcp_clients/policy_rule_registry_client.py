"""Registry-based policy rule client: uses RuleRegistry instead of a data file."""

from typing import Optional

import rules as rls

from .mcp_client import McpRuleClient


class PolicyRuleRegistryClient(McpRuleClient):
    """Rule client that resolves active policy LookupRules from the registry.

    Mirrors the ClaimAppealRuleMcpClient pattern: backed by RuleRegistry rather
    than a data file, so rules are versioned, DAG-ordered, and effective-date aware.

    Use find() for single-match lookups (returns the highest-priority matching rule).
    Use rules directly for full iteration.
    """

    @property
    def rules(self) -> list[rls.LookupRule]:
        """Active policy LookupRules in DAG topological / priority order."""
        return [r for r in self._registry.get_active("policy") if isinstance(r, rls.LookupRule)]

    def find(self, context: dict) -> Optional[rls.LookupRule]:
        """Return the first active policy rule whose match_keys are all satisfied by context.

        Rules are evaluated in priority-descending order (highest priority first).
        Returns None if no rule matches.

        Args:
            context: Flat dict of facts, e.g. {"claim_type": "auto_collision",
                     "attribute": "is_fraud", "value": "true"}.
        """
        for rule in self.rules:
            if rule.matches(context):
                return rule
        return None
