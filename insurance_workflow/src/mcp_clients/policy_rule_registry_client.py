"""Registry-based policy rule client: uses RuleRegistry instead of a data file."""

import uuid
from typing import Optional

import rules as rls
import services as svc

from .mcp_client import McpRuleClient


class PolicyRuleRegistryClient(McpRuleClient):
    """Rule client that resolves active policy LookupRules from the registry.

    Backed by RuleRegistry.execute() so policy lookups produce a full DAG
    execution audit record consistent with other domains.

    Use find() for single-match lookups (returns the first triggered rule).
    Use rules directly for full iteration.
    """

    @property
    def rules(self) -> list[rls.LookupRule]:
        """Active policy LookupRules in DAG topological / priority order."""
        return [r for r in self._registry.get_effective("policy") if isinstance(r, rls.LookupRule)]

    def find(self, context: dict, trace_id: str = "",
             executed_by: str = "policy_rule_registry_client") -> Optional[rls.LookupRule]:
        """Execute the policy domain and return the first triggered rule.

        Replaces the linear matches() loop with RuleRegistry.execute() so every
        policy lookup produces an evaluations log and an audit record. All policy
        rules are roots (no upstream producers); execute() evaluates them in
        priority order and stops after the first match writes its outputs.

        Args:
            context: Flat dict of facts, e.g. {"claim_type": "auto_collision",
                     "attribute": "is_fraud", "value": "true"}.
            trace_id: Optional workflow trace ID threaded from the caller.
            executed_by: Executor label recorded in the audit metadata.

        Returns:
            First triggered LookupRule, or None if no rule matched.
        """
        result = self._registry.execute(
            "policy", dict(context),
            trace_id=trace_id or f"policy-{uuid.uuid4()}",
            executed_by=executed_by,
        )
        svc.RuleExecutionAuditService().log(result)
        return result.triggered[0] if result.triggered else None
