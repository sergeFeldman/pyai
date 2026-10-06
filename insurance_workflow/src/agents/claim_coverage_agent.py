"""Claim coverage verification agent related classes."""

import dataclasses

import mcp_clients as mcp
import models as mdl
import rules as rls
import services as svc

from .base_agent import McpEnabledAgent


class ClaimCoverageAgentConfig(mdl.WorkflowBaseModel):
    """Configuration model for ClaimCoverageAgent."""


class ClaimCoverageAgent(McpEnabledAgent[ClaimCoverageAgentConfig, mcp.PolicyCoverageRuleMcpClient]):
    """Agent responsible for policy coverage verification against the policy_coverage rule domain.

    Builds a shared execution context from claim, customer, and policy objects, then
    delegates to RuleRegistry.execute() which walks the DAG in topological order.
    Uses the negative-gating model: if policy_coverage.disqualified is written by any
    rule, coverage is denied; if no disqualifier fires, coverage is verified.
    """

    _config_data_type = ClaimCoverageAgentConfig

    def __init__(self, config: ClaimCoverageAgentConfig):
        """Initialize the claim coverage agent.

        Args:
            config (ClaimCoverageAgentConfig): Validated agent configuration.
        """
        super().__init__(config, mcp.PolicyCoverageRuleMcpClient())

    def _build_context(self, claim: mdl.Claim, customer: mdl.Customer,
                       policy: mdl.Policy) -> dict:
        """Build the execution context dict from claim, customer, and policy objects.

        Uses dataclasses.fields() + getattr() to preserve original Python types for
        correct threshold coercion in DecisionRules. Adds a derived boolean field
        claim.below_deductible computed from claim.amount vs policy.deductible_amount.

        Args:
            claim: Claim to include in context.
            customer: Customer context to include.
            policy: Policy record to include.

        Returns:
            dict: Flat context keyed by "claim.<field>", "customer.<field>", "policy.<field>",
                  plus derived field "claim.below_deductible".
        """
        ctx = {f"claim.{f.name}": getattr(claim, f.name)
               for f in dataclasses.fields(claim)}
        ctx.update({f"customer.{f.name}": getattr(customer, f.name)
                    for f in dataclasses.fields(customer)})
        ctx.update({f"policy.{f.name}": getattr(policy, f.name)
                    for f in dataclasses.fields(policy)})
        ctx["claim.below_deductible"] = claim.amount < policy.deductible_amount
        return ctx

    def check_coverage(self, claim: mdl.Claim, customer: mdl.Customer,
                       policy: mdl.Policy, trace_id: str = "") -> mdl.PolicyCoverageResult:
        """Verify whether the claim is covered by the associated policy.

        Executes the policy_coverage rule domain via RuleRegistry. Rules are
        evaluated in topological + priority order; the DAG chain
        pc_high_value_flag -> pc_high_value_repeat_claimant propagates intermediate
        outputs before the terminal disqualifier check.

        Args:
            claim: Claim to evaluate.
            customer: Customer context for the claim.
            policy: Policy record for the claim's customer.
            trace_id: Workflow trace ID threaded from the orchestrator.

        Returns:
            mdl.PolicyCoverageResult: Coverage result with reason.
        """
        context = self._build_context(claim, customer, policy)
        result = rls.RuleRegistry().execute(
            "policy_coverage", context, trace_id=trace_id, executed_by="claim_coverage_agent",
            entities={
                "claim": claim.to_dict(),
                "customer": customer.to_dict(),
                "policy": policy.to_dict(),
            },
        )
        svc.RuleExecutionAuditService().log(result)
        if "policy_coverage.disqualified" in result.outputs:
            reason = next(
                (r.reason for r in result.triggered if "policy_coverage.disqualified" in r.output),
                None,
            )
            return mdl.PolicyCoverageResult(claim.claim_id, False, reason)
        return mdl.PolicyCoverageResult(claim.claim_id, True, "Claim is covered under the active policy.")

    def get_coverage_message(self, claim: mdl.Claim, customer: mdl.Customer,
                             policy: mdl.Policy, trace_id: str = "") -> str:
        """Build a user-facing coverage verification message.

        Args:
            claim: Claim to evaluate.
            customer: Customer context for the claim.
            policy: Policy record for the claim's customer.
            trace_id: Workflow trace ID threaded from the orchestrator.

        Returns:
            str: User-facing coverage verification message.
        """
        result = self.check_coverage(claim, customer, policy, trace_id=trace_id)
        covered_str = "covered" if result.verified else "not covered"
        message = f"Claim {result.claim_id} is {covered_str} under the policy."
        if not result.verified:
            message += f" {result.reason}"
        return message
