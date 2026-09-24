"""Claim appeal eligibility agent related classes."""

import dataclasses

import mcp_clients as mcp
import models as mdl
import rules as rls
import services as svc

from .base_agent import McpEnabledAgent


class ClaimAppealAgentConfig(mdl.WorkflowBaseModel):
    """Configuration model for ClaimAppealAgent."""


class ClaimAppealAgent(McpEnabledAgent[ClaimAppealAgentConfig, mcp.ClaimAppealRuleMcpClient]):
    """Configurable agent class responsible for claim appeal eligibility checks.

    Builds a shared execution context from the claim and customer objects, then
    delegates to RuleRegistry.execute() which walks the DAG in topological order,
    enforces input preconditions between rules, and propagates intermediate outputs.
    """

    _config_data_type = ClaimAppealAgentConfig

    def __init__(self, config: ClaimAppealAgentConfig):
        """Initialize the configurable claim appeal agent.

        Args:
            config (ClaimAppealAgentConfig): Validated claim appeal agent configuration.
        """
        super().__init__(config, mcp.ClaimAppealRuleMcpClient())

    def _build_context(self, claim: mdl.Claim, customer: mdl.Customer) -> dict:
        """Build the execution context dict from claim and customer domain objects.

        Uses dataclasses.fields() + getattr() to preserve original Python types.
        to_dict() must not be used here; it converts bool to "true"/"false" strings
        and Enum to .value, both of which would break DecisionRule._coerce().

        Args:
            claim: Claim to include in context.
            customer: Customer context to include.

        Returns:
            dict: Flat context keyed by "claim.<field>" and "customer.<field>".
        """
        ctx = {f"claim.{f.name}": getattr(claim, f.name)
               for f in dataclasses.fields(claim)}
        ctx.update({f"customer.{f.name}": getattr(customer, f.name)
                    for f in dataclasses.fields(customer)})
        return ctx

    def check_eligibility(self, claim: mdl.Claim,
                          customer: mdl.Customer,
                          trace_id: str = "") -> mdl.ClaimAppealResult:
        """Check whether the claim is eligible for appeal.

        Executes the claim_appeal rule domain via RuleRegistry. Rules are evaluated
        in topological + priority order; each rule's input preconditions are checked
        before evaluation so that dependent rules only trigger when their upstream
        outputs are present in the context. Claim and customer entity snapshots are
        captured at execution time and included in the audit record alongside the
        ordered per-rule evaluation log (triggered / skipped_precondition / skipped_no_match).
        The execution result is persisted to the rule execution audit log.

        Args:
            claim (mdl.Claim): Claim to evaluate.
            customer (mdl.Customer): Customer context for the claim.
            trace_id (str): Workflow trace ID threaded from the orchestrator.

        Returns:
            mdl.ClaimAppealResult: Eligibility result with reason.
        """
        context = self._build_context(claim, customer)
        result = rls.RuleRegistry().execute(
            "claim_appeal", context, trace_id=trace_id, executed_by="claim_appeal_agent",
            entities={"claim": claim.to_dict(), "customer": customer.to_dict()},
        )
        svc.RuleExecutionAuditService().log(result)
        if "appeal.disqualified" in result.outputs:
            reason = next(
                (r.reason for r in result.triggered if "appeal.disqualified" in r.output),
                None,
            )
            return mdl.ClaimAppealResult(claim.claim_id, False, reason)
        return mdl.ClaimAppealResult(claim.claim_id, True, "Claim is eligible for appeal.")

    def get_eligibility_message(self, claim: mdl.Claim, customer: mdl.Customer,
                                trace_id: str = "") -> str:
        """Build a user-facing appeal eligibility message for the given claim and customer.

        Args:
            claim (mdl.Claim): Claim to evaluate.
            customer (mdl.Customer): Customer context for the claim.
            trace_id (str): Workflow trace ID threaded from the orchestrator.

        Returns:
            str: User-facing appeal eligibility message.
        """
        result = self.check_eligibility(claim, customer, trace_id=trace_id)
        eligible_str = "eligible" if result.eligible else "not eligible"
        message = f"Claim {result.claim_id} is {eligible_str} for appeal."
        if not result.eligible:
            message += f" {result.reason}"
        return message
