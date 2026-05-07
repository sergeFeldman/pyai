"""Claim agent related classes."""

from typing import Optional

import mcp_clients as mcp
import models as mdl

from .base_agent import McpEnabledAgent


class ClaimAgentConfig(mdl.WorkflowBaseModel):
    """Configuration model for ClaimAgent."""

    claim_mcp_client_config: mcp.ClaimMcpClientConfig


class ClaimAgent(
    McpEnabledAgent[
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

    _REVIEW_ELIGIBILITY_MAPPING = {
        mdl.ClaimStatus.OPEN: False,
        mdl.ClaimStatus.UNDER_REVIEW: True,
        mdl.ClaimStatus.APPROVED: False,
        mdl.ClaimStatus.DENIED: True,
    }

    _EXPLANATIONS_MAPPING = {
        ("status", mdl.ClaimStatus.OPEN): "This claim is currently open.",
        ("status", mdl.ClaimStatus.UNDER_REVIEW): "This claim is currently under review.",
        ("status", mdl.ClaimStatus.APPROVED): "This claim was approved.",
        ("status", mdl.ClaimStatus.DENIED): "This claim was denied and may require further review.",
        ("is_fraud", True): "This claim was flagged as potentially fraudulent.",
        ("is_fraud", False): "This claim was not flagged as fraudulent.",
    }

    _NEXT_STEPS_MAPPING = {
        ("status", mdl.ClaimStatus.OPEN): "No further action is required at this time.",
        ("status", mdl.ClaimStatus.UNDER_REVIEW): "Please wait for the review to complete or request an update.",
        ("status", mdl.ClaimStatus.APPROVED): "No further action is required at this time.",
        ("status", mdl.ClaimStatus.DENIED): "You may request a manual review.",
        ("is_fraud", True): "Please contact your claims representative.",
        ("is_fraud", False): "",
    }

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

    def explain(self, claim: mdl.Claim, attributes: list[str]) -> list[mdl.AttributeExplanation]:
        """Build a list of attribute explanations for the given claim.

        Args:
            claim (mdl.Claim): Claim object to explain.
            attributes (list[str]): Attribute names to explain.

        Returns:
            list[mdl.AttributeExplanation]: One explanation entry per requested attribute.
        """
        results = []
        for attribute in attributes:
            value = getattr(claim, attribute)
            str_value = value.value if hasattr(value, "value") else str(value).lower()
            results.append(mdl.AttributeExplanation(
                attribute_values={attribute: str_value},
                explanation=self._EXPLANATIONS_MAPPING.get((attribute, value), ""),
                next_steps=self._NEXT_STEPS_MAPPING.get((attribute, value), ""),
            ))
        return results

    def get_explanation(self, request: mdl.ClaimExplanationRequest) -> Optional[mdl.ClaimExplanationResult]:
        """Build a structured explanation result for the requested claim attributes.

        Args:
            request (mdl.ClaimExplanationRequest): Claim explanation request object.

        Returns:
            Optional[mdl.ClaimExplanationResult]: Structured explanation result,
                if the claim is found.

        Raises:
            ValueError: Raised when any requested attribute is not explainable.
        """
        claim = self.get_obj(mdl.ClaimRequest(claim_id=request.claim_id))
        if claim is None:
            return None
        invalid = self.validate_attributes(request.attributes)
        if invalid:
            raise ValueError(f"Non-explainable attributes: {invalid}")
        return mdl.ClaimExplanationResult(
            claim=claim,
            explanations=self.explain(claim, request.attributes),
            review_eligible=self._REVIEW_ELIGIBILITY_MAPPING[claim.status],
        )

    def get_explanation_message(self, request: mdl.ClaimExplanationRequest) -> str:
        """Build a user-facing explanation message for the requested claim attributes.

        Args:
            request (mdl.ClaimExplanationRequest): Claim explanation request object.

        Returns:
            str: User-facing explanation message.
        """
        result = self.get_explanation(request)
        if result is None:
            return f"Claim {request.claim_id} was not found."
        return self.get_explanation_message_from_result(result)

    def get_explanation_message_from_result(self, result: mdl.ClaimExplanationResult) -> str:
        """Build a user-facing explanation message from an already-assembled result.

        Args:
            result (mdl.ClaimExplanationResult): Assembled claim explanation result.

        Returns:
            str: User-facing explanation message.
        """
        parts = [f"{e.explanation} {e.next_steps}".strip() for e in result.explanations]
        review_msg = (
            "This claim is eligible for review."
            if result.review_eligible
            else "This claim is not eligible for review."
        )
        if result.customer_context:
            parts.append(f"Customer context: {result.customer_context}")
        return f"{' '.join(parts)} {review_msg}".strip()

    def validate_attributes(self, requested: list[str]) -> set[str]:
        """Return the set of requested attributes that are not explainable.

        Args:
            requested (list[str]): Attribute names requested for explanation.

        Returns:
            set[str]: Attribute names not eligible for explanation. Empty if all are valid.
        """
        return set(requested) - mdl.Claim.explainable_attributes()

