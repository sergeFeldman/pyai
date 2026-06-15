"""Internal dataclasses used by workflow processing."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

import shared.core as shd_core


# Enums

class ClaimStatus(Enum):
    """Supported claim status values."""

    # In-progress states
    OPEN = "open"
    UNDER_REVIEW = "under_review"

    # Final states
    APPROVED = "approved"
    DENIED = "denied"


# Classes

@dataclass
class AttributeExplanation:
    """Explanation covering one or more correlated attribute-value pairs of a domain object.

    A single explanation may span multiple attributes when their values are
    meaningfully related - for example, status=denied combined with is_fraud=true
    produces a richer, correlated explanation than explaining each in isolation.

    attribute_values maps each attribute name to its string-represented value:
        {"status": "denied", "is_fraud": "true"}
    """

    attribute_values: dict[str, str]
    explanation: str
    next_steps: str


@dataclass
class Claim(shd_core.ExplainableMixin, shd_core.SerializableMixin):
    """Normalized claim data returned by the claims workflow."""

    claim_id: str
    claim_type: str
    customer_id: str
    amount: float
    date: str
    repair_shop: str
    status: Annotated[ClaimStatus, shd_core.Explainable()]
    is_fraud: Annotated[bool, shd_core.Explainable()] = False


@dataclass
class ClaimRequest:
    """Input data required to retrieve a claim."""

    claim_id: str


@dataclass
class ClaimExplanationRequest:
    """Input data required for the claim explanation workflow."""

    claim_id: str
    attributes: list[str]


@dataclass
class ClaimExplanationResult:
    """Structured result returned by the claim explanation workflow.

    Attributes:
        claim: The resolved claim record.
        explanations: Per-attribute explanations generated for the requested attributes.
        review_eligible: True when the claim status allows a manual review request.
        customer_context: Optional human-readable summary of customer relationship context,
            enriched by the orchestrator when customer data is available.
        policy_basis: Optional policy section reference explaining the denial basis,
            enriched by the orchestrator when a matching policy rule is found.
        escalation_required: True when the claim warrants immediate escalation,
            set by the orchestrator based on fraud flags or review status.
    """

    claim: Claim
    explanations: list[AttributeExplanation]
    review_eligible: bool = False
    customer_context: Optional[str] = None
    policy_basis: Optional[str] = None
    escalation_required: bool = False


@dataclass
class Customer(shd_core.SerializableMixin):
    """Customer relationship context used to enrich the explanation workflow."""

    customer_id: str
    tenure_years: int
    active_policy_count: int
    prior_claim_count: int
    last_interaction_date: str
    preferred_contact_method: str
    escalation_history_count: int


@dataclass
class CustomerRequest:
    """Input data required to retrieve customer context."""

    customer_id: str


@dataclass
class PolicyRule(shd_core.SerializableMixin):
    """Policy rule returned by the policy rules lookup."""

    policy_rule_id: str
    claim_type: str
    attribute: str
    value: str
    denial_basis: str
    next_steps: str
    policy_section: str


@dataclass
class PolicyRuleRequest:
    """Input data required to retrieve a policy rule by its primary key."""

    policy_rule_id: str


@dataclass
class PolicyRuleFilterRequest:
    """Input data required to retrieve a policy rule by claim context."""

    claim_type: str
    attribute: str
    value: str


@dataclass
class ClaimAppealRule:
    """Disqualification rule evaluated during the claim appeal eligibility check."""

    _OPS = {
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">":  lambda a, b: a > b,
        "<":  lambda a, b: a < b,
    }

    claim_appeal_rule_id: str
    subject: str
    field: str
    operator: str
    threshold: str
    reason: str

    def matches(self, value) -> bool:
        """Return True if the provided value satisfies this disqualification rule.

        Args:
            value: Actual field value from claim or customer; threshold is cast to its type.

        Returns:
            bool: True if the rule condition is met.
        """
        return self._OPS[self.operator](value, type(value)(self.threshold))


@dataclass
class ClaimAppealRuleRequest:
    """Input data required to retrieve a claim appeal rule by its primary key."""

    claim_appeal_rule_id: str


@dataclass
class ClaimAppealResult:
    """Result of a claim appeal eligibility check."""

    claim_id: str
    eligible: bool
    reason: str


@dataclass
class UserRequest:
    """Normalized user input passed into the workflow layer."""

    message: str
    attributes: Optional[list[str]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class UserResponse:
    """User-facing response produced by the workflow."""

    message: str
    trace_id: str


@dataclass
class WorkflowContext:
    """Per-request metadata shared across workflow components."""

    # Trace metadata travels with the request through the workflow.
    trace_id: str
    started_at: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class AuditRecord:
    """Immutable audit entry recorded at the end of each workflow request.

    Contains no PII - only operational metadata needed for compliance and debugging.
    Raw message payload and customer data are intentionally excluded.
    """

    trace_id: str
    request_type: str
    agent_names: list[str]
    response: str
    timestamp: datetime
