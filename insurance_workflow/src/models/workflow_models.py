"""Internal dataclasses used by workflow processing."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

import core


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
    meaningfully related — for example, status=denied combined with is_fraud=true
    produces a richer, correlated explanation than explaining each in isolation.

    attribute_values maps each attribute name to its string-represented value:
        {"status": "denied", "is_fraud": "true"}
    """

    attribute_values: dict[str, str]
    explanation: str
    next_steps: str


@dataclass
class Claim(core.ExplainableMixin, core.SerializableMixin):
    """Normalized claim data returned by the claims workflow."""

    claim_id: str
    claim_type: str
    customer_id: str
    amount: float
    date: str
    repair_shop: str
    status: Annotated[ClaimStatus, core.Explainable()]
    is_fraud: Annotated[bool, core.Explainable()] = False


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
    """Structured result returned by the claim explanation workflow."""

    claim: Claim
    explanations: list[AttributeExplanation]
    review_eligible: bool = False
    customer_context: Optional[str] = None
    policy_basis: Optional[str] = None
    escalation_required: bool = False


@dataclass
class CustomerContext(core.SerializableMixin):
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
class PolicyRule(core.SerializableMixin):
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
