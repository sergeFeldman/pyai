"""Internal dataclasses used by workflow processing."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

import shared.core as shd_core

from .base_model import WorkflowBaseModel


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

class DomainConfig(WorkflowBaseModel):
    """Configuration for a single rule domain.

    Attributes:
        domain: Domain key, e.g. "claim_appeal".
        terminal_outputs: Fields produced by rules in this domain that are
            consumed by the agent rather than by other rules. The graph
            validation linter suppresses orphan producer warnings for declared
            terminal fields; a misspelling in a rule's output will not match
            and will still warn. Agents and APIs use this as the single source
            of truth for domain output contracts instead of hardcoding field names.
        context_inputs: External context prefixes (before the first dot) that
            rules in this domain consume but no rule produces, e.g. ["claim",
            "customer"]. When declared, the graph validation linter uses an
            explicit allowlist for phantom consumer detection: any consumed
            prefix not in this list and not produced by an active rule is
            flagged. Without this field, the linter falls back to namespace
            scoping and prefix typos in context fields go undetected.
    """

    domain: str
    terminal_outputs: list[str] = []
    context_inputs: list[str] = []


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
class ClaimAppealResult:
    """Result of a claim appeal eligibility check."""

    claim_id: str
    eligible: bool
    reason: str


@dataclass
class Policy(shd_core.SerializableMixin):
    """Policy data used for coverage verification."""

    policy_id: str
    customer_id: str
    policy_type: str
    policy_status: str
    coverage_limit: float
    deductible_amount: float


@dataclass
class PolicyRequest:
    """Input data required to retrieve a policy by customer ID."""

    customer_id: str


@dataclass
class PolicyCoverageResult:
    """Result of a policy coverage verification check."""

    claim_id: str
    verified: bool
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
