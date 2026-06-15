"""Rule abstractions for the insurance workflow decision engine."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone

import shared.core as shd_core


def _now() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(kw_only=True)
class Rule(ABC, shd_core.SerializableMixin):
    """Abstract base for all rule types in the decision engine.

    Rules are versioned domain entities identified by id and group. The engine
    evaluates is_active to gate execution and the ETL pipeline uses is_changed
    to decide whether to bump the version on re-import.

    Attributes:
        kind: Rule subtype identifier included in serialized output for informational purposes.
            Not required in raw input - RuleFactory.detect_type() infers it from field presence.
        id: Unique rule identifier within a domain.
        domain: Top-level business domain this rule belongs to, e.g. "claim_appeal".
        group: Execution cluster within the domain. Rules sharing a group are evaluated together.
        priority: Execution order within a group. Higher value = higher priority; executed first within a group.
        effective_from: UTC ISO 8601 timestamp when the rule becomes active.
        effective_to: UTC ISO 8601 timestamp when the rule expires.
        input: Dot-notation attribute references consumed by this rule, e.g. ["claim.status"].
        output: Attribute names produced by this rule, e.g. ["is_eligible"].
        metadata: Audit and version metadata managed by the ETL pipeline.
    """

    id: str = ""
    domain: str = ""
    group: str = ""
    priority: int = 0  # higher value = higher priority; executed first within a group
    effective_from: str = ""
    effective_to: str = ""
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    kind: str = ""
    metadata: shd_core.EntityMetadata = field(default_factory=shd_core.EntityMetadata)

    @property
    def is_active(self) -> bool:
        """Return True if the current UTC time falls within the rule's effective window."""
        now = _now()
        return self.effective_from <= now <= self.effective_to

    def is_changed(self, other: Rule) -> bool:
        """Return True if any non-metadata field differs from other.

        Compares all business fields, excluding metadata, so callers can
        determine whether the rule content has meaningfully changed.

        Args:
            other: Another rule instance to compare against.

        Returns:
            bool: True if any business field (excluding metadata) has changed.
        """
        return any(
            getattr(self, f) != getattr(other, f)
            for f in self.__dataclass_fields__
            if f != "metadata"
        )


@dataclass(kw_only=True)
class DecisionRule(Rule):
    """A rule that evaluates a single attribute against a threshold.

    Extends Rule with the condition fields needed to test a specific
    attribute on a subject domain object.

    Attributes:
        subject: Domain object the rule applies to, e.g. "claim" or "customer".
        attribute: Attribute on the subject being tested, e.g. "status".
        operator: Comparison operator: one of >=, <=, ==, !=, >, <.
        threshold: Value the attribute is compared against, always stored as a string.
        reason: Human-readable explanation returned when the rule matches.
    """

    kind: str = "decision"
    subject: str = ""
    attribute: str = ""
    operator: str = ""
    threshold: str = ""
    reason: str = ""


@dataclass(kw_only=True)
class LookupRule(Rule):
    """A rule that matches a context against a set of keys and returns a payload.

    Generic and reusable across any domain. The engine matches incoming context
    values against match_keys and returns the associated output_values when all
    keys match. Both match_keys and output_values are open-ended dicts, so no
    schema change is needed when a domain introduces new lookup dimensions or
    output fields.

    Attributes:
        match_keys: Key-value pairs the context must satisfy for this rule to fire,
            e.g. {"claim_type": "auto_collision", "attribute": "is_fraud", "value": "true"}.
        output_values: Payload returned when all match_keys are satisfied,
            e.g. {"denial_basis": "...", "next_steps": "...", "policy_section": "12.1"}.
    """

    kind: str = "lookup"
    match_keys: dict[str, str] = field(default_factory=dict)
    output_values: dict[str, str] = field(default_factory=dict)
