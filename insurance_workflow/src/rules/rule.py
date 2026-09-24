"""Rule abstractions for the insurance workflow decision engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import shared.core as shd_core



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
    reason: str = ""
    kind: str = ""
    metadata: shd_core.EntityMetadata = field(default_factory=shd_core.EntityMetadata)

    @property
    def is_active(self) -> bool:
        """Return True if the current UTC time falls within the rule's effective window.

        Returns False if either date is missing or not a valid ISO 8601 string.
        """
        try:
            now = datetime.now(timezone.utc)
            return datetime.fromisoformat(self.effective_from) <= now <= datetime.fromisoformat(self.effective_to)
        except (ValueError, TypeError):
            return False

    def ready(self, context: dict) -> bool:
        """Return True if all declared input preconditions are satisfied in context.

        Checks that every field declared in self.input is present in context.
        Subclasses extend this to add type-specific checks (e.g. the subject.attribute
        key for DecisionRule, or match_keys presence for LookupRule).

        Args:
            context: Shared execution context dict.

        Returns:
            bool: True if all preconditions for evaluation are met.
        """
        return all(f in context for f in self.input)

    @abstractmethod
    def evaluate(self, context: dict) -> tuple[bool, dict]:
        """Evaluate against context and return the outputs to write on match.

        Callers must check ready() first. evaluate() assumes all preconditions
        are satisfied and raises if required keys are missing.

        Args:
            context: Shared execution context dict.

        Returns:
            tuple[bool, dict]: (matched, outputs_to_write). outputs is empty when
                matched is False.
        """

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

    _OPS = {
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">":  lambda a, b: a > b,
        "<":  lambda a, b: a < b,
    }

    kind: str = "decision"
    subject: str = ""
    attribute: str = ""
    operator: str = ""
    threshold: str = ""

    def _coerce(self, value):
        """Coerce threshold string to the type of value.

        bool requires special handling: bool("False") == True in Python because
        any non-empty string is truthy.
        """
        if isinstance(value, bool):
            return self.threshold.lower() in ("true", "1")
        return type(value)(self.threshold)

    def ready(self, context: dict) -> bool:
        """Return True if all input preconditions and the subject.attribute key are in context."""
        return super().ready(context) and f"{self.subject}.{self.attribute}" in context

    def matches(self, value) -> bool:
        """Return True if the provided value satisfies this rule's condition.

        Args:
            value: Actual attribute value from the subject; threshold is cast to its type.

        Returns:
            bool: True if the condition is met.
        """
        return self._OPS[self.operator](value, self._coerce(value))

    def evaluate(self, context: dict) -> tuple[bool, dict]:
        """Evaluate the condition and return output fields set to True on match.

        Args:
            context: Shared execution context dict; must contain subject.attribute key.

        Returns:
            tuple[bool, dict]: (matched, {output_field: True, ...}) or (False, {}).
        """
        if self.matches(context[f"{self.subject}.{self.attribute}"]):
            return True, {f: True for f in self.output}
        return False, {}


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

    def ready(self, context: dict) -> bool:
        """Return True if all input preconditions and every match_keys key are in context."""
        return super().ready(context) and all(k in context for k in self.match_keys)

    def matches(self, context: dict) -> bool:
        """Return True if every key in match_keys is present in context with the expected value."""
        return all(context.get(k) == v for k, v in self.match_keys.items())

    def evaluate(self, context: dict) -> tuple[bool, dict]:
        """Evaluate the lookup and return output_values as a copy on match.

        Args:
            context: Shared execution context dict; must contain all match_keys keys.

        Returns:
            tuple[bool, dict]: (matched, dict(output_values)) or (False, {}).
        """
        if self.matches(context):
            return True, dict(self.output_values)
        return False, {}
