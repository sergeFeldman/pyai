"""Factory for creating Rule instances by kind."""

import dataclasses

import shared.core as shd_core

from .rule import DecisionRule, LookupRule, Rule


class RuleFactory(metaclass=shd_core.Singleton):
    """Singleton factory for creating Rule instances by kind.

    _TYPES_MAPPING is the single source of truth mapping each kind key to its
    concrete Rule subclass. The kind key matches the value each subclass declares
    for its kind field (e.g. DecisionRule.kind = "decision" -> key "decision").
    Add a new entry here when introducing a new Rule subclass.
    """

    _TYPES_MAPPING: dict[str, type] = {
        "decision": DecisionRule,
        "lookup": LookupRule,
    }

    def detect_type(self, data: dict) -> str:
        """Detect the rule kind from raw data fields without requiring a kind key.

        Compares data keys against the unique fields of each registered rule class
        (fields not declared on the base Rule). The first class whose unique fields
        overlap with the data keys wins.

        Args:
            data: Raw rule dict, typically from an input JSON file.

        Returns:
            Kind string, e.g. "decision" or "lookup".

        Raises:
            ValueError: If no registered class matches the data fields.
        """
        base_fields = {f.name for f in dataclasses.fields(Rule)}
        for kind, rule_class in self._TYPES_MAPPING.items():
            unique = {f.name for f in dataclasses.fields(rule_class)} - base_fields
            if unique & data.keys():
                return kind
        raise ValueError(f"Cannot detect rule type from fields: {sorted(data.keys())}")

    def get_class(self, kind: str) -> type:
        """Return the concrete Rule subclass for the given kind key.

        Args:
            kind: Kind key, e.g. "decision".

        Returns:
            type: Concrete Rule subclass.

        Raises:
            ValueError: If the kind key is not registered.
        """
        if kind not in self._TYPES_MAPPING:
            raise ValueError(f"Unknown rule kind '{kind}'. Known: {sorted(self._TYPES_MAPPING)}")
        return self._TYPES_MAPPING[kind]

    def from_dict(self, data: dict) -> Rule:
        """Deserialize a rule dict into the appropriate typed Rule instance.

        Resolves the concrete subclass via detect_type(), filters to valid
        fields only, and reconstructs any nested EntityMetadata dict.

        Args:
            data: Raw or persisted rule dict.

        Returns:
            Rule: Fully constructed concrete Rule subclass instance.
        """
        rule_class = self.get_class(self.detect_type(data))
        valid = {f.name for f in dataclasses.fields(rule_class)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = shd_core.EntityMetadata(**kwargs["metadata"])
        return rule_class(**kwargs)
