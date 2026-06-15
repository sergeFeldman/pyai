"""Rule registry backed by the generic KeyedRegistry."""

from __future__ import annotations

import shared.core as shd_core

from .rule import Rule


class RuleRegistry(shd_core.KeyedRegistry[Rule]):
    """Registry for all rule versions, keyed by domain.

    Extends KeyedRegistry with rule-specific queries: get_latest() resolves the
    highest-versioned rule for a given id, and get_active() returns the latest
    active version of each rule in a domain sorted by priority.

    The registry is append-only - add() never replaces an existing rule.
    All versions of a rule coexist so the full audit trail is preserved.
    """

    def __init__(self) -> None:
        super().__init__(Rule, key_field="domain")

    def get_latest(self, id: str, domain: str) -> Rule | None:
        """Return the highest-versioned rule matching id within the given domain.

        Args:
            id: Rule identifier.
            domain: Domain to search within.

        Returns:
            Rule with the highest metadata.version, or None if not found.
        """
        matches = [r for r in self.get_by_key(domain) if r.id == id]
        return max(matches, key=lambda r: r.metadata.version) if matches else None

    def get_active(self, domain: str, group: str = "") -> list[Rule]:
        """Return active rules for a domain, optionally filtered by group.

        Resolves the latest version per rule id before filtering by is_active,
        so each rule id appears at most once. Results are sorted by priority
        descending - higher priority value executes first.

        Args:
            domain: Domain to retrieve rules for.
            group: Optional execution cluster filter within the domain.

        Returns:
            Active Rule instances sorted by priority descending.
        """
        rules = self.get_by_key(domain)
        if group:
            rules = [r for r in rules if r.group == group]

        by_id: dict[str, Rule] = {}
        for r in rules:
            if r.id not in by_id or r.metadata.version > by_id[r.id].metadata.version:
                by_id[r.id] = r

        return sorted(
            [r for r in by_id.values() if r.is_active],
            key=lambda r: r.priority,
            reverse=True,
        )

    def all(self) -> list[Rule]:
        """Return all rule versions across all domains, sorted by id then version descending.

        Sorted so all versions of the same rule are grouped together with the
        latest version first - consistent ordering for the output JSON file.

        Returns:
            Flat list of all Rule instances across all domains and versions.
        """
        rules = [r for items in self._registry.values() for r in items]
        return sorted(rules, key=lambda r: (r.id, -r.metadata.version))
