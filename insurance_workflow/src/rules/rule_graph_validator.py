"""Graph validation linter for the rule engine DAG."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from .rule import Rule


class RuleValidationSeverity(StrEnum):
    """Severity level of a rule graph validation finding."""

    ERROR   = "error"
    WARNING = "warning"


class RuleValidationKind(StrEnum):
    """Category of a rule graph validation finding."""

    PHANTOM_CONSUMER = "phantom_consumer"
    ORPHAN_PRODUCER  = "orphan_producer"


@dataclass
class RuleGraphFinding:
    """One finding produced by the rule graph validator.

    Attributes:
        rule_id: ID of the rule that triggered the finding.
        severity: ERROR for phantom consumers (rule excluded from DAG);
            WARNING for orphan producers (rule kept but noted).
        kind: PHANTOM_CONSUMER when a rule declares an input field no active
            rule produces; ORPHAN_PRODUCER when a rule produces a field no
            active rule consumes.
        domain: Rule domain the finding belongs to.
        field: The field name involved in the broken or unmatched link.
        message: Human-readable description of the finding.
    """

    rule_id:  str
    severity: RuleValidationSeverity
    kind:     RuleValidationKind
    domain:   str
    field:    str
    message:  str


@dataclass
class RuleGraphValidationReport:
    """Aggregated result of a rule graph validation run.

    Attributes:
        findings: All findings produced across all domains.
        validated_at: UTC timestamp when validation ran.
    """

    findings:     list[RuleGraphFinding] = field(default_factory=list)
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def errors(self) -> list[RuleGraphFinding]:
        return [f for f in self.findings if f.severity == RuleValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[RuleGraphFinding]:
        return [f for f in self.findings if f.severity == RuleValidationSeverity.WARNING]

    def is_valid(self) -> bool:
        return not self.errors


class RuleGraphValidator:
    """Validates the field-link integrity of an active rule graph.

    For each domain, a single pass over the active rules builds two maps:
        produced: field -> [rule_ids] that write it in output
        consumed: field -> [rule_ids] that declare it in input

    Two set differences derive the findings:
        phantom_fields = consumed.keys() - produced.keys()  -> ERROR per consumer rule
        orphan_fields  = produced.keys() - consumed.keys()  -> WARNING per producer rule

    Phantom consumer detection has two modes. Explicit mode (when context_inputs is
    declared for the domain): any consumed field whose prefix is not in context_inputs
    and is not produced by an active rule is flagged. This catches prefix typos in
    external context fields (e.g. "cliam.amount" instead of "claim.amount").
    Namespace-scoping fallback (no context_inputs declared): only fields whose prefix
    matches the domain's own produced-field prefixes are checked; external context
    fields are excluded because their prefix never appears in any rule's output.

    Orphan producers include legitimate terminal outputs (e.g. appeal.disqualified)
    that are consumed by the agent rather than by other rules.
    """

    def validate(
        self,
        rules: list[Rule],
        terminal_outputs: dict[str, set[str]] | None = None,
        context_inputs: dict[str, set[str]] | None = None,
    ) -> RuleGraphValidationReport:
        """Validate field-link integrity across all domains in the rule list.

        Args:
            rules: Active rules to validate, typically the effective rule set
                returned by RuleRegistry._get_effective() across all domains.
            terminal_outputs: Optional map of domain -> set of declared terminal
                output field names (from DomainConfig). Orphan producer warnings
                are suppressed for fields present in their domain's terminal set;
                a misspelling in a rule's output will not match and will still warn.
                Defaults to empty (all orphan producers warn).
            context_inputs: Optional map of domain -> set of external context
                prefixes (e.g. {"claim", "customer"}) declared in DomainConfig.
                When provided for a domain, phantom consumer detection uses an
                explicit allowlist: any consumed field whose prefix is not in
                context_inputs and not produced by an active rule is flagged.
                Without this, the fallback namespace-scoping mode applies and
                prefix typos in context fields go undetected.

        Returns:
            RuleGraphValidationReport with all findings and a validated_at timestamp.
        """
        sev_error    = RuleValidationSeverity.ERROR
        sev_warning  = RuleValidationSeverity.WARNING
        kind_phantom = RuleValidationKind.PHANTOM_CONSUMER
        kind_orphan  = RuleValidationKind.ORPHAN_PRODUCER

        findings: list[RuleGraphFinding] = []

        by_domain: dict[str, list[Rule]] = defaultdict(list)
        for rule in rules:
            by_domain[rule.domain].append(rule)

        for domain, domain_rules in by_domain.items():
            produced: dict[str, list[str]] = defaultdict(list)
            consumed: dict[str, list[str]] = defaultdict(list)

            for rule in domain_rules:
                for f in rule.output:
                    produced[f].append(rule.id)
                for f in rule.input:
                    consumed[f].append(rule.id)

            domain_context = (context_inputs or {}).get(domain, None)
            if domain_context is not None:
                # Explicit mode: flag any consumed field whose prefix is not a
                # declared context input and is not produced by an active rule.
                phantom_fields = {
                    f for f in consumed.keys() - produced.keys()
                    if f.split(".")[0] not in domain_context
                }
            else:
                # Namespace-scoped fallback: only flag fields whose prefix matches
                # the domain's own produced-field prefixes (external context fields
                # like claim.amount are excluded because their prefix never appears
                # in any rule's output).
                internal_prefixes = {f.split(".")[0] for f in produced}
                phantom_fields = {
                    f for f in consumed.keys() - produced.keys()
                    if f.split(".")[0] in internal_prefixes
                }
            for f in phantom_fields:
                for rule_id in consumed[f]:
                    findings.append(RuleGraphFinding(
                        severity=sev_error,
                        kind=kind_phantom,
                        domain=domain,
                        rule_id=rule_id,
                        field=f,
                        message=(
                            f"Rule '{rule_id}' declares input '{f}' but no active rule "
                            f"in domain '{domain}' produces it. This rule will always be "
                            f"skipped_precondition and has been excluded from the DAG."
                        ),
                    ))

            domain_terminals = (terminal_outputs or {}).get(domain, set())
            orphan_fields = produced.keys() - consumed.keys()
            for f in orphan_fields:
                if f in domain_terminals:
                    continue
                for rule_id in produced[f]:
                    findings.append(RuleGraphFinding(
                        severity=sev_warning,
                        kind=kind_orphan,
                        domain=domain,
                        rule_id=rule_id,
                        field=f,
                        message=(
                            f"Rule '{rule_id}' produces '{f}' but no active rule "
                            f"in domain '{domain}' consumes it. "
                            f"If this is a terminal output consumed by the agent, "
                            f"declare it in domain_config.json to suppress this warning."
                        ),
                    ))

        return RuleGraphValidationReport(findings=findings)

