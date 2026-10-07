"""Tests for RuleGraphValidator — phantom consumer and orphan producer detection."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import models as mdl
from rules import (
    DecisionRule,
    RuleGraphFinding,
    RuleGraphValidationReport,
    RuleGraphValidator,
    RuleRegistry,
    RuleValidationKind,
    RuleValidationSeverity,
)
from shared.core import EntityMetadata, Singleton


def _ts(delta_days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=delta_days)).isoformat()


def _rule(rule_id: str, domain: str = "d",
          inp: list[str] | None = None,
          out: list[str] | None = None) -> DecisionRule:
    return DecisionRule(
        id=rule_id, domain=domain, priority=0,
        input=inp or [], output=out or [],
        subject="claim", attribute="status", operator="==", threshold="open",
        effective_from=_ts(-1), effective_to=_ts(365),
    )


@pytest.fixture(autouse=True)
def reset_singletons():
    yield
    Singleton._instances.pop(RuleRegistry, None)


# ── TestRuleGraphValidator ────────────────────────────────────────────────────

class TestRuleGraphValidator:

    def test_clean_graph_no_errors(self):
        # r1 produces d.flag, r2 consumes it — valid internal link, no phantom consumers.
        # r2 produces d.done which nothing consumes (terminal output) → warning only.
        rules = [_rule("r1", out=["d.flag"]), _rule("r2", inp=["d.flag"], out=["d.done"])]
        report = RuleGraphValidator().validate(rules)
        assert report.errors == []
        assert report.is_valid()

    def test_no_input_output_no_findings(self):
        rules = [_rule("r1"), _rule("r2")]
        report = RuleGraphValidator().validate(rules)
        assert report.findings == []

    def test_phantom_consumer_single(self):
        # r_prod establishes the "d" namespace; r1 consumes a field that namespace produces nothing for.
        rules = [_rule("r_prod", out=["d.something"]), _rule("r1", inp=["d.missing"])]
        report = RuleGraphValidator().validate(rules)
        assert len(report.errors) == 1
        e = report.errors[0]
        assert e.severity == RuleValidationSeverity.ERROR
        assert e.kind == RuleValidationKind.PHANTOM_CONSUMER
        assert e.rule_id == "r1"
        assert e.field == "d.missing"

    def test_phantom_consumer_multiple_fields_on_one_rule(self):
        rules = [_rule("r_prod", out=["d.something"]), _rule("r1", inp=["d.missing_a", "d.missing_b"])]
        report = RuleGraphValidator().validate(rules)
        assert len(report.errors) == 2
        fields = {e.field for e in report.errors}
        assert fields == {"d.missing_a", "d.missing_b"}

    def test_orphan_producer_terminal_output(self):
        rules = [_rule("r1", out=["d.disqualified"])]
        report = RuleGraphValidator().validate(rules)
        assert report.is_valid()
        assert len(report.warnings) == 1
        w = report.warnings[0]
        assert w.severity == RuleValidationSeverity.WARNING
        assert w.kind == RuleValidationKind.ORPHAN_PRODUCER
        assert w.rule_id == "r1"
        assert w.field == "d.disqualified"

    def test_typo_in_output_produces_both_findings(self):
        # r1 produces typo, r2 expects correct name -> orphan on r1, phantom on r2.
        # Both fields share the domain's internal namespace prefix "d", so both are detected.
        rules = [
            _rule("r1", out=["d.flaged"]),   # typo
            _rule("r2", inp=["d.flagged"]),  # correct name, no producer
        ]
        report = RuleGraphValidator().validate(rules)
        assert len(report.errors) == 1
        assert report.errors[0].rule_id == "r2"
        assert report.errors[0].field == "d.flagged"
        assert len(report.warnings) == 1
        assert report.warnings[0].rule_id == "r1"
        assert report.warnings[0].field == "d.flaged"

    def test_external_context_field_not_flagged_as_phantom(self):
        # claim.amount is an external context field — its prefix "claim" is not
        # in the domain's internal namespace, so it is not a phantom consumer.
        rules = [
            _rule("r1", out=["d.flag"], inp=["claim.amount"]),
        ]
        report = RuleGraphValidator().validate(rules)
        assert report.errors == []

    def test_multi_domain_isolation(self):
        rules = [
            _rule("r0", domain="appeal", out=["appeal.something"]),           # establishes "appeal" namespace
            _rule("r1", domain="appeal", inp=["appeal.something", "appeal.missing"]),  # consumes r0; missing is phantom
            _rule("r2", domain="coverage", out=["coverage.flag"]),
        ]
        report = RuleGraphValidator().validate(rules)
        # one phantom consumer error in appeal, no bleed into coverage
        assert len(report.errors) == 1
        assert report.errors[0].domain == "appeal"
        assert report.errors[0].field == "appeal.missing"
        # only coverage orphan warning — appeal has no orphan producers
        assert all(w.domain != "appeal" for w in report.warnings)

    def test_report_has_validated_at_timestamp(self):
        report = RuleGraphValidator().validate([])
        assert isinstance(report.validated_at, datetime)

    def test_is_valid_false_when_errors(self):
        rules = [_rule("r_prod", out=["d.something"]), _rule("r1", inp=["d.missing"])]
        report = RuleGraphValidator().validate(rules)
        assert not report.is_valid()

    def test_errors_and_warnings_properties(self):
        rules = [
            _rule("r_prod", out=["d.something"]),  # establishes namespace
            _rule("r1", inp=["d.missing"]),         # phantom consumer -> error
            _rule("r2", out=["d.terminal"]),        # orphan producer -> warning
        ]
        report = RuleGraphValidator().validate(rules)
        assert len(report.errors) == 1
        assert len(report.warnings) == 2  # d.something and d.terminal are both orphan producers

    def test_declared_terminal_suppresses_orphan_warning(self):
        rules = [_rule("r1", out=["d.disqualified"])]
        report = RuleGraphValidator().validate(
            rules, terminal_outputs={"d": {"d.disqualified"}}
        )
        assert report.warnings == []
        assert report.is_valid()

    def test_undeclared_orphan_still_warns_when_terminals_declared(self):
        rules = [_rule("r1", out=["d.disqualified"]), _rule("r2", out=["d.other"])]
        report = RuleGraphValidator().validate(
            rules, terminal_outputs={"d": {"d.disqualified"}}
        )
        assert len(report.warnings) == 1
        assert report.warnings[0].field == "d.other"

    def test_misspelled_terminal_still_warns(self):
        rules = [_rule("r1", out=["d.disqualifeid"])]  # typo
        report = RuleGraphValidator().validate(
            rules, terminal_outputs={"d": {"d.disqualified"}}
        )
        assert len(report.warnings) == 1
        assert report.warnings[0].field == "d.disqualifeid"

    def test_no_terminal_outputs_arg_all_orphans_warn(self):
        rules = [_rule("r1", out=["d.disqualified"])]
        report = RuleGraphValidator().validate(rules)
        assert len(report.warnings) == 1


# ── TestRuleRegistryExclusion ─────────────────────────────────────────────────

class TestRuleRegistryExclusion:

    _APPEAL_FILE   = Path(__file__).parent.parent.parent / "data" / "out" / "claim_appeal_rules.json"
    _COVERAGE_FILE = Path(__file__).parent.parent.parent / "data" / "out" / "policy_coverage_rules.json"
    _POLICY_FILE   = Path(__file__).parent.parent.parent / "data" / "out" / "policy_rules.json"

    def test_phantom_rule_excluded_from_get_effective(self):
        registry = RuleRegistry()
        registry.load([
            _rule("r1", out=["d.flag"]),
            _rule("r2", inp=["d.missing"], out=["d.done"]),  # phantom consumer
        ])
        registry._excluded_rule_ids = {"r2"}
        effective_ids = {r.id for r in registry._get_effective("d")}
        assert "r2" not in effective_ids
        assert "r1" in effective_ids

    def test_phantom_rule_preserved_in_registry_all(self):
        registry = RuleRegistry()
        registry.load([
            _rule("r1", out=["d.flag"]),
            _rule("r2", inp=["d.missing"]),
        ])
        registry._excluded_rule_ids = {"r2"}
        all_ids = {r.id for r in registry.all()}
        assert "r2" in all_ids

    def test_phantom_rules_excluded_after_load_from(self):
        registry = RuleRegistry.load_from(
            str(self._APPEAL_FILE),
            str(self._COVERAGE_FILE),
            str(self._POLICY_FILE),
        )
        assert "ca_test_phantom_consumer" in registry._excluded_rule_ids

    def test_validation_report_populated_after_load_from(self):
        registry = RuleRegistry.load_from(
            str(self._APPEAL_FILE),
            str(self._COVERAGE_FILE),
            str(self._POLICY_FILE),
        )
        assert isinstance(registry._validation_report, RuleGraphValidationReport)
        assert isinstance(registry._validation_report.validated_at, datetime)


# ── TestLoadFromValidation ────────────────────────────────────────────────────

class TestLoadFromValidation:

    _APPEAL_FILE   = Path(__file__).parent.parent.parent / "data" / "out" / "claim_appeal_rules.json"
    _COVERAGE_FILE = Path(__file__).parent.parent.parent / "data" / "out" / "policy_coverage_rules.json"
    _POLICY_FILE   = Path(__file__).parent.parent.parent / "data" / "out" / "policy_rules.json"

    def test_production_rules_load_without_exception(self):
        RuleRegistry.load_from(
            str(self._APPEAL_FILE),
            str(self._COVERAGE_FILE),
            str(self._POLICY_FILE),
        )

    def test_production_rules_report_phantom_consumer_error(self):
        registry = RuleRegistry.load_from(
            str(self._APPEAL_FILE),
            str(self._COVERAGE_FILE),
            str(self._POLICY_FILE),
        )
        error_ids = {f.rule_id for f in registry._validation_report.errors}
        assert "ca_test_phantom_consumer" in error_ids

    def test_production_rules_have_orphan_producer_warnings(self):
        registry = RuleRegistry.load_from(
            str(self._APPEAL_FILE),
            str(self._COVERAGE_FILE),
            str(self._POLICY_FILE),
        )
        assert len(registry._validation_report.warnings) > 0
        assert all(
            w.kind == RuleValidationKind.ORPHAN_PRODUCER
            for w in registry._validation_report.warnings
        )

    def test_phantom_consumer_excluded_not_exception(self, tmp_path):
        broken_rule = {
            "id": "r_broken",
            "domain": "test_domain",
            "priority": 1,
            "input": ["test_domain.ghost_field"],
            "output": ["test_domain.result"],
            "subject": "claim",
            "attribute": "status",
            "operator": "==",
            "threshold": "open",
            "effective_from": (datetime.now(timezone.utc) + timedelta(days=-1)).isoformat(),
            "effective_to":   (datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
        }
        import json
        rule_file = tmp_path / "broken_rules.json"
        rule_file.write_text(json.dumps([broken_rule]))

        registry = RuleRegistry.load_from(str(rule_file))
        assert "r_broken" in registry._excluded_rule_ids
        assert "r_broken" not in {r.id for r in registry._get_effective("test_domain")}
        assert "r_broken" in {r.id for r in registry.all()}
