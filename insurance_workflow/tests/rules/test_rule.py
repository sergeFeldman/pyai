"""Tests for Rule base class, DecisionRule, and LookupRule."""

from datetime import datetime, timedelta, timezone

import pytest

from rules import DecisionRule, LookupRule
from shared.core import EntityMetadata


def _ts(delta_days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=delta_days)).isoformat()


def _decision(**kwargs) -> DecisionRule:
    defaults = dict(
        id="r1", domain="test", priority=0,
        subject="claim", attribute="status", operator="==", threshold="open",
        effective_from=_ts(-1), effective_to=_ts(365),
    )
    return DecisionRule(**(defaults | kwargs))


class TestIsActive:
    def test_within_range_is_true(self):
        assert _decision(effective_from=_ts(-1), effective_to=_ts(1)).is_active is True

    def test_before_effective_from_is_false(self):
        assert _decision(effective_from=_ts(1), effective_to=_ts(2)).is_active is False

    def test_after_effective_to_is_false(self):
        assert _decision(effective_from=_ts(-2), effective_to=_ts(-1)).is_active is False

    def test_missing_effective_from_is_false(self):
        assert _decision(effective_from="").is_active is False

    def test_missing_effective_to_is_false(self):
        assert _decision(effective_to="").is_active is False

    def test_invalid_iso_string_is_false(self):
        assert _decision(effective_from="not-a-date").is_active is False


class TestIsChanged:
    def test_identical_rules_not_changed(self):
        r1 = _decision(effective_to=_ts(100))
        r2 = _decision(effective_to=_ts(100))
        assert r1.is_changed(r2) is False

    def test_different_priority_is_changed(self):
        assert _decision(priority=10).is_changed(_decision(priority=20)) is True

    def test_different_effective_to_is_changed(self):
        assert _decision(effective_to=_ts(1)).is_changed(_decision(effective_to=_ts(365))) is True

    def test_metadata_difference_not_changed(self):
        r1 = _decision()
        r2 = _decision()
        r2.metadata = EntityMetadata(version=5, updated_by="etl")
        assert r1.is_changed(r2) is False


class TestDecisionRuleMatches:
    def _r(self, operator: str, threshold: str) -> DecisionRule:
        return DecisionRule(
            id="r", domain="d",
            subject="claim", attribute="amount",
            operator=operator, threshold=threshold,
        )

    def test_gte_true(self):
        assert self._r(">=", "100").matches(100) is True

    def test_gte_false(self):
        assert self._r(">=", "100").matches(99) is False

    def test_lte_true(self):
        assert self._r("<=", "50").matches(50) is True

    def test_lte_false(self):
        assert self._r("<=", "50").matches(51) is False

    def test_eq_string(self):
        assert self._r("==", "open").matches("open") is True

    def test_neq_string(self):
        assert self._r("!=", "closed").matches("open") is True

    def test_gt_float(self):
        assert self._r(">", "1.5").matches(2.0) is True

    def test_lt_float(self):
        assert self._r("<", "3.0").matches(2.9) is True

    def test_bool_false_string_coerced_correctly(self):
        # bool("False") is True in Python — _coerce must handle this
        assert self._r("==", "False").matches(False) is True

    def test_bool_true_string_coerced_correctly(self):
        assert self._r("==", "true").matches(True) is True


class TestDecisionRuleReady:
    def _r(self, inp: list[str] | None = None) -> DecisionRule:
        return DecisionRule(id="r", domain="d", subject="claim", attribute="amount",
                            operator="<", threshold="500", input=inp or [])

    def test_true_when_key_in_context(self):
        assert self._r().ready({"claim.amount": 400}) is True

    def test_false_when_key_missing(self):
        assert self._r().ready({"claim.status": "denied"}) is False

    def test_false_when_input_dep_missing(self):
        assert self._r(inp=["appeal.fraud_flagged"]).ready({"claim.amount": 400}) is False

    def test_true_when_key_and_input_dep_present(self):
        assert self._r(inp=["appeal.fraud_flagged"]).ready(
            {"claim.amount": 400, "appeal.fraud_flagged": True}
        ) is True


class TestDecisionRuleEvaluate:
    def _r(self, output: list[str] | None = None) -> DecisionRule:
        return DecisionRule(id="r", domain="d", subject="claim", attribute="amount",
                            operator="<", threshold="500", output=output or ["appeal.disqualified"])

    def test_match_returns_true_with_output_fields(self):
        matched, outputs = self._r().evaluate({"claim.amount": 400})
        assert matched is True
        assert outputs == {"appeal.disqualified": True}

    def test_no_match_returns_false_empty_dict(self):
        matched, outputs = self._r().evaluate({"claim.amount": 5000})
        assert matched is False
        assert outputs == {}

    def test_multiple_output_fields_all_set_to_true(self):
        matched, outputs = self._r(output=["appeal.disqualified", "appeal.fraud_flagged"]).evaluate(
            {"claim.amount": 400}
        )
        assert matched is True
        assert outputs == {"appeal.disqualified": True, "appeal.fraud_flagged": True}


class TestLookupRuleReady:
    def _r(self, match_keys: dict | None = None, inp: list[str] | None = None) -> LookupRule:
        return LookupRule(id="r", domain="d",
                          match_keys=match_keys or {"claim.claim_type": "theft"},
                          output_values={"appeal.tier": "high"},
                          input=inp or [])

    def test_true_when_all_match_keys_in_context(self):
        assert self._r().ready({"claim.claim_type": "theft"}) is True

    def test_false_when_match_key_missing_from_context(self):
        assert self._r().ready({"claim.status": "denied"}) is False

    def test_false_when_input_dep_missing(self):
        assert self._r(inp=["appeal.fraud_flagged"]).ready({"claim.claim_type": "theft"}) is False

    def test_true_when_all_conditions_met(self):
        r = self._r(match_keys={"claim.claim_type": "theft", "claim.status": "denied"})
        assert r.ready({"claim.claim_type": "theft", "claim.status": "denied"}) is True

    def test_false_when_only_partial_match_keys_present(self):
        r = self._r(match_keys={"claim.claim_type": "theft", "claim.status": "denied"})
        assert r.ready({"claim.claim_type": "theft"}) is False


class TestLookupRuleEvaluate:
    def _r(self, match_keys: dict | None = None, output_values: dict | None = None) -> LookupRule:
        return LookupRule(id="r", domain="d",
                          match_keys=match_keys or {"claim.claim_type": "theft"},
                          output_values=output_values or {"appeal.disqualified": "true",
                                                          "appeal.reason": "theft not eligible"})

    def test_match_returns_true_with_output_values(self):
        matched, outputs = self._r().evaluate({"claim.claim_type": "theft"})
        assert matched is True
        assert outputs == {"appeal.disqualified": "true", "appeal.reason": "theft not eligible"}

    def test_no_match_returns_false_empty_dict(self):
        matched, outputs = self._r().evaluate({"claim.claim_type": "auto"})
        assert matched is False
        assert outputs == {}

    def test_returned_dict_is_copy_not_reference(self):
        r = self._r(output_values={"x": "1"})
        _, outputs = r.evaluate({"claim.claim_type": "theft"})
        outputs["x"] = "modified"
        assert r.output_values["x"] == "1"


class TestLookupRuleMatches:
    def _rule(self, match_keys: dict) -> LookupRule:
        return LookupRule(id="r", domain="d", match_keys=match_keys, output_values={"basis": "x"})

    def test_all_keys_match(self):
        assert self._rule({"claim_type": "auto", "status": "denied"}).matches(
            {"claim_type": "auto", "status": "denied"}
        ) is True

    def test_partial_match_is_false(self):
        assert self._rule({"claim_type": "auto", "status": "denied"}).matches(
            {"claim_type": "auto"}
        ) is False

    def test_wrong_value_is_false(self):
        assert self._rule({"claim_type": "auto"}).matches({"claim_type": "property_damage"}) is False

    def test_empty_context_is_false(self):
        assert self._rule({"claim_type": "auto"}).matches({}) is False

    def test_extra_context_keys_ignored(self):
        assert self._rule({"claim_type": "auto"}).matches(
            {"claim_type": "auto", "unrelated": "value"}
        ) is True
