"""Tests for ClaimAppealAgent — precondition gating and disqualification paths."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agents.claim_appeal_agent import ClaimAppealAgent, ClaimAppealAgentConfig
import models as mdl
from rules import DecisionRule, RuleCondition, RuleFactory, RuleLogic, RuleOperator, RuleRegistry
from services import RuleExecutionAuditService
from shared.core import Singleton


def _ts(delta_days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=delta_days)).isoformat()


def _rule(rule_id: str, subject: str, attribute: str, operator: str, threshold: str,
          inp: list[str] | None = None, out: list[str] | None = None,
          priority: int = 0, reason: str = "") -> DecisionRule:
    return DecisionRule(
        id=rule_id, domain="claim_appeal", priority=priority,
        input=inp or [], output=out or [],
        subject=subject, attribute=attribute,
        operator=operator, threshold=threshold,
        reason=reason,
        effective_from=_ts(-1), effective_to=_ts(365),
    )


def _claim(**kwargs) -> mdl.Claim:
    defaults = dict(
        claim_id="claim_1",
        claim_type="auto_collision",
        customer_id="cust_1",
        amount=5000.0,
        date="2026-01-01",
        repair_shop="shop_1",
        status=mdl.ClaimStatus.DENIED,
        is_fraud=False,
    )
    return mdl.Claim(**(defaults | kwargs))


def _customer(**kwargs) -> mdl.Customer:
    defaults = dict(
        customer_id="cust_1",
        tenure_years=5,
        active_policy_count=1,
        prior_claim_count=0,
        last_interaction_date="2026-01-01",
        preferred_contact_method="email",
        escalation_history_count=0,
    )
    return mdl.Customer(**(defaults | kwargs))


def _cond(subject: str, attribute: str, operator: str, threshold: str) -> RuleCondition:
    return RuleCondition(subject=subject, attribute=attribute,
                         operator=RuleOperator(operator), threshold=threshold)


def _group(logic: str, conditions: list) -> RuleCondition:
    return RuleCondition(logic=RuleLogic(logic), conditions=conditions)


def _compound_rule(rule_id: str, conditions: list, logic: str = "AND",
                   inp: list | None = None, out: list | None = None,
                   priority: int = 0, reason: str = "") -> DecisionRule:
    return DecisionRule(
        id=rule_id, domain="claim_appeal", priority=priority,
        input=inp or [],
        output=out or ["appeal.disqualified"],
        conditions=conditions,
        logic=RuleLogic(logic),
        reason=reason,
        effective_from=_ts(-1), effective_to=_ts(365),
    )


@pytest.fixture(autouse=True)
def reset_singletons():
    yield
    Singleton._instances.pop(RuleRegistry, None)
    Singleton._instances.pop(RuleFactory, None)
    Singleton._instances.pop(RuleExecutionAuditService, None)


class TestClaimAppealAgent:
    def _agent(self) -> ClaimAppealAgent:
        return ClaimAppealAgent(ClaimAppealAgentConfig())

    def test_no_rules_fire_claim_eligible(self):
        RuleRegistry().load([
            _rule("r1", "claim", "amount", "<", "0", out=["appeal.disqualified"]),
        ])
        result = self._agent().check_eligibility(_claim(), _customer())
        assert result.eligible is True

    def test_simple_rule_match_disqualifies(self):
        RuleRegistry().load([
            _rule("r1", "claim", "amount", "<", "1000",
                  out=["appeal.disqualified"],
                  reason="Claim amount too low for appeal."),
        ])
        result = self._agent().check_eligibility(_claim(amount=500.0), _customer())
        assert result.eligible is False
        assert result.reason == "Claim amount too low for appeal."

    def test_fraud_alone_disqualifies(self):
        RuleRegistry().load([
            _rule("r1", "claim", "is_fraud", "==", "True",
                  out=["appeal.disqualified"],
                  reason="Fraud claims are not eligible for appeal."),
        ])
        result = self._agent().check_eligibility(_claim(is_fraud=True), _customer())
        assert result.eligible is False
        assert result.reason == "Fraud claims are not eligible for appeal."

    def test_repeat_claimant_with_escalation_disqualifies(self):
        RuleRegistry().load([
            _compound_rule(
                "r1",
                [_cond("customer", "prior_claim_count", ">=", "5"),
                 _cond("customer", "escalation_history_count", ">=", "2")],
                reason="Repeat claimant with multiple escalation attempts.",
            ),
        ])
        result = self._agent().check_eligibility(
            _claim(),
            _customer(prior_claim_count=6, escalation_history_count=2),
        )
        assert result.eligible is False
        assert result.reason == "Repeat claimant with multiple escalation attempts."

    def test_repeat_claimant_low_escalation_eligible(self):
        RuleRegistry().load([
            _compound_rule(
                "r1",
                [_cond("customer", "prior_claim_count", ">=", "5"),
                 _cond("customer", "escalation_history_count", ">=", "2")],
            ),
        ])
        result = self._agent().check_eligibility(
            _claim(),
            _customer(prior_claim_count=6, escalation_history_count=1),
        )
        assert result.eligible is True

    def test_low_amount_low_tenure_disqualifies(self):
        RuleRegistry().load([
            _compound_rule(
                "r1",
                [_cond("claim", "amount", "<", "500"),
                 _cond("customer", "tenure_years", "<", "5")],
                reason="Low-value claim requires minimum 5 years customer tenure for appeal.",
            ),
        ])
        result = self._agent().check_eligibility(_claim(amount=400.0), _customer(tenure_years=3))
        assert result.eligible is False
        assert result.reason == "Low-value claim requires minimum 5 years customer tenure for appeal."

    def test_low_amount_sufficient_tenure_eligible(self):
        RuleRegistry().load([
            _compound_rule(
                "r1",
                [_cond("claim", "amount", "<", "500"),
                 _cond("customer", "tenure_years", "<", "5")],
            ),
        ])
        result = self._agent().check_eligibility(_claim(amount=400.0), _customer(tenure_years=10))
        assert result.eligible is True

    def test_nested_or_group_and_leaf_disqualifies(self):
        RuleRegistry().load([
            _compound_rule(
                "r1",
                [_group("OR", [
                    _cond("claim", "is_fraud", "==", "True"),
                    _cond("customer", "prior_claim_count", ">=", "5"),
                 ]),
                 _cond("customer", "escalation_history_count", ">=", "2")],
                inp=["claim.is_fraud", "customer.prior_claim_count",
                     "customer.escalation_history_count"],
                reason="High-risk claimant with escalation history.",
            ),
        ])
        result = self._agent().check_eligibility(
            _claim(is_fraud=True),
            _customer(prior_claim_count=1, escalation_history_count=2),
        )
        assert result.eligible is False
        assert result.reason == "High-risk claimant with escalation history."

    def test_nested_or_group_escalation_below_threshold_eligible(self):
        RuleRegistry().load([
            _compound_rule(
                "r1",
                [_group("OR", [
                    _cond("claim", "is_fraud", "==", "True"),
                    _cond("customer", "prior_claim_count", ">=", "5"),
                 ]),
                 _cond("customer", "escalation_history_count", ">=", "2")],
                inp=["claim.is_fraud", "customer.prior_claim_count",
                     "customer.escalation_history_count"],
            ),
        ])
        result = self._agent().check_eligibility(
            _claim(is_fraud=True),
            _customer(prior_claim_count=1, escalation_history_count=1),
        )
        assert result.eligible is True

    def test_result_claim_id_matches_input(self):
        RuleRegistry().load([])
        result = self._agent().check_eligibility(_claim(claim_id="claim_42"), _customer())
        assert result.claim_id == "claim_42"


class TestClaimAppealAgentEvaluations:
    _AUDIT_FILE = Path(__file__).parent.parent.parent / "data" / "test" / "audit" / "rule_executions.jsonl"

    @pytest.fixture(autouse=True)
    def clear_audit_file(self):
        if self._AUDIT_FILE.exists():
            self._AUDIT_FILE.write_text("")

    def _agent(self) -> ClaimAppealAgent:
        return ClaimAppealAgent(ClaimAppealAgentConfig())

    def _last_record(self) -> dict:
        return RuleExecutionAuditService().list("claim_appeal")[0]

    def test_evaluations_has_one_entry_per_rule(self):
        RuleRegistry().load([
            _rule("r1", "claim", "amount", "<", "0", out=["appeal.disqualified"]),
            _rule("r2", "claim", "status", "==", "open", out=["appeal.x"]),
        ])
        self._agent().check_eligibility(_claim(), _customer(), trace_id="t1")
        assert len(self._last_record()["evaluations"]) == 2

    def test_all_evaluations_have_rule_id_and_outcome(self):
        RuleRegistry().load([
            _rule("r1", "claim", "amount", "<", "9999", out=["appeal.disqualified"]),
        ])
        self._agent().check_eligibility(_claim(), _customer(), trace_id="t1")
        for ev in self._last_record()["evaluations"]:
            assert "rule_id" in ev
            assert ev["outcome"] in {"triggered", "skipped_no_match", "skipped_precondition"}

    def test_triggered_rule_appears_in_evaluations(self):
        RuleRegistry().load([
            _rule("r1", "claim", "amount", "<", "9999", out=["appeal.disqualified"]),
        ])
        self._agent().check_eligibility(_claim(amount=100.0), _customer(), trace_id="t1")
        outcomes = {e["rule_id"]: e["outcome"] for e in self._last_record()["evaluations"]}
        assert outcomes["r1"] == "triggered"

    def test_entities_contains_claim_and_customer_keys(self):
        RuleRegistry().load([])
        self._agent().check_eligibility(_claim(), _customer(), trace_id="t1")
        record = self._last_record()
        assert "claim" in record["entities"]
        assert "customer" in record["entities"]

    def test_entities_claim_id_matches_input(self):
        RuleRegistry().load([])
        self._agent().check_eligibility(_claim(claim_id="claim_xyz"), _customer(), trace_id="t1")
        assert self._last_record()["entities"]["claim"]["claim_id"] == "claim_xyz"

    def test_entities_customer_id_matches_input(self):
        RuleRegistry().load([])
        self._agent().check_eligibility(_claim(), _customer(customer_id="cust_xyz"), trace_id="t1")
        assert self._last_record()["entities"]["customer"]["customer_id"] == "cust_xyz"


class TestClaimAppealAgentIntegration:
    """Integration tests loading production rules from data/out/claim_appeal_rules.json."""

    _RULES_FILE = Path(__file__).parent.parent.parent / "data" / "out" / "claim_appeal_rules.json"
    _AUDIT_FILE = Path(__file__).parent.parent.parent / "data" / "test" / "audit" / "rule_executions.jsonl"

    @pytest.fixture(autouse=True)
    def load_production_rules(self):
        RuleRegistry.load_from(str(self._RULES_FILE))
        yield
        if self._AUDIT_FILE.exists():
            self._AUDIT_FILE.write_text("")

    def _agent(self) -> ClaimAppealAgent:
        return ClaimAppealAgent(ClaimAppealAgentConfig())

    def test_commercial_claim_disqualified(self):
        result = self._agent().check_eligibility(
            _claim(claim_type="commercial"),
            _customer(tenure_years=10, prior_claim_count=0),
        )
        assert result.eligible is False

    def test_blacklisted_shop_disqualified(self):
        result = self._agent().check_eligibility(
            _claim(repair_shop="shop_rouge"),
            _customer(tenure_years=10),
        )
        assert result.eligible is False

    def test_fraud_claim_disqualified(self):
        result = self._agent().check_eligibility(
            _claim(is_fraud=True),
            _customer(tenure_years=10, escalation_history_count=0),
        )
        assert result.eligible is False

    def test_theft_low_tenure_disqualified(self):
        result = self._agent().check_eligibility(
            _claim(claim_type="theft"),
            _customer(tenure_years=3),
        )
        assert result.eligible is False

    def test_theft_sufficient_tenure_not_disqualified_by_tenure_rule(self):
        result = self._agent().check_eligibility(
            _claim(claim_type="theft", amount=5000.0, is_fraud=False,
                   repair_shop="shop_1", status=mdl.ClaimStatus.DENIED),
            _customer(tenure_years=5, prior_claim_count=0, escalation_history_count=0),
        )
        assert result.eligible is True

    def test_non_denied_status_disqualified(self):
        result = self._agent().check_eligibility(
            _claim(status=mdl.ClaimStatus.UNDER_REVIEW),
            _customer(tenure_years=10),
        )
        assert result.eligible is False

    def test_standard_clean_claim_eligible(self):
        result = self._agent().check_eligibility(
            _claim(claim_type="auto_collision", amount=5000.0, is_fraud=False,
                   repair_shop="shop_1", status=mdl.ClaimStatus.DENIED),
            _customer(tenure_years=10, prior_claim_count=0, escalation_history_count=0),
        )
        assert result.eligible is True
