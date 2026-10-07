"""Tests for ClaimCoverageAgent — negative-gating model and DAG chain paths."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agents.claim_coverage_agent import ClaimCoverageAgent, ClaimCoverageAgentConfig
import models as mdl
from rules import DecisionRule, LookupRule, RuleFactory, RuleRegistry
from services import RuleExecutionAuditService
from shared.core import EntityMetadata, Singleton


def _ts(delta_days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=delta_days)).isoformat()


def _decision(rule_id: str, subject: str, attribute: str, operator: str, threshold: str,
              inp: list[str] | None = None, out: list[str] | None = None,
              priority: int = 0, reason: str = "") -> DecisionRule:
    return DecisionRule(
        id=rule_id, domain="policy_coverage", priority=priority,
        input=inp or [], output=out or [],
        subject=subject, attribute=attribute,
        operator=operator, threshold=threshold,
        reason=reason,
        effective_from=_ts(-1), effective_to=_ts(365),
    )


def _lookup(rule_id: str, match_keys: dict, inp: list[str] | None = None,
            out: list[str] | None = None, priority: int = 0, reason: str = "") -> LookupRule:
    return LookupRule(
        id=rule_id, domain="policy_coverage", priority=priority,
        input=inp or [], output=out or ["policy_coverage.disqualified"],
        match_keys=match_keys,
        output_values={"policy_coverage.disqualified": True},
        reason=reason,
        effective_from=_ts(-1), effective_to=_ts(365),
    )


def _claim(**kwargs) -> mdl.Claim:
    defaults = dict(
        claim_id="claim_1",
        claim_type="auto_collision",
        customer_id="cust_1",
        amount=2000.0,
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


def _policy(**kwargs) -> mdl.Policy:
    defaults = dict(
        policy_id="pol_cust_1",
        customer_id="cust_1",
        policy_type="comprehensive",
        policy_status="active",
        coverage_limit=50000.0,
        deductible_amount=1000.0,
    )
    return mdl.Policy(**(defaults | kwargs))


@pytest.fixture(autouse=True)
def reset_singletons():
    yield
    Singleton._instances.pop(RuleRegistry, None)
    Singleton._instances.pop(RuleFactory, None)
    Singleton._instances.pop(RuleExecutionAuditService, None)


class TestClaimCoverageAgent:
    def setup_method(self):
        RuleRegistry()._domain_configs["policy_coverage"] = mdl.DomainConfig(
            domain="policy_coverage", terminal_outputs=["policy_coverage.disqualified"]
        )

    def _agent(self) -> ClaimCoverageAgent:
        return ClaimCoverageAgent(ClaimCoverageAgentConfig())

    def test_no_rules_fire_claim_verified(self):
        RuleRegistry().load([
            _decision("r1", "policy", "policy_status", "==", "cancelled",
                      out=["policy_coverage.disqualified"]),
        ])
        result = self._agent().check_coverage(_claim(), _customer(), _policy())
        assert result.verified is True

    def test_cancelled_policy_disqualifies(self):
        RuleRegistry().load([
            _decision("pc_cancelled", "policy", "policy_status", "==", "cancelled",
                      out=["policy_coverage.disqualified"],
                      reason="Policy has been cancelled."),
        ])
        result = self._agent().check_coverage(_claim(), _customer(), _policy(policy_status="cancelled"))
        assert result.verified is False
        assert result.reason == "Policy has been cancelled."

    def test_lapsed_policy_disqualifies(self):
        RuleRegistry().load([
            _decision("pc_lapsed", "policy", "policy_status", "==", "lapsed",
                      out=["policy_coverage.disqualified"],
                      reason="Policy has lapsed."),
        ])
        result = self._agent().check_coverage(_claim(), _customer(), _policy(policy_status="lapsed"))
        assert result.verified is False
        assert result.reason == "Policy has lapsed."

    def test_active_policy_not_cancelled(self):
        RuleRegistry().load([
            _decision("pc_cancelled", "policy", "policy_status", "==", "cancelled",
                      out=["policy_coverage.disqualified"]),
        ])
        result = self._agent().check_coverage(_claim(), _customer(), _policy(policy_status="active"))
        assert result.verified is True

    def test_auto_collision_third_party_only_disqualifies(self):
        RuleRegistry().load([
            _lookup("pc_auto_tp", {"claim.claim_type": "auto_collision", "policy.policy_type": "third_party_only"},
                    reason="Auto collision not covered under third-party-only."),
        ])
        result = self._agent().check_coverage(
            _claim(claim_type="auto_collision"),
            _customer(),
            _policy(policy_type="third_party_only"),
        )
        assert result.verified is False
        assert result.reason == "Auto collision not covered under third-party-only."

    def test_theft_basic_auto_disqualifies(self):
        RuleRegistry().load([
            _lookup("pc_theft_basic", {"claim.claim_type": "theft", "policy.policy_type": "basic_auto"},
                    reason="Theft not covered under basic auto."),
        ])
        result = self._agent().check_coverage(
            _claim(claim_type="theft"),
            _customer(),
            _policy(policy_type="basic_auto"),
        )
        assert result.verified is False
        assert result.reason == "Theft not covered under basic auto."

    def test_auto_collision_comprehensive_verified(self):
        RuleRegistry().load([
            _lookup("pc_auto_tp", {"claim.claim_type": "auto_collision", "policy.policy_type": "third_party_only"}),
        ])
        result = self._agent().check_coverage(
            _claim(claim_type="auto_collision"),
            _customer(),
            _policy(policy_type="comprehensive"),
        )
        assert result.verified is True

    def test_below_deductible_disqualifies(self):
        RuleRegistry().load([
            _decision("pc_below_ded", "claim", "below_deductible", "==", "True",
                      out=["policy_coverage.disqualified"],
                      reason="Claim amount is below the policy deductible."),
        ])
        result = self._agent().check_coverage(
            _claim(amount=300.0),
            _customer(),
            _policy(deductible_amount=500.0),
        )
        assert result.verified is False
        assert result.reason == "Claim amount is below the policy deductible."

    def test_above_deductible_verified(self):
        RuleRegistry().load([
            _decision("pc_below_ded", "claim", "below_deductible", "==", "True",
                      out=["policy_coverage.disqualified"]),
        ])
        result = self._agent().check_coverage(
            _claim(amount=1500.0),
            _customer(),
            _policy(deductible_amount=500.0),
        )
        assert result.verified is True

    def test_result_claim_id_matches_input(self):
        RuleRegistry().load([])
        result = self._agent().check_coverage(_claim(claim_id="claim_42"), _customer(), _policy())
        assert result.claim_id == "claim_42"

    def test_high_value_chain_no_prior_claims_verified(self):
        """pc_high_value_flag fires but pc_high_value_repeat_claimant threshold not met."""
        RuleRegistry().load([
            _decision("pc_hv_flag", "claim", "amount", ">", "10000",
                      inp=[], out=["policy_coverage.high_value_flagged"], priority=4),
            _decision("pc_hv_repeat", "customer", "prior_claim_count", ">=", "3",
                      inp=["policy_coverage.high_value_flagged"],
                      out=["policy_coverage.disqualified"], priority=3,
                      reason="High-value repeat claimant."),
        ])
        result = self._agent().check_coverage(
            _claim(amount=15000.0),
            _customer(prior_claim_count=2),
            _policy(),
        )
        assert result.verified is True

    def test_high_value_chain_with_prior_claims_disqualifies(self):
        """pc_high_value_flag fires, pc_high_value_repeat_claimant fires via DAG chain."""
        RuleRegistry().load([
            _decision("pc_hv_flag", "claim", "amount", ">", "10000",
                      inp=[], out=["policy_coverage.high_value_flagged"], priority=4),
            _decision("pc_hv_repeat", "customer", "prior_claim_count", ">=", "3",
                      inp=["policy_coverage.high_value_flagged"],
                      out=["policy_coverage.disqualified"], priority=3,
                      reason="High-value repeat claimant."),
        ])
        result = self._agent().check_coverage(
            _claim(amount=15000.0),
            _customer(prior_claim_count=5),
            _policy(),
        )
        assert result.verified is False
        assert result.reason == "High-value repeat claimant."

    def test_high_value_flag_not_produced_consumer_pruned(self):
        """pc_high_value_flag does not fire; pc_high_value_repeat_claimant is pruned."""
        RuleRegistry().load([
            _decision("pc_hv_flag", "claim", "amount", ">", "10000",
                      inp=[], out=["policy_coverage.high_value_flagged"], priority=4),
            _decision("pc_hv_repeat", "customer", "prior_claim_count", ">=", "3",
                      inp=["policy_coverage.high_value_flagged"],
                      out=["policy_coverage.disqualified"], priority=3),
        ])
        result = self._agent().check_coverage(
            _claim(amount=5000.0),
            _customer(prior_claim_count=10),
            _policy(),
        )
        assert result.verified is True


class TestClaimCoverageAgentAudit:
    _AUDIT_FILE = Path(__file__).parent.parent.parent / "data" / "test" / "audit" / "rule_executions.jsonl"

    @pytest.fixture(autouse=True)
    def clear_audit_file(self):
        if self._AUDIT_FILE.exists():
            self._AUDIT_FILE.write_text("")

    def _agent(self) -> ClaimCoverageAgent:
        return ClaimCoverageAgent(ClaimCoverageAgentConfig())

    def _last_record(self) -> dict:
        return RuleExecutionAuditService().list("policy_coverage")[0]

    def test_audit_record_written_on_coverage_check(self):
        RuleRegistry().load([
            _decision("r1", "policy", "policy_status", "==", "cancelled",
                      out=["policy_coverage.disqualified"]),
        ])
        self._agent().check_coverage(_claim(), _customer(), _policy(), trace_id="t1")
        record = self._last_record()
        assert record["domain"] == "policy_coverage"

    def test_entities_contain_claim_customer_policy(self):
        RuleRegistry().load([])
        self._agent().check_coverage(_claim(), _customer(), _policy(), trace_id="t1")
        record = self._last_record()
        assert "claim" in record["entities"]
        assert "customer" in record["entities"]
        assert "policy" in record["entities"]

    def test_entities_claim_id_matches_input(self):
        RuleRegistry().load([])
        self._agent().check_coverage(_claim(claim_id="claim_xyz"), _customer(), _policy(), trace_id="t1")
        assert self._last_record()["entities"]["claim"]["claim_id"] == "claim_xyz"

    def test_all_evaluations_have_rule_id_and_outcome(self):
        RuleRegistry().load([
            _decision("r1", "policy", "policy_status", "==", "active",
                      out=["policy_coverage.disqualified"]),
        ])
        self._agent().check_coverage(_claim(), _customer(), _policy(), trace_id="t1")
        for ev in self._last_record()["evaluations"]:
            assert "rule_id" in ev
            assert ev["outcome"] in {"triggered", "skipped_no_match", "skipped_precondition", "pruned"}


class TestClaimCoverageAgentIntegration:
    """Integration tests loading production rules from data/out/policy_coverage_rules.json."""

    _RULES_FILE = Path(__file__).parent.parent.parent / "data" / "out" / "policy_coverage_rules.json"
    _AUDIT_FILE = Path(__file__).parent.parent.parent / "data" / "test" / "audit" / "rule_executions.jsonl"
    _DOMAIN_CONFIG_FILE = Path(__file__).parent.parent.parent / "data" / "out" / "domain_config.json"

    @pytest.fixture(autouse=True)
    def load_production_rules(self):
        RuleRegistry.load_from(
            str(self._RULES_FILE),
            domain_config_path=str(self._DOMAIN_CONFIG_FILE),
        )
        yield
        if self._AUDIT_FILE.exists():
            self._AUDIT_FILE.write_text("")

    def _agent(self) -> ClaimCoverageAgent:
        return ClaimCoverageAgent(ClaimCoverageAgentConfig())

    def test_cancelled_policy_disqualified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="auto_collision", amount=2000.0),
            _customer(),
            _policy(policy_status="cancelled"),
        )
        assert result.verified is False

    def test_lapsed_policy_disqualified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="auto_collision", amount=2000.0),
            _customer(),
            _policy(policy_status="lapsed"),
        )
        assert result.verified is False

    def test_auto_collision_third_party_only_disqualified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="auto_collision", amount=2000.0),
            _customer(),
            _policy(policy_type="third_party_only"),
        )
        assert result.verified is False

    def test_theft_basic_auto_disqualified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="theft", amount=2000.0),
            _customer(),
            _policy(policy_type="basic_auto"),
        )
        assert result.verified is False

    def test_property_damage_third_party_only_disqualified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="property_damage", amount=2000.0),
            _customer(),
            _policy(policy_type="third_party_only"),
        )
        assert result.verified is False

    def test_below_deductible_disqualified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="auto_collision", amount=300.0),
            _customer(),
            _policy(policy_type="comprehensive", deductible_amount=500.0),
        )
        assert result.verified is False

    def test_high_value_repeat_claimant_disqualified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="auto_collision", amount=15000.0),
            _customer(prior_claim_count=5),
            _policy(policy_type="comprehensive", deductible_amount=500.0),
        )
        assert result.verified is False

    def test_clean_claim_comprehensive_verified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="auto_collision", amount=2000.0),
            _customer(prior_claim_count=0),
            _policy(policy_type="comprehensive", policy_status="active", deductible_amount=500.0),
        )
        assert result.verified is True

    def test_theft_comprehensive_verified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="theft", amount=2000.0),
            _customer(prior_claim_count=0),
            _policy(policy_type="comprehensive", policy_status="active", deductible_amount=500.0),
        )
        assert result.verified is True

    def test_high_value_low_prior_claims_verified(self):
        result = self._agent().check_coverage(
            _claim(claim_type="auto_collision", amount=15000.0),
            _customer(prior_claim_count=2),
            _policy(policy_type="comprehensive", policy_status="active", deductible_amount=500.0),
        )
        assert result.verified is True
