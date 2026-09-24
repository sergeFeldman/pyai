"""Tests for RuleRegistry — active rule resolution, graph construction, and execution order."""

import json
from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest

from rules import DecisionRule, LookupRule, RuleFactory, RuleRegistry
from shared.core import EntityMetadata, Singleton


def _ts(delta_days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=delta_days)).isoformat()


def _drule(rule_id: str, subject: str, attribute: str, operator: str, threshold: str,
           inp: list[str] | None = None, out: list[str] | None = None,
           domain: str = "d", priority: int = 0) -> DecisionRule:
    return DecisionRule(
        id=rule_id, domain=domain, priority=priority,
        input=inp or [], output=out or [],
        subject=subject, attribute=attribute,
        operator=operator, threshold=threshold,
        effective_from=_ts(-1), effective_to=_ts(365),
    )


def _rule(rule_id: str, domain: str = "d", priority: int = 0,
          inp: list[str] | None = None, out: list[str] | None = None,
          version: int = 0, active: bool = True, group: str = "") -> DecisionRule:
    dates = (
        {"effective_from": _ts(-1), "effective_to": _ts(365)}
        if active
        else {"effective_from": _ts(-10), "effective_to": _ts(-1)}
    )
    return DecisionRule(
        id=rule_id, domain=domain, priority=priority, group=group,
        input=inp or [], output=out or [],
        subject="claim", attribute="status", operator="==", threshold="open",
        metadata=EntityMetadata(version=version),
        **dates,
    )


def _lrule(rule_id: str, match_keys: dict, output_values: dict,
           inp: list[str] | None = None, out: list[str] | None = None,
           domain: str = "d", priority: int = 0) -> LookupRule:
    return LookupRule(
        id=rule_id, domain=domain, priority=priority,
        input=inp or [],
        output=out or list(output_values.keys()),
        match_keys=match_keys, output_values=output_values,
        effective_from=_ts(-1), effective_to=_ts(365),
    )


@pytest.fixture(autouse=True)
def reset_singletons():
    yield
    Singleton._instances.pop(RuleRegistry, None)
    Singleton._instances.pop(RuleFactory, None)


class TestGetActiveRules:
    def test_highest_version_selected(self):
        reg = RuleRegistry()
        reg.load([_rule("r1", version=0), _rule("r1", version=1)])
        active = reg._get_active_rules("d")
        assert len(active) == 1
        assert active[0].metadata.version == 1

    def test_inactive_rule_excluded(self):
        reg = RuleRegistry()
        reg.load([_rule("r1", active=False)])
        assert reg._get_active_rules("d") == []

    def test_active_and_inactive_versions_selects_active(self):
        reg = RuleRegistry()
        # v0 inactive, v1 active — v1 is highest and active
        reg.load([_rule("r1", version=0, active=False), _rule("r1", version=1, active=True)])
        active = reg._get_active_rules("d")
        assert len(active) == 1
        assert active[0].metadata.version == 1

    def test_group_filter_applied(self):
        reg = RuleRegistry()
        reg.load([_rule("r1", group="g1"), _rule("r2", group="g2")])
        result = reg._get_active_rules("d", group="g1")
        assert [r.id for r in result] == ["r1"]


class TestBuildGraph:
    def test_edge_added_on_field_overlap(self):
        reg = RuleRegistry()
        G = reg._build_graph([_rule("r1", out=["x"]), _rule("r2", inp=["x"])])
        assert G.has_edge("r1", "r2")

    def test_multi_field_edge_carries_all_fields(self):
        reg = RuleRegistry()
        G = reg._build_graph([_rule("r1", out=["x", "y"]), _rule("r2", inp=["x", "y"])])
        assert set(G["r1"]["r2"]["fields"]) == {"x", "y"}

    def test_single_field_edge_carries_one_field(self):
        reg = RuleRegistry()
        G = reg._build_graph([_rule("r1", out=["x"]), _rule("r2", inp=["x"])])
        assert G["r1"]["r2"]["fields"] == ["x"]

    def test_no_edge_without_field_overlap(self):
        reg = RuleRegistry()
        G = reg._build_graph([_rule("r1", out=["x"]), _rule("r2", inp=["y"])])
        assert not G.has_edge("r1", "r2")

    def test_root_node_has_zero_indegree(self):
        reg = RuleRegistry()
        G = reg._build_graph([_rule("r1", inp=["external"], out=["x"]), _rule("r2", inp=["x"])])
        assert G.in_degree("r1") == 0

    def test_consumer_has_correct_indegree(self):
        reg = RuleRegistry()
        G = reg._build_graph([_rule("r1", out=["x"]), _rule("r2", inp=["x"])])
        assert G.in_degree("r2") == 1

    def test_cyclic_graph_detected(self):
        reg = RuleRegistry()
        G = reg._build_graph([_rule("r1", inp=["y"], out=["x"]), _rule("r2", inp=["x"], out=["y"])])
        assert not nx.is_directed_acyclic_graph(G)


class TestGetActive:
    def test_producer_before_consumer(self):
        reg = RuleRegistry()
        reg.load([_rule("r1", out=["x"]), _rule("r2", inp=["x"])])
        order = [r.id for r in reg.get_active("d")]
        assert order.index("r1") < order.index("r2")

    def test_priority_descending_within_same_level(self):
        reg = RuleRegistry()
        reg.load([_rule("r1", priority=10), _rule("r2", priority=20)])
        order = [r.id for r in reg.get_active("d")]
        assert order == ["r2", "r1"]

    def test_id_tiebreaker_on_equal_priority(self):
        reg = RuleRegistry()
        reg.load([_rule("b", priority=5), _rule("a", priority=5)])
        order = [r.id for r in reg.get_active("d")]
        assert order == ["a", "b"]

    def test_only_active_rules_returned(self):
        reg = RuleRegistry()
        reg.load([_rule("r1", active=True), _rule("r2", active=False)])
        ids = [r.id for r in reg.get_active("d")]
        assert ids == ["r1"]


class TestGetDag:
    def test_cache_hit_returns_same_object(self):
        reg = RuleRegistry()
        reg.load([_rule("r1")])
        assert reg.get_dag("d") is reg.get_dag("d")

    def test_replace_with_new_forces_rebuild(self):
        reg = RuleRegistry()
        reg.load([_rule("r1")])
        g1 = reg.get_dag("d")
        g2 = reg.get_dag("d", replace_with_new=True)
        assert g1 is not g2

    def test_load_clears_dag_cache(self):
        reg = RuleRegistry()
        reg.load([_rule("r1")])
        g1 = reg.get_dag("d")
        reg.load([_rule("r1"), _rule("r2")])
        g2 = reg.get_dag("d")
        assert g1 is not g2
        assert len(g2.nodes) == 2

    def test_group_keyed_separately_from_domain(self):
        reg = RuleRegistry()
        reg.load([_rule("r1", group="g1"), _rule("r2", group="g2")])
        g_all = reg.get_dag("d")
        g_g1 = reg.get_dag("d", group="g1")
        assert len(g_all.nodes) == 2
        assert len(g_g1.nodes) == 1


class TestExecute:
    def test_root_rule_fires_and_writes_output(self):
        reg = RuleRegistry()
        rule = _drule("r1", "claim", "amount", "<", "500", out=["appeal.amount_tier"])
        reg.load([rule])
        result = reg.execute("d", {"claim.amount": 400})
        assert result.triggered == [rule]
        assert result.outputs == {"appeal.amount_tier": True}

    def test_precondition_gates_downstream_when_upstream_does_not_fire(self):
        reg = RuleRegistry()
        r1 = _drule("r1", "claim", "is_fraud", "==", "True",
                    out=["appeal.fraud_flagged"], priority=2)
        r2 = _drule("r2", "customer", "escalation_history_count", ">=", "2",
                    inp=["appeal.fraud_flagged"], out=["appeal.disqualified"], priority=1)
        reg.load([r1, r2])
        result = reg.execute("d", {"claim.is_fraud": False,
                                   "customer.escalation_history_count": 3})
        assert result.triggered == []
        assert "appeal.disqualified" not in result.outputs

    def test_upstream_output_enables_downstream(self):
        reg = RuleRegistry()
        r1 = _drule("r1", "claim", "is_fraud", "==", "True",
                    out=["appeal.fraud_flagged"], priority=2)
        r2 = _drule("r2", "customer", "escalation_history_count", ">=", "2",
                    inp=["appeal.fraud_flagged"], out=["appeal.disqualified"], priority=1)
        reg.load([r1, r2])
        result = reg.execute("d", {"claim.is_fraud": True,
                                   "customer.escalation_history_count": 3})
        assert [r.id for r in result.triggered] == ["r1", "r2"]
        assert result.outputs == {"appeal.fraud_flagged": True, "appeal.disqualified": True}

    def test_no_rules_match_empty_triggered_and_outputs(self):
        reg = RuleRegistry()
        rule = _drule("r1", "claim", "amount", "<", "500", out=["appeal.disqualified"])
        reg.load([rule])
        result = reg.execute("d", {"claim.amount": 5000})
        assert result.triggered == []
        assert result.outputs == {}

    def test_intermediate_rule_fires_without_disqualifying(self):
        reg = RuleRegistry()
        rule = _drule("r1", "claim", "is_fraud", "==", "True",
                      out=["appeal.fraud_flagged"])
        reg.load([rule])
        result = reg.execute("d", {"claim.is_fraud": True})
        assert result.triggered == [rule]
        assert "appeal.fraud_flagged" in result.outputs
        assert "appeal.disqualified" not in result.outputs

    def test_missing_context_key_skips_rule(self):
        reg = RuleRegistry()
        rule = _drule("r1", "claim", "amount", "<", "500", out=["appeal.disqualified"])
        reg.load([rule])
        result = reg.execute("d", {"claim.status": "denied"})
        assert result.triggered == []
        assert result.outputs == {}

    def test_execution_metadata_populated(self):
        reg = RuleRegistry()
        reg.load([_rule("r1")])
        result = reg.execute("d", {"claim.status": "open"},
                             trace_id="trace-123", executed_by="test_agent")
        assert result.metadata.trace_id == "trace-123"
        assert result.metadata.executed_by == "test_agent"
        assert result.metadata.executed_timestamp != ""

    def test_result_domain_set(self):
        reg = RuleRegistry()
        reg.load([_rule("r1")])
        result = reg.execute("d", {})
        assert result.domain == "d"

    def test_lookup_rule_triggers_and_writes_output_values(self):
        reg = RuleRegistry()
        rule = _lrule("l1", match_keys={"claim.claim_type": "theft"},
                      output_values={"appeal.tier": "high"})
        reg.load([rule])
        result = reg.execute("d", {"claim.claim_type": "theft"})
        assert result.triggered == [rule]
        assert result.outputs == {"appeal.tier": "high"}

    def test_lookup_rule_no_match_is_skipped_no_match(self):
        reg = RuleRegistry()
        rule = _lrule("l1", match_keys={"claim.claim_type": "theft"},
                      output_values={"appeal.tier": "high"})
        reg.load([rule])
        result = reg.execute("d", {"claim.claim_type": "auto"})
        assert result.triggered == []
        assert result.outputs == {}
        assert result.evaluations == [{"rule_id": "l1", "outcome": "skipped_no_match"}]

    def test_lookup_rule_missing_match_key_is_skipped_precondition(self):
        reg = RuleRegistry()
        rule = _lrule("l1", match_keys={"claim.claim_type": "theft"},
                      output_values={"appeal.tier": "high"})
        reg.load([rule])
        result = reg.execute("d", {"claim.status": "denied"})
        assert result.triggered == []
        assert result.evaluations == [{"rule_id": "l1", "outcome": "skipped_precondition"}]

    def test_decision_rule_enables_downstream_lookup_rule(self):
        reg = RuleRegistry()
        r1 = _drule("r1", "claim", "is_fraud", "==", "True",
                    out=["appeal.fraud_flagged"], priority=2)
        l1 = _lrule("l1", match_keys={"appeal.fraud_flagged": True},
                    output_values={"appeal.disqualified": True},
                    inp=["appeal.fraud_flagged"], priority=1)
        reg.load([r1, l1])
        result = reg.execute("d", {"claim.is_fraud": True})
        assert [r.id for r in result.triggered] == ["r1", "l1"]
        assert result.outputs == {"appeal.fraud_flagged": True, "appeal.disqualified": True}

    def test_seeded_keys_excluded_from_outputs(self):
        reg = RuleRegistry()
        rule = _lrule("l1", match_keys={"claim.claim_type": "theft"},
                      output_values={"appeal.tier": "high"})
        reg.load([rule])
        result = reg.execute("d", {"claim.claim_type": "theft"})
        assert "claim.claim_type" not in result.outputs
        assert result.outputs == {"appeal.tier": "high"}


class TestExecuteEvaluations:
    def test_every_rule_has_exactly_one_evaluation_entry(self):
        reg = RuleRegistry()
        r1 = _drule("r1", "claim", "amount", "<", "500", out=["appeal.amount_tier"], priority=2)
        r2 = _drule("r2", "claim", "status", "==", "denied", out=["appeal.x"], priority=1)
        r3 = _drule("r3", "customer", "tenure_years", "<", "3",
                    inp=["appeal.amount_tier"], out=["appeal.disqualified"])
        reg.load([r1, r2, r3])
        result = reg.execute("d", {"claim.amount": 400, "claim.status": "denied",
                                   "customer.tenure_years": 2})
        assert len(result.evaluations) == 3
        assert {e["rule_id"] for e in result.evaluations} == {"r1", "r2", "r3"}

    def test_triggered_outcome_string(self):
        reg = RuleRegistry()
        rule = _drule("r1", "claim", "amount", "<", "500", out=["appeal.flag"])
        reg.load([rule])
        result = reg.execute("d", {"claim.amount": 400})
        assert result.evaluations == [{"rule_id": "r1", "outcome": "triggered"}]

    def test_skipped_no_match_outcome_string(self):
        reg = RuleRegistry()
        rule = _drule("r1", "claim", "amount", "<", "500", out=["appeal.flag"])
        reg.load([rule])
        result = reg.execute("d", {"claim.amount": 5000})
        assert result.evaluations == [{"rule_id": "r1", "outcome": "skipped_no_match"}]

    def test_pruned_outcome_when_upstream_fails(self):
        reg = RuleRegistry()
        r1 = _drule("r1", "claim", "is_fraud", "==", "True",
                    out=["appeal.fraud_flagged"], priority=2)
        r2 = _drule("r2", "customer", "escalation_history_count", ">=", "2",
                    inp=["appeal.fraud_flagged"], out=["appeal.disqualified"], priority=1)
        reg.load([r1, r2])
        result = reg.execute("d", {"claim.is_fraud": False,
                                   "customer.escalation_history_count": 3})
        outcomes = {e["rule_id"]: e["outcome"] for e in result.evaluations}
        assert outcomes["r1"] == "skipped_no_match"
        assert outcomes["r2"] == "pruned"

    def test_evaluations_ordered_topologically_then_priority(self):
        reg = RuleRegistry()
        r_hi  = _drule("r_hi",  "claim", "amount", ">", "0", out=["appeal.flag"],    priority=10)
        r_lo  = _drule("r_lo",  "claim", "status", "==", "denied", out=["appeal.x"], priority=1)
        r_dep = _drule("r_dep", "claim", "amount", ">", "0",
                       inp=["appeal.flag"], out=["appeal.disqualified"],              priority=5)
        reg.load([r_hi, r_lo, r_dep])
        result = reg.execute("d", {"claim.amount": 100, "claim.status": "denied"})
        ids = [e["rule_id"] for e in result.evaluations]
        assert ids.index("r_hi") < ids.index("r_lo"),  "higher priority executes first within level"
        assert ids.index("r_hi") < ids.index("r_dep"), "level-0 rules execute before level-1"
        assert ids.index("r_lo") < ids.index("r_dep"), "level-0 rules execute before level-1"

    def test_entities_stored_as_passed(self):
        reg = RuleRegistry()
        reg.load([_rule("r1")])
        entities = {"claim": {"claim_id": "c1", "amount": 500},
                    "customer": {"customer_id": "cust_1"}}
        result = reg.execute("d", {}, entities=entities)
        assert result.entities == entities

    def test_entities_defaults_to_empty_dict(self):
        reg = RuleRegistry()
        reg.load([_rule("r1")])
        result = reg.execute("d", {})
        assert result.entities == {}


class TestLoadFrom:
    def _rule_dict(self, rule_id: str, domain: str,
                   inp: list[str] | None = None,
                   out: list[str] | None = None) -> dict:
        return {
            "id": rule_id, "domain": domain,
            "effective_from": _ts(-1), "effective_to": _ts(365),
            "input": inp or [], "output": out or [],
            "subject": "claim", "attribute": "status",
            "operator": "==", "threshold": "open",
        }

    def test_single_file_loads_its_rules(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps([
            self._rule_dict("r1", "dom_a"),
            self._rule_dict("r2", "dom_a"),
        ]))
        reg = RuleRegistry.load_from(str(f))
        assert len(reg.get_active("dom_a")) == 2

    def test_multiple_files_load_all_domains_with_correct_counts(self, tmp_path):
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text(json.dumps([
            self._rule_dict("r1", "dom_a"),
            self._rule_dict("r2", "dom_a"),
        ]))
        f2.write_text(json.dumps([
            self._rule_dict("r3", "dom_b"),
        ]))
        reg = RuleRegistry.load_from(str(f1), str(f2))
        assert len(reg.get_active("dom_a")) == 2
        assert len(reg.get_active("dom_b")) == 1

    def test_cycle_in_first_file_raises(self, tmp_path):
        f = tmp_path / "rules.json"
        f.write_text(json.dumps([
            self._rule_dict("r1", "dom_cycle", inp=["y"], out=["x"]),
            self._rule_dict("r2", "dom_cycle", inp=["x"], out=["y"]),
        ]))
        with pytest.raises(ValueError, match="Cycle detected"):
            RuleRegistry.load_from(str(f))

    def test_cycle_in_second_file_raises(self, tmp_path):
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        f1.write_text(json.dumps([self._rule_dict("r1", "dom_a")]))
        f2.write_text(json.dumps([
            self._rule_dict("r2", "dom_cycle", inp=["y"], out=["x"]),
            self._rule_dict("r3", "dom_cycle", inp=["x"], out=["y"]),
        ]))
        with pytest.raises(ValueError, match="Cycle detected"):
            RuleRegistry.load_from(str(f1), str(f2))
