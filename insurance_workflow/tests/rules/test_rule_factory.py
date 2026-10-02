"""Tests for RuleFactory — type detection and deserialization."""

import pytest

from rules import DecisionRule, LookupRule, RuleCondition, RuleFactory, RuleLogic, RuleOperator
from shared.core import EntityMetadata, Singleton


@pytest.fixture(autouse=True)
def reset_factory():
    yield
    Singleton._instances.pop(RuleFactory, None)


class TestDetectType:
    def test_detects_decision_from_subject_field(self):
        assert RuleFactory().detect_type({"id": "r1", "subject": "claim", "operator": ">="}) == "decision"

    def test_detects_lookup_from_match_keys_field(self):
        assert RuleFactory().detect_type({"id": "r1", "match_keys": {"type": "auto"}}) == "lookup"

    def test_detects_decision_from_conditions_field(self):
        assert RuleFactory().detect_type({"id": "r1", "conditions": [{"subject": "claim", "attribute": "amount", "operator": "<", "threshold": "500"}]}) == "decision"

    def test_unknown_fields_raises_value_error(self):
        with pytest.raises(ValueError):
            RuleFactory().detect_type({"id": "r1", "unknown_field": "x"})


class TestFromDict:
    def test_decision_rule_deserialized(self):
        data = {
            "id": "r1", "domain": "claim", "priority": 5,
            "subject": "claim", "attribute": "amount",
            "operator": ">=", "threshold": "100",
            "effective_from": "2026-01-01T00:00:00+00:00",
            "effective_to": "2027-12-31T23:59:59+00:00",
            "input": ["claim.amount"], "output": ["is_eligible"],
        }
        rule = RuleFactory().from_dict(data)
        assert isinstance(rule, DecisionRule)
        assert rule.id == "r1"
        assert rule.threshold == "100"
        assert rule.input == ["claim.amount"]

    def test_lookup_rule_deserialized(self):
        data = {
            "id": "r2", "domain": "claim",
            "match_keys": {"claim_type": "auto"},
            "output_values": {"denial_basis": "fraud"},
        }
        rule = RuleFactory().from_dict(data)
        assert isinstance(rule, LookupRule)
        assert rule.match_keys == {"claim_type": "auto"}
        assert rule.output_values == {"denial_basis": "fraud"}

    def test_metadata_dict_reconstructed_as_entity_metadata(self):
        data = {
            "id": "r1", "domain": "claim",
            "subject": "claim", "attribute": "status",
            "operator": "==", "threshold": "open",
            "metadata": {
                "version": 2,
                "created_by": "etl",
                "created_timestamp": "2026-01-01T00:00:00+00:00",
                "updated_by": "etl",
                "updated_timestamp": "2026-06-01T00:00:00+00:00",
            },
        }
        rule = RuleFactory().from_dict(data)
        assert isinstance(rule.metadata, EntityMetadata)
        assert rule.metadata.version == 2
        assert rule.metadata.updated_by == "etl"

    def test_unknown_fields_ignored(self):
        data = {
            "id": "r1", "domain": "claim",
            "subject": "claim", "attribute": "status",
            "operator": "==", "threshold": "open",
            "nonexistent_field": "should_be_ignored",
        }
        rule = RuleFactory().from_dict(data)
        assert isinstance(rule, DecisionRule)
        assert not hasattr(rule, "nonexistent_field")

    def test_compound_rule_conditions_deserialized_as_rule_conditions(self):
        data = {
            "id": "r1", "domain": "claim_appeal",
            "logic": "AND",
            "conditions": [
                {"subject": "customer", "attribute": "prior_claim_count", "operator": ">=", "threshold": "5"},
                {"subject": "customer", "attribute": "escalation_history_count", "operator": ">=", "threshold": "2"},
            ],
            "input": ["customer.prior_claim_count", "customer.escalation_history_count"],
            "output": ["appeal.disqualified"],
        }
        rule = RuleFactory().from_dict(data)
        assert isinstance(rule, DecisionRule)
        assert rule.logic == RuleLogic.AND
        assert len(rule.conditions) == 2
        assert all(isinstance(c, RuleCondition) for c in rule.conditions)
        assert rule.conditions[0].operator == RuleOperator.GTE
        assert rule.conditions[0].threshold == "5"

    def test_compound_rule_nested_group_deserialized_recursively(self):
        data = {
            "id": "r1", "domain": "claim_appeal",
            "logic": "AND",
            "conditions": [
                {"logic": "OR", "conditions": [
                    {"subject": "claim", "attribute": "is_fraud", "operator": "==", "threshold": "True"},
                    {"subject": "customer", "attribute": "prior_claim_count", "operator": ">=", "threshold": "5"},
                ]},
                {"subject": "customer", "attribute": "escalation_history_count", "operator": ">=", "threshold": "2"},
            ],
            "input": ["claim.is_fraud", "customer.prior_claim_count", "customer.escalation_history_count"],
            "output": ["appeal.disqualified"],
        }
        rule = RuleFactory().from_dict(data)
        assert len(rule.conditions) == 2
        group_node = rule.conditions[0]
        assert isinstance(group_node, RuleCondition)
        assert group_node.logic == RuleLogic.OR
        assert len(group_node.conditions) == 2
        assert all(isinstance(c, RuleCondition) for c in group_node.conditions)

    def test_compound_rule_evaluates_correctly_after_deserialization(self):
        data = {
            "id": "r1", "domain": "claim_appeal",
            "logic": "AND",
            "conditions": [
                {"subject": "customer", "attribute": "prior_claim_count", "operator": ">=", "threshold": "5"},
                {"subject": "customer", "attribute": "escalation_history_count", "operator": ">=", "threshold": "2"},
            ],
            "input": ["customer.prior_claim_count", "customer.escalation_history_count"],
            "output": ["appeal.disqualified"],
        }
        rule = RuleFactory().from_dict(data)
        matched, outputs = rule.evaluate(
            {"customer.prior_claim_count": 6, "customer.escalation_history_count": 3}
        )
        assert matched is True
        assert outputs == {"appeal.disqualified": True}
        matched_not, _ = rule.evaluate(
            {"customer.prior_claim_count": 6, "customer.escalation_history_count": 1}
        )
        assert matched_not is False
