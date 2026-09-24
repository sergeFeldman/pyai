"""Tests for RuleFactory — type detection and deserialization."""

import pytest

from rules import DecisionRule, LookupRule, RuleFactory
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
