"""Session-wide test fixtures."""

from pathlib import Path

import pytest

import services as svc

_TEST_AUDIT_FILE = Path(__file__).parent.parent / "data" / "test" / "audit" / "rule_executions.jsonl"


@pytest.fixture(autouse=True)
def test_audit_service():
    svc.RuleExecutionAuditService(file_path=_TEST_AUDIT_FILE)
