"""Audit service for rule execution results."""

from pathlib import Path

import shared.core as shd_core
import shared.data as shd_data
import rules as rls

_AUDIT_FILE = Path(__file__).parent.parent.parent / "data" / "audit" / "rule_executions.jsonl"


class RuleExecutionAuditService(metaclass=shd_core.Singleton):
    """Persists rule execution results to a JSONL audit file.

    Each call to log() appends one JSON record capturing the execution metadata,
    domain, triggered rules in execution order, ordered per-rule evaluations
    (outcome per rule: triggered/skipped_precondition/skipped_no_match), entity
    snapshots (claim and customer objects at execution time), and outputs produced.
    Self-initializes on first use with the default audit file path.
    """

    def __init__(self, file_path: str | Path = _AUDIT_FILE):
        """Initialize the audit service with a target file path.

        Args:
            file_path: Path to the JSONL audit file. Defaults to
                data/audit/rule_executions.jsonl relative to the project root.
        """
        self._storage = shd_data.JsonlDataStorage(
            shd_data.JsonlDataStorageConfig(
                model_class=dict,
                file_path=str(file_path),
                key_field="metadata.trace_id",
            )
        )

    def log(self, result: rls.RuleExecutionResult) -> None:
        """Append a rule execution result to the audit file.

        Args:
            result: Execution result from RuleRegistry.execute().
        """
        self._storage.append(result.to_dict())

    def list(self, domain: str) -> list[dict]:
        """Return all execution records for a domain, newest first.

        Records with an empty trace_id (e.g. from test runs) are excluded.

        Args:
            domain: Domain key to filter by (e.g. "claim_appeal").

        Returns:
            list[dict]: Matching records in reverse chronological order.
        """
        return [
            r for r in reversed(self._storage.read())
            if r.get("domain") == domain and r.get("metadata", {}).get("trace_id")
        ]
