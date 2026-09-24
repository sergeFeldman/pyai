"""Execution metadata for traceable runtime results."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionMetadata:
    """Trace and audit metadata for runtime execution results.

    Composed into execution result types in the same way EntityMetadata is
    composed into persistent domain entities — keeps business output and
    execution provenance cleanly separated.

    Lifecycle:
        executed_timestamp is auto-stamped at creation and never modified.
        trace_id and executed_by are empty until populated by the caller.

    Attributes:
        trace_id: Workflow request trace ID. Empty until threaded from the orchestrator.
        executed_by: Agent or process that triggered the execution,
            e.g. "claim_appeal_agent".
        executed_timestamp: UTC ISO 8601 timestamp auto-stamped at creation.
    """

    trace_id: str = ""
    executed_by: str = ""
    executed_timestamp: str = field(default_factory=_now)
