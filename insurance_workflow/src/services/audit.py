"""Audit service for workflow request logging."""

import csv
from datetime import datetime, timezone
from pathlib import Path

import models as mdl

_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "out"
_FIELDS = ["trace_id", "request_type", "agent_names", "response", "timestamp"]


class AuditService:
    """Service class responsible for appending audit records to a session CSV file.

    The file is created once at service startup with a timestamp suffix:
    audit_YYYY-MM-DDTHH-MM-SS.csv. All records for the lifetime of the
    service are written to that file.
    Records contain no PII - only operational metadata.
    """

    def __init__(self):
        """Initialize the audit service and set the session audit file path."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self._path = _OUTPUT_DIR / f"audit_{timestamp}.csv"

    def log(self, record: mdl.AuditRecord) -> None:
        """Append an audit record to the session audit CSV file.

        Args:
            record (mdl.AuditRecord): Audit record to persist.
        """
        write_header = not self._path.exists()
        with open(self._path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "trace_id": record.trace_id,
                "request_type": record.request_type,
                "agent_names": ",".join(record.agent_names),
                "response": record.response,
                "timestamp": record.timestamp.isoformat(),
            })
