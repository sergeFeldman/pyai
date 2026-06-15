"""Versioned entity metadata for auditable domain entities."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EntityMetadata:
    """Audit and version metadata for any domain entity.

    Composed into domain entities rather than inherited, keeping business
    attributes and metadata cleanly separated. The ETL pipeline is responsible
    for setting and bumping these fields - application code reads them only.

    Lifecycle:
        created_* fields are set once when the entity is first persisted and
        never modified again. updated_* fields are empty on creation and set
        on every subsequent change via bump().

    Attributes:
        version: Monotonically increasing version number, starting at 0.
            Incremented by bump() each time a business field changes.
        created_by: Identifier of the agent or process that first persisted the entity.
        created_timestamp: UTC ISO 8601 timestamp of initial creation. Never changed.
        updated_by: Identifier of the agent or process that last changed the entity.
            Empty until the first update.
        updated_timestamp: UTC ISO 8601 timestamp of the most recent change.
            Empty until the first update. Always set together with updated_by.
    """

    version: int = 0
    created_by: str = ""
    created_timestamp: str = field(default_factory=_now)
    updated_by: str = ""
    updated_timestamp: str = ""

    def bump(self, updated_by: str) -> None:
        """Increment the version and update audit fields in place.

        Args:
            updated_by: Identifier of the agent or process triggering the version bump.
        """
        self.version += 1
        self.updated_by = updated_by
        self.updated_timestamp = _now()
