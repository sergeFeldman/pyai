"""Serializable mixin for domain model dataclasses."""

import dataclasses
from enum import Enum


class SerializableMixin:
    """Mixin that adds to_dict() to any dataclass.

    Converts all fields to primitives; Enum fields are serialized to their value.

    Usage:

        @dataclass
        class Claim(SerializableMixin):
            status: ClaimStatus

        claim.to_dict()  # {"status": "denied", ...}
    """

    def to_dict(self) -> dict:
        """Return a shallow dict of field name → primitive value.

        Returns:
            dict: All dataclass fields serialized to primitives.
        """
        result = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Enum):
                result[f.name] = v.value
            elif isinstance(v, bool):
                result[f.name] = str(v).lower()
            else:
                result[f.name] = v
        return result
