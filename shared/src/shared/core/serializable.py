"""Serializable mixin for domain model dataclasses."""

import dataclasses
from enum import Enum


class SerializableMixin:
    """Mixin that adds to_dict() to any dataclass.

    Converts all fields to primitives recursively. Handles nested dataclasses,
    lists of dataclasses, Enum fields, and bool fields.

    Usage:

        @dataclass
        class Rule(SerializableMixin):
            metadata: EntityMetadata

        rule.to_dict()  # {"metadata": {"version": 0, ...}, ...}
    """

    def to_dict(self) -> dict:
        """Return a fully recursive dict of field name → primitive value.

        Returns:
            dict: All dataclass fields serialized to primitives, including
                nested dataclasses and lists.
        """
        return {
            f.name: SerializableMixin._serialize(getattr(self, f.name))
            for f in dataclasses.fields(self)
        }

    @staticmethod
    def _serialize(v) -> object:
        """Recursively serialize a single value to a primitive.

        Args:
            v: Value to serialize.

        Returns:
            Serialized primitive: dict for dataclasses, list for lists,
            str for Enum and bool, original value otherwise.
        """
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, bool):
            return str(v).lower()
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {
                f.name: SerializableMixin._serialize(getattr(v, f.name))
                for f in dataclasses.fields(v)
            }
        if isinstance(v, list):
            return [SerializableMixin._serialize(item) for item in v]
        return v
