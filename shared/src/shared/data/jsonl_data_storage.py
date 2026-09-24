"""Append-only JSONL data storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .data_storage import DataStorage, DataStorageConfig


class JsonlDataStorageConfig(DataStorageConfig):
    """Configuration model for JSONL data storage.

    Attributes:
        file_path: Path to the JSONL file, relative to the working directory.
        key_field: Field name (or dot-notation path) used as the primary key for
            read_by_key(), e.g. "trace_id" or "metadata.trace_id". Defaults to "trace_id".
    """

    file_path: str
    key_field: str = "trace_id"


class JsonlDataStorage(DataStorage[JsonlDataStorageConfig]):
    """Append-only data storage backed by a JSON Lines (JSONL) file.

    Each record occupies exactly one line. write() appends records rather than
    overwriting, preserving the full audit history across calls.

    read() returns raw dicts parsed from each line. read_by_key() scans all
    records for the first match on the configured key field.
    """

    _config_data_type = JsonlDataStorageConfig

    def __init__(self, config: JsonlDataStorageConfig):
        """Initialize the JSONL storage instance.

        Args:
            config: Validated JSONL storage configuration.
        """
        super().__init__(config)
        self._file_path = Path(config.file_path)

    @property
    def file_path(self) -> Path:
        """Configured JSONL file path."""
        return self._file_path

    def read(self) -> list[Any]:
        """Read all records from the JSONL file as raw dicts.

        Records are returned as plain dicts regardless of model_class; no
        deserialization into model instances is performed. Returns an empty
        list if the file does not exist.

        Returns:
            list[Any]: All records parsed from the file, one dict per line.
        """
        if not self._file_path.exists():
            return []
        with self._file_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def read_by_key(self, key: str) -> Optional[Any]:
        """Return the first record whose key field matches the given value.

        key_field supports dot-notation for nested fields, e.g. "metadata.trace_id"
        traverses record["metadata"]["trace_id"].

        Args:
            key: Value to match against config.key_field.

        Returns:
            Optional[Any]: First matching record as a dict, or None.
        """
        key_field = self.config.key_field
        for record in self.read():
            if isinstance(record, dict) and self._get_nested(record, key_field) == key:
                return record
        return None

    @staticmethod
    def _get_nested(record: dict, path: str) -> Any:
        """Traverse a dot-notation path in a nested dict.

        Args:
            record: Source dict.
            path: Dot-separated key path, e.g. "metadata.trace_id".

        Returns:
            Any: Value at the path, or None if any segment is missing.
        """
        value: Any = record
        for segment in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(segment)
        return value

    def write(self, objects: list[Any]) -> None:
        """Append records to the JSONL file.

        Unlike JSON storage, this method appends rather than overwrites,
        preserving existing records. Creates parent directories if needed.
        Each object is serialized via to_dict() if available, otherwise
        passed directly to json.dumps.

        Args:
            objects: Records to append. Each must be a dict or implement to_dict().
        """
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_path.open("a", encoding="utf-8") as f:
            for obj in objects:
                record = obj.to_dict() if hasattr(obj, "to_dict") else obj
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, record: Any) -> None:
        """Append a single record to the JSONL file.

        Convenience wrapper over write() for single-record use cases.

        Args:
            record: Record to append. Should be a dict or implement to_dict() for correct JSON serialization.
        """
        self.write([record])
