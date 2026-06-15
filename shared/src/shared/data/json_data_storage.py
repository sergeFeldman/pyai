"""Generic JSON-backed data storage."""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
import re
import types
from typing import Annotated, Any, Optional, Union, get_args, get_origin, get_type_hints

from pydantic import computed_field

from .data_storage import DataStorage, DataStorageConfig


class JsonDataStorageConfig(DataStorageConfig):
    """Configuration model for JSON-backed data storage.

    Attributes:
        file_path: Path to the JSON file, relative to the working directory.
        key_field: Field name used as the primary key for read_by_key().
            Defaults to ``{model_type}_id`` if empty (e.g. ``"claim_id"``).
            Override when the model uses a generic key name such as ``"id"``.
    """

    file_path: str
    key_field: str = ""

    @computed_field
    @property
    def model_type(self) -> str:
        """Logical model type derived from the model class name (CamelCase → snake_case)."""
        return re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', self.model_class.__name__).lower()

    @computed_field
    @property
    def resolved_key_field(self) -> str:
        """Effective primary key field name, applying the default convention if key_field is empty."""
        return self.key_field or f"{self.model_type}_id"


class JsonDataStorage(DataStorage[JsonDataStorageConfig]):
    """Generic data storage implementation backed by JSON files.

    Handles nested dataclasses (e.g. EntityMetadata inside DecisionRule) by
    recursively deserializing dicts into their declared field types on read,
    and delegates serialization to each model's to_dict() on write.

    read() returns an empty list rather than raising FileNotFoundError when the
    file does not exist - the ETL startup load treats a missing output file as
    an empty registry rather than an error.
    """

    _config_data_type = JsonDataStorageConfig

    def __init__(self, config: JsonDataStorageConfig):
        """Initialize the JSON storage instance.

        Args:
            config (JsonDataStorageConfig): Validated JSON storage configuration.
        """
        super().__init__(config)
        self._file_path = Path(config.file_path)

    @property
    def file_path(self) -> Path:
        """Configured JSON file path.

        Returns:
            Path: Configured JSON file path.
        """
        return self._file_path

    def read(self) -> list[Any]:
        """Read and deserialize model instances from the JSON file.

        Recursively constructs nested dataclasses from embedded dicts.
        Returns an empty list if the file does not exist.

        Returns:
            list[Any]: Deserialized model instances.
        """
        if not self._file_path.exists():
            return []

        with self._file_path.open("r", encoding="utf-8") as f:
            raw_list = json.load(f)

        hints = get_type_hints(self.model_class)
        return [self._from_dict(self.model_class, hints, item) for item in raw_list]

    def read_by_key(self, key: str) -> Optional[Any]:
        """Read a single model instance by its primary key.

        Args:
            key (str): Primary key value to match.

        Returns:
            Optional[Any]: Matching model instance, or None if not found.
        """
        key_field = self.config.resolved_key_field
        for instance in self.read():
            if getattr(instance, key_field) == key:
                return instance
        return None

    def read_as_dicts(self) -> list[dict[str, Any]]:
        """Read the JSON file and return raw dicts without model deserialization.

        Used by the ETL pipeline to read raw input that does not yet carry metadata.
        Returns an empty list if the file does not exist.

        Returns:
            list[dict[str, Any]]: Raw JSON entries as plain dicts.
        """
        if not self._file_path.exists():
            return []
        with self._file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write(self, objects: list[Any]) -> None:
        """Serialize and write model instances to the JSON file.

        Creates parent directories if they do not exist. Overwrites the file.

        Args:
            objects (list[Any]): Model instances to persist. Each must implement to_dict().
        """
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_path.open("w", encoding="utf-8") as f:
            json.dump(
                [obj.to_dict() for obj in objects],
                f,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def _from_dict(model_class: Any, hints: dict[str, Any], data: dict) -> Any:
        """Reconstruct a dataclass instance from a dict.

        Fields absent from data are omitted so the model's declared defaults apply.
        Extra keys in data that have no matching field are silently ignored.

        Args:
            model_class: Target dataclass type.
            hints: Resolved type hints for model_class (from get_type_hints).
            data: Source dict, typically a parsed JSON object.

        Returns:
            Any: Constructed model_class instance.
        """
        kwargs = {}
        for field in dataclasses.fields(model_class):
            if field.name not in data:
                continue
            field_type = hints.get(field.name, field.type)
            kwargs[field.name] = JsonDataStorage._deserialize(data[field.name], field_type)
        return model_class(**kwargs)

    @staticmethod
    def _deserialize(value: Any, target_type: Any) -> Any:
        """Recursively deserialize a JSON value to the declared field type.

        Handles Annotated, Optional/Union wrappers, list[T], nested dataclasses,
        and Enum. Primitive types (str, int, float, bool) are returned as-is since
        json.load already produces the correct Python type.

        Args:
            value: Parsed JSON value.
            target_type: Declared field type, possibly wrapped in Annotated or Optional.

        Returns:
            Any: Value cast or constructed to match target_type.
        """
        if value is None:
            return None

        origin = get_origin(target_type)

        if origin is Annotated:
            target_type = get_args(target_type)[0]
            origin = get_origin(target_type)

        if origin in {Union, types.UnionType}:
            args = [a for a in get_args(target_type) if a is not type(None)]
            if args:
                target_type = args[0]
                origin = get_origin(target_type)

        if origin is list:
            args = get_args(target_type)
            item_type = args[0] if args else Any
            return [JsonDataStorage._deserialize(item, item_type) for item in value]

        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            nested_hints = get_type_hints(target_type)
            return JsonDataStorage._from_dict(target_type, nested_hints, value)

        if isinstance(target_type, type) and issubclass(target_type, Enum):
            return target_type(value)

        return value
