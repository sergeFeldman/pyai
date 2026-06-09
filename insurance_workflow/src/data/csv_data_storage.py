"""CSV-backed data storage related classes."""

from __future__ import annotations

import csv
from dataclasses import fields
from enum import Enum
from pathlib import Path
import types
from typing import Annotated, Any, Optional, Union, get_args, get_origin

import models as mdl

from .data_storage import DataModelType, DataStorage, DataStorageConfig


class CsvDataStorageConfig(DataStorageConfig):
    """Configuration model for CSV-backed data storage."""

    file_path: str


class CsvDataStorage(DataStorage[CsvDataStorageConfig]):
    """Data storage implementation backed by CSV files.

    The storage is configured with a CSV file path and logical model type,
    and supports reading and writing collections of that model from and to
    the file.
    """

    _MODEL_TYPES_MAPPING = {
        DataModelType.CLAIM: mdl.Claim,
        DataModelType.CUSTOMER: mdl.CustomerContext,
        DataModelType.POLICY_RULE: mdl.PolicyRule,
        DataModelType.CLAIM_APPEAL_RULE: mdl.ClaimAppealRule,
    }
    _config_data_type = CsvDataStorageConfig

    def __init__(self, config: CsvDataStorageConfig):
        """Initialize the CSV storage instance.

        Args:
            config (CsvDataStorageConfig): Validated CSV storage configuration.
        """
        super().__init__(config)
        self._file_path = Path(config.file_path)

    @property
    def file_path(self) -> Path:
        """Property accessor for the configured CSV file path.

        Returns:
            Path: Configured CSV file path.
        """
        return self._file_path

    def read(self) -> list[Any]:
        """Read model instances from the configured CSV file.

        Returns:
            list[Any]: Collection of deserialized model instances.

        Raises:
            FileNotFoundError: Raised if the target CSV file does not exist.
            ValueError: Raised if the CSV headers do not match model fields.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(f"Persistence file does not exist: {self._file_path}")

        field_map = {field.name: field for field in fields(self._model_type)}
        expected_headers = list(field_map.keys())

        with self._file_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            actual_headers = reader.fieldnames or []
            if actual_headers != expected_headers:
                raise ValueError(
                    "CSV headers do not match dataclass fields. "
                    f"Expected {expected_headers}, got {actual_headers}"
                )

            instances: list[Any] = []
            for row in reader:
                attributes = {
                    field_name: self._deserialize_value(row[field_name], field_map[field_name].type)
                    for field_name in expected_headers
                }
                instances.append(self._model_type(**attributes))

        return instances

    def read_by_key(self, key: str) -> Optional[Any]:
        """Read a single model instance from the configured CSV file by key.

        Args:
            key (str): Domain object key value.

        Returns:
            Optional[Any]: Matching model instance, if found.
        """
        key_field_name = f"{self.model_type.value}_id"

        for instance in self.read():
            if getattr(instance, key_field_name) == key:
                return instance

        return None

    def read_as_dicts(self) -> list[dict[str, Any]]:
        """Read model instances and return them as dictionaries.

        Returns:
            list[dict[str, Any]]: Collection of serialized instance dictionaries.
        """
        return [instance.to_dict() for instance in self.read()]

    def write(self, objects: list[Any]) -> None:
        """Write model instances to the configured CSV file.

        Args:
            objects (list[Any]): Collection of model instances to persist.
        """
        if not objects:
            return

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        field_names = [field.name for field in fields(self._model_type)]

        with self._file_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()

            for instance in objects:
                writer.writerow(instance.to_dict())

    @staticmethod
    def _deserialize_value(value: str, target_type: Any) -> Any:
        """Convert a CSV string value to the declared dataclass field type.

        Args:
            value (str): Raw CSV value.
            target_type (Any): Declared dataclass field type.

        Returns:
            Any: Converted field value.

        Raises:
            ValueError: Raised if the value cannot be converted correctly.
        """
        resolved_type, is_optional = CsvDataStorage._resolve_type(target_type)

        if value == "" and is_optional:
            return None

        if resolved_type is str:
            return value
        if resolved_type is int:
            return int(value)
        if resolved_type is float:
            return float(value)
        if resolved_type is bool:
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
            raise ValueError(f"Cannot convert '{value}' to bool")
        if isinstance(resolved_type, type) and issubclass(resolved_type, Enum):
            return resolved_type(value)

        return value

    @staticmethod
    def _resolve_type(target_type: Any) -> tuple[Any, bool]:
        """Resolve Annotated, Optional, and Union wrappers around a target field type.

        Args:
            target_type (Any): Declared dataclass field type.

        Returns:
            tuple[Any, bool]: Resolved base type and whether None is allowed.
        """
        origin = get_origin(target_type)
        if origin is Annotated:
            return CsvDataStorage._resolve_type(get_args(target_type)[0])
        if origin in {Union, types.UnionType}:
            args = [arg for arg in get_args(target_type) if arg is not type(None)]
            if len(args) == 1:
                return args[0], True
        return target_type, False
