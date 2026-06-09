"""Data storage related base classes and enumerations."""

from abc import abstractmethod
from enum import Enum
from typing import Any, Optional, Type, TypeVar

import core
import models as mdl

T = TypeVar("T", bound="DataStorageConfig")


class DataModelType(Enum):
    """Supported logical data model types for storage backends."""

    CLAIM = "claim"
    CUSTOMER = "customer"
    POLICY_RULE = "policy_rule"
    CLAIM_APPEAL_RULE = "claim_appeal_rule"


class DataStorageId(Enum):
    """Supported data storage backend identifiers."""

    API = "api"
    CSV = "csv"
    DB = "db"


class DataStorageConfig(mdl.WorkflowBaseModel):
    """Base configuration model for data storage implementations.

    Concrete storage configuration models should inherit from this class.
    The shared `model_type` field identifies the logical entity type handled
    by the storage implementation.
    """

    model_type: DataModelType


class DataStorage(core.Configurable[T]):
    """Abstract configurable base class for data storage implementations.

    Concrete subclasses are responsible for reading from and writing to a
    specific storage backend, such as an external database, API-based system,
    or file-backed storage mechanism.

    Implementations are initialized with validated configuration data and are
    expected to operate on the logical entity type identified by the config
    model's `model_type` field.
    """

    _MODEL_TYPES_MAPPING = None

    def __init__(self, config: T):
        """Initialize the data storage instance.

        Args:
            config (T): Validated data storage configuration.
        """
        super().__init__(config)
        self._model_type = self._resolve_model_type(config.model_type)

    @property
    def model_type(self) -> DataModelType:
        """Property accessor for the configured logical model type.

        Returns:
            DataModelType: Logical entity type handled by the storage instance.
        """
        return self.config.model_type

    @property
    def model_class(self) -> Type[Any]:
        """Property accessor for the resolved concrete model class.

        Returns:
            Type[Any]: Concrete model class handled by the storage instance.
        """
        return self._model_type

    @classmethod
    def _resolve_model_type(cls, model_type: DataModelType) -> Type[Any]:
        """Resolve a logical model type enum to a concrete model class.

        Args:
            model_type (DataModelType): Logical model type identifier.

        Returns:
            Type[Any]: Concrete model class associated with the provided enum.

        Raises:
            ValueError: Raised when the provided model type is not supported.
        """
        if cls._MODEL_TYPES_MAPPING is None:
            raise ValueError(f"Given storage implementation {cls.__name__} is invalid, "
                             "since _MODEL_TYPES_MAPPING is not assigned")

        if model_type not in cls._MODEL_TYPES_MAPPING:
            raise ValueError(f"Provided model type is not supported: {model_type}")

        return cls._MODEL_TYPES_MAPPING[model_type]

    @abstractmethod
    def read(self) -> list[Any]:
        """Read all objects from the configured storage backend.

        Returns:
            list[Any]: Collection of objects returned by the concrete storage backend.
        """

    @abstractmethod
    def read_by_key(self, key: str) -> Optional[Any]:
        """Read a single object from the configured storage backend by key.

        Args:
            key (str): Domain object key value.

        Returns:
            Optional[Any]: Matching object, if found.
        """

    @abstractmethod
    def write(self, objects: list[Any]) -> None:
        """Write objects to the configured storage backend.

        Args:
            objects (list[Any]): Collection of objects to persist.
        """
