"""Data storage related base classes."""

from abc import abstractmethod
from enum import Enum
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict

import shared.core as shd_core


class DataStorageId(Enum):
    """Supported data storage backend identifiers."""

    API = "api"
    CSV = "csv"
    DB = "db"
    JSON = "json"


T = TypeVar("T", bound="DataStorageConfig")


class DataStorageConfig(BaseModel):
    """Base configuration model for data storage implementations.

    Every concrete storage operates against a single model class - the dataclass
    it deserializes records into. ``model_class`` captures that contract here so
    all backends share the same field regardless of their specific storage medium.
    """

    model_class: Type[Any]
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class DataStorage(shd_core.Configurable[T]):
    """Abstract configurable base class for data storage implementations.

    Concrete subclasses are responsible for reading from and writing to a
    specific storage backend, such as a database, API, or file system.
    """

    def __init__(self, config: T):
        """Initialize the data storage instance.

        Args:
            config (T): Validated data storage configuration.
        """
        super().__init__(config)

    @property
    def model_class(self) -> Type[Any]:
        """The model class this storage instance deserializes records into.

        Returns:
            Type[Any]: Concrete model class declared in the storage config.
        """
        return self.config.model_class

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
