"""Base MCP client classes."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, cast

import core
import data
import models as mdl

if TYPE_CHECKING:
    from data.data_storage import DataStorage

TConfig = TypeVar("TConfig", bound="MpcClientConfig")
TRequest = TypeVar("TRequest")
TObject = TypeVar("TObject")


class MpcClientConfig(mdl.WorkflowBaseModel):
    """Base configuration model for MCP clients backed by a data storage."""

    data_storage_id: data.DataStorageId
    data_storage_config: dict


class MpcClient(core.Configurable[TConfig], Generic[TConfig, TRequest, TObject]):
    """Abstract base class for MCP clients backed by a configurable data storage.

    Subclasses declare _primary_key_field for simple key-based lookup, or
    override get_obj() entirely for custom retrieval logic.
    """

    _primary_key_field: str

    def __init__(self, config: TConfig):
        """Initialize the MCP client.

        Args:
            config (TConfig): Validated MCP client configuration.
        """
        super().__init__(config)
        self._storage = cast("DataStorage",
                             data.DataStorageFactory().get_obj(
                                 config.data_storage_id.value,
                                 config.data_storage_config))

    def get_obj(self, request: TRequest) -> Optional[TObject]:
        """Retrieve a domain object by primary key extracted from the request.

        Args:
            request (TRequest): Domain request object.

        Returns:
            Optional[TObject]: Matching domain object, if found.
        """
        return self._storage.read_by_key(getattr(request, self._primary_key_field))

    def get_obj_by_filter(self, request: TRequest) -> Optional[TObject]:
        """Retrieve a domain object by filter criteria extracted from the request.

        Override in subclasses that support filter-based lookup.

        Args:
            request (TRequest): Domain request object.

        Returns:
            Optional[TObject]: Matching domain object, if found.

        Raises:
            NotImplementedError: Raised if the subclass does not support filter-based lookup.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support filter-based lookup")
