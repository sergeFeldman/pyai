"""Customer MCP client related classes."""

from typing import TYPE_CHECKING, Optional, cast

import core
import data
import models as mdl

if TYPE_CHECKING:
    from data.data_storage import DataStorage


class CustomerMcpClientConfig(mdl.WorkflowBaseModel):
    """Configuration model for CustomerMcpClient."""

    data_storage_id: data.DataStorageId
    data_storage_config: dict


class CustomerMcpClient(core.Configurable[CustomerMcpClientConfig]):
    """Configurable client class responsible for retrieving customer context records.

    This client delegates customer-context loading to a configured DataStorage
    implementation and exposes customer-specific retrieval behavior.
    """

    _config_data_type = CustomerMcpClientConfig

    def __init__(self, config: CustomerMcpClientConfig):
        """Initialize the configurable customer client.

        Args:
            config (CustomerMcpClientConfig): Validated customer-client configuration.
        """
        super().__init__(config)
        self._storage = cast("DataStorage",
                             data.DataStorageFactory().get_obj(config.data_storage_id.value,
                                                               config.data_storage_config))

    def get_obj(self, request: mdl.CustomerRequest) -> Optional[mdl.CustomerContext]:
        """Retrieve a customer context record by customer identifier.

        Args:
            request (mdl.CustomerRequest): Customer context lookup request object.

        Returns:
            Optional[mdl.CustomerContext]: Matching customer context record, if found.
        """
        return self._storage.read_by_key(request.customer_id)
