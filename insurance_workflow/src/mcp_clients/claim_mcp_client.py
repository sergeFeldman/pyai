"""Claim MCP client related classes."""

from typing import TYPE_CHECKING, Optional, cast

import core
import data
import models as mdl

if TYPE_CHECKING:
    from data.data_storage import DataStorage


class ClaimMcpClientConfig(mdl.WorkflowBaseModel):
    """Configuration model for ClaimMcpClient."""

    data_storage_id: data.DataStorageId
    data_storage_config: dict


class ClaimMcpClient(core.Configurable[ClaimMcpClientConfig]):
    """Configurable client class responsible for retrieving claim records.

    This client delegates claim-record loading to a configured DataStorage
    implementation and exposes claim-specific retrieval behavior.
    """

    _config_data_type = ClaimMcpClientConfig

    def __init__(self, config: ClaimMcpClientConfig):
        """Initialize the configurable claim client.

        Args:
            config (ClaimMcpClientConfig): Validated claim-client configuration.
        """
        super().__init__(config)
        self._storage = cast("DataStorage",
                             data.DataStorageFactory().get_obj(config.data_storage_id.value,
                                                               config.data_storage_config))

    def get_obj(self, request: mdl.ClaimRequest) -> Optional[mdl.Claim]:
        """Retrieve a claim record by claim identifier.

        Args:
            request (mdl.ClaimRequest): Claim lookup request object.

        Returns:
            Optional[mdl.Claim]: Matching claim record, if found.
        """
        return self._storage.read_by_key(request.claim_id)
