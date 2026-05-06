"""Policy rule MCP client related classes."""

from typing import TYPE_CHECKING, Optional, cast

import core
import data
import models as mdl

if TYPE_CHECKING:
    from data.data_storage import DataStorage


class PolicyRuleMcpClientConfig(mdl.WorkflowBaseModel):
    """Configuration model for PolicyRuleMcpClient."""

    data_storage_id: data.DataStorageId
    data_storage_config: dict


class PolicyRuleMcpClient(core.Configurable[PolicyRuleMcpClientConfig]):
    """Configurable client class responsible for retrieving policy rule records.

    Unlike claim and customer clients, policy rule lookup requires filtering
    across all rules by (claim_type, attribute, value) rather than a single
    primary key.
    """

    _config_data_type = PolicyRuleMcpClientConfig

    def __init__(self, config: PolicyRuleMcpClientConfig):
        """Initialize the configurable policy rule client.

        Args:
            config (PolicyRuleMcpClientConfig): Validated policy rule client configuration.
        """
        super().__init__(config)
        self._storage = cast("DataStorage",
                             data.DataStorageFactory().get_obj(config.data_storage_id.value,
                                                               config.data_storage_config))

    def get_obj(self, request: mdl.PolicyRuleRequest) -> Optional[mdl.PolicyRule]:
        """Retrieve the policy rule matching the provided claim type, attribute, and value.

        Args:
            request (mdl.PolicyRuleRequest): Policy rule lookup request object.

        Returns:
            Optional[mdl.PolicyRule]: Matching policy rule, if found.
        """
        for rule in self._storage.read():
            if (
                rule.claim_type == request.claim_type
                and rule.attribute == request.attribute
                and rule.value == request.value
            ):
                return rule
        return None
