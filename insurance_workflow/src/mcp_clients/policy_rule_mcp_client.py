"""Policy rule MCP client related classes."""

from typing import Optional

import models as mdl

from .mcp_client import MpcClient, MpcClientConfig


class PolicyRuleMcpClientConfig(MpcClientConfig):
    """Configuration model for PolicyRuleMcpClient."""


class PolicyRuleMcpClient(MpcClient[PolicyRuleMcpClientConfig, mdl.PolicyRuleRequest, mdl.PolicyRule]):
    """Configurable client class responsible for retrieving policy rule records."""

    _config_data_type = PolicyRuleMcpClientConfig
    _primary_key_field = "policy_rule_id"

    def get_obj_by_filter(self, request: mdl.PolicyRuleFilterRequest) -> Optional[mdl.PolicyRule]:
        """Retrieve the policy rule matching the provided claim type, attribute, and value.

        Args:
            request (mdl.PolicyRuleFilterRequest): Policy rule lookup request object.

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
