"""Claim appeal rule MCP client related classes."""

import models as mdl

from .mcp_client import MpcClient, MpcClientConfig


class ClaimAppealRuleMcpClientConfig(MpcClientConfig):
    """Configuration model for ClaimAppealRuleMcpClient."""


class ClaimAppealRuleMcpClient(
    MpcClient[ClaimAppealRuleMcpClientConfig, mdl.ClaimAppealRuleRequest, mdl.ClaimAppealRule]
):
    """Configurable client class responsible for retrieving claim appeal rules."""

    _config_data_type = ClaimAppealRuleMcpClientConfig
    _primary_key_field = "claim_appeal_rule_id"

    def __init__(self, config: ClaimAppealRuleMcpClientConfig):
        """Initialize the client and eagerly load all appeal rules from storage.

        Args:
            config (ClaimAppealRuleMcpClientConfig): Validated MCP client configuration.
        """
        super().__init__(config)
        self._rules = self._storage.read()

    @property
    def rules(self) -> list[mdl.ClaimAppealRule]:
        """All claim appeal disqualification rules, loaded once at initialization.

        Returns:
            list[mdl.ClaimAppealRule]: All appeal disqualification rules.
        """
        return self._rules
