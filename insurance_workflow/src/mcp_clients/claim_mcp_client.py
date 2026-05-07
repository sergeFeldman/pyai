"""Claim MCP client related classes."""

import models as mdl

from .mcp_client import MpcClient, MpcClientConfig


class ClaimMcpClientConfig(MpcClientConfig):
    """Configuration model for ClaimMcpClient."""


class ClaimMcpClient(MpcClient[ClaimMcpClientConfig, mdl.ClaimRequest, mdl.Claim]):
    """Configurable client class responsible for retrieving claim records."""

    _config_data_type = ClaimMcpClientConfig
    _primary_key_field = "claim_id"
