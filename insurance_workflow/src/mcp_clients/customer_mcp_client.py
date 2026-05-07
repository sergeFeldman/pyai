"""Customer MCP client related classes."""

import models as mdl

from .mcp_client import MpcClient, MpcClientConfig


class CustomerMcpClientConfig(MpcClientConfig):
    """Configuration model for CustomerMcpClient."""


class CustomerMcpClient(MpcClient[CustomerMcpClientConfig, mdl.CustomerRequest, mdl.CustomerContext]):
    """Configurable client class responsible for retrieving customer context records."""

    _config_data_type = CustomerMcpClientConfig
    _primary_key_field = "customer_id"
