"""Customer MCP client related classes."""

import models as mdl

from .mcp_client import McpStorageClient, McpStorageClientConfig


class CustomerMcpClientConfig(McpStorageClientConfig):
    """Configuration model for CustomerMcpClient."""


class CustomerMcpClient(McpStorageClient[CustomerMcpClientConfig, mdl.CustomerRequest, mdl.Customer]):
    """Configurable client class responsible for retrieving customer context records."""

    _config_data_type = CustomerMcpClientConfig
    _primary_key_field = "customer_id"
