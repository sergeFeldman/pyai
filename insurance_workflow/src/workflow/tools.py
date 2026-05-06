"""LangChain tool loading via MCP server for the claim explanation workflow."""

import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


async def create_claim_explanation_tools() -> list:
    """Load LangChain tools from the CSV MCP server.

    Connects to the csv-server MCP process via stdio, discovers its exposed
    tools (get_claim, get_customer, get_policy_rule), and returns
    them as LangChain-compatible tool objects for use by the ReAct agent.

    Returns:
        list: LangChain tools loaded from the CSV MCP server.
    """
    client = MultiServerMCPClient({
        "csv": {
            "command": sys.executable,
            "args": ["src/mcp_clients/servers/csv_mcp_server.py"],
            "transport": "stdio",
            "env": {**os.environ, "PYTHONPATH": "src"},
        },
    })
    return await client.get_tools()
