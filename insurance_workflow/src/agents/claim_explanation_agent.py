"""Claim explanation agent related classes."""

import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

import models as mdl

from .base_agent import LlmAgentConfig, LlmEnabledAgent

_src = Path(__file__).parent.parent
_server = str(_src / "mcp_clients" / "servers" / "csv_mcp_server.py")


class ClaimExplanationAgentConfig(LlmAgentConfig):
    """Configuration model for ClaimExplanationAgent."""


class ClaimExplanationAgent(LlmEnabledAgent[ClaimExplanationAgentConfig]):
    """LLM-backed agent for claim explanation via ReAct + MCP tools."""

    _config_data_type = ClaimExplanationAgentConfig

    @classmethod
    async def _load_tools(cls) -> list:
        """Load LangChain tools from the CSV MCP server subprocess.

        Returns:
            list: LangChain-compatible MCP tools.
        """
        client = MultiServerMCPClient({  # type: ignore[arg-type]
            "csv": {
                "command": sys.executable,
                "args": [_server],
                "transport": "stdio",
                "env": {**os.environ, "PYTHONPATH": str(_src)},
            },
        })
        return await client.get_tools()

    async def get_explanation_message(self, request: mdl.UserRequest) -> str:
        """Build a user-facing explanation using a ReAct agent over MCP tools.

        Args:
            request (mdl.UserRequest): Normalized user request object.

        Returns:
            str: LLM-synthesized explanation message.
        """
        query = (
            f"Explain the following attributes {request.attributes} "
            f"for claim {request.message}. Include policy basis and customer context."
        )
        result = await self._executor.ainvoke({"input": query})
        return result["output"]
