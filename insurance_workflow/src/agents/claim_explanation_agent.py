"""Claim explanation agent related classes."""

import models as mdl

from .base_agent import LlmAgentConfig, LlmEnabledAgent


class ClaimExplanationAgentConfig(LlmAgentConfig):
    """Configuration model for ClaimExplanationAgent."""


class ClaimExplanationAgent(LlmEnabledAgent[ClaimExplanationAgentConfig]):
    """LLM-backed agent for claim explanation via ReAct + MCP tools."""

    _config_data_type = ClaimExplanationAgentConfig

    @classmethod
    async def _load_tools(cls) -> list:
        """Load MCP tools from the MCP server.

        Returns:
            list: LangChain-compatible MCP tools.
        """
        from workflow.tools import create_claim_explanation_tools
        return await create_claim_explanation_tools()

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
