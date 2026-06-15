"""Base agent related classes."""

from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain import hub
from pydantic import BaseModel

import shared.core as shd_core


TConfig = TypeVar("TConfig")
TMcpClient = TypeVar("TMcpClient")
TRequest = TypeVar("TRequest")
TObject = TypeVar("TObject")


class LlmAgentConfig(BaseModel):
    """Base configuration for LLM-enabled agents.

    Attributes:
        llm_provider: LLM backend to use. Supported values: "anthropic", "groq", "ollama".
        model: Model identifier passed to the provider, e.g. "claude-sonnet-4-6".
        prompt_name: LangChain Hub prompt identifier pulled at agent creation time.
    """

    llm_provider: str
    model: str
    prompt_name: str


def _create_llm(provider: str, model: str):
    """Instantiate a LangChain chat model for the given provider and model name.

    Args:
        provider: LLM backend identifier. Supported: "anthropic", "groq", "ollama".
        model: Provider-specific model name, e.g. "claude-sonnet-4-6".

    Returns:
        A LangChain BaseChatModel instance for the requested provider.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model)
    raise ValueError(f"Unknown LLM provider: {provider}")


TLlmConfig = TypeVar("TLlmConfig", bound=LlmAgentConfig)


class LlmEnabledAgent(shd_core.Configurable[TLlmConfig], Generic[TLlmConfig]):
    """Abstract base class for LLM-backed agents that load tools dynamically.

    Subclasses implement _load_tools() to supply the agent with the
    appropriate tools. The async factory method create() handles tool
    loading and executor assembly, keeping __init__ synchronous.

    Usage:

        class ClaimExplanationAgent(LlmEnabledAgent[ClaimExplanationAgentConfig]):

            @classmethod
            async def _load_tools(cls) -> list:
                return await create_claim_explanation_tools()
    """

    def __init__(self, config: TLlmConfig, executor: AgentExecutor):
        """Initialize the LLM-enabled agent.

        Args:
            config (TLlmConfig): Validated agent configuration.
            executor (AgentExecutor): Assembled LangChain agent executor.

        Raises:
            ValueError: Raised if the provided executor is None.
        """
        super().__init__(config)
        if executor is None:
            raise ValueError("Executor parameter cannot be None")
        self._executor = executor

    @classmethod
    async def create(cls, config: TLlmConfig) -> "LlmEnabledAgent":
        """Async factory method - loads tools and assembles the executor.

        Args:
            config (TLlmConfig): Validated agent configuration.

        Returns:
            LlmEnabledAgent: Fully initialized agent instance.
        """
        tools = await cls._load_tools()
        llm = _create_llm(config.llm_provider, config.model)
        agent = create_structured_chat_agent(llm, tools, hub.pull(config.prompt_name))
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        return cls(config, executor)

    @classmethod
    @abstractmethod
    async def _load_tools(cls) -> list:
        """Return the tools for this agent.

        Returns:
            list: LangChain-compatible tools.
        """


class McpEnabledAgent(shd_core.Configurable[TConfig], Generic[TConfig, TMcpClient, TRequest, TObject]):
    """Abstract configurable base class for MCP-enabled domain agents.

    Concrete subclasses are responsible for constructing the appropriate MCP
    client instance and passing it into this base class during initialization.
    """

    def __init__(self, config: TConfig, mcp_client: TMcpClient):
        """Initialize the MCP-enabled agent.

        Args:
            config (TConfig): Validated agent configuration.
            mcp_client (TMcpClient): Instantiated MCP client dependency.

        Raises:
            ValueError: Raised if the provided MCP client is None.
        """
        super().__init__(config)
        if mcp_client is None:
            raise ValueError("MCP client parameter cannot be None")
        self._mcp_client = mcp_client

    def get_obj(self, request: TRequest) -> Optional[TObject]:
        """Retrieve the domain object associated with the provided request.

        Args:
            request (TRequest): Domain request object.

        Returns:
            Optional[TObject]: Matching domain object, if found.
        """
        return self._mcp_client.get_obj(request)
