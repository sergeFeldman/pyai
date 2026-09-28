---
name: workflow-agents
description: Agents in src/agents and their factory - the LlmEnabledAgent, McpEnabledAgent and McpStorageAgent base classes, AgentFactory registration and sync versus async creation, agent config models, LLM provider selection from config/agents.yaml, the LangChain ReAct claim explanation agent and its stdio MCP tools, the rule-driven ClaimAppealAgent, and the user-facing messages agents build. Use when adding a new agent, changing an agent's behavior, config, or messages, switching the LLM provider or model, or debugging agent creation or the explanation and appeal flows.
---

# Workflow Agents

## Responsibility

Owns the agent class hierarchy, `AgentFactory`, agent config models, LLM setup, and the behavior and messages of each agent.

Does not own MCP client internals (mcp-clients), rule semantics or the appeal eligibility rule (rule-engine), or audit persistence (audit-and-trace).

## Hierarchy

All agents are `Configurable` components with a `<Agent>Config` model and `_config_data_type`.

| Base | Use for | Contract |
|---|---|---|
| `LlmEnabledAgent[TLlmConfig]` | LangChain agents with tools | Implement `@classmethod async _load_tools(cls) -> list`. `create(config)` loads tools, builds the LLM with `_create_llm(provider, model)`, pulls the prompt from LangChain Hub, and wraps a structured chat agent in `AgentExecutor`. Config extends `LlmAgentConfig` (`llm_provider`, `model`, `prompt_name`). |
| `McpEnabledAgent[TConfig, TMcpClient]` | Agents backed by any MCP client | Pass the client instance to `super().__init__(config, client)`; `None` raises `ValueError`. |
| `McpStorageAgent[TConfig, TMcpClient, TRequest, TObject]` | Agents that fetch one domain object by request | Adds `get_obj(request)` delegating to the client. |

Rules:

- An agent reaches data and rules only through its own MCP client, never through `DataStorage`, `DataStorageFactory`, or `RuleRegistry` directly.
  Status: partial (`ClaimAppealAgent` calls `RuleRegistry().execute()` directly; its `ClaimAppealRuleMcpClient` is created but unused)
- Each agent constructs its own client in `__init__`, from a nested client config when the client needs one (rule clients take none). Clients are not factory-managed; `AgentFactory` caching keeps one client per agent.
- Agents hold no per-request state. One cached instance serves every request.

## AgentFactory

`AgentFactory` (`src/agents/agent_factory.py`) is a `ConfigurableObjectFactory` singleton keyed by agent id: `claim`, `claim_appeal`, `claim_explanation`, `customer`, `policy_rule`.

- Register a new agent by adding it to `_TYPES_MAPPING`. No other factory change is needed.
- `_create_obj_async` detects LLM agents with `issubclass(cls, LlmEnabledAgent)` and awaits `cls.create(config)`. Always obtain LLM agents with `await AgentFactory().get_obj_async(...)`; the sync `get_obj` cannot build them because it passes no executor.
- Instances are cached by id; the first config wins for the life of the process.

## LLM configuration

- Provider and model come from `config/agents.yaml`. `_create_llm` supports `anthropic`, `groq`, `ollama`, and `gemini`; any other value raises `ValueError`. Add a provider by adding a branch with a lazy import and the matching `langchain-*` package in `requirements.txt`.
- `src/app/main.py` loads `.env` into the process environment with `load_dotenv()`; provider SDKs read their keys from there, and the MCP server subprocess inherits the environment. Never pass keys through config or code.
- LangSmith tracing is enabled by environment variables only; see [docs/implementation/setup.md](../../../docs/implementation/setup.md).

## ClaimExplanationAgent

- `_load_tools` starts `src/mcp_clients/servers/csv_mcp_server.py` as a stdio subprocess through `MultiServerMCPClient`, with `PYTHONPATH` set to `src` and `../shared/src`, and returns its tools.
- `get_explanation_message(request)` asks the executor to explain `request.attributes` for claim `request.message`, including policy basis and customer context, and returns the LLM output.

Status: planned (validating requested attributes against `Claim.explainable_attributes()`; `ClaimExplanationRequest`, `ClaimExplanationResult`, and `AttributeExplanation` are unused). Status: planned (trace ID passed into the agent and ReAct intermediate steps captured; see audit-and-trace).

## ClaimAppealAgent

- `_build_context(claim, customer)` builds `{"claim.<field>": value, "customer.<field>": value}` with `dataclasses.fields()` and `getattr()`, keeping native bool, Enum, int, and float types. Never use `to_dict()` here; it turns bools into strings and Enums into values, which breaks threshold coercion.
- `check_eligibility(claim, customer, trace_id)` executes the `claim_appeal` domain with `executed_by="claim_appeal_agent"` and `entities={"claim": claim.to_dict(), "customer": customer.to_dict()}`, logs the result to `RuleExecutionAuditService`, and applies the eligibility rule in [rule-engine](../rule-engine/SKILL.md) (claim_appeal domain output contract) to return `ClaimAppealResult`.

## Messages

User-facing text must be exact for every path:

| Agent method | Case | Message |
|---|---|---|
| `ClaimAgent.get_status_message` | found | `Claim {claim_id} is currently {status.value}.` |
| `ClaimAgent.get_status_message` | not found | `Claim {claim_id} was not found.` |
| `ClaimAppealAgent.get_eligibility_message` | eligible | `Claim {claim_id} is eligible for appeal.` |
| `ClaimAppealAgent.get_eligibility_message` | not eligible | `Claim {claim_id} is not eligible for appeal. {reason}` |

## Legacy

`PolicyRuleAgent` and its CSV-backed `PolicyRuleMcpClient` are registered and configured but used by no workflow. Policy lookups go through `PolicyRuleRegistryClient`. Keep them as they are; do not extend them or wire them into new workflows.

## Adding an agent

1. Create `src/agents/<name>_agent.py` with `<Name>AgentConfig` and `<Name>Agent` on the right base.
2. Set `_config_data_type`; for MCP agents, build the client from a nested client config in `__init__`.
3. Add it to `AgentFactory._TYPES_MAPPING`.
4. Add a config builder and `_AGENT_CONFIGS` entry in `src/app/dependencies.py`.
5. Add tests under `tests/agents/`.

## Verification

- `python -m pytest tests/agents -q`.
- For LLM agents, run `POST /claim-explanation` with a synthetic claim id and check the answer against the claim, customer, and policy data.

## Examples

- Normal: claim `claim_1` denied, `auto_collision`, amount 5000.0, no fraud; customer tenure 6, no prior claims or escalations: `Claim claim_1 is eligible for appeal.`
- Edge: same claim with amount 500.0: `Claim claim_1 is not eligible for appeal. Claim value too low to qualify for appeal.`
- Edge: `AgentFactory().get_obj("claim_explanation", cfg)` fails; use `get_obj_async`.

## Background

- [docs/implementation/dataflow.md](../../../docs/implementation/dataflow.md): call sequence for all three use cases.
- [docs/architecture/component-deep-dive.md](../../../docs/architecture/component-deep-dive.md): target Domain Agents design.
