# Implementation Dataflow

## Module Structure

| Package | Responsibility |
|---|---|
| `app` | FastAPI application, routes, dependencies (HTTP boundary) |
| `app.routes` | One module per use case; validates request, calls handler |
| `app.dependencies` | Builds `_AGENT_CONFIGS` constant and wires all shared objects at startup |
| `workflow` | Orchestrates use cases; owns agent configs; no LangChain or framework details |
| `agents` | LLM and rule-based agents; `AgentFactory` creates and caches instances |
| `agents.base_agent` | `LlmEnabledAgent` base class with async factory (`create()`); `McpEnabledAgent` base class for MCP-backed agents |
| `mcp_clients` | MCP clients for claim, customer, policy rule, and claim appeal rule data; `MpcClient` base provides `get_obj()` and `get_obj_by_filter()` |
| `mcp_clients.servers` | `csv_mcp_server.py`: stdio MCP server backed by CSV files; exposes `get_claim`, `get_customer`, `get_policy_rule`, `get_policy_rule_by_filter` tools |
| `models` | Pydantic/dataclass models: `Claim`, `CustomerContext`, `PolicyRule`, `ClaimAppealRule`, `ClaimAppealResult`, `UserRequest`, `UserResponse` |
| `data` | CSV data storage layer (`DataStorageFactory`, `CsvDataStorage`) |
| `services` | `TraceService`: creates and stores trace contexts |
| `core` | Cross-cutting concerns: `Configurable`, `ConfigurableObjectFactory`, `Singleton`, `SerializableMixin` |
| `handlers` | Request handler wiring HTTP layer to orchestrator |
| `config/agents.yaml` | Non-secret agent settings: `llm_provider`, `model`, `prompt_name` per agent |
| `config/storage.yaml` | Storage backend settings: `storage_type`, `model_type`, `file_path` per domain entity |

---

## Use Case 1: Claim Status

**Endpoint:** `POST /claim-status`

| Step | Component | Action |
|---|---|---|
| 1 | `app.routes.claim_status` | Receives HTTP request; extracts `claim_id` |
| 2 | `handlers.RequestHandler` | Delegates to `WorkflowOrchestrator.get_claim_status()` |
| 3 | `workflow.WorkflowOrchestrator` | Creates trace context; resolves `ClaimAgent` from `AgentFactory` using `self._agent_configs["claim"]` |
| 4 | `agents.AgentFactory` | Returns cached agent instance (Singleton cache) |
| 5 | `agents.ClaimAgent` | Calls `ClaimMcpClient.get_obj(ClaimRequest)` |
| 6 | `mcp_clients.ClaimMcpClient` | Calls `CsvDataStorage.read_by_key(claim_id)` |
| 7 | `data.CsvDataStorage` | Parses CSV row into `models.Claim` dataclass |
| 8 | `workflow.WorkflowOrchestrator` | Assembles `models.UserResponse` with status message and trace ID |
| 9 | `handlers.RequestHandler` | Returns response to route |
| 10 | `app.routes.claim_status` | Returns HTTP 200 JSON response |

---

## Use Case 2: Claim Explanation

**Endpoint:** `POST /claim-explanation`

| Step | Component | Action |
|---|---|---|
| 1 | `app.routes.claim_explanation` | Receives HTTP request; deserializes `models.UserRequest` (claim ID + attributes) |
| 2 | `handlers.RequestHandler` | Delegates to `WorkflowOrchestrator.get_claim_explanation()` |
| 3 | `workflow.WorkflowOrchestrator` | Creates trace context; calls `AgentFactory.get_obj_async("claim_explanation", self._agent_configs["claim_explanation"])` |
| 4 | `agents.AgentFactory` | On cache miss, detects `ClaimExplanationAgent` is a `LlmEnabledAgent` subclass; calls `ClaimExplanationAgent.create(config)` |
| 5 | `agents.ClaimExplanationAgent.create()` | Calls `_load_tools()` to get MCP-backed LangChain tools |
| 6 | `agents.ClaimExplanationAgent._load_tools()` | Spawns `csv_mcp_server.py` subprocess via `MultiServerMCPClient` (stdio transport) |
| 7 | `langchain_mcp_adapters` | Discovers `get_claim`, `get_customer`, `get_policy_rule`, `get_policy_rule_by_filter` tools from MCP server |
| 8 | `agents.ClaimExplanationAgent.create()` | Calls `_create_llm(llm_provider, model)` to instantiate the configured LLM; builds structured chat agent with tools + `hub.pull(prompt_name)` |
| 9 | `langchain.agents.AgentExecutor` | Wraps ReAct agent for multi-step invocation |
| 10 | `agents.ClaimExplanationAgent.get_explanation_message()` | Builds natural-language query string; calls `executor.ainvoke()` |
| 11 | `AgentExecutor` (ReAct loop) | LLM reasons over claim data; calls MCP tools iteratively to gather context |
| 12 | `mcp_clients.servers.csv_mcp_server` | Each tool call reads the relevant CSV and returns a dict |
| 13 | Configured LLM (e.g. `ChatGroq`) | Generates final natural-language explanation grounded in policy rules and customer context |
| 14 | `workflow.WorkflowOrchestrator` | Wraps LLM output in `models.UserResponse`; returns to HTTP layer |

---

## Use Case 3: Claim Appeal Eligibility

**Endpoint:** `POST /claim-appeal`

| Step | Component | Action |
|---|---|---|
| 1 | `app.routes.claim_appeal` | Receives HTTP request; extracts `claim_id` |
| 2 | `handlers.RequestHandler` | Delegates to `WorkflowOrchestrator.get_claim_appeal_eligibility()` |
| 3 | `workflow.WorkflowOrchestrator` | Creates trace context; resolves `ClaimAgent`, `CustomerAgent`, `ClaimAppealAgent` from `AgentFactory` |
| 4 | `agents.ClaimAgent` | Calls `ClaimMcpClient.get_obj(ClaimRequest)` → returns `models.Claim` |
| 5 | `agents.CustomerAgent` | Calls `CustomerMcpClient.get_obj(CustomerRequest)` using `claim.customer_id` → returns `models.CustomerContext` |
| 6 | `agents.ClaimAppealAgent` | Calls `get_eligibility_message(claim, customer)` |
| 7 | `agents.ClaimAppealAgent.check_eligibility()` | Iterates `ClaimAppealRuleMcpClient.rules` (loaded once at init); calls `rule.matches(value)` for each rule |
| 8 | `models.ClaimAppealRule.matches()` | Evaluates field value against operator and threshold using `_OPS` class constant; returns `True` if rule fires |
| 9 | `agents.ClaimAppealAgent` | First matching rule → `ClaimAppealResult(eligible=False, reason)`; no match → `ClaimAppealResult(eligible=True)` |
| 10 | `workflow.WorkflowOrchestrator` | Wraps message in `models.UserResponse`; returns to HTTP layer |
| 11 | `app.routes.claim_appeal` | Returns HTTP 200 JSON response |

---

## Key Design Patterns

| Pattern | Where Used | Purpose |
|---|---|---|
| `Singleton` metaclass | `WorkflowOrchestrator`, `AgentFactory` | One instance per process; safe shared state |
| `_AGENT_CONFIGS` constant | `app.dependencies` | All agent configs built at import time and owned by the orchestrator; routes never touch configs |
| `Configurable[TConfig]` base class | All agents and MCP clients | Typed config injection; config stored as `_config` |
| `ConfigurableObjectFactory` | `AgentFactory`, `DataStorageFactory` | Hash-keyed object cache; `get_obj` / `get_obj_async` |
| `issubclass(LlmEnabledAgent)` check | `AgentFactory._create_obj_async()` | LLM-backed agents detected by type, not by name; adding a new agent requires no factory changes |
| Async factory (`create()` classmethod) | `LlmEnabledAgent` | Async tool loading during construction; `__init__` stays sync |
| `create_structured_chat_agent` | `LlmEnabledAgent.create()` | Handles tools with JSON schema `args_schema`; required for MCP tools |
| `LlmAgentConfig(BaseModel)` | `ClaimExplanationAgentConfig` | Plain-string config (`llm_provider`, `model`, `prompt_name`); no live objects in config |
| YAML config + `_create_llm()` factory | `agents.base_agent` | Provider decoupled from code; switching LLM requires only `config/agents.yaml` change |
| `MpcClient` base class | `mcp_clients` | Shared `get_obj()` (primary key) and `get_obj_by_filter()` (criteria-based); filter raises `NotImplementedError` by default |
| `_primary_key_field` class attribute | `ClaimMcpClient`, `CustomerMcpClient`, `PolicyRuleMcpClient`, `ClaimAppealRuleMcpClient` | Declares the primary key field name; base `get_obj()` uses it for key-based lookup |
| `SerializableMixin.to_dict()` | `Claim`, `CustomerContext`, `PolicyRule` | Enum-safe, bool-safe dict serialization for MCP tool return values |
| MCP stdio transport | `ClaimExplanationAgent._load_tools()` | LangChain tools backed by a subprocess MCP server; isolated data access; path resolved relative to `__file__` |
| FastAPI `Depends` | `app.dependencies` | Decouples route handlers from object creation; enables testability |
| Eager rule loading | `ClaimAppealRuleMcpClient.__init__()` | Static disqualification rules loaded once at client construction; served from memory via `rules` property |
| All-disqualifiers pattern | `ClaimAppealAgent.check_eligibility()` | Rules are disqualifiers only; first match → not eligible; no match → eligible; eliminates ordering dependency |
| `matches()` on model | `ClaimAppealRule` | Rule owns its own evaluation logic; `_OPS` dict is a class constant to avoid per-call instantiation |
| `comprehensive_hashing=True` | `DataStorageFactory` | Required when multiple storage backends share the same type id (e.g. `"csv"`) but differ by `file_path` or `model_type` |
