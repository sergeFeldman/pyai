# Implementation Dataflow

## Module Structure

| Package | Responsibility |
|---|---|
| `app` | FastAPI application, routes, dependencies — HTTP boundary |
| `app.routes` | One module per use case; validates request, calls orchestrator |
| `app.dependencies` | FastAPI `Depends` factories for shared objects (config, services) |
| `workflow` | Orchestrates use cases; no LangChain or framework details |
| `workflow.tools` | Creates MCP-backed LangChain tool sets for LLM agents |
| `agents` | LLM and rule-based agents; `AgentFactory` creates instances |
| `agents.base_agent` | `LlmEnabledAgent` base class with async factory (`create()`) |
| `mcp_clients` | HTTP MCP clients for claim, customer, and policy rule data |
| `mcp_clients.servers` | `csv_mcp_server.py` — stdio MCP server backed by CSV files |
| `models` | Pydantic/dataclass models: `Claim`, `CustomerContext`, `PolicyRule`, `UserRequest`, `UserResponse` |
| `data` | CSV data storage layer (`DataStorageFactory`, `CsvDataStorage`) |
| `services` | `TraceService` — creates and stores trace contexts |
| `core` | Cross-cutting concerns: `Configurable`, `ConfigurableObjectFactory`, `Singleton`, `SerializableMixin` |
| `handlers` | Request handler wiring HTTP layer to orchestrator |
| `config/agents.yaml` | Non-secret agent settings: `llm_provider`, `model`, `prompt_name` per agent |

---

## Use Case 1: Claim Status

**Endpoint:** `GET /claim/status/{claim_id}`

| Step | Component | Action |
|---|---|---|
| 1 | `app.routes.claim_status` | Receives HTTP request; extracts `claim_id` |
| 2 | `app.dependencies` | Injects `TraceService` singleton and agent configs via `Depends` |
| 3 | `handlers.RequestHandler` | Delegates to `WorkflowOrchestrator.get_claim_status()` |
| 4 | `workflow.WorkflowOrchestrator` | Creates trace context; resolves `ClaimAgent`, `CustomerAgent`, `PolicyRuleAgent` from `AgentFactory` |
| 5 | `agents.AgentFactory` | Returns cached agent instances keyed by config hash (Singleton cache) |
| 6 | `agents.ClaimAgent` | Calls `mcp_clients.ClaimMcpClient.get_claim(claim_id)` |
| 7 | `mcp_clients.ClaimMcpClient` | Sends MCP HTTP request to the server |
| 8 | `mcp_clients.servers.csv_mcp_server` | `get_claim` tool reads `data/in/claim.csv` via `DataStorageFactory` |
| 9 | `data.CsvDataStorage` | Parses CSV row into `models.Claim` dataclass |
| 10 | `agents.CustomerAgent` | Calls `ClaimMcpClient.get_customer(customer_id)` from claim |
| 11 | `agents.PolicyRuleAgent` | Calls `get_policy_rule(claim_type, attribute, value)` for each relevant attribute |
| 12 | `models.Claim.to_dict()` | `SerializableMixin` serializes enum fields to `.value` strings |
| 13 | `workflow.WorkflowOrchestrator` | Assembles `models.UserResponse` with explanation message and trace ID |
| 14 | `handlers.RequestHandler` | Returns response to route |
| 15 | `app.routes.claim_status` | Returns HTTP 200 JSON response |

---

## Use Case 2: Claim Explanation

**Endpoint:** `POST /claim/explanation`

| Step | Component | Action |
|---|---|---|
| 1 | `app.routes.claim_explanation` | Receives HTTP request; deserializes `models.UserRequest` (claim ID + attributes) |
| 2 | `app.dependencies` | Injects `TraceService` singleton and `ClaimExplanationAgentConfig` dict via `Depends` |
| 3 | `handlers.RequestHandler` | Delegates to `WorkflowOrchestrator.get_claim_explanation()` |
| 4 | `workflow.WorkflowOrchestrator` | Creates trace context; calls `AgentFactory.get_obj_async("claim_explanation", config)` |
| 5 | `agents.AgentFactory` | On cache miss, calls `ClaimExplanationAgent.create(config)` |
| 6 | `agents.ClaimExplanationAgent.create()` | Calls `_load_tools()` to get MCP-backed LangChain tools |
| 7 | `workflow.tools.create_claim_explanation_tools()` | Spawns `csv_mcp_server.py` subprocess via `MultiServerMCPClient` (stdio transport) |
| 8 | `langchain_mcp_adapters` | Discovers `get_claim`, `get_customer`, `get_policy_rule` tools from MCP server |
| 9 | `agents.ClaimExplanationAgent.create()` | Calls `_create_llm(llm_provider, model)` to instantiate the configured LLM; builds structured chat agent with tools + `hub.pull(prompt_name)` |
| 10 | `langchain.agents.AgentExecutor` | Wraps ReAct agent for multi-step invocation |
| 11 | `agents.ClaimExplanationAgent.get_explanation_message()` | Builds natural-language query string; calls `executor.ainvoke()` |
| 12 | `AgentExecutor` (ReAct loop) | LLM reasons over claim data; calls MCP tools iteratively to gather context |
| 13 | `mcp_clients.servers.csv_mcp_server` | Each tool call reads the relevant CSV and returns a dict |
| 14 | Configured LLM (e.g. `ChatGroq`) | Generates final natural-language explanation grounded in policy rules and customer context |
| 15 | `workflow.WorkflowOrchestrator` | Wraps LLM output in `models.UserResponse`; returns to HTTP layer |

---

## Key Design Patterns

| Pattern | Where Used | Purpose |
|---|---|---|
| `Singleton` metaclass | `WorkflowOrchestrator`, `AgentFactory` | One instance per process; safe shared state |
| `Configurable[TConfig]` base class | All agents | Typed config injection; config stored as `_config` |
| `ConfigurableObjectFactory` | `AgentFactory` | Hash-keyed object cache; `get_obj` / `get_obj_async` |
| Async factory (`create()` classmethod) | `LlmEnabledAgent` | Async tool loading during construction; `__init__` stays sync |
| `create_structured_chat_agent` | `LlmEnabledAgent.create()` | Handles tools with JSON schema `args_schema`; required for MCP tools — plain ReAct (`create_react_agent`) does not support structured tool inputs |
| `LlmAgentConfig(BaseModel)` | `ClaimExplanationAgentConfig` | Plain-string config (`llm_provider`, `model`, `prompt_name`) — no live objects in config |
| YAML config + `_create_llm()` factory | `agents.base_agent` | Provider decoupled from code; switching LLM = change `config/agents.yaml` + matching `*_API_KEY` in `.env` |
| `SerializableMixin.to_dict()` | `Claim`, `CustomerContext`, `PolicyRule` | Enum-safe dict serialization for MCP tool return values |
| MCP stdio transport | `workflow.tools` | LangChain tools backed by a subprocess MCP server; isolated data access |
| Lazy import in `_load_tools()` | `ClaimExplanationAgent` | Breaks circular import between `agents` and `workflow` packages |
| FastAPI `Depends` | `app.dependencies` | Decouples route handlers from object creation; enables testability |
