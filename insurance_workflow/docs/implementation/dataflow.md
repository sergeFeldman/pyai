# Implementation Dataflow

## Module Structure

| Package | Responsibility |
|---|---|
| `app` | FastAPI application, routes, dependencies (HTTP boundary) |
| `app.routes` | One module per use case; validates request, calls handler |
| `app.dependencies` | Builds `_AGENT_CONFIGS` constant, loads `RuleRegistry` at startup, wires all shared objects |
| `workflow` | Orchestrates use cases; owns agent configs; no LangChain or framework details |
| `agents` | LLM and rule-based agents; `AgentFactory` creates and caches instances |
| `agents.base_agent` | `LlmEnabledAgent` base class with async factory (`create()`); `McpEnabledAgent` base class for MCP-backed agents |
| `mcp_clients` | MCP clients for claim, customer, policy rule, claim appeal rule, and policy rule registry; `McpRuleClient` base serves rules from `RuleRegistry` |
| `mcp_clients.servers` | `csv_mcp_server.py`: stdio MCP server; exposes `get_claim`, `get_customer`, `get_policy_rule`, `get_policy_rule_by_filter` tools backed by registry and CSV |
| `rules` | Rule taxonomy (`DecisionRule`, `LookupRule`), `RuleRegistry` (versioned singleton, DAG-aware, executor), `RuleFactory` (type detection and instantiation), `RuleExecutionResult` (execution result with metadata, triggered rules, ordered per-rule evaluations, entity snapshots, and outputs) |
| `etl` | `RuleEtl`: versioning ETL pipeline; reads raw JSON, detects changes, bumps versions, writes full audit history; driven by `config/etl.yaml` |
| `models` | Pydantic/dataclass models: `Claim`, `Customer`, `PolicyRule`, `ClaimAppealResult`, `UserRequest`, `UserResponse`, `AuditRecord` |
| `services` | `TraceService` (trace context creation), `AuditService` (request-level audit log), `RuleExecutionAuditService` (rule execution audit log; appends to JSONL) |
| `core` | Cross-cutting concerns: `Configurable`, `ConfigurableObjectFactory`, `Singleton`, `SerializableMixin`, `EntityMetadata`, `ExecutionMetadata`; provided by the [`shared`](../../../shared/docs/core.md) foundation library |
| `handlers` | Request handler wiring HTTP layer to orchestrator |
| `config/agents.yaml` | Non-secret agent settings: `llm_provider`, `model`, `prompt_name` per agent |
| `config/storage.yaml` | Storage backend settings: `storage_type`, `model_type`, `file_path` per domain entity and rule registry |
| `config/etl.yaml` | ETL pipeline settings: `input_file_path`, `output_file_path`, `updated_by` per domain |

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
| 5 | `agents.CustomerAgent` | Calls `CustomerMcpClient.get_obj(CustomerRequest)` using `claim.customer_id` → returns `models.Customer` |
| 6 | `agents.ClaimAppealAgent` | Calls `get_eligibility_message(claim, customer)` |
| 7 | `agents.ClaimAppealAgent._build_context()` | Builds flat execution context from claim + customer using `dataclasses.fields()` + `getattr()`, preserving `bool` and `Enum` types for correct threshold coercion |
| 8 | `rules.RuleRegistry.execute("claim_appeal", context, entities=...)` | Walks the DAG in topological generation order; gates each `DecisionRule` on its declared `input` preconditions; writes triggered rule outputs to the shared context as `True`; records every rule evaluation with its outcome (triggered / skipped_precondition / skipped_no_match); returns `RuleExecutionResult` |
| 9 | `services.RuleExecutionAuditService.log(result)` | Appends one JSONL record to `data/audit/rule_executions.jsonl` capturing trace ID, domain, triggered rules in execution order, ordered per-rule evaluations, entity snapshots (claim and customer), and outputs |
| 10 | `agents.ClaimAppealAgent.check_eligibility()` | If `"appeal.disqualified"` in `result.outputs` → `ClaimAppealResult(eligible=False, reason)`; else → `ClaimAppealResult(eligible=True)` |
| 11 | `workflow.WorkflowOrchestrator` | Wraps message in `models.UserResponse`; returns to HTTP layer |
| 12 | `app.routes.claim_appeal` | Returns HTTP 200 JSON response |

---

## Rule Architecture

Business rules (appeal eligibility, policy lookup, routing decisions, pricing) follow a structured taxonomy: Decision, Lookup, and Extraction. See [Rule Architecture](../architecture/rules.md).

---

## Key Design Patterns

| Pattern | Where Used | Purpose |
|---|---|---|
| `Singleton` metaclass | `WorkflowOrchestrator`, `AgentFactory`, `RuleRegistry` | One instance per process; safe shared state |
| `_AGENT_CONFIGS` constant | `app.dependencies` | All agent configs built at import time and owned by the orchestrator; routes never touch configs |
| `Configurable[TConfig]` base class | All agents and MCP clients | Typed config injection; config stored as `_config` |
| `ConfigurableObjectFactory` | `AgentFactory`, `DataStorageFactory` | Hash-keyed object cache; `get_obj` / `get_obj_async` |
| `issubclass(LlmEnabledAgent)` check | `AgentFactory._create_obj_async()` | LLM-backed agents detected by type, not by name; adding a new agent requires no factory changes |
| Async factory (`create()` classmethod) | `LlmEnabledAgent` | Async tool loading during construction; `__init__` stays sync |
| `create_structured_chat_agent` | `LlmEnabledAgent.create()` | Handles tools with JSON schema `args_schema`; required for MCP tools |
| `LlmAgentConfig(BaseModel)` | `ClaimExplanationAgentConfig` | Plain-string config (`llm_provider`, `model`, `prompt_name`); no live objects in config |
| YAML config + `_create_llm()` factory | `agents.base_agent` | Provider decoupled from code; switching LLM requires only `config/agents.yaml` change |
| `McpRuleClient` base class | `ClaimAppealRuleMcpClient`, `PolicyRuleRegistryClient` | Serves active rules from `RuleRegistry`; `rules` property is the only required override |
| `McpStorageClient` base class | `ClaimMcpClient`, `CustomerMcpClient`, `PolicyRuleMcpClient` | Shared `get_obj()` (primary key) and `get_obj_by_filter()` (criteria-based) backed by `DataStorage` |
| `SerializableMixin.to_dict()` | `Claim`, `Customer`, `PolicyRule` | Enum-safe, bool-safe dict serialization for MCP tool return values |
| MCP stdio transport | `ClaimExplanationAgent._load_tools()` | LangChain tools backed by a subprocess MCP server; isolated data access; path resolved relative to `__file__` |
| FastAPI `Depends` | `app.dependencies` | Decouples route handlers from object creation; enables testability |
| `RuleRegistry` append-only versioning | `etl.RuleEtl`, `app.dependencies` | ETL bumps versions on change and preserves all history; startup loads current active versions |
| `RuleRegistry.get_active()` topological sort | `ClaimAppealRuleMcpClient.rules`, `app.routes.rules` | Rules returned in DAG execution order: root rules first, priority descending within each level |
| `RuleRegistry.execute()` DAG execution | `ClaimAppealAgent.check_eligibility()` | Walks rules in topological generation order; gates each rule on declared `input` preconditions; propagates outputs through shared context; records every evaluated rule with outcome (triggered / skipped_precondition / skipped_no_match); returns `RuleExecutionResult` |
| `_build_context()` with `dataclasses.fields()` | `ClaimAppealAgent.check_eligibility()` | Preserves Python types (`bool`, `Enum`) for correct threshold coercion; `to_dict()` must not be used here as it converts `bool` to `"true"` string |
| `entities` in `RuleRegistry.execute()` | `ClaimAppealAgent.check_eligibility()` | `claim.to_dict()` and `customer.to_dict()` passed as entity snapshots; captured at execution time so the audit record carries full domain object state regardless of later mutations |
| `ExecutionMetadata` / `RuleExecutionResult` | `RuleRegistry.execute()` | Audit metadata (trace ID, executor, timestamp) composed into execution results, mirroring `EntityMetadata` on persistent entities |
| `DecisionRule.matches(value)` | `RuleRegistry.execute()` | Rule owns its own evaluation logic; `_OPS` dict is a class constant to avoid per-call instantiation |
| `LookupRule.matches(context)` | `PolicyRuleRegistryClient.find()` | Checks all `match_keys` against a context dict; used to find the applicable policy rule for a claim attribute |
| `comprehensive_hashing=True` | `DataStorageFactory` | Required when multiple storage backends share the same type id (e.g. `"json"`) but differ by `file_path` |
| `RuleExecutionAuditService` Singleton | `ClaimAppealAgent.check_eligibility()` | Self-initializes on first use; appends `RuleExecutionResult.to_dict()` to `data/audit/rule_executions.jsonl` via `JsonlDataStorage` after every `execute()` call; each record includes evaluations and entity snapshots |
| `JsonlDataStorage` append-only write | `RuleExecutionAuditService` | `write()` appends rather than overwrites; `read_by_key()` supports dot-notation path for nested fields (e.g. `metadata.trace_id`) |
