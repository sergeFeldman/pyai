---
name: insurance-workflow
description: Project-wide rules for the insurance_workflow FastAPI claims platform and the ../shared library it builds on. Covers the layer map (routes, RequestHandler, WorkflowOrchestrator, agents, MCP clients, storage and rule registry), startup wiring in app/dependencies.py and config/*.yaml, the orchestrator use-case pattern, HTTP error mapping, Python code style, and how to run and write tests. Use for any code change, review, refactor, bug fix, or test in this project or ../shared, and whenever adding a new endpoint or use case, even if the request names only one file.
---

# Insurance Workflow Platform

## Responsibility

Owns the rules that apply to every Python change in `src/`, `tests/`, and `../shared/src`: how the layers fit together, how objects are wired at startup, the per-use-case orchestration pattern, code style, and test conventions.

Does not own the internals of any layer. Follow the specialized routes at the end for foundation classes, agents, MCP clients, the rule engine, and audit and trace.

## Layer map

Use-case requests flow in one direction, and each layer calls only the layer directly below it (cross-cutting services and models are usable from any layer). The read-only `/rules/*` and `/executions` routes are the exception: they read `RuleRegistry` and `RuleExecutionAuditService` directly and have no handler or orchestrator method.

| Layer | Location | Role |
|---|---|---|
| HTTP | `src/app/routes/`, `src/ui/dashboard.py` | One module per use case. Validates an `*HttpRequest`, calls `RequestHandler.handle()`, returns an `*HttpResponse`. |
| Handler | `src/handlers/request_handler.py` | Maps `request_type` to an orchestrator method via `_WORKFLOWS_MAPPING`; builds `UserRequest`; awaits the method if it is a coroutine. |
| Orchestration | `src/workflow/orchestrator.py` | `WorkflowOrchestrator` singleton. One method per use case. No LangChain, storage, or rule logic here. |
| Agents | `src/agents/` | Domain behavior. Obtained only through `AgentFactory`. |
| Tool access | `src/mcp_clients/` | MCP clients over `DataStorage` or `RuleRegistry`, plus the stdio MCP server. |
| Engine and data | `src/rules/`, `../shared/src/shared/data/` | Rule registry and storage backends. |
| Cross-cutting | `src/services/`, `src/models/`, `../shared/src/shared/core/` | Trace, audit, models, foundation classes. |

Routes never read config or construct agents. Agents never import from `workflow`, `handlers`, or `app`.

## Startup wiring

`src/app/dependencies.py` runs at import time:

1. Loads the rule registry once with `RuleRegistry.load_from(...)`, using the `rule_registry` and `rule_registry_policy` paths from `config/storage.yaml`. A cycle among any domain's effective rules raises here, before any request is served.
2. Builds `_AGENT_CONFIGS`, one dict per agent key (MCP agents' dicts hold an already built client config object), from `config/storage.yaml` and `config/agents.yaml`.
3. `get_workflow_orchestrator()` returns the orchestrator singleton holding `_AGENT_CONFIGS`; `get_request_handler()` wraps it for FastAPI `Depends`.

Config files hold non-secret settings only:

- `config/agents.yaml`: `llm_provider`, `model`, `prompt_name` per LLM agent.
- `config/storage.yaml`: `storage_type`, `model_type`, `file_path` per entity, plus rule registry file paths.
- `config/etl.yaml`: rule ETL input and output paths per domain.

## Orchestrator use-case pattern

Every orchestrator method follows the same steps:

1. `context = self._trace_service.create_context(request)`.
2. Resolve each agent with `agt.AgentFactory().get_obj(key, self._agent_configs[key])`, or `get_obj_async` for LLM agents.
3. Call agent methods. When a required entity is missing, return a `UserResponse` whose message names what was not found, for example `Claim claim_9 was not found.`
4. Build `UserResponse(message, trace_id=context.trace_id)`.
5. Record the request audit entry, following [audit-and-trace](../audit-and-trace/SKILL.md).

Current use cases:

| Endpoint | `request_type` | Orchestrator method | Agents |
|---|---|---|---|
| `POST /claim-status` | `claim_status` | `get_claim_status` | claim |
| `POST /claim-explanation` | `claim_explanation` | `get_claim_explanation` (async) | claim_explanation |
| `POST /claim-appeal` | `claim_appeal` | `get_claim_appeal_eligibility` | claim, customer, claim_appeal |

To add a use case, read [references/new-use-case.md](references/new-use-case.md).

## Error handling

`src/app/main.py` maps exceptions to HTTP responses: `ValueError` to 400, `FileNotFoundError` and `RuntimeError` to 500. Raise `ValueError` for bad client input (such as an unknown request type) and let infrastructure errors propagate. Endpoints that look up a domain or rule return `HTTPException(404)` when it is not loaded. Never swallow an exception to return a success message.

## Code style

Match the surrounding code. The conventions below hold across `src/` and `../shared/src`.

- Start every module with a one-line docstring, for example `"""Claim agent related classes."""`.
- Use Google-style docstrings with `Args:`, `Returns:`, and `Raises:` on public classes and methods.
- In `src/`, import sibling packages whole, with these fixed aliases (a few older lines, such as `import workflow` in `request_handler.py`, predate this; tests import names directly, for example `from rules import RuleRegistry`):

  | Package | Alias |
  |---|---|
  | `shared.core` | `shd_core` |
  | `shared.data` | `shd_data` |
  | `models` | `mdl` |
  | `mcp_clients` | `mcp` |
  | `rules` | `rls` |
  | `services` | `svc` |
  | `agents` | `agt` |
  | `workflow` | `wfl` |
  | `handlers` | `hdl` |

  Inside a package, use relative imports (`from .base_agent import McpStorageAgent`).
- Re-export each package's public names from its `__init__.py` and list them in `__all__`.
- Name a component's config model `<Component>Config`. In `src/`, extend `mdl.WorkflowBaseModel` (Pydantic, `extra="forbid"`), or `LlmAgentConfig` for LLM agents. (`RuleEtlConfig` extends `BaseModel` with `extra="forbid"` directly; storage configs in `../shared` extend `DataStorageConfig`.)
- Model domain entities and requests as `@dataclass` types in `src/models/workflow_models.py`, adding `shd_core.SerializableMixin` or `shd_core.ExplainableMixin` when needed. HTTP payloads are Pydantic models in `src/models/api_models.py`.
- Store injected state in protected attributes (`self._config`, `self._mcp_client`), and add a read-only property only when callers outside the class need it.
- Name class-level constants with a leading underscore and upper case (`_TYPES_MAPPING`, `_OPS`, `_WORKFLOWS_MAPPING`).
- Use `TConfig`, `TRequest`, `TObject` style type variables for generic bases.
- Keep comments sparse and explain why, not what.
- When code behavior changes, update the docstring and any inline comments that describe that behavior in the same change. A docstring that describes superseded behavior is actively misleading.
- Do not use em-dashes in comments, docstrings, docs, or user-facing text.

## Tests

Run the suite from the project root:

```bash
python -m pytest tests/ -q
```

The root `conftest.py` adds `src` and `../shared/src` to `sys.path`. Before writing or changing tests, read [references/testing.md](references/testing.md).

Status: partial (tests cover `rules` and `ClaimAppealAgent` only; no tests yet for `../shared`, MCP clients, the orchestrator, handlers, or routes)

## Verification

- Run the full suite and confirm it passes.
- For an endpoint change, start the app from the project root with `PYTHONPATH=src` and `python -m uvicorn app.main:app` (data paths in `config/*.yaml` are relative to the project root) and call the endpoint (examples in [docs/implementation/setup.md](../../../docs/implementation/setup.md)).
- When behavior changes, update the owning skill and the matching section of [docs/implementation/dataflow.md](../../../docs/implementation/dataflow.md) in the same change.

## Background

- [docs/implementation/dataflow.md](../../../docs/implementation/dataflow.md): per-use-case call sequence and design pattern table.
- [docs/implementation/setup.md](../../../docs/implementation/setup.md): installation, environment variables, running the app.
- [docs/architecture/agentic-platform-overview.md](../../../docs/architecture/agentic-platform-overview.md): target enterprise architecture this POC is a subset of.

## Specialized skills

- Use [shared-foundation](../shared-foundation/SKILL.md) to build or change a Configurable component, a ConfigurableObjectFactory subclass, a Singleton, a KeyedRegistry, a dataclass using SerializableMixin or ExplainableMixin, EntityMetadata or ExecutionMetadata, or a DataStorage backend in ../shared.
- Use [workflow-agents](../workflow-agents/SKILL.md) to add, change, or debug an agent, its config model, or its AgentFactory registration, including the LLM-backed claim explanation agent and the rule-driven claim appeal agent.
- Use [mcp-clients](../mcp-clients/SKILL.md) to add or change a storage-backed or rule-registry-backed MCP client, its storage.yaml entry, or a tool exposed by the csv_mcp_server stdio server.
- Use [rule-engine](../rule-engine/SKILL.md) to add, change, or debug rule types, rule data in data/in, RuleRegistry DAG ordering and execute() semantics, the rule ETL and versioning, or the rules API and dashboard.
- Use [audit-and-trace](../audit-and-trace/SKILL.md) to change trace ID creation, request-level audit logging, rule execution audit records, the executions API, or what the platform persists for compliance.
