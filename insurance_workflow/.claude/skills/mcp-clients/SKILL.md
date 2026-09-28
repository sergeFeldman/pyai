---
name: mcp-clients
description: The MCP client layer in src/mcp_clients - McpStorageClient (DataStorage-backed lookup by primary key or filter), McpRuleClient (RuleRegistry-backed rule access per domain), the concrete claim, customer, claim appeal rule, and policy rule registry clients, how client configs are built from config/storage.yaml, and the csv_mcp_server stdio server whose tools the claim explanation agent calls. Use when adding or changing a client, a lookup method, a storage.yaml entity, a model type mapping, or an MCP server tool, or when debugging a record that is not found or a tool that returns an empty dict.
---

# MCP Clients

## Responsibility

Owns the tool access layer between agents and data: client base classes, concrete clients, their config, and the stdio MCP server.

Does not own storage backend behavior (shared-foundation), rule semantics (rule-engine), or how agents use clients (workflow-agents).

## Storage-backed clients

`McpStorageClient[TConfig, TRequest, TObject]` (`src/mcp_clients/mcp_client.py`) is a `Configurable` over one `DataStorage`.

- Config extends `McpStorageClientConfig`: `data_storage_id` (a `DataStorageId`) and `data_storage_config` (a dict with `model_class` and `file_path`). The storage instance comes from `DataStorageFactory` at construction.
- Declare `_config_data_type` and `_primary_key_field`, the request attribute that holds the key (for example `claim_id`). `get_obj(request)` passes that value to `read_by_key`, which matches the storage's own key field (`<model_type>_id` for CSV), and returns the record or `None`. Keep the two names equal.
- Override `get_obj_by_filter(request)` for criteria lookups; the base raises `NotImplementedError`.
- Return domain dataclasses, never dicts. Return `None` for not found; never raise for a missing record.

Concrete: `ClaimMcpClient` (`claim_id`), `CustomerMcpClient` (`customer_id`), and the legacy `PolicyRuleMcpClient` (see workflow-agents, Legacy).

## Rule-backed clients

`McpRuleClient` holds the `RuleRegistry` singleton. Subclasses expose rules for one domain through a `rules` property built on `get_effective(domain)`, so results are always the latest effective versions in execution order.

- `ClaimAppealRuleMcpClient`: effective `claim_appeal` rules of every type (its `list[DecisionRule]` type hint is wrong; the domain also holds `LookupRule`s). It must also be the agent's path to executing the domain, so `ClaimAppealAgent` never touches the registry itself.
  Status: partial (the client has no execution method yet and `ClaimAppealAgent` calls `RuleRegistry` directly)
- `PolicyRuleRegistryClient`: effective `policy` `LookupRule`s; `find(context)` returns the first matching rule or `None`.

## Config from storage.yaml

`_build_mcp_client_config(key, config_type)` in `src/app/dependencies.py` reads a `config/storage.yaml` entry (`storage_type`, `model_type`, `file_path`), maps `storage_type` to `DataStorageId` and `model_type` to a class through `_MODEL_CLASS_MAPPING`. A new entity needs a `storage.yaml` entry, a `_MODEL_CLASS_MAPPING` entry, and a client config class.

## csv_mcp_server

`src/mcp_clients/servers/csv_mcp_server.py` is a FastMCP stdio server, started as a subprocess by `ClaimExplanationAgent`. It runs in its own process, so it loads its own `RuleRegistry` from the policy rules output file.

Tools: `get_claim(claim_id)`, `get_customer(customer_id)`, `get_policy_rule(policy_rule_id)`, `get_policy_rule_by_filter(claim_type, attribute, value)`.

- Every tool returns a plain dict, or `{}` when nothing matches. Never raise for not found; the LLM reads `{}` as "no record". `get_claim` and `get_customer` build it with `to_dict()` (bools as `"true"`/`"false"`, Enums as values); the policy tools build it from the rule id, the lookup keys, and the rule's `output_values`.
- `get_policy_rule` returns the latest version of the rule by id, even if that version is not effective. `get_policy_rule_by_filter` uses `find()` and sees only effective rules.
- Write each tool docstring for the LLM: what it returns, argument formats with synthetic examples, and the empty-dict case.
- The server reads its entity and rule file paths from `config/storage.yaml`, the same source as the app.
  Status: partial (paths are hardcoded as `data/in/claim.csv`, `data/in/customer_context.csv`, and `data/out/policy_rules.json`, relative to the working directory)

## Verification

- No automated tests cover this layer yet. Status: planned
- For a storage client change, construct the client with a synthetic config in a quick script or new test and call `get_obj`.
- For a server tool change, run `POST /claim-explanation` and confirm the tool is called and returns the expected dict.

## Examples

- Normal: `ClaimMcpClient(cfg).get_obj(mdl.ClaimRequest(claim_id="claim_1"))` returns a `Claim` with `status` as a `ClaimStatus`.
- Edge: an unknown claim id returns `None` from the client and `{}` from `get_claim`.
- Edge: `get_policy_rule_by_filter("auto_collision", "is_fraud", "True")` returns `{}` because match values are lowercase strings (`"true"`).

## Background

- [docs/implementation/dataflow.md](../../../docs/implementation/dataflow.md): key design patterns table.
- [docs/architecture/component-deep-dive.md](../../../docs/architecture/component-deep-dive.md): target MCP Servers design.
