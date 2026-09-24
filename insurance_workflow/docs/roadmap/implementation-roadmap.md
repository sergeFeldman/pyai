# Implementation Roadmap

## Phase 1: Foundation and Core Use Cases

### Scope

- Shared foundation library (`Configurable`, `Singleton`, `SerializableMixin`, `EntityMetadata`, `KeyedRegistry`, `ExplainableMixin`) extracted into `shared/` and reused across all projects
- Use Case 1: Claim Status — deterministic, sequential; claim record retrieved via MCP, policy rule looked up, structured response assembled
- Use Case 2: Claim Explanation — agentic, LLM-driven; ReAct agent backed by Groq/Anthropic autonomously calls MCP tools, synthesizes a natural-language explanation grounded in policy rules and customer context

### Status

Complete.

---

## Phase 2: Rule Engine and Use Case 3

### Scope

- Rule taxonomy: `DecisionRule` (single-condition scalar comparison with type coercion), `LookupRule` (keyed match returning a fixed output payload)
- `RuleRegistry`: versioned singleton with append-only history, latest-version resolution, active rule ordering, and DAG construction from `input`/`output` field declarations
- `RuleFactory`: type detection from raw dict field presence, instantiation of the correct `Rule` subclass
- ETL pipeline: reads raw input rules, detects field-level changes, bumps versions, writes full audit history to the output file; driven by `config/etl.yaml`
- Rules Dashboard: DAG visualization with hierarchical left-to-right layout, priority-ordered nodes, rule detail panel, version history
- Use Case 3: Claim Appeal Eligibility — rule-driven, deterministic; evaluates all active claim appeal rules from the registry in topological + priority order

### Status

Complete.

---

## Phase 3: Rule Executor and Execution Traceability

### Scope

- `RuleRegistry.execute()`: walks the DAG in topological generation order; gates each `DecisionRule` on its declared `input` preconditions before evaluation; writes fired rule outputs to the shared context as `True`, enabling downstream rules
- `ExecutionMetadata`: trace ID, executor name, UTC timestamp; mirrors `EntityMetadata` on persistent entities
- `RuleExecutionResult`: carries `ExecutionMetadata`, the domain, triggered rules in execution order, and intermediate/terminal outputs; seeded claim/customer context excluded from result
- `ClaimAppealAgent._build_context()`: builds the execution context from claim and customer using `dataclasses.fields()` + `getattr()` to preserve Python types for correct threshold coercion
- `RuleExecutionAuditService`: Singleton audit service; appends one `RuleExecutionResult` record per `execute()` call to `data/audit/rule_executions.jsonl` via `JsonlDataStorage`; trace ID threaded from the orchestrator through `ClaimAppealAgent` into `ExecutionMetadata`
- `JsonlDataStorage`: append-only `DataStorage` subclass; `read()` returns raw dicts; `read_by_key()` supports dot-notation path traversal for nested fields; registered in `DataStorageFactory` as `"jsonl"`
- Test coverage: `TestExecute` (precondition gating, output propagation, metadata, domain) and `TestClaimAppealAgent` (fraud-check-alone vs fraud-with-escalation, amount-tier chains)

### Status

Complete.

---

## Phase 4: Rule Engine Depth

### Scope

- Compound `DecisionRule`: replace the single `operator`/`threshold` pair with a condition list supporting AND/OR/IN logic; `matches()` evaluates the condition tree
- Positive qualification rules: `appeal.qualified` output alongside `appeal.disqualified`; result includes both disqualifying and qualifying rules that fired
- Per-rule audit detail: extend the execution audit record with per-rule outcome (fired, skipped on precondition, skipped on no-match), rule version, and output values; enables the dashboard audit view to show the full rule evaluation sequence

### Status

Pending.

---

## Phase 5: Data and Traceability Hardening

### Scope

- Extraction rules: `ExtractionRule` with a path expression (JSONPath or dot-notation) and a target field name; applied as a preprocessing step before Decision or Lookup rules are evaluated; relevant when integrating with backends that return complex JSON payloads
- PII tokenization service: tokenize sensitive values in rule inputs and audit output before persistence

### Status

Pending.
