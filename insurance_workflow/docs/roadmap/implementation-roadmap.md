# Implementation Roadmap

## Phase 1: Foundation and Core Use Cases

### Scope

- Shared foundation library (`Configurable`, `Singleton`, `SerializableMixin`, `EntityMetadata`, `KeyedRegistry`, `ExplainableMixin`) extracted into `shared/` and reused across all projects
- Use Case 1: Claim Status: deterministic, sequential; claim record retrieved via MCP, policy rule looked up, structured response assembled
- Use Case 2: Claim Explanation: agentic, LLM-driven; ReAct agent backed by Groq/Anthropic autonomously calls MCP tools, synthesizes a natural-language explanation grounded in policy rules and customer context

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
- Use Case 3: Claim Appeal Eligibility: rule-driven, deterministic; evaluates all active claim appeal rules from the registry in topological + priority order

### Status

Complete.

---

## Phase 3: Rule Executor and Execution Traceability

### Scope

- `RuleRegistry.execute()`: walks the DAG in topological generation order; before evaluating each rule, checks whether all producers of its required internal inputs have failed or been pruned. If so, the rule is pruned and never evaluated, making the traversal true DAG branch pruning rather than a linear sweep. Rules that pass the pruning check are evaluated against context; fired rule outputs are written to the shared context, enabling downstream rules. Every rule is recorded in the evaluation log with one of four outcomes: `triggered`, `skipped_no_match`, `pruned`, or `skipped_precondition`
- `ExecutionMetadata`: trace ID, executor name, UTC timestamp; mirrors `EntityMetadata` on persistent entities
- `RuleExecutionResult`: carries `ExecutionMetadata`, the domain, triggered rules in execution order, and intermediate/terminal outputs; seeded claim/customer context excluded from result
- `ClaimAppealAgent._build_context()`: builds the execution context from claim and customer using `dataclasses.fields()` + `getattr()` to preserve Python types for correct threshold coercion
- `RuleExecutionAuditService`: Singleton audit service; appends one `RuleExecutionResult` record per `execute()` call to `data/audit/rule_executions.jsonl` via `JsonlDataStorage`; trace ID threaded from the orchestrator through `ClaimAppealAgent` into `ExecutionMetadata`
- `JsonlDataStorage`: append-only `DataStorage` subclass; `read()` returns raw dicts; `read_by_key()` supports dot-notation path traversal for nested fields; registered in `DataStorageFactory` as `"jsonl"`
- Test coverage: `TestExecute` (DAG branch pruning, output propagation, metadata, domain) and `TestClaimAppealAgent` (fraud-check-alone vs fraud-with-escalation, amount-tier chains)

### Status

Complete.

---

## Phase 4: Rule Engine Depth

### Scope

- Compound `DecisionRule`: replace the single `operator`/`threshold` pair with a condition list supporting AND/OR/IN logic; `matches()` evaluates the condition tree
- Positive qualification rules: `appeal.qualified` output alongside `appeal.disqualified`; result includes both disqualifying and qualifying rules that fired
- Per-rule audit detail: extend the execution audit record with per-rule outcome (triggered, skipped_no_match, pruned, skipped_precondition), rule version, and output values; enables the dashboard audit view to show the full rule evaluation sequence

### Status

Pending.

---

## Phase 5: Data and Traceability Hardening

### Scope

- Extraction rules: `ExtractionRule` with a path expression (JSONPath or dot-notation) and a target field name; applied as a preprocessing step before Decision or Lookup rules are evaluated; relevant when integrating with backends that return complex JSON payloads
- PII tokenization service: tokenize sensitive values in rule inputs and audit output before persistence

### Status

Pending.

---

## Phase 6: Agent Reasoning Traceability

### Scope

- Capture LangChain ReAct agent intermediate steps from `AgentExecutor` result for the `claim_explanation` workflow; currently the reasoning trace (tool calls, inputs, responses) is only visible in the terminal and optionally in LangSmith, with nothing stored in the application
- Store intermediate steps in a new audit record per `claim_explanation` request, alongside the final response
- Add a dashboard execution view for `claim_explanation` showing the full reasoning chain: each tool called, its input, the data returned, and the final answer; gives reviewers full visibility into why the agent said what it said

### Status

Pending.
