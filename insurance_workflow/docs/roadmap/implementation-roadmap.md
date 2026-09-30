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

- `RuleRegistry.execute()`: walks the DAG using a ready-queue model (Kahn's variant); a rule enters the queue only when all upstream producers have settled; cascade-prunes consumers whose required inputs will never arrive; stops on first terminal output; records every rule that entered the queue or was cascade-pruned with one of five outcomes: `triggered`, `skipped_no_match`, `pruned`, `skipped_precondition`, or `not_evaluated`
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

- **Ready-queue + eager cascade**: Replace the topological generation sweep in `RuleRegistry.execute()` with a queue-based model (Kahn's variant). A rule enters the queue only when all its producers are settled (evaluated, failed, or pruned). When a rule enters `failed` or `pruned`, immediately walk its consumers and cascade pruning to any whose remaining producers are all decided; pruned rules never enter the queue at all. Precompute and cache the producers map, consumers map, and sorted generation lists on the DAG at build time rather than rebuilding on every `execute()` call. A hard disqualifier firing in an early generation causes immediate early exit without visiting any downstream rules; cascade pruning completes before the next rule is dequeued
- Compound `DecisionRule`: replace the single `operator`/`threshold` pair with a condition list supporting AND/OR/IN logic; `matches()` evaluates the condition tree
- Positive qualification rules: `appeal.qualified` output alongside `appeal.disqualified`; result includes both disqualifying and qualifying rules that fired
- Per-rule audit detail: extend each entry in the execution audit record's `evaluations` list with the rule version and the output values written. The per-rule outcome (triggered, skipped_no_match, pruned, skipped_precondition) is already recorded (done in Phase 3)

### Status

In progress. Ready-queue + eager cascade complete. Compound DecisionRule, appeal.qualified, and per-rule audit detail pending.

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

---

## Phase 7: DAG Execution Model

### Scope

- **Graph validation linter**: Extend `load_from()` beyond cycle detection to catch phantom consumers (a rule declaring an `input` field no active rule produces), orphan producers (fields produced but never consumed by any rule), and cross-domain field mismatches. Validation runs at load time and returns a structured report; phantom consumers are errors that block startup, orphan producers are warnings. Prevents silent `skipped_precondition` decisions at scale where a typo in a field name goes undetected across thousands of executions
- **Parallel execution**: Fire all rules currently in the ready queue concurrently via `asyncio.gather`. Rules in the queue at the same moment have no unsettled dependencies on each other and are safe to evaluate in parallel. Primary payoff is when rules call external services such as fraud scoring APIs, compliance watchlists, and credit bureaus; independent calls in the same topological wave run simultaneously rather than serially. Depends on ready-queue + eager cascade completed in Phase 4

### Status

Pending.

---

## Phase 8: Rule Governance

### Scope

- **Decision replay / shadow testing**: Before a rule change goes live, re-run all historical decisions stored in `rule_executions.jsonl` against a candidate rule set and report which outcomes flip; for example, "37 appeals would move from eligible to disqualified." The `entities` snapshot already stored in every `RuleExecutionResult` provides the frozen claim/customer context required for replay. Combined with `nx.descendants` to show the structural blast radius (which rules in the graph a given change touches), this is the compliance checkpoint before any rule deployment in a regulated environment
- **Goal-directed evaluation**: Accept a `goal` parameter on `execute()`. Use `nx.ancestors` to build the subgraph of rules that can contribute to the requested output, then run only that subgraph. At production scale different call sites need different slices of the graph (a fraud check API, a quick eligibility pre-check, a full appeal evaluation), and each should traverse only the rules it actually needs. Enables partial evaluation for mid-workflow agent checks without running the full domain

### Status

Pending.

---

## Phase 9: Cross-domain Architecture and Explainability

### Scope

- **Field registry / cross-domain provenance**: A versioned map of which domain produces which field, enforced at rule load time. Required once claim rules reference policy outputs (e.g. `policy.appeal_permitted`, `policy.commercial_class`). The graph validation linter introduced in Phase 7 cannot catch phantom consumers that span domain boundaries without a cross-domain field contract. Field renames in one domain must be validated against consumers in other domains before deployment
- **Why-pruned explanation trace**: Given a completed `RuleExecutionResult`, reconstruct the causal chain behind each pruned rule: which producer failed, what field it was supposed to produce, and which downstream rule was pruned as a result. Pure graph traversal over the existing evaluation log and DAG; no new execution machinery required. Extends the per-rule audit detail introduced in Phase 4 to include full pruning provenance. Required for regulatory explainability of adverse appeal decisions where "claim not eligible" is insufficient and the full decision path must be auditable

### Status

Pending.

---

## Phase 10: Long-lived Claim Support

### Scope

- **Incremental re-evaluation**: When claim data changes after an initial decision (fraud investigation result arrives, repair estimate revised, new escalation logged), re-run only the rules downstream of the changed fields rather than the full domain. The engine maintains evaluation state per claim identifier; a delta of changed context keys triggers a targeted subgraph re-execution via `nx.descendants` of the affected fields. Requires per-claim state storage, change detection on context keys, and subgraph result merging. Most complex item in the roadmap; relevant once the system has sufficient operational history to confirm the long-lived claim pattern

### Status

Pending.
