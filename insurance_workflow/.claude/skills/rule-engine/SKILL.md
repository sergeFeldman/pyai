---
name: rule-engine
description: The deterministic rule engine in src/rules and src/etl - Rule, DecisionRule and LookupRule semantics, RuleFactory type detection, RuleRegistry versioning, effective-date filtering, DAG construction and topological/priority ordering, execute() with branch pruning and the four evaluation outcomes, domain output contracts (including the claim appeal eligibility rule), the versioning ETL from data/in to data/out, and the /rules API and /dashboard page. Use when adding or editing rules or rule data, adding a rule type, changing execution or pruning, debugging why a rule fired, was skipped, or was pruned, running the ETL, or changing the DAG dashboard.
---

# Rule Engine

## Responsibility

Owns rule types, rule data, the registry and its DAG, execution semantics, domain output contracts, the rule ETL, the `/rules/*` API, and the `/dashboard` page.

Does not own how an agent builds the execution context or turns a result into a user message (workflow-agents), rule clients (mcp-clients), or persistence of execution results and the `/executions` API (audit-and-trace).

## Rule model

All rules are `@dataclass(kw_only=True)` subclasses of `Rule` (`src/rules/rule.py`), versioned entities keyed by `id` within a `domain`.

- `priority`: higher value runs first within a topological level.
- `effective_from` / `effective_to`: timezone-aware ISO 8601 strings. `is_effective` is true only when now is inside the window; a missing, invalid, or naive timestamp makes the rule inactive, never an error.
- `input`: every context key the rule needs, including `subject.attribute` for decision rules and any intermediate field produced by another rule. `output`: the fields it writes on match. These two lists define the DAG, so declare them exactly; a wrong declaration causes a rule to be pruned or skipped silently.
- `reason`: text shown to the customer when this rule decides an outcome. Write it as a complete sentence.
- `metadata`: `EntityMetadata`, written only by the ETL.

Rule types:

- `DecisionRule` compares `context["<subject>.<attribute>"]` with `threshold` using `operator` (`>=`, `<=`, `==`, `!=`, `>`, `<`). The threshold is stored as a string and coerced to the value's type: `"true"`/`"1"` for bool (never `bool(str)`), `type(value)(threshold)` otherwise, so an Enum value compares against `Enum(threshold)`. On match every `output` field is set to `True`.
- `LookupRule` matches when every `match_keys` entry equals the context value, and writes a copy of `output_values`. It writes the `output_values` keys, not the declared `output` list, so when a lookup rule feeds another rule in `execute()`, its `output_values` keys must equal its `output` fields.

Callers check `ready(context)` before `evaluate(context)`. `ready` requires every `input` key (plus `subject.attribute` or every `match_keys` key) to be present; `evaluate` assumes it.

## Adding a rule type

1. Add a `@dataclass(kw_only=True)` subclass of `Rule` with a `kind` default and at least one field no other rule type has.
2. Implement `ready` (call `super().ready`) and `evaluate`.
3. Register it in `RuleFactory._TYPES_MAPPING` and export it from `src/rules/__init__.py`.

`RuleFactory.detect_type` picks the first registered class whose unique fields (fields not on `Rule`) appear in the raw dict; the `kind` key is not used. Keep unique field names disjoint across types.

Status: planned (compound AND/OR/IN decision conditions, `appeal.qualified` qualification rules, `ExtractionRule`; see [docs/roadmap/implementation-roadmap.md](../../../docs/roadmap/implementation-roadmap.md))

## Registry

`RuleRegistry` (`src/rules/rule_registry.py`) is a `KeyedRegistry[Rule]` singleton keyed by domain.

- Append-only: `add()` never replaces; every version of a rule coexists.
- `get_latest(id, domain)` returns the highest `metadata.version`.
- `get_effective(domain, group="")` takes the latest version per id, then drops it if that version is not effective, and orders the rest by topological generation, then priority descending, then id. An older effective version is never used as a fallback: if the latest version has expired, the rule is out.
- `get_dag(domain, group="")` builds an `nx.DiGraph` (node data `rule`) with one edge A to B when `A.output` and `B.input` share at least one field; the edge's `fields` lists every shared field. Graphs are cached per `domain` or `domain:group`; `load()` clears the cache and `replace_with_new=True` rebuilds. `add()` does not clear it, so a graph built before an `add()` is stale until rebuilt.
- `load_from(*paths)` loads every file in one pass and raises `ValueError` if any domain's effective-rule graph has a cycle (expired or superseded versions are not checked). The app calls it once at startup; the MCP server subprocess calls it for the policy file.

Concurrency: the registry, its DAG cache, and `RuleFactory` are process-wide singletons with no locking. Never call `load()` while requests are being served, and never mutate a rule object held by the registry.

## Execution

`execute(domain, context, trace_id, executed_by, group, entities)` walks the DAG and returns a `RuleExecutionResult` with `metadata`, `domain`, `triggered` (in execution order), `evaluations`, `entities`, and `outputs` (only keys added during execution). It mutates `context` in place.

Every rule gets exactly one outcome: `triggered`, `skipped_no_match`, `pruned`, `skipped_precondition`, or `not_evaluated` (in the queue when early exit fired).

Pruning happens in `_cascade()` before enqueue; the in-loop prune check in `execute()` is a defensive fallback. For the pruning algorithm, outcome definitions, and worked examples, read [references/execution-semantics.md](references/execution-semantics.md) before changing `execute()` or debugging an outcome.

Status: partial (evaluations record `rule_id` and `outcome` only; rule version and output values per evaluation are planned). Graph validation linter, parallel execution, decision replay, goal-directed evaluation, and incremental re-evaluation are planned.

## Domain output contracts

Each domain's rules write agreed field names. Agents and APIs read only these fields.

**claim_appeal**

- Mixes `DecisionRule`s and `LookupRule`s (for example `ca_commercial_type` matches `claim.claim_type == "commercial"`). All rules write boolean `True` to `appeal.disqualified`; lookup rules declare this in `output_values` as a JSON boolean (`true`), not a string.
- Terminal: `appeal.disqualified`. Intermediate: `appeal.risk_flagged` (produced by more than one rule), `appeal.amount_tier`.
- Eligibility rule: a claim is eligible for appeal if and only if executing the `claim_appeal` domain produces no `appeal.disqualified` output. Key presence decides, never its value. When it is produced, the decision reason is the `reason` of the first rule in `triggered` whose `output` contains `appeal.disqualified`.
- Context keys are `claim.<field>` and `customer.<field>` with native Python types.

Status: implemented

**policy**

- `LookupRule`s with `match_keys` `claim_type`, `attribute`, `value` (strings, bools as `"true"`/`"false"`) and `output_values` `denial_basis`, `next_steps`, `policy_section`.
- Not run through `execute()`. `PolicyRuleRegistryClient.find(context)` calls `matches()` on each rule in `get_effective` order and returns the first match, without `ready()`.

Status: implemented

## Rule data and ETL

Raw rules live in `data/in/<domain>_rules.json`; the versioned output in `data/out/<domain>_rules.json` is what the app loads. Never edit output files by hand. For the ETL steps, versioning rules, and the command, read [references/etl-and-versioning.md](references/etl-and-versioning.md).

## Rules API and dashboard

- `GET /rules/domains`: sorted list of loaded domains.
- `GET /rules/dag/{domain}?group=&replace_with_new=`: nodes in `get_effective` order and edges `{from, to, fields}`; 404 for an unknown domain.
- `GET /rules/history/{domain}/{rule_id}`: all versions newest first with `changed_fields` against the next older version (metadata excluded); 404 if absent.
- `GET /dashboard`: `src/ui/templates/dashboard.html`, a vis-network page with a Rules tab (DAG, rule detail, version history) and an Executions tab that reads `/executions` (owned by audit-and-trace). Keep the page consistent with both APIs when either changes.

## Verification

- `python -m pytest tests/rules tests/agents -q`. `tests/rules/test_rule_registry.py::TestExecute` covers pruning and propagation; `tests/agents/test_claim_appeal_agent.py` covers appeal chains end to end.
- For rule data changes, run the ETL, then `GET /rules/dag/claim_appeal` and confirm node count, edges, and order.
- When writing tests for `execute()`: a field produced by a rule but consumed by no other rule in the test DAG is a terminal output -- if it appears in context, execution stops immediately and any rules still in the queue get no evaluation entry. To prevent early exit before all intended rules run, ensure each intermediate output is consumed by at least one downstream rule in the test DAG.

## Examples

- Normal: `ca_min_amount` (`claim.amount < 1000`) on a denied 500.0 claim that no other rule disqualifies triggers and writes `appeal.disqualified: True`; the claim is not eligible, with that rule's reason.
- Edge: a claim with `is_fraud=True` and `escalation_history_count=0` triggers `ca_fraud_check` (`appeal.risk_flagged`) but `ca_fraud_escalation_limit` gets `skipped_no_match`; no disqualification comes from the fraud branch.
- Edge: when both `ca_fraud_check` and `ca_repeat_claimant_flag` fail, `ca_fraud_escalation_limit` is `pruned`; if either triggers, it is evaluated.
- Edge: a threshold of `"False"` against a bool field coerces to `False`, not `True`.

## Background

- [docs/architecture/rules.md](../../../docs/architecture/rules.md): taxonomy, engine overview, claim appeal DAG.
- [docs/architecture/component-deep-dive.md](../../../docs/architecture/component-deep-dive.md): Rule Engine design decisions and trade-offs.
- [docs/roadmap/implementation-roadmap.md](../../../docs/roadmap/implementation-roadmap.md): planned engine phases.
