---
name: rule-engine
description: The deterministic rule engine in src/rules and src/etl - Rule, DecisionRule and LookupRule semantics, RuleFactory type detection, RuleRegistry versioning, effective-date filtering, DAG construction and topological/priority ordering, execute() with branch pruning and the four evaluation outcomes, domain output contracts (including the claim appeal eligibility rule), the versioning ETL from data/in to data/out, and the /rules API and /dashboard page. Use when adding or editing rules or rule data, adding a rule type, changing execution or pruning, debugging why a rule fired, was skipped, or was pruned, running the ETL, or changing the DAG dashboard.
---

# Rule Engine

## Responsibility

Owns rule types, rule data, the registry and its DAG, execution semantics, domain output contracts, the rule ETL, and the `/rules/*` API.

Does not own how an agent builds the execution context or turns a result into a user message (workflow-agents), rule clients (mcp-clients), persistence of execution results and the `/executions` API (audit-and-trace), or the dashboard HTML/JS (see the dashboard skill).

## Rule model

All rules are `@dataclass(kw_only=True)` subclasses of `Rule` (`src/rules/rule.py`), versioned entities keyed by `id` within a `domain`.

- `priority`: higher value runs first within a topological level.
- `effective_from` / `effective_to`: timezone-aware ISO 8601 strings. `is_effective` is true only when now is inside the window; a missing, invalid, or naive timestamp makes the rule inactive, never an error.
- `input`: every context key the rule needs, including `subject.attribute` for decision rules and any intermediate field produced by another rule. `output`: the fields it writes on match. These two lists define the DAG, so declare them exactly; a wrong declaration causes a rule to be pruned or skipped silently.
- `reason`: text shown to the customer when this rule decides an outcome. Write it as a complete sentence.
- `metadata`: `EntityMetadata`, written only by the ETL.

Rule types:

- `DecisionRule` supports two modes. **Simple mode** (empty `conditions`): compares `context["<subject>.<attribute>"]` with `threshold` using `operator` (`>=`, `<=`, `==`, `!=`, `>`, `<`). **Compound mode** (non-empty `conditions`): evaluates a `RuleCondition` tree where leaf nodes are scalar comparisons and group nodes combine sub-conditions with `RuleLogic.AND` or `RuleLogic.OR`; groups nest arbitrarily (e.g. `(fraud == True OR prior_claim_count >= 5) AND escalation_history_count >= 2`). In both modes the threshold is coerced to the value's type via module-level `_coerce()`: `"true"`/`"1"` for bool (never `bool(str)`), `type(value)(threshold)` otherwise. `RuleOperator` and `RuleLogic` are `StrEnum`s; `RuleCondition` is a dataclass with leaf fields (`subject`, `attribute`, `operator: RuleOperator`, `threshold`) and group fields (`logic: RuleLogic`, `conditions: list[RuleCondition]`). Module-level `_eval_condition()` recurses the tree. `RuleFactory._build_conditions()` converts raw condition dicts to `RuleCondition` trees on deserialization. On match every `output` field is set to `True`.
- `LookupRule` matches when every `match_keys` entry equals the context value, and writes a copy of `output_values`. It writes the `output_values` keys, not the declared `output` list, so when a lookup rule feeds another rule in `execute()`, its `output_values` keys must equal its `output` fields.

Callers check `ready(context)` before `evaluate(context)`. `ready` requires every `input` key to be present. For `DecisionRule` in simple mode, it also requires `subject.attribute` to be present. For `LookupRule`, it also requires every `match_keys` key. `evaluate` assumes `ready` passed.

## Adding a rule type

1. Add a `@dataclass(kw_only=True)` subclass of `Rule` with a `kind` default and at least one field no other rule type has.
2. Implement `ready` (call `super().ready`) and `evaluate`.
3. Register it in `RuleFactory._TYPES_MAPPING` and export it from `src/rules/__init__.py`.

`RuleFactory.detect_type` picks the first registered class whose unique fields (fields not on `Rule`) appear in the raw dict; the `kind` key is not used. Keep unique field names disjoint across types.

Status: `ExtractionRule` planned; see [docs/roadmap/implementation-roadmap.md](../../../docs/roadmap/implementation-roadmap.md)

## Registry

`RuleRegistry` (`src/rules/rule_registry.py`) is a `KeyedRegistry[Rule]` singleton keyed by domain.

- Append-only: `add()` never replaces; every version of a rule coexists.
- `get_latest(id, domain)` returns the highest `metadata.version`.
- `get_effective(domain, group="")` takes the latest version per id, then drops it if that version is not effective, and orders the rest by topological generation, then priority descending, then id. An older effective version is never used as a fallback: if the latest version has expired, the rule is out.
- `get_dag(domain, group="")` builds an `nx.DiGraph` (node data `rule`) with one edge A to B when `A.output` and `B.input` share at least one field; the edge's `fields` lists every shared field. Graphs are cached per `domain` or `domain:group`; `load()` clears the cache and `replace_with_new=True` rebuilds. `add()` does not clear it, so a graph built before an `add()` is stale until rebuilt.
- `load_from(*paths, domain_config_path=None)` loads every file in one pass, raises `ValueError` if any domain's effective-rule graph has a cycle (expired or superseded versions are not checked), then runs three graph validations on the effective rules: (1) cycle detection (raises), (2) phantom consumer detection (a rule whose `input` declares a field no active rule in the same namespace produces — excluded from the DAG, logged at ERROR), (3) orphan producer detection (a field produced but consumed by no other rule — logged at WARNING; declared terminal outputs are suppressed). Namespace scoping limits phantom consumer checks to fields whose prefix matches the domain's produced-field prefixes, so external context fields such as `claim.amount` are not flagged. After validation, the registry stores `_excluded_rule_ids: set[str]` (rules excluded from execution) and `_validation_report: RuleGraphValidationReport` (all findings). Excluded rules remain in `all()` but never appear in `get_effective()` or the DAG. The app calls it once at startup with `domain_config_path`; the MCP server subprocess calls it for the policy file (no domain config needed there).
- `terminal_outputs(domain)` returns `set[str]` of declared terminal output fields for a domain, or an empty set if no `DomainConfig` was loaded. Agents and APIs use this instead of hardcoding field names.

Concurrency: the registry, its DAG cache, and `RuleFactory` are process-wide singletons with no locking. Never call `load()` while requests are being served, and never mutate a rule object held by the registry.

## Execution

`execute(domain, context, trace_id, executed_by, group, entities)` walks the DAG and returns a `RuleExecutionResult` with `metadata`, `domain`, `triggered` (in execution order), `evaluations`, `entities`, and `outputs` (only keys added during execution). It mutates `context` in place.

A rule appears in `evaluations` if and only if it entered the ready queue or was cascade-pruned. Rules still waiting on a producer that was cut off by early exit never became ready and get no entry. Of rules that do appear, each gets exactly one outcome: `triggered`, `skipped_no_match`, `pruned`, `skipped_precondition`, or `not_evaluated` (was queued when early exit fired).

Priority orders rules only within a cascade wave. A dependent rule always runs after all its producers regardless of its own priority, so the first triggered disqualifier depends on graph depth first and priority second.

Pruning happens in `_cascade()` before enqueue; the in-loop prune check in `execute()` is a defensive fallback. For the pruning algorithm, outcome definitions, and worked examples, read [references/execution-semantics.md](references/execution-semantics.md) before changing `execute()` or debugging an outcome.

Status: partial (evaluations record `rule_id` and `outcome` only; rule version and output values per evaluation are planned). Parallel execution, decision replay, goal-directed evaluation, and incremental re-evaluation are planned.

## DomainConfig

`mdl.DomainConfig` (`src/models/workflow_models.py`, a `WorkflowBaseModel`) declares the terminal output fields for one domain. Terminal outputs are fields produced by rules but consumed by the agent rather than by other rules.

- Declared in `data/in/domain_config.json` (array of `{domain, terminal_outputs}`), copied to `data/out/` by the ETL. Never version-managed; copied as-is.
- Loaded by `load_from(domain_config_path=...)` into `_domain_configs: dict[str, mdl.DomainConfig]` via `mdl.DomainConfig.model_validate(dc)`.
- Used in three places: (1) the graph validation linter suppresses orphan producer warnings for declared terminal fields — a misspelling in a rule's output will not match and will still warn; (2) agents call `terminal_outputs(domain)` to decide eligibility without hardcoding field names; (3) the DAG API endpoint includes `terminal_outputs` in its response so the dashboard can colour nodes correctly.
- Adding a new terminal output: add the field to `data/in/domain_config.json` and run the ETL. No code change required.

## Domain output contracts

Each domain's rules write agreed field names. Agents and APIs read only these fields via `terminal_outputs(domain)`.

**claim_appeal**

- Mixes `DecisionRule`s and `LookupRule`s (for example `ca_commercial_type` matches `claim.claim_type == "commercial"`). All rules write boolean `True` to `appeal.disqualified`; lookup rules declare this in `output_values` as a JSON boolean (`true`), not a string.
- Terminal: `appeal.disqualified`. Intermediates: `appeal.high_value_flagged` (produced by `ca_high_value_flag`, consumed by `ca_high_value_repeat_claimant`) and `appeal.low_value_flagged` (produced by `ca_low_value_flag`, consumed by `ca_low_tier_tenure_check`, `ca_low_value_escalation`, and `ca_low_value_repeat_claimant`). These two producer/consumer chains are the only DAG edges in the domain.
- Eligibility rule: a claim is eligible for appeal if and only if executing the `claim_appeal` domain produces no terminal output (checked via `terminal_outputs("claim_appeal")`). Key presence decides, never its value. When a terminal field is produced, the decision reason is the `reason` of the first rule in `triggered` whose `output` intersects the terminal set.
- Context keys are `claim.<field>` and `customer.<field>` with native Python types.

Status: implemented

**policy_coverage**

- Mixes `DecisionRule`s and `LookupRule`s. Rules check `policy.policy_status`, claim type vs policy type compatibility, `claim.below_deductible`, and a 2-rule DAG chain: `pc_high_value_flag` (produces `policy_coverage.high_value_flagged`) → `pc_high_value_repeat_claimant` (consumes it, produces `policy_coverage.disqualified`).
- Terminal: `policy_coverage.disqualified`. Intermediate: `policy_coverage.high_value_flagged`.
- Eligibility rule: coverage is verified if executing the `policy_coverage` domain produces no terminal output. Context keys are `claim.<field>`, `customer.<field>`, and `policy.<field>` with native Python types.

Status: implemented

**policy**

- `LookupRule`s with `match_keys` `claim_type`, `attribute`, `value` (strings, bools as `"true"`/`"false"`) and `output_values` `denial_basis`, `next_steps`, `policy_section`.
- Not run through `execute()`. `PolicyRuleRegistryClient.find(context)` calls `matches()` on each rule in `get_effective` order and returns the first match, without `ready()`.

Status: implemented

## Rule data and ETL

Raw rules live in `data/in/<domain>_rules.json`; the versioned output in `data/out/<domain>_rules.json` is what the app loads. Never edit output files by hand. For the ETL steps, versioning rules, and the command, read [references/etl-and-versioning.md](references/etl-and-versioning.md).

## Rules API and dashboard

- `GET /rules/domains`: sorted list of loaded domains.
- `GET /rules/dag/{domain}?group=&replace_with_new=`: nodes in `get_effective` order, edges `{from, to, fields}`, and `terminal_outputs` (list of declared terminal field names for the domain); 404 for an unknown domain.
- `GET /rules/history/{domain}/{rule_id}`: all versions newest first with `changed_fields` against the next older version (metadata excluded); 404 if absent.
- `GET /rules/validation-report`: returns the `RuleGraphValidationReport` produced at startup — `is_valid`, `error_count`, `warning_count`, `validated_at`, and a `findings` list (each finding has `rule_id`, `severity`, `kind`, `domain`, `field`, `message`). Consumed by the dashboard Health tab.
- `GET /dashboard`: `src/ui/templates/dashboard.html`, a vis-network page with a Rules tab (DAG, rule detail, version history), an Executions tab that reads `/executions` (owned by audit-and-trace), and a Health tab. The Health tab shows the validation report filtered to the selected domain (domain dropdown required to display findings); the badge in the header is only visible while on the Health tab. Keep the page consistent with all three APIs when any of them changes.

## Verification

- `python -m pytest tests/rules tests/agents -q`. `tests/rules/test_rule_registry.py::TestExecute` covers pruning and propagation; `tests/agents/test_claim_appeal_agent.py` covers appeal chains end to end.
- For rule data changes, run the ETL, then `GET /rules/dag/claim_appeal` and confirm node count, edges, and order.
- When writing tests for `execute()`: a field produced by a rule but consumed by no other rule in the test DAG is a terminal output -- if it appears in context, execution stops immediately and any rules still in the queue get no evaluation entry. To prevent early exit before all intended rules run, ensure each intermediate output is consumed by at least one downstream rule in the test DAG.

## Examples

- Normal: `ca_min_amount` (simple, `claim.amount < 1000`) on a denied claim with `amount=500.0` triggers and writes `appeal.disqualified: True`; the claim is not eligible, with that rule's reason.
- Normal: `ca_fraud` (simple, `claim.is_fraud == True`) fires immediately for a fraud claim and triggers early exit; all lower-priority rules receive `not_evaluated`.
- Normal: `ca_theft_min_tenure` (compound AND, `claim.claim_type == "theft" AND customer.tenure_years < 4`) fires for a theft claim with 3 years tenure; a theft claim with 5 years tenure does not match.
- Normal: `ca_low_value_flag` (simple, `claim.amount < 500`) produces `appeal.low_value_flagged`; this unlocks `ca_low_tier_tenure_check`, `ca_low_value_escalation`, and `ca_low_value_repeat_claimant`. If amount is not below 500 the flag is never set and all three consumers are cascade-pruned.
- Normal: `ca_fraud_escalation_limit` (compound AND, `prior_claim_count >= 5 AND escalation_history_count >= 2`) fires for a repeat claimant with 6 prior claims and 2 escalations; with only 1 escalation it gets `skipped_no_match`.
- Edge: a threshold of `"False"` against a bool field coerces to `False`, not `True`.

## Background

- [docs/architecture/rules.md](../../../docs/architecture/rules.md): taxonomy, engine overview, claim appeal DAG.
- [docs/architecture/component-deep-dive.md](../../../docs/architecture/component-deep-dive.md): Rule Engine design decisions and trade-offs.
- [docs/roadmap/implementation-roadmap.md](../../../docs/roadmap/implementation-roadmap.md): planned engine phases.
