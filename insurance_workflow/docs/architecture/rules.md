# Rule Architecture

## Overview

Rules encode business logic that determines outcomes (eligibility, pricing, routing, and data transformation) without requiring code changes. This document describes the rule taxonomy used across the platform, the current implementation, and the extension path as complexity grows.

---

## Rule Categories

### Decision

Evaluates one condition against input facts. If the condition matches, the rule produces an output: a disqualification reason, a flag, a routing decision, or a premium adjustment.

**Example:**
```
IF claim.amount < 1000
THEN appeal.disqualified, reason = "Claim value too low to qualify for appeal."
```

**Current implementation:** `DecisionRule` in the `rules` package. Supports two modes:

- **Simple mode** (empty `conditions` list): evaluates a single `subject.attribute` comparison against `threshold` using `operator`. Operator support: `>=`, `<=`, `==`, `!=`, `>`, `<` via module-level `_OPS`. Type coercion handles bool, int, and float thresholds automatically via module-level `_coerce()`.
- **Compound mode** (non-empty `conditions` list): evaluates a `RuleCondition` tree where leaf nodes are scalar comparisons and group nodes combine sub-conditions with `AND` or `OR`. Groups nest arbitrarily; for example `(is_fraud == True OR prior_claim_count >= 5) AND escalation_history_count >= 2`. Evaluated recursively by module-level `_eval_condition()`.

Shared types: `RuleOperator` (StrEnum: `>=`, `<=`, `==`, `!=`, `>`, `<`), `RuleLogic` (StrEnum: `AND`, `OR`), `RuleCondition` (dataclass with leaf fields `subject/attribute/operator/threshold` and group fields `logic/conditions`). `RuleFactory._build_conditions()` converts raw condition dicts into `RuleCondition` trees on deserialization. On match every `output` field is set to `True`.

---

### Lookup

Retrieves a fixed output payload from a keyed rule set based on one or more provided keys. The retrieved values are used by a downstream rule or workflow step.

**Example:**
```
GIVEN claim_type = "auto_collision" AND status = "denied"
LOOKUP policy_rules[claim_type][status]
SET denial_basis, next_steps, policy_section = result
```

**Current implementation:** `LookupRule` in the `rules` package. Each rule carries `match_keys` (the lookup criteria dict) and `output_values` (the payload to return on match). A `matches(context: dict)` method checks that every key in `match_keys` is present in the provided context with the expected value. `PolicyRuleRegistryClient` wraps `RuleRegistry` and uses `LookupRule.matches()` to find the applicable rule for a given context; it is used by the `csv_mcp_server` policy tools, which the claim explanation agent calls. LookupRules are also used inside the `claim_appeal` domain (for example `ca_commercial_type`), where they write `appeal.disqualified`.

**Extension path:** Support range-keyed lookups (e.g. `credit_tier IN ["A", "B"]`) and multi-output matrix lookups.

---

### Extraction

Extracts a specific value from a nested or list-structured payload and assigns it to a named field for use in downstream rules.

**Example:**
```
GIVEN rateMetrics list from backend response
EXTRACT rateMetrics[0].rateMetricId
SET selectedRateMetricId = result
```

**Current implementation:** Not yet implemented. The current data model is flat; CSV rows map directly to dataclasses with no nesting. Extraction becomes relevant when the platform integrates with real insurance backends that return complex JSON payloads.

**Extension path:** Introduce an `ExtractionRule` model with a `path` expression (e.g. JSONPath or dot notation) and a target field name. Applied as a preprocessing step before Decision or Lookup rules are evaluated.

---

## Rule Engine

The rule engine is a domain-generic framework for loading, versioning, executing, and visualizing rules.

### RuleRegistry

A singleton registry that stores all rule versions across all domains. Key capabilities:

- **Append-only versioning**: `add()` appends a new version without removing prior versions. `load()` replaces the full registry from a versioned output file.
- **Latest-version resolution**: `get_latest(rule_id, domain)` returns the highest-version rule for a given id and domain.
- **Effective rule ordering**: `get_effective(domain, group)` returns effective rules in topological execution order, root rules first and dependent rules after, with priority descending within each topological level.
- **DAG construction**: `get_dag(domain, group)` builds and caches a NetworkX `DiGraph` from the active rules. Edges are annotated with the list of field names that connect producers to consumers (the `fields` key).
- **Execution**: `execute(domain, context, trace_id, executed_by, group, entities)` walks the DAG using a ready-queue model (Kahn's variant): a rule enters the queue only when all its upstream producers have settled. When a producer is decided, `_cascade()` immediately walks its consumers -- if a consumer's required input will never arrive (all producers failed or pruned), it is cascade-pruned before being enqueued. Execution stops as soon as any terminal output appears in context; rules still in the queue at that point receive a `not_evaluated` outcome. Rules still waiting on a producer that was itself cut off never became ready and get no entry in `evaluations`. Priority orders rules only within a cascade wave -- a dependent rule always runs after all its producers regardless of its own priority, so the first triggered disqualifier depends on graph depth first and priority second. Returns a `RuleExecutionResult` carrying an `ExecutionMetadata` record (trace ID, executor name, UTC timestamp), the domain, the triggered rules in execution order, an ordered `evaluations` list recording the outcome of every rule that entered the queue or was cascade-pruned (`triggered`, `skipped_no_match`, `pruned`, `skipped_precondition`, or `not_evaluated`), `entities` (caller-supplied domain object snapshots, e.g. claim and customer), and the intermediate and terminal outputs produced. Input fields seeded by the caller (`claim.*`, `customer.*`) are excluded from the outputs.

### RuleFactory

Detects rule type from a raw dict (`detect_type()`) and instantiates the correct `Rule` subclass (`from_dict()`). Currently supports `decision` and `lookup` types.

### RuleETL

Versioning ETL pipeline that processes one domain per run:

1. Seeds the registry from the existing output file (skipped on first run).
2. Reads raw input rules. Inserts at version 0 if new; bumps the version if any business field has changed; skips unchanged rules.
3. Writes all versions (full audit history) back to the output file.

Run all configured domains with:

```bash
PYTHONPATH=src python src/etl/rule_etl.py
```

ETL domains are configured in `config/etl.yaml`.

### DAG

The registry derives a dependency graph from the `input` and `output` field declarations on each rule. An edge `A → B` exists when rule A produces an output field that rule B declares as an input. Edges are annotated with all shared field names.

The DAG is visualized at `GET /dashboard` using vis-network with a hierarchical left-to-right layout. Root rules (no upstream dependencies) appear in the first column; dependent rules appear in subsequent columns. Within each column, rules are ordered by priority descending, matching the topological execution order.

---

## Claim Appeal: DAG Execution

The claim appeal domain has sixteen rules. Twelve are topological Level 0 (roots); four are Level 1 consumers. Two roots (`ca_commercial_type`, `ca_blacklisted_shop`) are LookupRules; the rest are DecisionRules. Two use compound mode (`ca_fraud_escalation_limit`, `ca_theft_min_tenure`); the rest are simple single-condition rules.

Most rules produce `appeal.disqualified` directly. Two producer/consumer chains create the only DAG edges:

- `ca_high_value_flag` produces `appeal.high_value_flagged`; `ca_high_value_repeat_claimant` consumes it before writing `appeal.disqualified`.
- `ca_low_value_flag` produces `appeal.low_value_flagged`; `ca_low_tier_tenure_check`, `ca_low_value_escalation`, and `ca_low_value_repeat_claimant` each consume it before writing `appeal.disqualified`.

**Execution order — Level 0 roots (priority descending):** `ca_low_value_flag` (14), `ca_high_value_flag` (13), `ca_theft_min_tenure` (11), `ca_commercial_type` (10), `ca_blacklisted_shop` (9), `ca_fraud` (8), `ca_fraud_escalation_limit` (7), `ca_max_escalations` (6), `ca_status_denied` (5), `ca_max_prior_claims` (4), `ca_min_tenure` (2), `ca_min_amount` (1).

**Level 1 consumers (run after their producer fires):** `ca_high_value_repeat_claimant` (12, unlocked by `ca_high_value_flag`); `ca_low_value_escalation` (6), `ca_low_value_repeat_claimant` (5), `ca_low_tier_tenure_check` (3) (all unlocked by `ca_low_value_flag`).

`ClaimAppealAgent.check_eligibility()` builds a flat execution context from the claim and customer objects using `dataclasses.fields()` + `getattr()`, preserving Python types (`bool`, `Enum`) required for correct threshold coercion. It also captures entity snapshots (`claim.to_dict()`, `customer.to_dict()`) and passes them as `entities` to `RuleRegistry.execute()` so the audit record carries the full domain object state at the time of evaluation. The executor:

1. Enqueues all rules immediately (no upstream dependencies)
2. Gates each rule on its declared `input` fields before evaluating it
3. Writes each triggered rule's `output` to the shared context
4. Stops as soon as `appeal.disqualified` appears in context; remaining queued rules receive `not_evaluated`

Since `appeal.disqualified` is a terminal output, only the highest-priority matching rule determines the eligibility outcome and its reason.

---

## Extension Roadmap

| Phase | Capability | Status |
|---|---|---|
| Current | Single-condition Decision rules (scalar comparison) | ✅ Done |
| Current | Lookup rules with keyed match and output values | ✅ Done |
| Current | Rule registry with versioning and DAG construction | ✅ Done |
| Current | ETL pipeline with version detection and audit history | ✅ Done |
| Current | DAG dashboard with hierarchical priority-ordered visualization | ✅ Done |
| Current | Rule Executor: DAG-driven context propagation with branch pruning | ✅ Done |
| Phase 4 | Compound Decision rules (AND/OR, multi-condition, nested groups) | ✅ Done |
| Phase 2 | Tokenization service for PII in rule inputs and audit output | 🔲 Pending |
| Current | Execution audit trail: `RuleExecutionResult` persisted to JSONL per `execute()` call; trace ID threaded from orchestrator | ✅ Done |
| Current | Per-rule evaluation log: ordered `evaluations` list with outcome per rule (triggered / skipped_no_match / pruned / skipped_precondition / not_evaluated) | ✅ Done |
| Current | Entity snapshots: claim and customer objects captured at execution time and included in the audit record | ✅ Done |
| Phase 3 | Extraction rules for nested backend payloads | 🔲 Pending |
