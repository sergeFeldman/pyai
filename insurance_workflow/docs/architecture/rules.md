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

**Current implementation:** `DecisionRule` in the `rules` package. Single scalar comparison (`subject.attribute operator threshold`). Supports `>=`, `<=`, `==`, `!=`, `>`, `<` via the `_OPS` class constant. Type coercion handles bool, int, and float thresholds automatically. One condition per rule; no compound AND/OR logic yet.

**Extension path:** Support compound conditions by replacing the single `operator`/`threshold` pair with a list of conditions joined by `AND`/`OR`. The `matches()` method would evaluate the condition tree rather than a single comparison.

---

### Lookup

Retrieves a fixed output payload from a keyed rule set based on one or more provided keys. The retrieved values are used by a downstream rule or workflow step.

**Example:**
```
GIVEN claim_type = "auto_collision" AND status = "denied"
LOOKUP policy_rules[claim_type][status]
SET denial_basis, next_steps, policy_section = result
```

**Current implementation:** `LookupRule` in the `rules` package. Each rule carries `match_keys` (the lookup criteria dict) and `output_values` (the payload to return on match). A `matches(context: dict)` method checks that every key in `match_keys` is present in the provided context with the expected value. `PolicyRuleRegistryClient` wraps `RuleRegistry` and uses `LookupRule.matches()` to find the applicable rule for a given context; used by both the live claim explanation workflow and the MCP server tools.

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
- **Active rule ordering**: `get_active(domain, group)` returns active rules in topological execution order, root rules first and dependent rules after, with priority descending within each topological level.
- **DAG construction**: `get_dag(domain, group)` builds and caches a NetworkX `DiGraph` from the active rules. Edges are annotated with the list of field names that connect producers to consumers (the `fields` key).
- **Execution**: `execute(domain, context, trace_id, executed_by, group, entities)` walks the DAG in topological generation order. Within each generation, rules are sorted by priority descending. For each `DecisionRule`, all declared `input` fields must be present in the shared context dict before the rule is evaluated (precondition gate). On match, each declared `output` field is written to the context as `True`, enabling downstream rules. Returns a `RuleExecutionResult` carrying an `ExecutionMetadata` record (trace ID, executor name, UTC timestamp), the domain, the triggered rules in execution order, an ordered `evaluations` list recording the outcome of every evaluated rule (`triggered`, `skipped_precondition`, or `skipped_no_match`), `entities` (caller-supplied domain object snapshots, e.g. claim and customer), and the intermediate and terminal outputs produced. Input fields seeded by the caller (`claim.*`, `customer.*`) are excluded from the outputs.

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

The DAG is visualized at `GET /rules/dashboard` using vis-network with a hierarchical left-to-right layout. Root rules (no upstream dependencies) appear in the first column; dependent rules appear in subsequent columns. Within each column, rules are ordered by priority descending, matching the topological execution order.

---

## Claim Appeal: DAG Execution

The claim appeal domain has eight rules across two topological levels:

**Level 0 (roots):** `ca_fraud_check` (priority 8), `ca_max_escalations` (6), `ca_status_denied` (5), `ca_amount_tier` (4), `ca_min_tenure` (2), `ca_min_amount` (1)

**Level 1 (dependents):**
- `ca_fraud_escalation_limit` (priority 7): requires `appeal.fraud_flagged` produced by `ca_fraud_check`
- `ca_low_tier_tenure_check` (priority 3): requires `appeal.amount_tier` produced by `ca_amount_tier`

`ClaimAppealAgent.check_eligibility()` builds a flat execution context from the claim and customer objects using `dataclasses.fields()` + `getattr()`, preserving Python types (`bool`, `Enum`) required for correct threshold coercion. It also captures entity snapshots (`claim.to_dict()`, `customer.to_dict()`) and passes them as `entities` to `RuleRegistry.execute()` so the audit record carries the full domain object state at the time of evaluation. The executor:

1. Walks the DAG in topological generation order (Level 0 before Level 1)
2. Sorts rules within each generation by priority descending
3. Gates each `DecisionRule` on its declared `input` fields before evaluating it
4. Writes each triggered rule's `output` fields to the shared context as `True`, enabling downstream rules

This means `ca_fraud_escalation_limit` only evaluates after `ca_fraud_check` has triggered and written `appeal.fraud_flagged` to the context. A claim with `is_fraud=True` but `escalation_history_count < 2` passes the fraud check but is not disqualified; the escalation limit rule correctly gates on its precondition.

---

## Extension Roadmap

| Phase | Capability | Status |
|---|---|---|
| Current | Single-condition Decision rules (scalar comparison) | ✅ Done |
| Current | Lookup rules with keyed match and output values | ✅ Done |
| Current | Rule registry with versioning and DAG construction | ✅ Done |
| Current | ETL pipeline with version detection and audit history | ✅ Done |
| Current | DAG dashboard with hierarchical priority-ordered visualization | ✅ Done |
| Current | Rule Executor: DAG-driven context propagation with precondition gating | ✅ Done |
| Phase 2 | Compound Decision rules (AND/OR/IN, multi-condition) | 🔲 Pending |
| Phase 2 | Positive qualification rules (`appeal.qualified` output) | 🔲 Pending |
| Phase 2 | Tokenization service for PII in rule inputs and audit output | 🔲 Pending |
| Current | Execution audit trail: `RuleExecutionResult` persisted to JSONL per `execute()` call; trace ID threaded from orchestrator | ✅ Done |
| Current | Per-rule evaluation log: ordered `evaluations` list with outcome per rule (triggered / skipped_precondition / skipped_no_match) | ✅ Done |
| Current | Entity snapshots: claim and customer objects captured at execution time and included in the audit record | ✅ Done |
| Phase 3 | Extraction rules for nested backend payloads | 🔲 Pending |
