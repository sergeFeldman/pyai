# Rule Architecture

## Overview

Rules encode business logic that determines outcomes — eligibility, pricing, routing, and data transformation — without requiring code changes. This document describes the rule taxonomy used across the platform, the current implementation, and the extension path as complexity grows.

---

## Rule Categories

### Decision

Evaluates one or more conditions against input facts. If the conditions match, the rule produces an output — a disqualification reason, a flag, a routing decision, or a premium adjustment.

**Example:**
```
IF state = "NY" AND claim_type IN ("auto_collision", "theft") AND amount > 5000
THEN not_eligible, reason = "High-value NY claim requires manual review."
```

**Current implementation:** `ClaimAppealRule` with a single scalar comparison (`field operator threshold`). Supports `>=`, `<=`, `==`, `!=`, `>`, `<` via the `_OPS` class constant. One condition per rule only — no compound AND/OR logic yet.

**Extension path:** Support compound conditions by replacing the single `operator`/`threshold` pair with a list of conditions joined by `AND`/`OR`. The `matches()` method would evaluate the condition tree rather than a single comparison.

---

### Lookup

Retrieves a value from a data source — a matrix, table, or range — based on one or more provided keys. The retrieved value is used as an input to a downstream rule or workflow step.

**Example:**
```
GIVEN credit_tier = "A" AND claim_type = "auto_collision"
LOOKUP rate_matrix[credit_tier][claim_type]
SET initial_rate = result
```

**Current implementation:** Not yet implemented. The MCP client layer (`get_obj_by_filter()`) provides the retrieval mechanism — Lookup rules would be a structured wrapper around it.

**Extension path:** Introduce a `LookupRule` model that specifies the data source, key fields, and output field. The rule engine resolves the lookup via the appropriate MCP client.

---

### Extraction

Extracts a specific value from a nested or list-structured payload and assigns it to a named field for use in downstream rules.

**Example:**
```
GIVEN rateMetrics list from backend response
EXTRACT rateMetrics[0].rateMetricId
SET selectedRateMetricId = result
```

**Current implementation:** Not yet implemented. The current data model is flat — CSV rows map directly to dataclasses with no nesting. Extraction becomes relevant when the platform integrates with real insurance backends that return complex JSON payloads.

**Extension path:** Introduce an `ExtractionRule` model with a `path` expression (e.g. JSONPath or dot notation) and a target field name. Applied as a preprocessing step before Decision or Lookup rules are evaluated.

---

## Current Rule Engine: All-Disqualifiers Pattern

Appeal eligibility uses a specific application of Decision rules called the **all-disqualifiers pattern**:

- Every rule is a disqualifier — a condition that, if matched, makes the claim ineligible
- Rules are evaluated in order; the first match short-circuits evaluation
- If no rule matches, the claim is eligible
- No `priority` or `eligible` field is needed — eliminating ordering dependencies

This pattern is intentionally simple: it works correctly for any rule ordering and requires no coordination between rules.

```
claim_appeal_rule_1: customer.escalation_history_count >= 3  → not eligible
claim_appeal_rule_2: claim.status != denied                  → not eligible
claim_appeal_rule_3: customer.tenure_years < 3               → not eligible
claim_appeal_rule_4: claim.amount < 1000                     → not eligible
(no match)                                                   → eligible
```

---

## Extension Roadmap

| Phase | Capability |
|---|---|
| Current | Single-condition Decision rules (scalar comparison) |
| Phase 2 | Compound Decision rules (AND/OR/IN, multi-condition) |
| Phase 2 | Lookup rules backed by MCP client data sources |
| Phase 2 | Tokenization service for PII in rule inputs and audit output |
| Phase 3 | Extraction rules for nested backend payloads |
| Phase 3 | Rule versioning and audit trail per rule evaluation |
