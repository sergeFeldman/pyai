# Execution Semantics

Read this before changing `RuleRegistry.execute()` or when explaining why a rule got a given outcome.

## Order

`execute()` uses a ready-queue model (Kahn's variant):

1. `get_dag(domain, group)` returns the cached graph. Three structures are precomputed at build time and stored on the graph: `producers` (field to producer rule IDs), `consumers` (rule ID to consumer rule IDs), and `terminal_outputs` (fields produced but consumed by no rule).
2. `undecided[rule_id]` is the per-rule set of upstream producer IDs not yet decided. Rules with an empty set at the start form the initial queue, sorted by priority descending then id ascending.
3. When a rule is decided, `_cascade()` immediately walks its consumers. Each consumer discards the decided rule from its `undecided` set. When the set empties, the consumer is either cascade-pruned (if prunable) or enqueued. Cascade-pruned rules record a `pruned` outcome immediately and never enter the queue.
4. Within each cascade wave, newly ready rules are enqueued in priority descending, then id ascending order. The overall order is fully deterministic.
5. Execution stops as soon as any terminal output appears in context.

## Per-rule decision

Rules that are cascade-pruned by `_cascade()` record `pruned` immediately and never enter the queue. For rules that do enter the queue, in order:

1. **Prune check.** `_is_prunable(rule)`: for each `field_name` in `rule.input`, if `field_name` is not in context, has at least one producer in the DAG, and every producer is already failed or pruned, the rule is `pruned`. It is never evaluated and joins the pruned set; `_cascade()` runs.
2. **Precondition check.** If `rule.ready(context)` is false, the outcome is `skipped_precondition` and the rule joins the failed set. This happens when an external field the caller should have seeded is missing; `_cascade()` runs.
3. **Evaluate.** On match, outputs are merged into context, the rule is appended to `triggered`, and the outcome is `triggered`. Otherwise the outcome is `skipped_no_match` and the rule joins the failed set. `_cascade()` runs in both cases.

Invariants:

- Every rule that is dequeued or cascade-pruned gets exactly one evaluation entry. Rules still in the queue when early exit fires get no entry.
- A field already present in context never causes pruning, even if all its producers failed.
- A field with several producers keeps its consumers alive until every producer has failed or been pruned (the `undecided` set tracks this).
- `_cascade()` is iterative, not recursive -- avoids Python stack depth issues on deep graphs.
- Context only grows monotonically; a field absent when cascade runs cannot appear later to save a cascade-pruned rule.
- `outputs` contains only keys that were not in context when `execute()` started.
- `context` is mutated in place. Pass a fresh dict per execution.

## Outcomes

| Outcome | Preconditions met | Evaluated | Meaning |
|---|---|---|---|
| `triggered` | yes | yes, matched | Outputs written to context |
| `skipped_no_match` | yes | yes, no match | Rule ran and did not fire |
| `pruned` | not checked | no | All producers of a required internal input failed or were pruned |
| `skipped_precondition` | no | no | A required field is missing and no failed producer explains it |

## Worked example (claim_appeal)

Synthetic claim: `amount=5000.0`, `status=DENIED`, `is_fraud=True`, `claim_type="auto_collision"`, `repair_shop="shop_1"`; customer `escalation_history_count=3`, `prior_claim_count=0`, `tenure_years=6`.

- **Initial queue (no upstream producers):** `ca_commercial_type` no match, `ca_blacklisted_shop` no match, `ca_fraud_check` triggers (`appeal.risk_flagged`), `ca_repeat_claimant_flag` no match, `ca_max_escalations` no match (3 < 5), `ca_status_denied` no match, `ca_amount_tier` no match, `ca_min_tenure` no match, `ca_min_amount` no match.
- **After initial wave settles via cascade:** `ca_fraud_escalation_limit` -- last producer decided, `appeal.risk_flagged` in context, enqueued. `ca_low_tier_tenure_check` -- last producer (`ca_amount_tier`) decided as failed; cascade-pruned immediately, never enqueued.
- **`ca_fraud_escalation_limit` dequeued:** 3 >= 2, triggers `appeal.disqualified`. Early exit fires.
- Result: not eligible, reason "High-risk claim with multiple prior escalation attempts."

Change `is_fraud` to `False`: both `ca_fraud_check` and `ca_repeat_claimant_flag` fail. When the last producer of `appeal.risk_flagged` is decided, `_cascade()` finds `ca_fraud_escalation_limit` prunable and cascade-prunes it immediately. The claim is then eligible.
