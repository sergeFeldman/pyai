# Execution Semantics

Read this before changing `RuleRegistry.execute_async()` or when explaining why a rule got a given outcome.

## Order

`execute_async()` uses a wave-based ready-queue model (Kahn's variant):

1. `get_dag(domain, group)` returns the cached graph. Four structures are precomputed at build time and stored on the graph: `producers` (field to producer rule IDs), `consumers` (rule ID to consumer rule IDs), `terminal_outputs` (fields produced but consumed by no rule), and `pending_producers` (per-rule frozen set of upstream producer IDs).
2. `pending_producers[rule_id]` is a mutable per-call copy of the upstream producer set. Rules with an empty set at the start form the initial queue, sorted by priority descending then id ascending.
3. The ready queue is drained entirely into a **wave**. All rules in the wave are evaluated concurrently via `asyncio.gather`. No rule writes to context during evaluation -- each `_eval_rule` coroutine returns `(rule_id, outcome, outputs)` without mutating shared state.
4. After `gather` completes, outputs are applied to context in priority-descending order, then `_cascade()` runs for every decided rule. Cascade walks each rule's consumers: when a consumer's `pending_producers` set empties, it is either cascade-pruned (if `_is_prunable`) or enqueued for the next wave.
5. After cascade, early exit is checked: if any terminal output is now in context, execution stops. Every rule still in the queue at that point receives a `not_evaluated` entry.
6. Otherwise the queue is drained into the next wave and the cycle repeats.

Key consequence: all rules whose producers have settled form a single wave and run concurrently. A rule in wave N+1 always sees the complete merged outputs of wave N before it evaluates.

## Per-rule decision

Rules cascade-pruned by `_cascade()` record `pruned` immediately and never enter the queue. For rules that do enter the queue and are evaluated by `_eval_rule`:

1. **Prune check (defensive).** `_is_prunable(rule)`: for each `field_name` in `rule.input`, if `field_name` is not in context, has at least one producer in the DAG, and every producer's outcome in `decided` is one of `skipped_no_match`, `skipped_precondition`, or `pruned`, the rule is `pruned`. `_cascade()` detects prunable consumers before they are enqueued -- this check is a defensive fallback for edge cases only.
2. **Precondition check.** If `rule.ready(context)` is false, the outcome is `skipped_precondition` and `decided[rule_id]` is set accordingly. This happens when an external field the caller should have seeded is missing; since context is sealed at call time, a missing external field never appears mid-execution.
3. **Evaluate.** On match, outputs are returned (not yet written). Otherwise the outcome is `skipped_no_match`. In both cases `decided[rule_id]` is set and `_cascade()` runs after the full wave.

Context writes happen in the outer loop after `gather`, sorted by priority descending. If two rules in the same wave produce the same field the lower-priority rule wins (last write). Same-field conflicts are a DAG design error caught by the graph linter in practice.

## Invariants

- A rule appears in `evaluations` if and only if it entered the ready queue or was cascade-pruned. Rules still waiting on a producer cut off by early exit never became ready and get no entry.
- Of rules that do appear, each gets exactly one evaluation entry.
- All rules in a wave read from the same pre-wave context snapshot; no rule in a wave sees another wave-mate's outputs.
- A field already present in context never causes pruning, even if all its producers failed.
- A field with several producers keeps its consumers alive until every producer has failed or been pruned.
- `_cascade()` is iterative, not recursive -- avoids Python stack depth issues on deep graphs.
- Context only grows monotonically; a field absent when cascade runs cannot appear later to save a cascade-pruned rule.
- `outputs` contains only keys that were not in context when `execute_async()` started.
- `context` is mutated in place. Pass a fresh dict per execution.
- `not_evaluated` applies only to rules queued for the next wave when early exit fires; same-wave rules that co-fired with the terminal rule are fully evaluated, not skipped.

## Outcomes

| Outcome | Preconditions met | Evaluated | Meaning |
|---|---|---|---|
| `triggered` | yes | yes, matched | Outputs written to context |
| `skipped_no_match` | yes | yes, no match | Rule ran and did not fire |
| `pruned` | not checked | no | All producers of a required internal input failed or were pruned |
| `skipped_precondition` | no | no | A required external field is missing from context |
| `not_evaluated` | yes (was queued) | no | Early exit fired after the previous wave; this rule was queued for the next wave but never started |

Rules that never entered the queue because their producer was cut off by early exit get no entry at all -- they were never ready.

## Worked example (claim_appeal)

Synthetic claim: `amount=5000.0`, `status=DENIED`, `is_fraud=True`, `claim_type="auto_collision"`, `repair_shop="shop_1"`; customer `escalation_history_count=3`, `prior_claim_count=0`, `tenure_years=6`.

- **Wave 1 (all root rules, no upstream producers):** `ca_commercial_type` no match, `ca_blacklisted_shop` no match, `ca_fraud_check` triggers (`appeal.risk_flagged`), `ca_repeat_claimant_flag` no match, `ca_max_escalations` no match (3 < 5), `ca_status_denied` no match, `ca_amount_tier` no match (amount not < 500), `ca_min_tenure` no match, `ca_min_amount` no match. All run concurrently; outputs merged after gather.
- **After wave 1 cascade:** `ca_fraud_escalation_limit` -- both producers (`ca_fraud_check` triggered, `ca_repeat_claimant_flag` decided) settled, `appeal.risk_flagged` in context, enqueued for wave 2. `ca_low_tier_tenure_check` -- its producer `ca_amount_tier` failed; cascade-pruned, records `pruned`, never enqueued.
- **Wave 2:** `ca_fraud_escalation_limit` alone. `escalation_history_count=3 >= 2`, triggers `appeal.disqualified`. Early exit fires after wave 2. Queue is empty -- no `not_evaluated` entries.
- **Result:** not eligible, reason from `ca_fraud_escalation_limit`.

Change `is_fraud` to `False`: `ca_fraud_check` fails in wave 1. When the last producer of `appeal.risk_flagged` is decided, `_cascade()` finds `ca_fraud_escalation_limit` prunable (all producers failed) and cascade-prunes it immediately. No terminal output is ever produced. Claim is eligible.

**Demonstrating `not_evaluated`:** Add a second root rule `ca_unlock` that produces `d.unlock` (not a terminal output), and a dependent rule `ca_next` that consumes `d.unlock`. If `ca_fraud_check` (wave 1) produces a terminal output AND `ca_unlock` (also wave 1) fires and cascades `ca_next` into the queue, early exit fires after wave 1 and `ca_next` receives `not_evaluated` -- it was queued for wave 2 but the engine stopped.
