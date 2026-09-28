# Execution Semantics

Read this before changing `RuleRegistry.execute()` or when explaining why a rule got a given outcome.

## Order

1. `get_dag(domain, group)` returns the cached graph of effective rules.
2. Rules are visited generation by generation (`nx.topological_generations`), so every producer is visited before its consumers.
3. Within a generation, rules are sorted by priority descending, then id ascending. The order is fully deterministic.

## Per-rule decision

For each rule, in order:

1. **Prune check.** For each field `f` in `rule.input`: if `f` is not in context, `f` has at least one producer in this DAG, and every producer of `f` is already failed or pruned, the rule is `pruned`. It is never evaluated and joins the pruned set.
2. **Precondition check.** If `rule.ready(context)` is false, the outcome is `skipped_precondition` and the rule joins the failed set. This happens when an external field the caller should have seeded is missing.
3. **Evaluate.** On match, the outputs are merged into context, the rule is appended to `triggered`, and the outcome is `triggered`. Otherwise the outcome is `skipped_no_match` and the rule joins the failed set.

Invariants:

- Every effective rule in the DAG gets exactly one evaluation entry, in visit order.
- A field already present in context never causes pruning, even if all its producers failed.
- A field with several producers keeps its consumers alive until every producer has failed or been pruned.
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

- Generation 0: `ca_commercial_type` no match, `ca_blacklisted_shop` no match, `ca_fraud_check` triggers (`appeal.risk_flagged`), `ca_repeat_claimant_flag` no match, `ca_max_escalations` no match (3 < 5), `ca_status_denied` no match, `ca_amount_tier` no match, `ca_min_tenure` no match, `ca_min_amount` no match.
- Generation 1: `ca_fraud_escalation_limit` has `appeal.risk_flagged` in context and 3 >= 2, so it triggers `appeal.disqualified`. `ca_low_tier_tenure_check` needs `appeal.amount_tier`, whose only producer `ca_amount_tier` failed, so it is `pruned`.
- Result: not eligible, reason "High-risk claim with multiple prior escalation attempts."

Change `is_fraud` to `False` and `ca_fraud_escalation_limit` is also pruned, because both producers of `appeal.risk_flagged` failed. The claim is then eligible.
