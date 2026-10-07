# Rules Dashboard

The dashboard is a single-page application served at `GET /dashboard`. It visualizes the rule engine state across three tabs: Rules (DAG and rule detail), Executions (audit history and execution trace), and Health (graph validation findings). All tabs share a domain selector in the header.

---

## Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Header: domain selector · Rules | Executions | Health  [badge]  │
├───────────────────────────────────┬─────────────────────────────┤
│ Main area                         │ Detail panel                │
│ (DAG graph / session list /       │ (rule detail / execution    │
│  health findings)                 │  trace / collapsed)         │
├───────────────────────────────────┴─────────────────────────────┤
│ Footer: stats · legend · hint                                    │
└─────────────────────────────────────────────────────────────────┘
```

The detail panel collapses on the Health tab. The panel toggle (chevron) is hidden on the Health tab. The Health badge in the header is only visible while on the Health tab.

---

## Rules Tab

**Purpose:** Visualize the active rule dependency graph for the selected domain and inspect individual rules.

### Domain selector

Choosing a domain calls `GET /rules/dag/{domain}`, which returns:
- `nodes`: effective rules in topological + priority order, each carrying the full rule dict
- `edges`: `{from, to, fields}` — the shared field names connecting producer to consumer
- `terminal_outputs`: declared terminal output fields for the domain (from `DomainConfig`)

### DAG graph

Rendered with vis-network in a hierarchical left-to-right layout. Root rules (no upstream dependencies) appear in the leftmost column; consumers appear to the right of their producers. Within each column, rules are ordered by priority descending, matching topological execution order.

Node colours:
- Default: domain-neutral fill
- **Triggered** (Executions tab overlay): green
- **Skipped / no match**: grey
- **Pruned**: orange
- **Not evaluated**: light grey
- **Terminal output producer**: highlighted border (field is in `terminal_outputs`)

Clicking a node opens the rule detail panel.

### Rule detail panel

Shows the selected rule's fields: id, domain, priority, effective dates, input fields, output fields, condition (operator, threshold, or compound condition tree), reason, and metadata (version, created by, timestamps).

A **version history** button expands the full version list, fetched from `GET /rules/history/{domain}/{rule_id}`. Each version shows changed fields diff against the previous version (metadata excluded).

### Footer stats

While on the Rules tab, the footer shows the rule count, edge count, and a legend for node colours (triggered, no match, pruned, precondition).

---

## Executions Tab

**Purpose:** Browse the rule execution audit history for the selected domain and inspect individual runs.

### Session list

Calls `GET /executions?domain={domain}`, returns sessions newest-first. Each row shows:
- Trace ID (truncated)
- Timestamp
- Entity ID (claim or customer, extracted from the entity snapshot)
- Eligible / not eligible badge — derived from whether any terminal output field appeared in `result.outputs`
- Triggered rule count

An optional entity filter input narrows the list by entity ID prefix.

### Execution detail panel

Clicking a session opens the detail panel with two sections.

**Triggered rules** — rules that fired in execution order, each showing rule id, reason, and output fields. Terminal-output-producing rules are highlighted.

**Full evaluation trace** — every rule that entered the ready queue or was cascade-pruned, in execution order, with its outcome:

| Outcome | Meaning |
|---|---|
| `triggered` | Condition matched; outputs written to context |
| `skipped_no_match` | Condition did not match; no outputs written |
| `pruned` | Cascade-pruned; a required upstream field will never arrive |
| `skipped_precondition` | A required external context field was absent |
| `not_evaluated` | Was queued when early exit fired; never evaluated |

Rules still waiting on a producer that was cut off by early exit never became ready and have no entry.

**Entity snapshots** — the claim, customer, and (for `policy_coverage`) policy objects captured at execution time, shown as a collapsible JSON block. These reflect the exact input state at the time of the run.

### DAG overlay

While viewing a session, the DAG graph in the main area updates to colour each node by its outcome for that run, giving a visual trace of which branch of the graph fired and which was pruned.

---

## Health Tab

**Purpose:** Surface graph validation findings for the selected domain — phantom consumers (structural errors) and orphan producers (warnings).

### Findings panel

Calls `GET /rules/validation-report` once at page load (cached for the session). The report is filtered to the selected domain. Selecting a domain is required; without one, the panel shows a prompt.

The panel shows:
- Error count and warning count for the domain
- Validated-at timestamp
- Findings table: severity · kind · rule id · field · message

Findings are sorted errors first, then warnings, then alphabetically by rule id.

### Finding types

| Kind | Severity | Meaning |
|---|---|---|
| `phantom_consumer` | ERROR | Rule declares an `input` field no active rule in the domain produces. Likely a typo or a missing rule. The rule is excluded from the DAG and will never execute. |
| `orphan_producer` | WARNING | Rule produces a field no other rule consumes and it is not declared as a terminal output in `DomainConfig`. Likely a typo in an output field name or a stale rule. |

Terminal output fields (e.g. `appeal.disqualified`) are expected orphan producers — consumed by the agent, not by another rule — and are suppressed from warnings when declared in `data/in/domain_config.json`.

### Header badge

A small badge next to the Health tab label shows the error/warning count for the selected domain. It is only visible while on the Health tab; switching to another tab hides it. Colour: red for any errors, amber for warnings only, green for a clean domain.

---

## APIs

| Endpoint | Tab | Purpose |
|---|---|---|
| `GET /rules/domains` | All | Populate domain selector on load |
| `GET /rules/dag/{domain}?replace_with_new=` | Rules, Executions | DAG nodes, edges, terminal outputs |
| `GET /rules/history/{domain}/{rule_id}` | Rules | All versions of a rule with field-level diff |
| `GET /executions?domain={domain}` | Executions | Execution audit history |
| `GET /rules/validation-report` | Health | Graph validation findings |

---

## Tab-switching behaviour

- Switching from Health to Rules or Executions with a domain already selected re-renders the DAG (`loadDag` is called if the graph was never loaded, or `network.fit()` is called if it was).
- Switching to Executions with a domain selected reloads the session list (`loadExecutions`).
- Switching to Health renders the cached validation report filtered to the current domain; if the report has not loaded yet, it is fetched.
- The detail panel collapses on Health and reopens (if `panelOpen`) on Rules.
