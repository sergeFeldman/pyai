---
name: dashboard
description: The rules dashboard single-page application at src/ui/templates/dashboard.html — three-tab layout (Rules, Executions, Health), vis-network DAG rendering, JS state management, API integration, and tab-switching behaviour. Use when adding a tab, changing the graph, updating the session list or execution trace, modifying the Health findings panel, or changing any JS or HTML in the dashboard.
---

# Dashboard

## Responsibility

Owns `src/ui/templates/dashboard.html` — the single-page application served at `GET /dashboard`.

Does not own the APIs the dashboard consumes (rule engine owns `/rules/*`; audit-and-trace owns `/executions`). Does not own the FastAPI route that serves the file (`src/ui/dashboard.py`).

## Structure

All JS lives inline in `dashboard.html`. There is no build step, no bundler, and no external JS files. vis-network is loaded from cdnjs. Keep everything in one file.

### Key global state

| Variable | Type | Purpose |
|---|---|---|
| `network` | vis.Network \| null | vis-network instance; null until first `loadDag` completes |
| `currentDomain` | string \| null | Domain selected in the dropdown |
| `currentData` | object \| null | Last DAG response; null if domain was set while on Health tab |
| `terminalOutputs` | Set\<string\> | Declared terminal fields for the current domain; set from DAG response |
| `activeTab` | string | `'rules'` \| `'executions'` \| `'health'` |
| `validationReport` | object \| null | Cached `GET /rules/validation-report` response |
| `executionsData` | array \| null | Cached session list for the current domain |
| `selectedTraceId` | string \| null | Currently selected execution session |
| `panelOpen` | boolean | Whether the detail panel is expanded |

### Entry points

- `init()` — called on `DOMContentLoaded`; fetches domain list and auto-selects if only one domain exists.
- `switchTab(tab)` — called by tab button clicks; updates `activeTab`, shows/hides panels, triggers data loads.
- `loadDag(domain, replaceWithNew)` — fetches DAG for `domain`; returns early if `activeTab === 'health'` (updates health panel only); otherwise fetches, sets `currentData`, `terminalOutputs`, renders graph and stats.
- `loadExecutions(domain)` — fetches `/executions?domain=`; renders session list.
- `loadValidationReport()` — fetches `/rules/validation-report`; caches in `validationReport`.

## Three tabs

### Rules tab

- Calls `loadDag(domain)` on domain change.
- Renders the DAG with vis-network (`renderGraph(data)`): hierarchical left-to-right layout, nodes ordered by priority within each level.
- Node colours: default fill for static view; overlaid with outcome colours when a session is selected (triggered=green, skipped=grey, pruned=orange, not\_evaluated=light grey). Terminal-output-producing nodes get a highlighted border — determined by checking `rule.output.some(f => terminalOutputs.has(f))`.
- Clicking a node calls `showRuleDetail(rule)` — populates the detail panel with the rule's fields and a version history button that fetches `GET /rules/history/{domain}/{rule_id}`.
- `clearOverlay()` resets all node colours to the default palette.

### Executions tab

- Calls `loadExecutions(domain)` on domain change or tab switch.
- `loadDag` also calls `loadExecutions` when `activeTab === 'executions'` after a successful fetch, so the two loads are coordinated.
- Each session row shows trace ID, timestamp, entity ID (extracted from entity snapshots), eligible/not-eligible badge, triggered rule count.
- Clicking a session calls `renderSessionDetail(session)`: populates the detail panel with triggered rules (terminal-output rules highlighted), full evaluation trace, and entity snapshots. Also calls `applyOverlay(session)` to colour DAG nodes by outcome.
- Eligible badge: `not bool(outputs.keys() & terminalOutputs)` — computed server-side in the `/executions` response; do not recompute client-side.

### Health tab

- Calls `loadValidationReport()` if not yet cached; otherwise calls `renderHealthPanel(validationReport)` and `renderHealthBadge(validationReport, currentDomain)` immediately.
- Hides `#graph-wrap` and the detail panel; hides the panel toggle.
- `renderHealthPanel(report)` filters findings to `currentDomain`; renders a findings table (severity · kind · rule · field · message), sorted errors first then warnings then by rule id. When no domain is selected, shows a prompt instead.
- `renderHealthBadge(report, domain)` updates `#health-header-badge`: red for errors, amber for warnings, green for clean. Badge is only visible while `activeTab === 'health'`; `switchTab` hides it when leaving Health.

## Tab-switching rules

When switching to `rules` or `executions` with `currentDomain` set:
- If `currentData === null` (domain was selected while on Health tab — `loadDag` returned early): call `loadDag(currentDomain)`.
- If `currentData` exists (graph was rendered before Health was visited): call `setTimeout(() => network.fit(), 0)` so vis-network recalculates dimensions on the now-visible container.

When switching to `health`:
- `loadDag` early-return: update the health panel and return; do not fetch the DAG or render the graph.

## Adding a tab

1. Add a `<button id="tab-{name}">` in the header tab group.
2. Add a panel `<div id="{name}-panel" class="hidden">`.
3. In `switchTab`: add a `classList.toggle('hidden', tab !== '{name}')` for the panel; add a branch in the `if/else` chain to load data and set up the detail panel header.
4. Add any new API call following the fetch pattern: `showOverlay('Loading…')` → `fetch` → `clearOverlay()` or `showOverlay('⚠ ...')` on error.
5. Update the badge logic if the tab has a status indicator.

## APIs consumed

| Endpoint | Called by | When |
|---|---|---|
| `GET /rules/domains` | `init()` | Page load |
| `GET /rules/dag/{domain}?replace_with_new=` | `loadDag()` | Domain change; tab switch to rules/executions when `currentData` is null |
| `GET /rules/history/{domain}/{rule_id}` | rule detail panel | Version history button click |
| `GET /executions?domain=` | `loadExecutions()` | Domain change on Executions tab; after `loadDag` when `activeTab === 'executions'` |
| `GET /rules/validation-report` | `loadValidationReport()` | First switch to Health tab |

The DAG response shape: `{nodes, edges, terminal_outputs}`. Store `terminal_outputs` as `new Set(data.terminal_outputs || [])` immediately after fetch — used by graph rendering, session detail, and eligible badge.

## Verification

Start the app (`PYTHONPATH=src uvicorn src.app.main:app --reload` from the project root), open `http://localhost:8000/dashboard`, and:

1. Select a domain — DAG renders, footer stats update.
2. Switch to Executions — session list loads, graph updates with overlay when a session is clicked.
3. Switch to Health — findings table renders for the selected domain, badge shows.
4. Switch back to Rules/Executions from Health with domain still selected — graph renders without needing to re-select the domain.
5. Reload DAG button — graph refreshes.

## Background

- [docs/architecture/dashboard.md](../../../docs/architecture/dashboard.md): full tab reference, finding types, API table.
