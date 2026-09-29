---
name: skill-authoring
description: Create, revise, split, merge, rename, remove, or audit this project's Claude Code skills in .claude/skills/, and keep them wired into CLAUDE.md through "Use [skill](path) to ..." routes. Use whenever the user wants to capture a business rule, design pattern, procedure, or lesson from a working session as a skill, turn docs/ content into skills, reorganize existing skills, update skills after code changes, or check that skills are reachable, current, and consistent with the code, even if they never say the word "skill".
allowed-tools: Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/audit_skills.py *)
---

# Skill Authoring

This skill governs how the project's skills are written, organized, and kept current. It does no domain work itself. It produces and maintains the skills Claude Code loads when doing domain work.

## Relationship to skill-creator

If the `skill-creator` plugin is installed, use it for the general mechanics: interviewing the user, drafting a skill, writing test prompts, running evals, and tuning the description so the skill triggers reliably. This skill adds the project rules below on top of that. Keep skill-creator's `<skill-name>-workspace/` folders outside `.claude/skills/` and out of version control.

If skill-creator is not installed, these rules are enough on their own. Suggest installing it (`/plugin install skill-creator@claude-plugins-official`) only when the user wants measured evaluation of a skill.

## CLAUDE.md or a skill

`CLAUDE.md` loads in full at the start of every session. Keep in it only what nearly every task needs: a short project summary, the hard rules that must never be broken, the skill routes, and the route-following rules. Everything else, including patterns, procedures, and domain rules, goes into skills, which load only when relevant.

## Layout

Keep every skill as a direct child of this project's `.claude/skills/` folder, one folder per skill. The project's `CLAUDE.md` sits at the project root or inside `.claude/`. Never place a skill folder inside another skill folder. The audit checks only that folder.

The folder name becomes the skill's command (`.claude/skills/claims-intake/` gives `/claims-intake`), so use lowercase words joined by hyphens. If the frontmatter has a `name`, keep it identical to the folder name.

A skill folder holds `SKILL.md` and, only when needed:

- `references/` for detail Claude reads on demand
- `scripts/` for code Claude runs rather than reads
- `assets/` for templates or files used in output

## Routing

Claude always sees every skill's description and loads a skill when a task matches it. Routes add a second, explicit path from general to specific, so Claude reaches the right narrow skill and picks up the broad rules on the way. Express the hierarchy only with forward routes in this form:

```markdown
Use [skill-name](relative/path/to/SKILL.md) to <concrete task or trigger>.
```

- Put routes to broad skills in `CLAUDE.md`. Put routes to narrower skills in the broad skill that owns the concern.
- Write the action so Claude can decide from that one line whether to load the skill.
- A leaf skill has no routing section.
- Do not add parent pointers, ancestry metadata, or links back to the router. Forward routes are the only hierarchy.
- A link that does not follow the route form is a supporting link, not a route.
- Route examples inside fenced code blocks or inline code are illustrations, not routes.

Read [the routing contract](references/routing-contract.md) for the `CLAUDE.md` template, the exact route grammar, and frontmatter details.

A skill can also declare `paths` in frontmatter (for example `src/rules/**`) so Claude loads it while working on matching files. `paths` limits automatic loading to those files, so avoid it on skills that should also trigger from a plain request.

## Sources of truth

Three kinds of material describe the project, and each has a different job:

- Source code and tests define what the system does today.
- Skills tell Claude which rules to follow and what behavior is required, written as instructions.
- `docs/` explains the system to people: architecture, rationale, targets, roadmap.

Do not copy doc content into skills. State the rule in the skill and link to the doc for background.

When a doc, a skill, and the code disagree, do not quietly pick one. Report the conflict to the user and record the agreed outcome in the owning skill.

When a code change alters behavior, a pattern, or a rule that a skill describes, update that skill in the same change. A stale skill is worse than none, because Claude follows it confidently.

When a skill's behavioral description changes, check `docs/` for any files that mirror it and update them in the same change. Skills and docs describe the same system from different angles; letting them diverge defeats both.

Label any requirement the code does not yet meet with a short status note next to the rule: `Status: implemented`, `Status: partial (what is missing)`, or `Status: planned`. A skill must never describe planned behavior as if it exists. Update `Status:` labels as work completes, not at the end -- a label that still says `planned` while the feature is live is equivalent to describing planned behavior as implemented.

When a design invariant is discovered during implementation (a constraint, a correctness property, a reason a simpler approach was rejected), record it in the owning skill's reference file before the change is considered done. Invariants captured retroactively are invariants the next implementation phase starts without.

## What a skill should contain

Write for Claude, in the imperative, and cover whichever of these apply:

1. Responsibility: what the skill owns and what it explicitly does not.
2. Inputs and outputs, including data shapes and where they come from.
3. States, transitions, and ordering: for example a claim moving from submitted to approved or denied, or which rules must run before others.
4. Rules and invariants that must always hold, with the reason when it is not obvious.
5. Design patterns to follow: which base classes, factories, or registries to use and what a new component must declare.
6. Concurrency and shared state: singletons, thread safety, caching, and what must not be mutated.
7. Failure handling: errors, retries, escalation to a human, and what must never fail silently.
8. User-visible behavior: API responses and any text shown to customers, which must be accurate for every path that produces it.
9. Verification: the tests or commands that prove a change is correct, and test conventions to follow.
10. Examples: a normal case plus the edge cases that matter.

Keep it short. Once loaded, a skill stays in context for the rest of the session, so every line has an ongoing cost. Keep `SKILL.md` under about 500 lines and move long detail into `references/` with a note saying when to read it.

Describe behavior and ownership rather than listing code. Link to source files or tests when that helps Claude find them.

Put shared vocabulary and rules in the broadest skill that needs them, and specialized behavior in the narrowest skill that owns it. State each rule once.

Use synthetic data in examples. Never put credentials, secrets, or real customer data in a skill.

## Bootstrapping from docs/

When asked to create an initial set of skills from existing documentation and code, read [the bootstrapping guide](references/bootstrapping.md) first.

## Updating skills after a working session

When asked to capture what a session taught (in Claude Code, or from a conversation the user pastes):

1. List explicit requests, user corrections and stated preferences, bugs found with their reproduction steps, and code changes made. Also check Claude Code's auto memory notes for learnings that belong in a skill instead.
2. Classify each item as a durable rule, a reusable procedure, evidence, a known gap, or transient detail.
3. Drop transient detail: session IDs, timestamps, temporary paths, credentials, one-off output.
4. Map each durable item to the narrowest skill that owns it, and check it against the current code and tests before editing. A bug that was fixed usually becomes an invariant plus an edge-case example; a bug still open becomes a rule marked `Status: partial`.
5. Record intent as behavior ("appeal decisions include the triggering rule and its reason"), not as a retelling of the conversation.

## Workflow

1. Read `CLAUDE.md` and inventory `.claude/skills/*/SKILL.md`. For implementation tasks, also read the owning skill's `references/` files before starting work -- naming conventions, invariants, and design patterns are there, not in `SKILL.md`. If a reference file is stale (describes behavior the code no longer implements), update it first, then make the code change. Follow existing routes relevant to the task.
2. Decide ownership. Extend an existing skill if it already owns the concern. Create a new skill only for a cohesive responsibility no skill covers and that is worth loading on its own.
3. Pick the router: the narrowest existing skill that should hand work to the new one, or `CLAUDE.md` if the concern is broad.
4. Write or edit the skill using the content rules above. Write the description last. Make it specific, and slightly pushy about when to use it, because the description is what Claude matches tasks against.
5. Add the route to the router. Add no link back. When renaming or removing a skill, update or remove every route and link to it.
6. Run the audit and fix every error:

   ```bash
   python3 ${CLAUDE_SKILL_DIR}/scripts/audit_skills.py --tree
   ```

7. Report to the user: skills created, changed, or removed, the routing chain from `CLAUDE.md` with repository-relative paths, any conflicts found, and anything marked partial or planned.

## Done when

- The audit exits with no errors, and every warning is fixed or explained to the user.
- Every changed skill states its responsibility, required behavior, and how to verify it, with at least one normal and one edge-case example where behavior is involved.
- No rule is duplicated between a skill and its router, or between a skill and `CLAUDE.md`.
- Nothing planned is described as implemented.
- `CLAUDE.md` still contains the route-following rules from the routing contract.
- The user has seen the routing chain and any open conflicts.
- Any `docs/` files whose content mirrors the changed skill's behavioral descriptions have been checked and updated.
- The owning skill's `references/` files reflect the current behavior, including any invariants discovered during implementation. No reference file still describes the superseded approach.
