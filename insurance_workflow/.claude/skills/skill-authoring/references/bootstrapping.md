# Bootstrapping Skills from Docs and Code

Read this when creating the project's first set of skills, or when doing a large reorganization from existing documentation.

## 1. Gather

Read `CLAUDE.md`, any existing skills, and every file in `docs/`. Then read enough of the source and tests to see the real structure: base classes and what inherits from them, factories and registries, singletons and shared services, configuration models, and the test layout. If the project depends on shared code outside its own folder, read the parts it uses.

## 2. Sort what you found

Put each doc section and each code-level finding into one bucket:

- **rule**: a constraint that must hold. Goes into the owning skill.
- **pattern**: how a kind of component is built (for example, how a new agent or client is declared and registered). Goes into the owning skill as instructions.
- **procedure**: steps for a recurring task (running an ETL, adding a rule). Becomes steps in a skill.
- **background**: rationale, history, target architecture. Stays in the doc; the skill links to it.
- **planned**: described in docs but not in code. Goes into the skill marked `Status: planned`.
- **stale or contradicted by the code**: flag to the user. Do not encode it.

Also note code conventions that are consistent across the codebase (naming, import aliases, config class naming, test structure). They belong in the broadest skill that covers the code they apply to.

## 3. Propose a skill map and wait

Before writing any skill, show the user:

- the broad skills routed from `CLAUDE.md`, and the narrow skills under each
- one line on what each skill owns and what it excludes
- the exact route text for each
- which docs each skill links to
- any conflicts between docs and code, and anything that would be marked partial or planned
- proposed additions to `CLAUDE.md` (project summary and hard rules only)

Wait for the user's approval or changes. Do not create files until then.

## 4. Build in order

Create broad skills first, then narrow ones, adding each route as you go. Run the audit after each broad skill and its children, not only at the end.

## 5. Keep the first set small

Aim for skills that match tasks the user actually repeats. Three to eight skills is a reasonable first set for a single project. Coverage can grow later through session updates; a large first set that nobody maintains goes stale.
