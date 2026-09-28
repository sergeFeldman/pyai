# Routing Contract

Read this when setting up `CLAUDE.md`, writing or checking a route, or choosing frontmatter fields.

## CLAUDE.md block

`CLAUDE.md` loads at the start of every session, so it holds both the top-level routes and the rules for following routes, which then apply during ordinary work and not only while skill-authoring is loaded. Keep this block in the project's `CLAUDE.md` and adjust the route list. The second route below is an example; replace it with real broad skills.

```markdown
## Project skills

- Use [skill-authoring](.claude/skills/skill-authoring/SKILL.md) to create, revise, reorganize, or audit project skills, or to turn docs and session lessons into skills.
- Use [claims-platform](.claude/skills/claims-platform/SKILL.md) to build, change, or review any part of the claims workflow platform.

## Following skill routes

- When a task matches a `Use [skill](path) to ...` line, read that skill and apply it before starting the work.
- Inside a skill, keep following narrower routes while they still match the task. Stop when they no longer do.
- Rules picked up along a route stay in force. A narrower skill adds detail; it never relaxes a broader rule unless it names that rule and says so explicitly.
- Use skill-authoring whenever a skill is created, changed, moved, or removed.
- When a code change alters behavior, a pattern, or a rule that a skill describes, update that skill in the same change.
```

Route only broad concerns from `CLAUDE.md`. Everything narrower is routed from the skill that owns the broader concern.

## Skill-to-skill routes

Place routes in a section near the end of the routing skill. Paths are relative to the file that contains the route.

```markdown
## Specialized skills

- Use [claims-intake](../claims-intake/SKILL.md) to validate and register a new claim submission.
- Use [claim-appeal-rules](../claim-appeal-rules/SKILL.md) to add, change, or debug appeal eligibility rules and their dependency graph.
```

## Route grammar

The general routing rules are in `SKILL.md`. Mechanically, every route must:

1. Start with the word `Use`.
2. Use the target skill's folder name as the link text.
3. Link to the target's `SKILL.md` with a relative path.
4. Continue with `to` and a concrete action or trigger.
5. Sit on its own list line.

Prefer one router per skill. If two files route to the same skill, check whether ownership is really shared or whether one of the routes belongs elsewhere.

## Frontmatter

In Claude Code, every frontmatter field is optional, but `description` is strongly recommended because Claude uses it to decide when to load the skill. The folder name, not `name`, sets the command.

- `description`: what the skill does and when to use it. Put the main use case first. The combined text of `description` and `when_to_use` is cut off at 1,536 characters in the skill listing.
- `when_to_use`: optional extra trigger phrases, counted in the same limit.
- `allowed-tools`: tools Claude may use without asking during the turn that invokes the skill.
- `disable-model-invocation: true`: only the user can run the skill. Use for actions with side effects, such as deploying or sending messages.
- `user-invocable: false`: only Claude loads the skill. Use for background knowledge that is not a meaningful command.
- `paths`: glob patterns that limit automatic loading to matching files.
- `context: fork`: run the skill in an isolated subagent. Only suitable for skills with an explicit task, not for reference guidance.

Claude Code silently ignores field names it does not recognize, so a misspelled field fails without any error. The audit script warns about unknown fields.

If a skill might also be uploaded to claude.ai or used through the Skills API, restrict its frontmatter to the portable set: `name`, `description`, `license`, `compatibility`, `metadata`, `allowed-tools`. Other fields cause those uploads to fail.

Default for this project: `name` and `description` only, adding other fields when there is a specific reason.

## Status markers

Place a status note directly after any rule whose implementation state is not obvious:

```markdown
- Every appeal decision records the rule that triggered it and the reason.
  Status: implemented
- Escalate appeals over the review threshold to a human adjuster.
  Status: planned (see docs/roadmap/implementation-roadmap.md)
```

Use `implemented`, `partial (what is missing)`, or `planned`. Update the marker when the code changes.

## What the audit checks

`scripts/audit_skills.py` checks the mechanical rules:

- `CLAUDE.md` exists and every skill is reachable from it through routes
- every route resolves to a `SKILL.md` that is a direct child of `.claude/skills/`
- route link text matches the target folder name, and no skill routes to itself
- no routing cycles
- folder names are lowercase and hyphenated, and any `name` field matches the folder
- frontmatter starts on the first line, parses, and includes a `description`
- description plus `when_to_use` fits within 1,536 characters
- unknown frontmatter fields, very short route actions, skills over 500 lines, multiple routers, and links back to a router (warnings)
- relative links in `CLAUDE.md`, `SKILL.md` files, and skill reference files point at files that exist
- skill-creator workspaces or other stray folders inside `.claude/skills/` (warnings)

It cannot judge whether an action is specific enough, whether rules are duplicated, whether a skill matches the docs and code, or whether status markers are accurate. Review those by reading.
