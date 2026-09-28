# Insurance Workflow Platform

Agentic claims-processing POC: claim status, claim explanation, and claim appeal
eligibility, built on FastAPI with an agent factory and a deterministic rule engine.
Architecture and business documentation live in docs/.

## Project skills

- Use [skill-authoring](.claude/skills/skill-authoring/SKILL.md) to create, revise, reorganize, or audit project skills, or to turn docs and session lessons into skills.

## Following skill routes

- When a task matches a `Use [skill](path) to ...` line, read that skill and apply it before starting the work.
- Inside a skill, keep following narrower routes while they still match the task. Stop when they no longer do.
- Rules picked up along a route stay in force. A narrower skill adds detail; it never relaxes a broader rule unless it names that rule and says so explicitly.
- Use skill-authoring whenever a skill is created, changed, moved, or removed.
- When a code change alters behavior, a pattern, or a rule that a skill describes, update that skill in the same change.