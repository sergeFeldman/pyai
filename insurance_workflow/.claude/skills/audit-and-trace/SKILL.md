---
name: audit-and-trace
description: Traceability and compliance persistence in src/services and the /executions API - TraceService trace IDs and WorkflowContext, trace ID threading from the orchestrator into rule execution metadata, the request-level AuditService CSV log, the RuleExecutionAuditService JSONL log of rule executions with evaluations and entity snapshots, audit isolation in tests, and PII rules for everything persisted. Use when adding or changing an audit record or its fields, adding a workflow that must be audited, changing trace ID handling, changing the executions API or the dashboard Executions tab data, or reviewing what data the platform stores.
---

# Audit and Trace

## Responsibility

Owns trace ID creation and threading, both audit logs, the `/executions` API, and the rules for what may be persisted.

Does not own rule execution itself or the shape of `RuleExecutionResult` (rule-engine), or when an orchestrator method returns (insurance-workflow).

## Trace IDs

- `TraceService.create_context(request)` returns a `WorkflowContext` with `trace_id = "trace-<uuid4>"`, `started_at` (UTC), `user_id`, and `session_id`. Create exactly one per request, first thing in the orchestrator method.
- Every `UserResponse` carries the request's `trace_id`, and the HTTP response returns it.
- Thread the same `trace_id` into every downstream record: agent calls that execute rules pass it to `RuleRegistry.execute(trace_id=...)`, which stores it in `ExecutionMetadata`.
  Status: partial (the claim explanation agent does not receive the trace ID)

## Request audit log

`AuditService` (`src/services/audit.py`) is a singleton that appends to one CSV per process, `data/out/audit_<UTC timestamp>.csv`, with fields `trace_id`, `request_type`, `agent_names`, `response`, `timestamp`.

- Every request path writes exactly one `AuditRecord`, including not-found and early-return paths, with the `request_type` and the agent keys resolved so far.
  Status: partial (the claim appeal workflow returns early without an audit record when the claim or the customer is not found)
- Audit records hold operational metadata only. `response` must not carry customer data; store the user-facing message only when it contains no customer attributes, and otherwise a non-identifying summary.
  Status: partial (the claim explanation workflow stores the full LLM answer, which can include customer context such as tenure and contact preferences)
- Never log the raw request payload.

## Rule execution audit log

`RuleExecutionAuditService` (`src/services/execution_audit.py`) is a singleton over a `JsonlDataStorage` at `data/audit/rule_executions.jsonl` (`key_field="metadata.trace_id"`).

- `log(result)` appends `RuleExecutionResult.to_dict()`: metadata, domain, triggered rules, evaluations, entity snapshots, outputs. Call it once per `execute()` from the calling agent.
- `list(domain)` returns that domain's records newest first and skips records with an empty `trace_id`.
- The file is append-only. Never rewrite or delete records outside tests.
- Entity snapshots are persisted customer and claim data.
  Status: planned (PII tokenization of snapshots before persistence, roadmap Phase 5)

## Executions API

`GET /executions?domain=` (`src/app/routes/executions.py`) returns 404 for a domain not loaded in the registry, otherwise `{domain, sessions}` with, per record: `trace_id`, `executed_timestamp`, `executed_by`, `triggered_ids`, `triggered_rules` (id, reason, condition fields, input, output), `evaluations`, `entities`, `output_keys`, and `eligible`. The dashboard Executions tab reads this shape.

- The endpoint is domain-generic. It must derive each domain's outcome from that domain's output contract in [rule-engine](../rule-engine/SKILL.md), never from a hardcoded field name.
  Status: partial (`eligible` is computed from the hardcoded `appeal.disqualified` for every domain)

## Tests

`tests/conftest.py` points `RuleExecutionAuditService` at `data/test/audit/rule_executions.jsonl` before each test. Never construct it with the default path in tests. Tests that exercise orchestrator methods must also redirect or reset `AuditService`.

## Verification

- `python -m pytest tests/agents -q` (`TestClaimAppealAgentEvaluations` checks evaluations and entity snapshots in the audit record).
- After a workflow change, call the endpoint and confirm one new CSV row with the response `trace_id` and, for appeal, one new JSONL record with the same `trace_id`.

## Examples

- Normal: `POST /claim-appeal` for an existing claim writes one CSV row (`request_type=claim_appeal`, `agent_names=claim,customer,claim_appeal`) and one JSONL record, both with the returned `trace_id`.
- Edge: `POST /claim-appeal` for unknown `claim_9` returns `Claim claim_9 was not found.` and must still write one CSV row with `agent_names=claim,customer,claim_appeal`.
- Edge: a rule execution run from a test with `trace_id=""` is persisted but never returned by `list()`.

## Planned

- Capture LangChain ReAct intermediate steps for each claim explanation request and show them in the dashboard. Status: planned (roadmap Phase 6)
- Immutable audit storage and distributed tracing are target architecture only; see [docs/architecture/security-and-compliance.md](../../../docs/architecture/security-and-compliance.md).

## Background

- [docs/implementation/dataflow.md](../../../docs/implementation/dataflow.md): where audit calls happen in each use case.
- [docs/roadmap/implementation-roadmap.md](../../../docs/roadmap/implementation-roadmap.md): Phases 5 and 6.
