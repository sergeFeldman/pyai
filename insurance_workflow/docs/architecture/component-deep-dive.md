# Component Deep Dive

> **Note:** This document describes the target enterprise architecture. The current POC implements a subset of this vision. See [Implementation Dataflow](../implementation/dataflow.md) for what is currently built.

## Orchestrator Agent

### Purpose

The Orchestrator Agent is the entry point for all requests. It classifies user intent, determines domain ownership, applies policy and risk checks, assembles context, invokes the right workflow pattern, and synthesizes the final user-facing response.

### Design Decisions

- Keep the orchestrator thin and policy-driven
- Use structured outputs for intent, entities, confidence, risk, and next action
- Delegate all business execution to domain agents or workflow steps
- Prevent direct enterprise-system access from the orchestrator

### Implementation Considerations

- Deploy as a stateless service on EKS or ECS
- Persist workflow state in Temporal
- Store short-lived conversation context in Redis and durable session state in Aurora or DynamoDB
- Use lightweight routing models whenever possible

### Key Trade-Offs

- A thin orchestrator improves maintainability and latency
- Too much logic in the orchestrator leads to prompt sprawl and bottlenecks
- Too little logic causes over-delegation and unnecessary model cost

## Domain Agents

### Purpose

Domain agents provide specialized reasoning and policy-aware task execution for Claims, Service, Sales, and later HR or IT.

### Design Decisions

- One primary agent per major business domain
- Domain-specific prompts, tools, constraints, and escalation rules
- Structured outputs with answer, confidence, sources, and recommended next steps

### Implementation Considerations

- Claims Agent handles claim status, filing, notes, document lookups, payout explanations, and appeal preparation
- Service Agent handles billing, policy servicing, ticket workflows, and account maintenance
- Sales Agent handles quotes, plan comparisons, coverage explanation, and approved upsell logic
- Maintain domain-specific retrieval indexes to reduce token usage and improve grounding

### Key Trade-Offs

- Specialization improves accuracy and compliance
- Additional agents increase platform complexity and ownership overhead
- Avoid over-fragmenting domains until traffic and use-case maturity justify it

## Sub-Agents

### Purpose

Sub-agents are reusable specialists that support multiple domains without owning the primary end-user interaction.

### Recommended Sub-Agents

- Knowledge Agent for retrieval, grounding, summarization, and citation generation
- Data Query Agent for governed SQL generation and result interpretation
- Document Processing Agent for OCR, extraction, form validation, and redaction
- Escalation Agent for human handoff packet generation
- Optional later agents for fraud triage and QA or compliance review

### Implementation Considerations

- Favor deterministic or heavily constrained workflows
- Use typed input and output contracts
- Restrict write operations and require policy checks for regulated actions

### Key Trade-Offs

- Reuse lowers duplication and improves consistency
- Excessive sub-agent decomposition can increase latency and cognitive overhead

## Rule Engine

### Purpose

The Rule Engine encodes business logic (eligibility, disqualification, routing, pricing, and data transformation) as versioned, auditable rules organized in a dependency graph. Rules are evaluated deterministically without code changes, giving business and compliance teams a governed path to modify logic independently of agent or workflow code.

### Design Decisions

- **Rule taxonomy**: Decision rules (single-condition scalar comparison), Lookup rules (keyed match returning a fixed output payload), and Extraction rules (path-based extraction from nested payloads, planned)
- **DAG-driven execution**: rules declare `input` and `output` field names; the registry derives a dependency graph automatically; an edge A → B exists when A produces a field that B consumes
- **DAG branch pruning**: `execute()` walks topological generations in order; before evaluating each rule it checks whether all producers of its required internal inputs have failed or been pruned. If so, the rule is pruned and never evaluated. This makes the traversal true DAG execution rather than a linear sweep. External inputs (fields seeded by the caller) are never subject to pruning
- **Versioned registry**: the registry is append-only; every rule version is preserved for full audit history; the ETL pipeline detects field-level changes and bumps versions automatically
- **Execution provenance**: `RuleExecutionResult` carries an `ExecutionMetadata` record (trace ID, executor name, UTC timestamp), the triggered rules in execution order, an ordered evaluation log (outcome per rule: `triggered` / `skipped_no_match` / `pruned` / `skipped_precondition`), entity snapshots, and the outputs produced; mirrors `EntityMetadata` on persistent entities
- **Input context capture**: `execute()` operates on a flat context dict and has no access to the original domain objects. The caller (agent layer) is responsible for passing entity snapshots (the full claim and customer objects serialized at call time) alongside the execution request. These snapshots are embedded in the audit record so it is self-contained: the exact input state that drove every rule evaluation is preserved without needing to look up source records after the fact. Entity state can change after execution; capturing at call time ensures the audit reflects what actually happened

### Implementation Considerations

- `RuleRegistry` is a singleton; loaded at startup from a versioned JSON file via `load_from()`
- `execute(domain, context, trace_id, executed_by, entities)` is the primary execution entry point; callers seed `context` with domain object fields using dot-notation keys (`claim.amount`, `customer.tenure_years`) for rule evaluation, and pass `entities` (e.g. `{"claim": claim.to_dict(), "customer": customer.to_dict()}`) as the input context snapshot embedded in the audit record. These are kept separate because `context` must preserve Python types for correct threshold coercion while `entities` is serialized for storage
- `get_active(domain)` returns active rules in topological + priority order; used by the dashboard API for visualization
- `get_dag(domain)` returns a cached `nx.DiGraph`; all NetworkX queries (ancestors, descendants, cycle detection) are available on the returned graph directly
- `RuleFactory` detects rule type from raw dict field presence (`detect_type()`) and instantiates the correct subclass; adding a new rule type requires only a new subclass and a `detect_type()` case
- ETL pipeline (`RuleEtl`) processes one domain per run; driven by `config/etl.yaml`; output files in `data/out/` are the source of truth

### Key Trade-Offs

- Rule-driven logic is easier to audit and modify without code changes, but adds a layer of indirection compared to direct conditional logic
- DAG construction and topological ordering add startup cost; cached after first access per domain
- Append-only versioning preserves full history but requires periodic archiving as rule counts grow
- DAG branch pruning makes the traversal semantically correct and efficient, but requires rule authors to declare `input`/`output` fields accurately. Incorrect declarations will cause rules to be pruned when they should run, or evaluated when their prerequisites were never met

## MCP Servers

### Purpose

MCP servers provide a standardized, governable interface to backend systems. They hide infrastructure details and expose domain-relevant tool operations to agents.

### Design Decisions

- One MCP server per enterprise system or bounded integration domain
- Business-oriented tool names and typed schemas
- Full support for input validation, auth propagation, audit hooks, and rate limiting
- Separate read-only and write-capable tools

### Implementation Considerations

- Claims MCP: `get_claim_status`, `list_claim_documents`, `create_claim_note`
- ServiceNow MCP: `get_ticket`, `create_ticket`, `update_ticket_comment`
- CRM MCP: `get_customer_profile`, `list_interactions`, `update_contact_preference`
- Billing and policy MCPs for coverage, premium, and payment operations
- Snowflake MCP limited to curated analytics access

### Key Trade-Offs

- Strong standardization reduces long-term integration cost
- Requires initial investment in schema design and governance
- Delivers long-term platform consistency across teams

## Model Gateway

### Purpose

The model gateway abstracts providers and enforces policy for model selection, latency budgets, cost ceilings, and fallback behavior.

### Design Decisions

- Support multiple providers through a single control plane
- Route by task type, latency target, and cost policy
- Enforce model allowlists and token limits per use case

### Implementation Considerations

- Use Bedrock first to access Claude, Llama, and other approved models
- Route cheap classifiers to lower-cost models
- Reserve premium models for synthesis-heavy or ambiguous tasks
- Add semantic caching and provider fallback

### Key Trade-Offs

- Centralized control improves governance and cost visibility
- Adds an extra layer to platform operations
- Essential at large multiline insurer scale for cost and resilience management

## Prompt Registry

### Purpose

Treat prompts as versioned production assets with testing, promotion, rollback, and approval workflows.

### Design Decisions

- Version prompts, templates, tool lists, and evaluation datasets together
- Tie prompt versions to production traces and audits
- Support canary deployments and rollback

### Implementation Considerations

- Store metadata such as owners, rollout status, approval history, and linked test suites
- Require business and compliance approvals for regulated prompt changes

### Key Trade-Offs

- More process around prompt changes slows ad hoc editing
- Greatly improves reliability, repeatability, and compliance posture

## Observability and Audit

### Purpose

These services provide production visibility and traceability across agent reasoning, workflow execution, tool usage, and user outcomes.

### Design Decisions

- Use distributed tracing for every request path
- Capture structured events for prompts, model calls, tool invocations, and escalations
- Store immutable audit records for regulated operations

### Implementation Considerations

- OpenTelemetry instrumentation across all services
- Metrics for latency, tool failure, containment, and cost per interaction
- Immutable audit storage using S3 Object Lock and indexed metadata for investigations

### Key Trade-Offs

- More telemetry increases storage and processing cost
- Strong observability is mandatory for platform trust, tuning, and compliance
