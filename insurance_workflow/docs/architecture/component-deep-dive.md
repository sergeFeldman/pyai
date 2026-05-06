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
