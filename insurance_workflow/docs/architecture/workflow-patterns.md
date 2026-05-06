# Workflow Patterns and Examples

## Workflow Selection Guidance

### Sequential Execution

Use sequential execution when the request is contained within a single domain, requires a small number of tool calls, and needs the lowest possible latency.

Best fit:

- claim status
- billing balance lookup
- policy coverage explanation

### Parallel Execution

Use parallel execution when independent tool calls can run concurrently and the request spans more than one domain or backend.

Best fit:

- claims plus service lookups
- policy plus billing context checks
- multi-system customer summaries

### Conditional Routing

Use conditional routing when business rules, risk thresholds, or backend results determine whether the platform should continue autonomously, switch domains, or escalate.

Best fit:

- underwriting threshold checks
- approval-required updates
- unsupported self-service actions

### Human Escalation with Context Handoff

Use escalation when confidence is low, business impact is high, data is contradictory, or a human must authorize the outcome.

Best fit:

- claim appeals
- payout disputes
- fraud-sensitive or compliance-sensitive cases

### Collaborative Multi-Agent Workflow

Use collaborative workflows when the request requires multiple types of reasoning, such as data lookup, knowledge grounding, and document extraction, before the final answer can be assembled.

Best fit:

- denial explanation plus appeal eligibility
- claim plus billing interactions
- document-backed case review

## MVP Starting Point

The simplest workflow to implement first is `Simple Claim Status (Sequential)`.

Why this first:

- single domain
- read-only
- one primary backend call
- low compliance risk
- high business value
- ideal for validating orchestration, MCP integration, observability, and audit on a narrow vertical slice

## Workflow Examples

### Example 1: Simple Claim Status (Sequential)

```text
User: "What's the status of claim CLM-12345?"

Trace ID: trace-001

Step 1: Orchestrator
├── Intent: "claim_status" (confidence: 0.98)
├── Domain: "claims"
├── Risk: "low"
└── Route to: Claims Agent (45ms)

Step 2: Claims Agent
├── Tool call: claims.get_claim_status(claim_id="CLM-12345")
├── MCP Server: Claims MCP → Claims DB
├── Result: {status: "in_review", adjuster: "Smith", last_update: "2026-04-03"}
└── Response draft generation (150ms)

Step 3: Orchestrator
├── Add user-facing explanation
├── Attach source metadata
└── Synthesize final response (25ms)

Final Response:
"Your claim CLM-12345 is currently in review. The assigned adjuster is Smith, and the latest update was posted on April 3, 2026."

Total: 220ms
Cost: $0.0003
```

### Example 2: Cross-Domain Query (Parallel)

```text
User: "What's the status of all open claims and service tickets for John Doe?"

Trace ID: trace-002

Step 1: Orchestrator
├── Intent: "cross_domain_query" (confidence: 0.95)
├── Domains: ["claims", "service"]
├── Entity resolution: customer="John Doe"
└── Spawn parallel workflows

Parallel Execution:
┌──────────────────────────────────────────────────────────────┐
│ Claims Agent (185ms)                                        │
│ ├── Tool: claims.get_claims_by_customer("John Doe")         │
│ ├── MCP Server: Claims MCP                                  │
│ └── Returns: [claim1, claim2]                               │
├──────────────────────────────────────────────────────────────┤
│ Service Agent (210ms)                                       │
│ ├── Tool: service.get_tickets_by_customer("John Doe")       │
│ ├── MCP Server: ServiceNow MCP                              │
│ └── Returns: [ticket1]                                      │
└──────────────────────────────────────────────────────────────┘

Step 3: Orchestrator
├── Aggregate results: "2 open claims, 1 open ticket"
├── Reconcile customer context
└── Synthesize response (35ms)

Final Response:
"John Doe currently has 2 open claims and 1 open service ticket. I can also provide the latest status for each item if you'd like."

Total: 245ms
Cost: $0.0008
Parallelization Benefit: saved ~150ms versus sequential execution
```

### Example 3: Agentic Workflow with Multiple Tool Calls

```text
User: "File a new claim for damage to my 2022 Toyota Camry. Accident was yesterday."

Trace ID: trace-003

Step 1: Orchestrator
├── Intent: "file_claim" (confidence: 0.93)
├── Domain: "claims"
├── Risk: "medium"
└── Route to: Claims Agent

Step 2: Claims Agent — Multi-step workflow
├── Step 2a: Knowledge Agent
│   └── Retrieve claim filing procedures and required fields (120ms)
├── Step 2b: Tool call: policy.get_policy_by_vehicle(customer_id, "2022 Toyota Camry") (95ms)
├── Step 2c: Tool call: claims.get_claim_filing_requirements() (45ms)
├── Step 2d: Tool call: claims.validate_claim_details(vehicle="2022 Toyota Camry", date="2026-04-03") (30ms)
├── Step 2e: Tool call: claims.submit_claim(policy_id, details) (180ms)
└── Step 2f: Generate response with claim number and next steps (210ms)

Step 3: Orchestrator
├── Confirm submission status
├── Format next actions for user
└── Return final response

Final Response:
"Your new claim has been submitted successfully. Your claim number is CLM-90871. An adjuster will review it, and you can expect the next update within 1 business day."

Total: 680ms
Cost: $0.012
Execution Pattern: sequential with selective parallel retrieval and validation
```

### Example 4: Human Escalation (Low Confidence)

```text
User: "My claim was denied and I think it's a mistake. I've been a customer for 20 years."

Trace ID: trace-004

Step 1: Orchestrator
├── Intent: "claim_appeal" (confidence: 0.91)
├── Domain: "claims"
├── Risk: "high"
└── Route to: Claims Agent with escalation allowed

Step 2: Claims Agent
├── Tool: claims.get_claim_details(claim_id) (150ms)
├── Tool: claims.get_denial_reason(claim_id) (80ms)
├── Tool: customer.get_customer_tenure(customer_id) → "20 years" (45ms)
├── Tool: knowledge.get_appeal_policy() (60ms)
└── Confidence assessment: 0.45 (threshold: 0.70)
    └── Reason: "Complex denial appeal requires human judgment"

Step 3: Escalation Agent
├── Gather full conversation context
├── Build escalation summary
├── Identify queue: senior_claims_adjuster
└── Transfer with context bundle

Response to User:
"I'm connecting you with a senior claims adjuster who can review your case personally. Please hold while I transfer your case."

Human Agent Receives:
- Full chat history
- Claim details
- Denial reason
- Customer tenure: 20 years
- Relevant policy excerpts
- AI confidence score: 0.45
- Suggested next investigation steps

Total: 410ms before transfer
Cost: $0.0021
Outcome: human-in-the-loop escalation with full context handoff
```
