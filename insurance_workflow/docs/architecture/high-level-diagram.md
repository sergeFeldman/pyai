# High-Level Architecture Diagram

> **Note:** This document describes the target enterprise architecture. The current POC implements a subset of this vision. See [Implementation Dataflow](../implementation/dataflow.md) for what is currently built.

## Diagram

```text
[Channels]
Agent Desktop | Supervisor UI | IVR/Voice | Chat | Internal Portal | API Clients
        |
        v
[Experience/API Layer]
API Gateway / ALB / WebSocket Gateway
SSO / OAuth2 / Session Context / Rate Limits
        |
        v
[Orchestration Layer]
Orchestrator Agent
Intent + domain classification
Policy/risk checks
Context assembly
Workflow selection
Response synthesis
Human escalation trigger
        |
        +-------------------------------+
        |                               |
        v                               v
[Workflow Engine]                 [Model Gateway]
Temporal / Step Functions         Multi-model routing
Durable execution                 Cost/latency policies
Retries/timeouts                  Fallbacks / A-B tests
        |                               |
        v                               v
[Agent Layer]
Claims Agent | Service Agent | Sales Agent | HR/IT Agent
Knowledge Agent | Data Query Agent | Document Agent | Escalation Agent
        |
        v
[Tool Access Layer]
MCP Registry + MCP Servers
Claims MCP | ServiceNow MCP | CRM MCP | Billing MCP | Policy MCP
Snowflake MCP | Document MCP | HR MCP | Identity MCP
        |
        v
[Enterprise Systems]
Claims DB | Policy Admin | CRM | ServiceNow | Snowflake | Doc Stores | HR Systems
        |
        v
[Shared Platform Services]
Vector/RAG | Prompt Registry | Audit Store | Observability | Vault | Feature Flags
```

## Layer Descriptions

### Channels

The platform should support employee-facing and customer-facing interaction points without changing core orchestration. Supported channels include contact center desktops, supervisor consoles, web chat, IVR, internal portals, and partner APIs.

### Experience and API Layer

This layer terminates traffic, applies initial security and rate controls, manages sessions, and normalizes requests before they enter the orchestration tier.

### Orchestration Layer

The orchestrator classifies intent, determines risk, selects workflow patterns, and coordinates downstream agents. It is the control plane for reasoning, not the execution point for enterprise integrations.

### Agent Layer

Domain agents handle business-specific reasoning. Sub-agents provide reusable capabilities that cut across domains, such as retrieval, document extraction, and escalation packaging.

### Tool Access Layer

MCP servers standardize backend access and isolate enterprise integration logic from agent logic. This layer is essential for governance, rate limiting, auditing, and gradual modernization.

### Shared Platform Services

These services make the platform production-ready by providing traceability, prompt and model lifecycle management, retrieval, secrets handling, and operational controls.
