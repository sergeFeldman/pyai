# Agentic Workflow Platform Architecture Overview for a Large Multiline Insurer

> **Note:** This document describes the target enterprise architecture. The current POC implements a subset of this vision. See [Implementation Dataflow](../implementation/dataflow.md) for what is currently built.

## Executive Summary

The right architecture for a large multiline insurer is a layered agentic platform, not a single "super-agent." At the insurer's scale and regulatory posture, the platform should separate orchestration, domain reasoning, enterprise-system access, and cross-cutting controls. The core pattern is: an Orchestrator Agent receives every request, determines intent and risk, builds context, delegates to one or more specialized Domain Agents, and synthesizes a final answer. Domain and sub-agents never connect directly to enterprise systems; instead, they use standardized MCP servers that encapsulate tool access, auth, rate limiting, and audit. This keeps the platform maintainable, governable, and safe to evolve.

For cloud implementation, AWS is the best fit: EKS or ECS for agent runtime, API Gateway plus ALB for ingress, Temporal for workflow durability, Bedrock plus a model gateway for model abstraction, OpenSearch and/or Aurora pgvector for retrieval, MSK/SQS for async eventing, and CloudWatch/X-Ray/OpenTelemetry for observability. The design should support multi-model routing, human escalation, prompt/model governance, immutable audit, and gradual rollout alongside current systems. The target operating model is hybrid: deterministic workflow where possible, agentic delegation where valuable, and human review for low-confidence or regulated actions.

The most important design choice is to treat "agentic" as a controlled execution fabric rather than freeform autonomy. Routine work should be largely handled through constrained tools, typed outputs, retrieval, policy checks, and workflow guards. That is how a large multiline insurer gets to 80%+ automation, sub-2s P95 for common flows, and 99.9% availability without creating an expensive or ungovernable system.

## Architecture Summary

The platform is organized into:

- Experience and API ingress for agent desktop, chat, IVR, supervisor tools, and internal portals
- An Orchestrator Agent responsible for routing, context assembly, policy enforcement, and final synthesis
- Domain Agents for Claims, Service, Sales, and later HR and IT
- Reusable sub-agents for knowledge retrieval, data query, document processing, and escalation
- MCP servers that provide standardized, audited access to backend systems
- Shared platform services for model routing, prompts, observability, security, audit, and credential management

## Success Targets

- 80%+ autonomous handling of routine inquiries
- Cross-domain handling across claims, service, and sales
- Human escalation with complete context handoff
- P95 latency under 2 seconds for top-volume synchronous flows
- 99.9% availability
- Full auditability for regulated interactions

## Recommended Technology Baseline

- Runtime: AWS EKS or ECS
- Workflow orchestration: Temporal
- Model access: Amazon Bedrock with an internal model gateway
- Agent orchestration: LangGraph or a lightweight orchestration layer on top of Temporal
- MCP servers: Python/FastAPI or Node.js/TypeScript
- API ingress: API Gateway and ALB
- Caching: ElastiCache Redis
- Eventing and queues: SQS and MSK
- Data stores: Aurora PostgreSQL, DynamoDB, S3
- Retrieval: OpenSearch and vector search, optionally pgvector for bounded domain use cases
- Observability: OpenTelemetry, CloudWatch, Grafana, Langfuse or Arize
- Secrets: AWS Secrets Manager and KMS

## Design Principles

- Keep orchestration centralized but thin
- Make system access tool-based and policy-controlled
- Use deterministic workflows where possible and agentic delegation where valuable
- Separate domain reasoning from enterprise integration logic
- Treat prompts, tools, and models as governed production assets
- Optimize first for safety, traceability, and maintainability, then for autonomy
