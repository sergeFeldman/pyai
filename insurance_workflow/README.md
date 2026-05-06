# Agentic Workflow Platform: Insurance Claims POC

A proof-of-concept demonstrating how agentic AI can automate insurance claims processing, from deterministic rule-based decisions to LLM-driven, multi-step reasoning over live data.

## The Problem

Insurance claims handling is data-intensive and context-dependent. A claim decision involves the claim record itself, the applicable policy rules, the customer's history and relationship context, and often multiple attributes that need to be explained together. Today this reasoning is largely manual: adjusters look up data across systems, apply policy knowledge, and compose responses. The process is slow, inconsistent, and hard to scale.

The question this POC explores: **how much of that reasoning can be automated by an agentic AI platform, and what does the right architecture look like?**

## The Approach

Rather than a single monolithic AI model, the platform uses a layered agentic architecture:

- A **workflow orchestrator** routes each request to the right agent
- **Specialized agents** handle specific domains (claims, customers, policy rules)
- **MCP (Model Context Protocol) servers** provide standardized, audited access to data; the LLM never touches raw systems directly
- An **LLM-backed agent** (powered by Groq or Anthropic) autonomously decides which tools to call, reasons over the results, and synthesizes a natural-language response grounded in policy

This separation keeps the platform maintainable, auditable, and safe to evolve: the LLM is responsible for reasoning, not data access.

## Use Cases Implemented

### Use Case 1: Claim Status (`POST /claim-status`)

Deterministic, rule-based. Given a claim ID, the platform:
1. Retrieves the claim record via MCP
2. Fetches the customer's relationship context
3. Looks up the applicable policy rule for the claim's current status
4. Returns a structured, policy-grounded explanation

No LLM involved: fast, predictable, auditable.

### Use Case 2: Claim Explanation (`POST /claim-explanation`)

Agentic, LLM-driven. Given a claim ID and a list of attributes to explain (e.g. `status`, `is_fraud`), the platform:
1. Launches a ReAct agent backed by Groq (Llama 3.3) or Anthropic
2. The agent autonomously decides which MCP tools to call and in what order
3. It retrieves claim data, customer context, and all applicable policy rules
4. Synthesizes a natural-language explanation covering all requested attributes

The agent's reasoning steps are fully visible: every tool call and observation is logged.

## Key Design Patterns

| Pattern | Purpose |
|---|---|
| MCP (Model Context Protocol) | Standardized tool interface between the LLM and data sources |
| LangChain structured chat agent | Handles tools with JSON schemas; drives the ReAct reasoning loop |
| Configurable factory + Singleton | Agents are typed, config-driven, and cached for reuse |
| Provider-agnostic LLM | Switch between Groq, Anthropic, or Ollama via `config/agents.yaml`; no code changes |
| Async factory (`create()`) | Enables async tool loading during agent construction |
| `SerializableMixin` | Consistent, enum-safe serialization of domain models to MCP tool responses |

## Technology Stack

- **FastAPI**: HTTP API layer
- **LangChain**: agent orchestration and ReAct loop
- **MCP / langchain-mcp-adapters**: tool protocol between LLM and data
- **Groq / Anthropic / Ollama**: pluggable LLM providers
- **Python dataclasses + Pydantic**: typed domain and config models

## Documentation

- [Architecture Overview](docs/architecture/agentic-platform-overview.md)
- [Workflow Patterns](docs/architecture/workflow-patterns.md)
- [Component Deep Dive](docs/architecture/component-deep-dive.md)
- [Implementation Dataflow](docs/implementation/dataflow.md)
- [Setup and Running](docs/setup.md)
- [Scaling and Performance](docs/architecture/scaling-and-performance.md)
- [Security and Compliance](docs/architecture/security-and-compliance.md)
- [Implementation Roadmap](docs/roadmap/implementation-roadmap.md)
- [Risk Register](docs/risks/risk-register.md)
