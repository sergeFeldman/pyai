# Scaling and Performance

> **Note:** This document describes the target enterprise architecture. The current POC implements a subset of this vision. See [Implementation Dataflow](../implementation/dataflow.md) for what is currently built.

## Performance Targets

- P95 latency below 2 seconds for high-volume synchronous flows
- 99.9% availability
- Support for millions of LLM calls per day
- Support for 10,000+ concurrent sessions in later rollout phases

## Horizontal Scaling Strategy

- Run orchestrator, agents, and MCP services as stateless workloads on EKS or ECS
- Use separate autoscaling groups for API ingress, orchestration, document processing, and MCP workers
- Use queue-backed execution for long-running or bursty work
- Isolate high-cost or high-latency components so they do not impact low-latency flows

## Caching Strategy

### Recommended Layers

- In-memory cache for prompt metadata and tool registry data
- Redis for session context and hot entity lookups
- Semantic cache for repetitive grounded questions
- Retrieval cache for common knowledge chunks
- Response cache for low-risk read-only flows only

### Expected Outcomes

- 20% to 40% reduction in repetitive LLM traffic
- Lower median and P95 latency for common flows
- Better backend protection through reduced duplicate reads

## Data and Retrieval Scaling

- Use Aurora PostgreSQL for transactional metadata and workflow state
- Use DynamoDB where high-throughput session or interaction storage is needed
- Use OpenSearch for enterprise search, log analytics, and optionally vector retrieval
- Use pgvector for smaller domain-specific retrieval workloads where operational simplicity matters
- Partition retrieval indexes by domain and retention needs

## Queuing and Backpressure

- Use SQS for background tasks and retries
- Use MSK when event streaming and ordered processing are needed
- Use Temporal task queues for durable workflow execution
- Apply backpressure at the MCP layer to protect legacy enterprise systems

## Latency Budget Example

- API ingress, auth, and session lookup: 100 ms
- Orchestrator classification and workflow selection: 150 ms
- Parallel tool calls: 500 to 900 ms
- Final synthesis: 300 to 600 ms
- Formatting, logging, and response handling: 100 ms

This budget keeps the main synchronous path inside the 2-second P95 target for top-volume scenarios.

## Availability Strategy

- Deploy all critical services across multiple availability zones
- Isolate failure domains by service type
- Add fallback models and provider failover
- Support degraded modes such as search-only answers or direct escalation
- Maintain operational runbooks and automated health checks

## Cost Optimization

- Route tasks to the cheapest acceptable model
- Use lightweight classifiers for intent and routing
- Avoid unnecessary second-pass synthesis
- Cache aggressively where grounded answers repeat
- Keep document-heavy and low-urgency work asynchronous
- Track cost per resolved interaction and per domain
