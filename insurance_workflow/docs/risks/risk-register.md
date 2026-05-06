# Risk Register

## Hallucination or Unsafe Action

### Risk

The platform may produce incorrect guidance or take an action that is not authorized or policy-compliant.

### Mitigation

- Ground responses with retrieval and citations
- Constrain execution through tools and typed outputs
- Use policy checks before any write action
- Require human approval for high-impact workflows

## Latency Blow-Up from Agent Fan-Out

### Risk

Excessive delegation and too many serial or parallel tool calls can push synchronous workflows beyond target latency.

### Mitigation

- Set strict synchronous latency budgets
- Cap fan-out depth and width
- Use async handling for long-running tasks
- Parallelize only when calls are independent

## Legacy Backend Fragility

### Risk

Existing enterprise systems may not tolerate bursty AI-driven traffic patterns.

### Mitigation

- Add MCP-layer rate limiting and circuit breakers
- Use caches and queue buffers
- Isolate workloads with bulkheads
- Provide degraded-mode responses when systems are unhealthy

## Cost Overrun

### Risk

Large-scale LLM usage can become uneconomical if routing and caching are poorly controlled.

### Mitigation

- Use a model gateway with task-based routing
- Apply semantic and retrieval caching
- Minimize unnecessary token usage
- Track and review cost per resolved interaction

## Governance Bottlenecks

### Risk

Prompt and model changes may become too slow if governance is manual and fragmented.

### Mitigation

- Use a prompt registry and automated evaluation pipeline
- Define risk-tiered approval workflows
- Clarify ownership by domain and platform team

## Organizational Complexity

### Risk

The platform can become difficult to operate if it evolves as an experimental research stack instead of a productized platform.

### Mitigation

- Standardize on MCP, workflow orchestration, and a model gateway
- Keep agents narrow and tool-driven
- Favor maintainable platform patterns over bespoke experimental solutions
