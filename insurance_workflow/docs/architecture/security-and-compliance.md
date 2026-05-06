# Security and Compliance

## Security Principles

- Apply least privilege at every layer
- Enforce identity and authorization at the MCP tool level
- Minimize sensitive data sent to models
- Keep all regulated operations fully auditable
- Separate employee, customer, and service identities

## Authentication

- Employee access should use enterprise SSO through OIDC or SAML
- Customer-facing channels should use CIAM with MFA where appropriate
- Service-to-service communication should use mTLS and short-lived credentials
- Agent actions must be attributable to the initiating user or approved system identity

## Authorization

- Use RBAC and ABAC together
- Enforce domain boundaries, such as Claims, Service, Sales, HR, and IT
- Restrict high-impact actions to users or agents with explicit permissions
- Require approval for regulated write operations where policy demands it

## Data Privacy

- Apply PII and PCI redaction before logging and tracing
- Encrypt data in transit and at rest
- Use field-level encryption for highly sensitive records
- Send only the minimum necessary fields to language models
- Mask or synthesize data in lower environments

## Audit Logging

Every significant AI-assisted interaction should capture:

- request ID and conversation ID
- employee or customer identity
- model version and prompt version
- retrieved documents and citations
- tool calls and their outputs
- policy checks and approval steps
- final answer or action

Immutable audit storage should be implemented with S3 Object Lock or an equivalent tamper-evident mechanism. Searchable metadata can be indexed separately for investigations and reporting.

## Model Governance

- Maintain an approved model registry
- Assign risk tiers by use case
- Require offline evaluation before production promotion
- Monitor for drift, quality degradation, and unexpected cost patterns
- Provide kill switches for models and providers

## Prompt Governance

- Version prompts like code
- Tie prompts to evaluation datasets and expected outputs
- Use canary releases and rollback
- Require approval for prompt changes affecting regulated workflows

## Compliance Alignment

The architecture should support insurance-sector regulatory expectations around traceability, explainability, retention, and controlled access. It should also make legal review easier by preserving evidence of what the system saw, what tools it used, what model and prompt version generated the output, and whether a human approved the final action.
