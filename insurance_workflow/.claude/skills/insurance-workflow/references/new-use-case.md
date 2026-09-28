# Adding a New Use Case

Read this when adding a new endpoint that runs a new workflow. Every step below is required; skipping one leaves the use case unreachable or unaudited.

1. **HTTP models.** Add `<UseCase>HttpRequest` and `<UseCase>HttpResponse` to `src/models/api_models.py`, extending `WorkflowBaseModel`. The request carries `message` plus optional `user_id` and `session_id`; the response carries `message` and `trace_id`. Export both from `src/models/__init__.py`.
2. **Route.** Create `src/app/routes/<use_case>.py` with an `APIRouter`, a `POST /<use-case>` handler that takes the request model and `Annotated[hdl.RequestHandler, Depends(get_request_handler)]`, calls `request_handler.handle(request_type="<use_case>", ...)`, and maps the `UserResponse` to the response model. Add the router to `all_routers` in `src/app/routes/__init__.py`.
3. **Handler mapping.** Add `"<use_case>": "<orchestrator_method>"` to `RequestHandler._WORKFLOWS_MAPPING`.
4. **Orchestrator method.** Add the method to `WorkflowOrchestrator`, following the use-case pattern in the insurance-workflow skill. Make it `async` only if an agent it calls is async.
5. **Agent.** Add or reuse an agent, following the workflow-agents skill. New agents are registered in `AgentFactory._TYPES_MAPPING`.
6. **Config.** Add a `get_<agent>_agent_config()` builder in `src/app/dependencies.py` and an entry in `_AGENT_CONFIGS`. Put any new non-secret settings in `config/*.yaml`; secrets go in `.env`.
7. **Audit.** Log the request with a new `request_type` value and the list of agent keys used, following the audit-and-trace skill, on every return path.
8. **Docs.** Add a use-case section to `docs/implementation/dataflow.md` and an example call to `docs/implementation/setup.md`.
9. **Tests.** Add tests under `tests/` for the agent behavior and each not-found path, following [testing.md](testing.md).

## Example

A synthetic "claim payout status" use case would add `ClaimPayoutHttpRequest`/`ClaimPayoutHttpResponse`, `src/app/routes/claim_payout.py` at `POST /claim-payout`, the mapping `"claim_payout": "get_claim_payout"`, an orchestrator method that resolves the `claim` agent and returns `Claim claim_9 was not found.` when the claim is missing, and an audit record with `request_type="claim_payout"` and `agent_names=["claim"]` on both the found and not-found paths.
