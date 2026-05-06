"""Workflow orchestrator related classes."""

import agents as agt
import core
import models as mdl
import services as svc


class WorkflowOrchestrator(metaclass=core.Singleton):
    """Orchestrator responsible for routing and executing workflow patterns."""

    def __init__(self, trace_service: svc.TraceService):
        """Initialize the workflow orchestrator.

        Args:
            trace_service (svc.TraceService): Trace service instance.
        """
        self._trace_service = trace_service

    async def get_claim_explanation(self, request: mdl.UserRequest, agent_config: dict) -> mdl.UserResponse:
        """Execute claim explanation workflow via LangChain ReAct agent.

        The agent dynamically decides which tools to invoke — claim lookup,
        customer context, and policy rules — and synthesizes the final
        explanation using an LLM.

        Args:
            request (mdl.UserRequest): Normalized user request object.
            agent_config (dict): Claim explanation agent configuration.

        Returns:
            mdl.UserResponse: User-facing workflow response.
        """
        # Step 1: Create workflow context with trace ID for auditability.
        context = self._trace_service.create_context(request)

        # Step 2: Retrieve (or create) the claim explanation agent via factory.
        agent = await agt.AgentFactory().get_obj_async("claim_explanation", agent_config)

        # Step 3: Invoke agent — LLM decides which tools to call and in what order.
        message = await agent.get_explanation_message(request)

        # Step 4: Return LLM-synthesized response with trace ID.
        return mdl.UserResponse(message=message, trace_id=context.trace_id)

    def get_claim_status(self, request: mdl.UserRequest, agent_config: dict) -> mdl.UserResponse:
        """Execute claim-status workflow.

        Args:
            request (mdl.UserRequest): Normalized user request object.
            agent_config (dict): Claim-agent configuration passed to AgentFactory.

        Returns:
            mdl.UserResponse: User-facing workflow response.
        """
        # Step 1: Create workflow context with trace ID for auditability.
        context = self._trace_service.create_context(request)

        # Step 2: Fetch claim status and build user-facing message via ClaimAgent.
        claim_agent = agt.AgentFactory().get_obj("claim", agent_config)
        claim_request = mdl.ClaimRequest(claim_id=request.message)
        message = claim_agent.get_status_message(claim_request)

        # Step 3: Return response with trace ID.
        return mdl.UserResponse(message=message, trace_id=context.trace_id)
