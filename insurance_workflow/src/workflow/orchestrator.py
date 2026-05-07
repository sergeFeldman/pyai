"""Workflow orchestrator related classes."""

import agents as agt
import core
import models as mdl
import services as svc


class WorkflowOrchestrator(metaclass=core.Singleton):
    """Orchestrator responsible for routing and executing workflow patterns."""

    def __init__(self, trace_service: svc.TraceService, agent_configs: dict):
        """Initialize the workflow orchestrator.

        Args:
            trace_service (svc.TraceService): Trace service instance.
            agent_configs (dict): Agent configurations keyed by agent name.
        """
        self._trace_service = trace_service
        self._agent_configs = agent_configs

    async def get_claim_explanation(self, request: mdl.UserRequest) -> mdl.UserResponse:
        """Execute claim explanation workflow via LangChain ReAct agent.

        The agent dynamically decides which tools to invoke — claim lookup,
        customer context, and policy rules — and synthesizes the final
        explanation using an LLM.

        Args:
            request (mdl.UserRequest): Normalized user request object.

        Returns:
            mdl.UserResponse: User-facing workflow response.
        """
        context = self._trace_service.create_context(request)
        agent = await agt.AgentFactory().get_obj_async("claim_explanation", self._agent_configs["claim_explanation"])
        message = await agent.get_explanation_message(request)
        return mdl.UserResponse(message=message, trace_id=context.trace_id)

    def get_claim_status(self, request: mdl.UserRequest) -> mdl.UserResponse:
        """Execute claim-status workflow.

        Args:
            request (mdl.UserRequest): Normalized user request object.

        Returns:
            mdl.UserResponse: User-facing workflow response.
        """
        context = self._trace_service.create_context(request)
        claim_agent = agt.AgentFactory().get_obj("claim", self._agent_configs["claim"])
        claim_request = mdl.ClaimRequest(claim_id=request.message)
        message = claim_agent.get_status_message(claim_request)
        return mdl.UserResponse(message=message, trace_id=context.trace_id)
