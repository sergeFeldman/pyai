"""Request handler related classes."""

import inspect
from typing import Optional

import models as mdl
import workflow


class RequestHandler:
    """Class responsible for handling incoming user requests."""

    _WORKFLOWS_MAPPING = {
        "claim_explanation": "get_claim_explanation",
        "claim_status": "get_claim_status",
    }

    def __init__(self, workflow_orchestrator: workflow.WorkflowOrchestrator):
        """Initialize the request handler.

        Args:
            workflow_orchestrator (workflow.WorkflowOrchestrator): Workflow orchestrator instance.
        """
        self._workflow_orchestrator = workflow_orchestrator

    async def handle(self, request_type: str, message: str, agent_config: dict,
                     user_id: Optional[str], session_id: Optional[str],
                     attributes: Optional[list[str]] = None) -> mdl.UserResponse:
        """Handle a raw user request and delegate it to the matching workflow method.

        Args:
            request_type (str): Workflow request type identifier.
            message (str): Raw user message.
            agent_config (dict): Agent configuration passed to the orchestrator.
            user_id (str, optional): User identifier.
            session_id (str, optional): Session identifier.
            attributes (list[str], optional): Attribute names for explanation workflows.

        Returns:
            mdl.UserResponse: User-facing workflow response.

        Raises:
            ValueError: Raised when the provided request type is not supported.
        """
        if request_type not in self._WORKFLOWS_MAPPING:
            raise ValueError(f"Unsupported request type: {request_type}")

        request = mdl.UserRequest(message, attributes, user_id, session_id)

        # Resolve the workflow method by request type.
        workflow_method_name = self._WORKFLOWS_MAPPING[request_type]
        workflow_method = getattr(self._workflow_orchestrator, workflow_method_name)

        # Pass agent_config only if the workflow method declares it as a parameter.
        # MCP-backed workflows (e.g. claim_explanation) resolve their own data access
        # via the MCP server and do not need agent_config.
        sig = inspect.signature(workflow_method)
        args = (request, agent_config) if "agent_config" in sig.parameters or "agent_configs" in sig.parameters else (request,)

        # Dispatch — async workflow methods (LLM/MCP-backed) are awaited,
        # sync methods (rule-based) are called directly.
        if inspect.iscoroutinefunction(workflow_method):
            return await workflow_method(*args)
        return workflow_method(*args)
