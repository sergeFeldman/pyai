"""Tracing service for workflow requests."""

from datetime import datetime, timezone
from uuid import uuid4

import models as mdl


class TraceService:
    """Service class responsible for creating workflow tracing metadata.

    Centralizes trace identifier generation and WorkflowContext initialization
    for request-scoped processing.
    """

    def create_context(self, request: mdl.UserRequest) -> mdl.WorkflowContext:
        """Create a new WorkflowContext instance for the provided request.

        Args:
            request (mdl.UserRequest): Normalized user request object.

        Returns:
            mdl.WorkflowContext: Newly created workflow context populated
                with generated trace metadata and user/session identifiers.
        """
        return mdl.WorkflowContext(
            trace_id=self._generate_trace_id(),
            started_at=datetime.now(timezone.utc),
            user_id=request.user_id,
            session_id=request.session_id,
        )

    @staticmethod
    def _generate_trace_id() -> str:
        """Generate a unique trace identifier for a workflow request.

        Returns:
            str: Generated trace identifier value.
        """
        return f"trace-{uuid4()}"
