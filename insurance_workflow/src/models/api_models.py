"""FastAPI request and response models for claim endpoints."""

from typing import Optional

from .base_model import WorkflowBaseModel


class ClaimExplanationHttpRequest(WorkflowBaseModel):
    """HTTP request payload for the claim explanation workflow."""

    message: str
    attributes: list[str]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ClaimExplanationHttpResponse(WorkflowBaseModel):
    """HTTP response payload for the claim explanation workflow."""

    message: str
    trace_id: str


class ClaimAppealHttpRequest(WorkflowBaseModel):
    """HTTP request payload for the claim appeal eligibility workflow."""

    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ClaimAppealHttpResponse(WorkflowBaseModel):
    """HTTP response payload for the claim appeal eligibility workflow."""

    message: str
    trace_id: str


class ClaimStatusHttpRequest(WorkflowBaseModel):
    """HTTP request payload for the claim-status workflow."""

    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ClaimStatusHttpResponse(WorkflowBaseModel):
    """HTTP response payload for the claim-status workflow."""

    message: str
    trace_id: str
