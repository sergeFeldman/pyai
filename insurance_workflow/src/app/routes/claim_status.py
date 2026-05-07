"""Claim-status route related functions."""

from typing import Annotated

from fastapi import APIRouter, Depends

import handlers as hdl
import models as mdl
from app.dependencies import get_request_handler


router = APIRouter()


@router.post("/claim-status", response_model=mdl.ClaimStatusHttpResponse)
async def claim_status(
    payload: mdl.ClaimStatusHttpRequest,
    request_handler: Annotated[hdl.RequestHandler, Depends(get_request_handler)],
) -> mdl.ClaimStatusHttpResponse:
    """Handle the claim-status endpoint."""
    response = await request_handler.handle(
        request_type="claim_status",
        message=payload.message,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )

    return mdl.ClaimStatusHttpResponse(
        message=response.message,
        trace_id=response.trace_id,
    )
