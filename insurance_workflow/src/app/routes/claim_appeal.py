"""Claim appeal eligibility route related functions."""

from typing import Annotated

from fastapi import APIRouter, Depends

import handlers as hdl
import models as mdl
from app.dependencies import get_request_handler


router = APIRouter()


@router.post("/claim-appeal", response_model=mdl.ClaimAppealHttpResponse)
async def claim_appeal(
    payload: mdl.ClaimAppealHttpRequest,
    request_handler: Annotated[hdl.RequestHandler, Depends(get_request_handler)],
) -> mdl.ClaimAppealHttpResponse:
    """Handle the claim appeal eligibility endpoint."""
    response = await request_handler.handle(
        request_type="claim_appeal",
        message=payload.message,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )

    return mdl.ClaimAppealHttpResponse(
        message=response.message,
        trace_id=response.trace_id,
    )
