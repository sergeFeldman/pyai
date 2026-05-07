"""Claim-explanation route related functions."""

from typing import Annotated

from fastapi import APIRouter, Depends

import handlers as hdl
import models as mdl
from app.dependencies import get_request_handler


router = APIRouter()


@router.post("/claim-explanation", response_model=mdl.ClaimExplanationHttpResponse)
async def claim_explanation(
    payload: mdl.ClaimExplanationHttpRequest,
    request_handler: Annotated[hdl.RequestHandler, Depends(get_request_handler)],
) -> mdl.ClaimExplanationHttpResponse:
    """Handle the claim-explanation endpoint."""
    response = await request_handler.handle(
        request_type="claim_explanation",
        message=payload.message,
        user_id=payload.user_id,
        session_id=payload.session_id,
        attributes=payload.attributes,
    )

    return mdl.ClaimExplanationHttpResponse(
        message=response.message,
        trace_id=response.trace_id,
    )
