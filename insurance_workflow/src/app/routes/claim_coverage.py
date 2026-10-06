"""Claim coverage verification route related functions."""

from typing import Annotated

from fastapi import APIRouter, Depends

import handlers as hdl
import models as mdl
from app.dependencies import get_request_handler


router = APIRouter()


@router.post("/claim-coverage", response_model=mdl.ClaimCoverageHttpResponse)
async def claim_coverage(
    payload: mdl.ClaimCoverageHttpRequest,
    request_handler: Annotated[hdl.RequestHandler, Depends(get_request_handler)],
) -> mdl.ClaimCoverageHttpResponse:
    """Handle the claim coverage verification endpoint."""
    response = await request_handler.handle(
        request_type="claim_coverage",
        message=payload.message,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )

    return mdl.ClaimCoverageHttpResponse(
        message=response.message,
        trace_id=response.trace_id,
    )
