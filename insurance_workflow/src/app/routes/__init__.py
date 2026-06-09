"""Convenience exports for the ``app.routes`` package."""

from fastapi import APIRouter

from .claim_appeal import router as claim_appeal_router
from .claim_explanation import router as claim_explanation_router
from .claim_status import router as claim_status_router

all_routers: list[APIRouter] = [
    claim_appeal_router,
    claim_explanation_router,
    claim_status_router,
]

__all__ = [
    "all_routers",
]
