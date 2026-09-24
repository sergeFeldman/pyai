"""Convenience exports for the ``app.routes`` package."""

from fastapi import APIRouter

from .claim_appeal import router as claim_appeal_router
from .claim_explanation import router as claim_explanation_router
from .claim_status import router as claim_status_router
from .executions import router as executions_router
from .rules import router as rules_router
from ui.dashboard import router as dashboard_router

all_routers: list[APIRouter] = [
    claim_appeal_router,
    claim_explanation_router,
    claim_status_router,
    executions_router,
    rules_router,
    dashboard_router,
]

__all__ = [
    "all_routers",
]
