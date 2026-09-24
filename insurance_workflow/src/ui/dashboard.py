"""Dashboard UI route: serves the interactive DAG visualization page."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_template = Path(__file__).parent / "templates" / "dashboard.html"


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the rule registry DAG dashboard."""
    return HTMLResponse(content=_template.read_text(encoding="utf-8"))
