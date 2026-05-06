"""Application entry point."""

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

from app.routes import all_routers


app = FastAPI()
for router in all_routers:
    app.include_router(router)


@app.exception_handler(ValueError)
async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
    """Handle value-related application errors."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def handle_file_not_found(_: Request, exc: FileNotFoundError) -> JSONResponse:
    """Handle missing-file application errors."""
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def handle_runtime_error(_: Request, exc: RuntimeError) -> JSONResponse:
    """Handle generic runtime application errors."""
    return JSONResponse(status_code=500, content={"detail": str(exc)})
