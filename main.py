"""
FasalDoc FastAPI backend — main entry point.

Gap fixes:
  - /health endpoint for UptimeRobot warm-pinging (prevents Railway cold start)
  - Global exception handler returns JSON errors
  - CORSMiddleware with explicit origins
  - Request size limit middleware
  - slowapi rate limiter wired up
  - structlog logging
"""

import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from config import ALLOWED_ORIGINS
from services.model_service import model_service
from routers import predict, shops, history, products
from models.schemas import HealthResponse

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — load TFLite model once
    logger.info("Starting FasalDoc API")
    model_service.load()
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


# ── App ────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="FasalDoc API",
    description="Plant disease detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Middleware ─────────────────────────────────────────────────────────────
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ALLOWED_ORIGINS if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    max_bytes = 10 * 1024 * 1024  # 10 MB hard cap at middleware level
    if request.headers.get("content-length"):
        if int(request.headers["content-length"]) > max_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request too large."})
    return await call_next(request)


# ── Global error handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Health endpoint ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """
    Ping endpoint for UptimeRobot / cron-job.org.
    Set up free monitor to ping every 5 minutes to prevent Railway cold starts.
    """
    return {
        "status": "ok",
        "model_loaded": model_service.interpreter is not None,
        "version": "1.0.0",
    }


# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(shops.router)
app.include_router(history.router)
app.include_router(products.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
