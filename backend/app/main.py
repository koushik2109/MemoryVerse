"""
MemoryVerse Backend — FastAPI Application Entry Point.
"""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import router as api_router
from app.core.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Startup / shutdown lifecycle handler."""
    configure_logging()
    settings = get_settings()
    logger.info("MemoryVerse API starting", env=settings.environment)
    yield
    logger.info("MemoryVerse API shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="MemoryVerse API",
        description="AI-powered personal memory journaling backend.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "service": "MemoryVerse API"}

    return app


app = create_app()
