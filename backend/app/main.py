from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from app.config.settings import settings
from app.api import auth, profile, vaults, memories, media, invites, notifications, search, ai
from app.api import timeline
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="MemoryVerse API — Production Backend with Supabase Integration",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": {"code": str(exc.status_code), "message": exc.detail}},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all: handles postgrest APIError, database errors, etc.
    Logs the real error server-side but returns a clean 500 to the client.
    PGRST205 (table not found) and PGRST204 (column not found) are treated as
    temporary schema-setup errors and return appropriate empty responses.
    """
    err_str = str(exc)

    # PostgREST schema cache errors — DB tables not yet created
    if "PGRST205" in err_str or "PGRST204" in err_str or "schema cache" in err_str:
        logger.warning(f"Schema not yet set up for {request.method} {request.url.path}: {err_str}")
        path = request.url.path

        # Return sensible empty responses per endpoint type so the app
        # doesn't crash while waiting for the DB schema to be applied.
        if path.endswith("/media") and request.method == "GET":
            return JSONResponse(status_code=200, content=[])
        if path.endswith("/vaults") and request.method == "GET":
            return JSONResponse(status_code=200, content=[])
        if path.endswith("/memories") and request.method == "GET":
            return JSONResponse(status_code=200, content=[])
        if path.endswith("/timeline") and request.method == "GET":
            return JSONResponse(status_code=200, content={"groups": [], "total_items": 0})
        if path.endswith("/notifications") and request.method == "GET":
            return JSONResponse(status_code=200, content=[])
        if path.endswith("/profile") and request.method == "GET":
            # Return a minimal profile so the app can render
            return JSONResponse(status_code=200, content={
                "id": "unknown",
                "email": "setup@required.com",
                "full_name": "Setup Required",
                "username": None,
                "avatar_url": None,
                "bio": "⚠️ Run supabase_schema.sql in your Supabase dashboard.",
                "vault_count": 0,
                "media_count": 0,
                "created_at": None,
            })

        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": {
                    "code": "DB_SCHEMA_MISSING",
                    "message": "Database tables not found. Please run supabase_schema.sql in your Supabase SQL Editor.",
                },
            },
        )

    # All other unhandled errors — return generic 500
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred. Please try again.",
            },
        },
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,          prefix="/api/v1")
app.include_router(profile.router,       prefix="/api/v1")
app.include_router(vaults.router,        prefix="/api/v1")
app.include_router(memories.router,      prefix="/api/v1")
app.include_router(media.router,         prefix="/api/v1")
app.include_router(timeline.router,      prefix="/api/v1")
app.include_router(invites.router,       prefix="/api/v1")
app.include_router(notifications.router, prefix="/api/v1")
app.include_router(search.router,        prefix="/api/v1")
app.include_router(ai.router,            prefix="/api/v1")


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
