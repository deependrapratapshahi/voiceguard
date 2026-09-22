"""
VOICEGUARD backend entrypoint.

This is a defensive fraud-prevention / impersonation-detection
prototype. All model outputs are probabilistic risk indicators, never
certainty claims (see docs/ml_pipeline.md).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import alerts, calls, detection, models, risk, speakers
from app.config import get_settings
from app.models.database import init_models
from app.models.schemas import HealthResponse
from app.utils.logging import configure_logging, get_logger
from app.websocket import call_stream

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} in {settings.ENV} mode")
    try:
        await init_models()
        logger.info("Database tables verified/created.")
    except Exception as exc:
        # Don't crash startup if the DB isn't reachable yet (e.g. first
        # `docker compose up` before Postgres finishes initializing) --
        # log clearly instead so /api/v1/health can report the problem.
        logger.warning(f"Database initialization skipped/failed at startup: {exc}")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title="VOICEGUARD API",
    description=(
        "AI-powered realtime voice-cloning impersonation detection and "
        "fraud-prevention API. Defensive security tool for authorized "
        "testing and monitoring only. All detector outputs are "
        "probabilistic risk indicators, not certainty claims."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calls.router)
app.include_router(detection.router)
app.include_router(speakers.router)
app.include_router(risk.router)
app.include_router(alerts.router)
app.include_router(models.router)
app.include_router(call_stream.router)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    db_status = "unknown"
    redis_status = "unknown"

    try:
        from sqlalchemy import text
        from app.models.database import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "unavailable"

    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.REDIS_URL)
        await client.ping()
        redis_status = "connected"
        await client.close()
    except Exception:
        redis_status = "unavailable"

    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        env=settings.ENV,
        database=db_status,
        redis=redis_status,
    )


@app.get("/")
async def root():
    return {"message": "VOICEGUARD API is running. See /docs for the API reference."}
