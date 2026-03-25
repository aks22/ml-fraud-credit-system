"""
src/api/main.py
----------------
FastAPI application factory and startup configuration.

This is the entry point for the API server.

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Run via Docker:
    docker compose up
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.model_loader import model_store
from src.api.database import create_tables
from src.utils.logger import logger

# Prometheus instrumentation for /metrics endpoint
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-fastapi-instrumentator not installed — /metrics endpoint disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — runs startup and shutdown logic.

    Startup:
      1. Create database tables (idempotent — safe to run multiple times)
      2. Load ML models into memory

    Shutdown:
      - Graceful cleanup (logs farewell message)
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=== API Server Starting ===")

    # Create DB tables (won't fail if DB is unavailable — logs warning instead)
    try:
        create_tables()
    except Exception as e:
        logger.warning(f"Database initialisation failed: {e}. Prediction logging disabled.")

    # Load models into the global singleton
    logger.info("Loading ML models...")
    model_store.load_all()

    if model_store.fully_ready:
        logger.info("All models loaded. API is ready.")
    elif model_store.fraud_ready or model_store.credit_ready:
        logger.warning("Only some models loaded. Some endpoints will return 503.")
    else:
        logger.error(
            "No models loaded! "
            "Run 'python src/models/train.py' to train and save models first."
        )

    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("=== API Server Shutting Down ===")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="ML Fraud & Credit Risk System",
        description=(
            "Production-grade ML API combining fraud detection and credit risk scoring. "
            "Built with XGBoost, Random Forest, FastAPI, and MLflow. "
            "Part of an AI engineering portfolio demonstrating end-to-end ML deployment."
        ),
        version="1.0.0",
        contact={
            "name": "AI Engineering Portfolio",
            "url": "https://github.com/your-username/ml-fraud-credit-system",
        },
        lifespan=lifespan,
        docs_url="/docs",       # Swagger UI
        redoc_url="/redoc",     # ReDoc UI
        openapi_url="/openapi.json",
    )

    # CORS — allow all origins in development (restrict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(router)

    # Prometheus metrics endpoint at /metrics
    if PROMETHEUS_AVAILABLE:
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        logger.info("Prometheus metrics exposed at /metrics")

    return app


# Application instance (imported by uvicorn)
app = create_app()


if __name__ == "__main__":
    import uvicorn
    from config.settings import PORT
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=PORT, reload=True)
