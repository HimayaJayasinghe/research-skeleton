"""
gateway/main.py
API Gateway — Single entry point for the Skeleton Identification System.

Serves:
  - REST API endpoints (/api/*)
  - WebSocket for real-time streaming (/ws/stream)
  - Static files for the web dashboard (/dashboard)
"""
import os
import sys
import structlog
import warnings

# Suppress protobuf deprecation warnings coming from mediapipe/tensorflow
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype() is deprecated.*")

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from database.connection import MongoDB
from services.identification.predictor import Predictor
from services.identification.trainer import ModelTrainer
from gateway.routes import users, identification, stream

log = structlog.get_logger()

# ── Shared instances ──────────────────────────────────────────────────────────
predictor = Predictor(
    model_dir=settings.model_dir,
    svm_weight=settings.svm_weight,
    lstm_weight=settings.lstm_weight,
    confidence_threshold=settings.confidence_threshold,
)
trainer = ModelTrainer(model_dir=settings.model_dir)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    log.info("gateway_starting", port=settings.gateway_port)

    # Connect to MongoDB
    await MongoDB.connect(settings.mongodb_uri, settings.mongodb_db)

    # Load trained models (if available)
    predictor.load_models()

    # Share predictor with route modules
    identification.init_predictor(predictor, trainer)
    stream.set_predictor(predictor)

    log.info(
        "gateway_ready",
        svm=predictor.ensemble.svm_ready,
        lstm=predictor.ensemble.lstm_ready,
    )
    yield

    # Shutdown
    await MongoDB.close()
    log.info("gateway_stopped")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Skeleton-Based Person Identification System",
    description="Real-time person identification using skeletal bone structure and gait patterns",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(users.router)
app.include_router(identification.router)
app.include_router(stream.router)


# ── Dashboard static files ───────────────────────────────────────────────────
dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard/assets", StaticFiles(directory=str(dashboard_dir)), name="dashboard_assets")


@app.get("/")
async def root():
    """Serve the dashboard."""
    index = dashboard_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {
        "service": "Skeleton ID Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/dashboard",
    }


@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard HTML."""
    index = dashboard_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"error": "Dashboard not found"}


@app.get("/health")
async def health():
    """System-wide health check."""
    db_ok = await MongoDB.is_connected()
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "models": {
            "svm": predictor.ensemble.svm_ready,
            "lstm": predictor.ensemble.lstm_ready,
        },
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gateway.main:app",
        host="0.0.0.0",
        port=settings.gateway_port,
        reload=True,
        log_level="info",
    )
