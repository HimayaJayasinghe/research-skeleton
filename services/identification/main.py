"""
services/identification/main.py
FastAPI microservice for identity prediction and model training.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np

from .predictor import Predictor
from .trainer import ModelTrainer

app = FastAPI(title="Identification Service", version="1.0.0")

predictor = Predictor(model_dir="./models")
trainer = ModelTrainer(model_dir="./models")


# ── Request / Response Models ─────────────────────────────────────────────────

class IdentifyRequest(BaseModel):
    static_features: Optional[List[float]] = None
    gait_sequence: Optional[List[List[float]]] = None  # (seq_len, n_angles)
    top_k: int = 5


class IdentifyResponse(BaseModel):
    predicted_user: str
    confidence: float
    is_known: bool
    method: str = ""
    top_k: List[Dict[str, Any]] = []
    latency_ms: float = 0.0


class TrainSVMRequest(BaseModel):
    features: List[List[float]]
    labels: List[str]
    feature_names: Optional[List[str]] = None


class TrainLSTMRequest(BaseModel):
    sequences: List[List[List[float]]]  # (n, seq_len, n_angles)
    labels: List[str]
    epochs: int = 100
    batch_size: int = 32


class TrainResponse(BaseModel):
    success: bool
    message: str = ""
    metrics: Dict[str, Any] = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "identification",
        "svm_ready": predictor.ensemble.svm_ready,
        "lstm_ready": predictor.ensemble.lstm_ready,
    }


@app.post("/load-models")
async def load_models():
    """Load trained models from disk."""
    success = predictor.load_models()
    return {"success": success, "svm": predictor.ensemble.svm_ready, "lstm": predictor.ensemble.lstm_ready}


@app.post("/identify", response_model=IdentifyResponse)
async def identify(req: IdentifyRequest):
    """Identify a person from features."""
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded. POST /load-models first.")

    static = np.array(req.static_features) if req.static_features else None
    gait = np.array(req.gait_sequence) if req.gait_sequence else None

    result = predictor.identify(static_features=static, gait_sequence=gait, top_k=req.top_k)

    return IdentifyResponse(
        predicted_user=result["predicted_user"],
        confidence=result["confidence"],
        is_known=result["is_known"],
        method=result.get("method", ""),
        top_k=result.get("top_k", []),
        latency_ms=result.get("latency_ms", 0),
    )


@app.post("/train/svm", response_model=TrainResponse)
async def train_svm(req: TrainSVMRequest):
    """Train the SVM model."""
    X = np.array(req.features)
    y = np.array(req.labels)
    result = await trainer.train_svm(X, y, feature_names=req.feature_names)

    if result["success"]:
        predictor.load_models()

    return TrainResponse(
        success=result["success"],
        message=f"SVM trained: v{result.get('version', '?')}",
        metrics=result.get("metrics", {}),
    )


@app.post("/train/lstm", response_model=TrainResponse)
async def train_lstm(req: TrainLSTMRequest):
    """Train the LSTM model."""
    sequences = np.array(req.sequences)
    labels = np.array(req.labels)
    result = await trainer.train_lstm(
        sequences, labels, epochs=req.epochs, batch_size=req.batch_size
    )

    if result["success"]:
        predictor.load_models()

    return TrainResponse(
        success=result["success"],
        message=f"LSTM trained: v{result.get('version', '?')}",
        metrics=result.get("metrics", {}),
    )


@app.on_event("startup")
async def startup():
    """Try to load models on startup."""
    predictor.load_models()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
