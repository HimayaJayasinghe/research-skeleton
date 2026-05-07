"""
services/feature_extraction/main.py
FastAPI microservice for feature extraction from skeleton keypoints.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import numpy as np

from .static_features import StaticFeatureExtractor
from .gait_features import GaitFeatureExtractor

app = FastAPI(title="Feature Extraction Service", version="1.0.0")

static_extractor = StaticFeatureExtractor()
# Per-session gait extractors (in production, key by session/person ID)
gait_extractors: Dict[str, GaitFeatureExtractor] = {}


class StaticFeatureRequest(BaseModel):
    body_keypoints: Dict[str, Dict]


class StaticFeatureResponse(BaseModel):
    success: bool
    features: Optional[Dict[str, float]] = None
    feature_vector: Optional[List[float]] = None
    num_features: int = 0


class GaitFeatureRequest(BaseModel):
    session_id: str = "default"
    body_keypoints: Dict[str, Dict]
    angles: Dict[str, float]


class GaitFeatureResponse(BaseModel):
    success: bool
    ready: bool = False
    buffer_length: int = 0
    features: Optional[Dict[str, float]] = None
    feature_vector: Optional[List[float]] = None
    sequence_matrix: Optional[List[List[float]]] = None


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "feature_extraction"}


@app.post("/extract-static", response_model=StaticFeatureResponse)
async def extract_static(req: StaticFeatureRequest):
    """Extract static skeletal features from body keypoints."""
    features = static_extractor.extract_all(req.body_keypoints)
    if features is None:
        return StaticFeatureResponse(success=False)

    vector = static_extractor.to_vector(features).tolist()
    return StaticFeatureResponse(
        success=True,
        features=features,
        feature_vector=vector,
        num_features=len(features),
    )


@app.post("/extract-gait", response_model=GaitFeatureResponse)
async def extract_gait(req: GaitFeatureRequest):
    """Add frame to gait buffer and extract temporal features when ready."""
    # Get or create gait extractor for this session
    if req.session_id not in gait_extractors:
        gait_extractors[req.session_id] = GaitFeatureExtractor(
            window_size=30, fps=30.0
        )

    gait = gait_extractors[req.session_id]
    gait.add_frame(req.body_keypoints, req.angles)

    if not gait.is_ready():
        return GaitFeatureResponse(
            success=True,
            ready=False,
            buffer_length=gait.buffer_length(),
        )

    features = gait.extract_all()
    if features is None:
        return GaitFeatureResponse(success=False, ready=True)

    vector = gait.to_vector(features).tolist()
    seq_matrix = gait.get_sequence_matrix()

    return GaitFeatureResponse(
        success=True,
        ready=True,
        buffer_length=gait.buffer_length(),
        features=features,
        feature_vector=vector,
        sequence_matrix=seq_matrix.tolist() if seq_matrix is not None else None,
    )


@app.post("/reset-gait/{session_id}")
async def reset_gait(session_id: str):
    """Reset gait buffer for a session."""
    if session_id in gait_extractors:
        gait_extractors[session_id].reset()
        return {"message": f"Gait buffer reset for session {session_id}"}
    return {"message": f"No gait buffer found for session {session_id}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
