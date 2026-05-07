"""
database/schemas.py
Pydantic v2 models for request/response validation and MongoDB document shapes.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


# ═══════════════════════════════════════════════════════════════════════════════
#  USER SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    """Request body for creating a new user."""
    name: str = Field(..., min_length=1, max_length=100, examples=["John Doe"])
    email: Optional[str] = Field(None, examples=["john@example.com"])
    notes: Optional[str] = None


class UserResponse(BaseModel):
    """API response for user data."""
    user_id: str
    name: str
    email: Optional[str] = None
    enrollment_status: str
    enrollment_frames_count: int = 0
    created_at: datetime
    updated_at: datetime


class UserInDB(BaseModel):
    """Full user document as stored in MongoDB."""
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    enrollment_status: str = "pending"  # pending | in_progress | completed | failed
    enrollment_frames_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE PROFILE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureProfileCreate(BaseModel):
    """Request body for creating/updating feature profiles."""
    user_id: str
    static_features: List[float]
    gait_features: Optional[List[float]] = None


class FeatureProfileInDB(BaseModel):
    """Feature profile document in MongoDB."""
    user_id: str
    static_features: Dict[str, Any] = Field(default_factory=lambda: {
        "mean_vector": [],
        "std_vector": [],
        "samples": [],
    })
    gait_features: Dict[str, Any] = Field(default_factory=lambda: {
        "mean_vector": [],
        "std_vector": [],
        "samples": [],
    })
    sample_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTIFICATION SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class IdentificationResult(BaseModel):
    """Result of an identification attempt."""
    predicted_user: str
    confidence: float
    is_known: bool
    svm_prediction: Optional[Dict[str, Any]] = None
    lstm_prediction: Optional[Dict[str, Any]] = None
    top_k: List[Dict[str, Any]] = []
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IdentificationLog(BaseModel):
    """Log entry for identification attempts (stored in MongoDB)."""
    model_config = {"protected_namespaces": ()}
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    predicted_user_id: Optional[str] = None
    confidence: float = 0.0
    svm_confidence: float = 0.0
    lstm_confidence: float = 0.0
    feature_vector: List[float] = []
    model_version: str = ""
    latency_ms: float = 0.0
    was_correct: Optional[bool] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  KEYPOINT / POSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class KeypointData(BaseModel):
    """Single frame keypoint data."""
    keypoints: List[Dict[str, float]]
    frame_index: int = 0
    timestamp: float = 0.0


class PoseEstimationResult(BaseModel):
    """Result from pose estimation service."""
    detected: bool
    keypoints: Optional[List[Dict[str, float]]] = None
    body_keypoints: Optional[Dict[str, Dict[str, float]]] = None
    num_visible: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class TrainedModelRecord(BaseModel):
    """Record of a trained model stored in MongoDB."""
    model_config = {"protected_namespaces": ()}

    model_type: str  # svm | lstm | ensemble
    version: str
    trained_at: datetime = Field(default_factory=datetime.utcnow)
    num_classes: int = 0
    accuracy: float = 0.0
    f1_score: float = 0.0
    model_path: str = ""
    hyperparameters: Dict[str, Any] = {}
    is_active: bool = False
    metrics: Dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    """Request to trigger model training."""
    model_config = {"protected_namespaces": ()}

    model_type: str = "ensemble"  # svm | lstm | ensemble
    force_retrain: bool = False


class TrainResponse(BaseModel):
    """Response after model training."""
    model_config = {"protected_namespaces": ()}

    success: bool
    model_type: str
    version: str
    metrics: Dict[str, Any] = {}
    message: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  ENROLLMENT SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class EnrollmentStatus(BaseModel):
    """Enrollment progress status."""
    user_id: str
    name: str
    status: str
    frames_collected: int
    frames_required: int
    progress_percent: float
    message: str = ""
