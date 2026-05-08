"""
config.py
Centralized configuration using pydantic-settings.
Reads from .env file and environment variables.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application-wide settings loaded from .env"""

    # ── MongoDB ───────────────────────────────────────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "skeleton_id"
    use_local_db: bool = True
    local_db_path: str = "./data/local_db.json"

    # ── Service Ports ─────────────────────────────────────────────────────────
    gateway_port: int = 8000
    video_service_port: int = 8001
    pose_service_port: int = 8002
    feature_service_port: int = 8003
    identification_service_port: int = 8004

    # ── Model ─────────────────────────────────────────────────────────────────
    model_dir: str = "./models"
    confidence_threshold: float = 0.50
    svm_weight: float = 0.5
    lstm_weight: float = 0.5

    # ── Video ─────────────────────────────────────────────────────────────────
    camera_index: int = 0
    camera_url: str = ""  # IP camera URL (e.g., http://192.168.1.5:8080/video)
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    skip_frames: int = 1

    # ── IP Camera (Hikvision / ONVIF) ─────────────────────────────────────────
    ip_camera_host: str = ""
    ip_camera_user: str = "admin"
    ip_camera_pass: str = "admin"
    ip_camera_rtsp_url: str = ""
    ip_camera_snapshot_url: str = ""

    # ── Pose Estimation ───────────────────────────────────────────────────────
    mediapipe_model_complexity: int = 0  # 0=fastest, 1=balanced, 2=most accurate
    min_detection_confidence: float = 0.3
    min_tracking_confidence: float = 0.3

    # ── LSTM ──────────────────────────────────────────────────────────────────
    lstm_sequence_length: int = 30
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 0.001

    # ── Enrollment ────────────────────────────────────────────────────────────
    enrollment_duration_seconds: int = 60
    min_enrollment_frames: int = 50

    # ── Service URLs (computed) ───────────────────────────────────────────────
    @property
    def video_service_url(self) -> str:
        return f"http://localhost:{self.video_service_port}"

    @property
    def pose_service_url(self) -> str:
        return f"http://localhost:{self.pose_service_port}"

    @property
    def feature_service_url(self) -> str:
        return f"http://localhost:{self.feature_service_port}"

    @property
    def identification_service_url(self) -> str:
        return f"http://localhost:{self.identification_service_port}"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton
settings = Settings()
