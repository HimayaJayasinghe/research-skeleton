"""
services/feature_extraction/gait_features.py
Temporal / gait feature extraction using sliding window over keypoint sequences.

Extracts ~15 temporal features:
  - Joint angular velocities (8)
  - Gait cycle frequencies (2)
  - Gait amplitudes (2)
  - Gait symmetry (1)
  - Body sway (2)
"""
import numpy as np
import structlog
from collections import deque
from typing import Dict, List, Optional
from scipy.signal import find_peaks
from scipy.fft import fft

log = structlog.get_logger()


class GaitFeatureExtractor:
    """Extracts temporal gait features from sequences of skeleton keypoints."""

    TRACKED_ANGLES = [
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "left_elbow_angle", "right_elbow_angle",
        "left_shoulder_angle", "right_shoulder_angle",
    ]

    def __init__(self, window_size: int = 30, fps: float = 30.0):
        self.window_size = window_size
        self.fps = fps
        self.keypoint_buffer: deque = deque(maxlen=window_size)
        self.angle_buffer: deque = deque(maxlen=window_size)
        self._feature_order: Optional[List[str]] = None

    def add_frame(self, keypoints: Dict[str, Dict], angles: Dict[str, float]):
        """Add one frame of data to the sliding window buffer."""
        self.keypoint_buffer.append(keypoints)
        self.angle_buffer.append(angles)

    def is_ready(self) -> bool:
        """True once the buffer is full (enough frames for gait analysis)."""
        return len(self.keypoint_buffer) >= self.window_size

    def buffer_length(self) -> int:
        return len(self.keypoint_buffer)

    def _angle_timeseries(self, angle_name: str) -> np.ndarray:
        """Get time series for a specific angle across the sliding window."""
        return np.array([
            frame.get(angle_name, 0.0) for frame in self.angle_buffer
        ])

    def compute_angular_velocities(self) -> Dict[str, float]:
        """Mean and std of angular velocity for 4 key joint pairs (8 features)."""
        dt = 1.0 / self.fps
        velocities = {}

        for angle_name in self.TRACKED_ANGLES[:4]:  # knees + hips
            ts = self._angle_timeseries(angle_name)
            diffs = np.abs(np.diff(ts)) / dt
            velocities[f"{angle_name}_vel_mean"] = float(np.mean(diffs))
            velocities[f"{angle_name}_vel_std"] = float(np.std(diffs))

        return velocities

    def compute_gait_cycle(self) -> Dict[str, float]:
        """FFT-based gait cycle frequency and amplitude (5 features)."""
        features = {}

        for side in ["left", "right"]:
            ts = self._angle_timeseries(f"{side}_knee_angle")
            ts_centered = ts - np.mean(ts)

            N = len(ts_centered)
            if N < 4:
                features[f"{side}_gait_freq"] = 0.0
                features[f"{side}_gait_amp"] = 0.0
                continue

            yf = np.abs(fft(ts_centered))[: N // 2]
            freqs = np.linspace(0, self.fps / 2, N // 2)

            # Walking frequency band: 0.5–3.0 Hz
            valid = (freqs >= 0.5) & (freqs <= 3.0)
            if np.any(valid) and np.any(yf[valid] > 0):
                dom_idx = np.argmax(yf[valid])
                features[f"{side}_gait_freq"] = float(freqs[valid][dom_idx])
                features[f"{side}_gait_amp"] = float(yf[valid][dom_idx])
            else:
                features[f"{side}_gait_freq"] = 0.0
                features[f"{side}_gait_amp"] = 0.0

        # Gait symmetry
        lf = features.get("left_gait_freq", 0.0)
        rf = features.get("right_gait_freq", 0.0)
        features["gait_symmetry"] = lf / (rf + 1e-8) if rf > 0.01 else 1.0

        return features

    def compute_body_sway(self) -> Dict[str, float]:
        """Lateral body sway from hip center movement (2 features)."""
        hip_x = []
        for kps in self.keypoint_buffer:
            lh = kps.get("left_hip", {})
            rh = kps.get("right_hip", {})
            cx = (lh.get("x", 0.5) + rh.get("x", 0.5)) / 2.0
            hip_x.append(cx)

        hip_x = np.array(hip_x)
        return {
            "body_sway_std": float(np.std(hip_x)),
            "body_sway_range": float(np.ptp(hip_x)),
        }

    def extract_all(self) -> Optional[Dict[str, float]]:
        """Extract the complete temporal feature vector (~15 features).

        Returns None if buffer is not full yet.
        """
        if not self.is_ready():
            return None

        try:
            features = {}
            features.update(self.compute_angular_velocities())  # 8
            features.update(self.compute_gait_cycle())           # 5
            features.update(self.compute_body_sway())            # 2
            return features  # 15 features total
        except Exception as e:
            log.warning("gait_feature_extraction_failed", error=str(e))
            return None

    def to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict → ordered numpy vector."""
        if self._feature_order is None:
            self._feature_order = sorted(features.keys())
        return np.array(
            [features.get(k, 0.0) for k in self._feature_order], dtype=np.float64
        )

    def get_feature_names(self) -> List[str]:
        if self._feature_order is None:
            return []
        return list(self._feature_order)

    def get_sequence_matrix(self) -> Optional[np.ndarray]:
        """Get the full angle time-series matrix for LSTM input.

        Returns:
            (window_size, num_angles) matrix, or None if not ready.
        """
        if not self.is_ready():
            return None

        rows = []
        for frame_angles in self.angle_buffer:
            row = [frame_angles.get(a, 0.0) for a in self.TRACKED_ANGLES]
            rows.append(row)

        return np.array(rows, dtype=np.float64)  # (30, 8)

    def reset(self):
        """Clear all buffers."""
        self.keypoint_buffer.clear()
        self.angle_buffer.clear()
