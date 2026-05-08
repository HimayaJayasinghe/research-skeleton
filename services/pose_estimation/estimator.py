"""
services/pose_estimation/estimator.py
Extracts 33 skeleton keypoints using MediaPipe Pose (Tasks API).

Updated from legacy mp.solutions.pose to mediapipe.tasks.python.vision.PoseLandmarker
which is required for mediapipe >= 0.10.14.
"""
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
import numpy as np
import cv2
import structlog
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

log = structlog.get_logger()


@dataclass
class Keypoint:
    """Single skeleton keypoint with 3D coordinates and visibility."""
    index: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


class PoseEstimator:
    """MediaPipe PoseLandmarker wrapper for skeleton keypoint extraction.

    Uses the new Tasks API (mediapipe >= 0.10.14).
    """

    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]

    # Key body landmarks for identification (indices into the 33 landmarks)
    BODY_LANDMARK_INDICES = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
    }

    # Map model_complexity (0, 1, 2) to task model files
    _MODEL_FILES = {
        0: "pose_landmarker_lite.task",
        1: "pose_landmarker_full.task",
        2: "pose_landmarker_heavy.task",
    }

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        # Resolve model path
        model_dir = Path(__file__).resolve().parent.parent.parent / "models"
        model_file = self._MODEL_FILES.get(model_complexity, self._MODEL_FILES[0])
        model_path = model_dir / model_file

        if not model_path.exists():
            # Fall back to whichever model exists
            for complexity in [0, 1, 2]:
                fallback = model_dir / self._MODEL_FILES[complexity]
                if fallback.exists():
                    model_path = fallback
                    log.warning(
                        "pose_model_fallback",
                        requested=model_file,
                        using=fallback.name,
                    )
                    break
            else:
                raise FileNotFoundError(
                    f"No pose landmarker model found in {model_dir}. "
                    f"Download from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models"
                )

        running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=min_detection_confidence,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)
        self._running_mode = running_mode
        self._static = static_image_mode
        self._frame_timestamp_ms = 0

        log.info(
            "pose_estimator_initialized",
            model=model_path.name,
            complexity=model_complexity,
            det_conf=min_detection_confidence,
            mode=running_mode.name,
        )

    def estimate(self, rgb_frame: np.ndarray) -> Optional[List[Dict]]:
        """Extract all 33 keypoints from an RGB frame.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3)

        Returns:
            List of 33 keypoint dicts, or None if no person detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self._static:
            results = self.landmarker.detect(mp_image)
        else:
            self._frame_timestamp_ms += 33  # ~30fps
            results = self.landmarker.detect_for_video(
                mp_image, self._frame_timestamp_ms
            )

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        # Take the first detected pose
        landmarks = results.pose_landmarks[0]

        keypoints = []
        for idx, landmark in enumerate(landmarks):
            kp = Keypoint(
                index=idx,
                name=self.LANDMARK_NAMES[idx] if idx < len(self.LANDMARK_NAMES) else f"landmark_{idx}",
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=float(landmark.visibility),
            )
            keypoints.append(asdict(kp))

        return keypoints

    def get_body_keypoints(
        self, all_keypoints: List[Dict], min_visibility: float = 0.3
    ) -> Optional[Dict[str, Dict]]:
        """Extract the 12 body landmarks required for identification.

        Returns None if too many critical landmarks are not visible.
        """
        if all_keypoints is None:
            return None

        body_kps = {}
        low_visibility_count = 0

        for name, idx in self.BODY_LANDMARK_INDICES.items():
            kp = all_keypoints[idx]
            if kp["visibility"] < min_visibility:
                low_visibility_count += 1
            body_kps[name] = kp

        # Reject if more than 10 of 12 body landmarks have low visibility
        # (was 8 — still too strict for some close-up shots)
        if low_visibility_count > 10:
            return None

        return body_kps

    def get_all_keypoints_dict(
        self, all_keypoints: List[Dict]
    ) -> Dict[str, Dict]:
        """Convert list of keypoints to name-indexed dictionary."""
        return {kp["name"]: kp for kp in all_keypoints}

    def draw_skeleton(
        self, bgr_frame: np.ndarray, rgb_frame: np.ndarray,
        cached_keypoints: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Draw skeleton overlay on BGR frame for visualization.

        If cached_keypoints is provided, draws without re-running pose estimation.
        """
        annotated = bgr_frame.copy()

        if cached_keypoints is not None:
            # Use cached keypoints to avoid running pose estimation again
            return self.draw_on_frame_with_results(annotated, cached_keypoints)

        # Re-run estimation and draw
        all_kps = self.estimate(rgb_frame)
        if all_kps:
            return self.draw_on_frame_with_results(annotated, all_kps)

        return annotated

    def draw_on_frame_with_results(
        self, bgr_frame: np.ndarray, all_keypoints: List[Dict]
    ) -> np.ndarray:
        """Draw keypoints on frame without re-running pose estimation."""
        annotated = bgr_frame.copy()
        h, w = annotated.shape[:2]

        # Draw connections first (behind dots)
        connections = [
            (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 12), (23, 24),  # Shoulders, Hips
            (11, 23), (12, 24),  # Torso
            (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
            (0, 11), (0, 12),  # Head to shoulders
        ]
        for start, end in connections:
            if start < len(all_keypoints) and end < len(all_keypoints):
                kp1, kp2 = all_keypoints[start], all_keypoints[end]
                if kp1["visibility"] > 0.2 and kp2["visibility"] > 0.2:
                    p1 = (int(kp1["x"] * w), int(kp1["y"] * h))
                    p2 = (int(kp2["x"] * w), int(kp2["y"] * h))
                    cv2.line(annotated, p1, p2, (0, 200, 255), 2)

        # Draw keypoints
        for kp in all_keypoints:
            if kp["visibility"] > 0.2:
                cx = int(kp["x"] * w)
                cy = int(kp["y"] * h)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        return annotated

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
        log.info("pose_estimator_closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
