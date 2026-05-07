"""
services/pose_estimation/estimator.py
Extracts 33 skeleton keypoints using MediaPipe Pose.
"""
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
except ImportError:
    import mediapipe as mp
    mp_pose, mp_drawing, mp_drawing_styles = None, None, None
import numpy as np
import structlog
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
    """MediaPipe Pose wrapper for skeleton keypoint extraction."""

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

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp_pose or mp.solutions.pose
        self.mp_drawing = mp_drawing or mp.solutions.drawing_utils
        self.mp_drawing_styles = mp_drawing_styles or mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        log.info(
            "pose_estimator_initialized",
            complexity=model_complexity,
            det_conf=min_detection_confidence,
        )

    def estimate(self, rgb_frame: np.ndarray) -> Optional[List[Dict]]:
        """Extract all 33 keypoints from an RGB frame.

        Args:
            rgb_frame: RGB image as numpy array (H, W, 3)

        Returns:
            List of 33 keypoint dicts, or None if no person detected.
        """
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks is None:
            return None

        keypoints = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            kp = Keypoint(
                index=idx,
                name=self.LANDMARK_NAMES[idx],
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

        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 200, 255), thickness=2
                ),
            )
        return annotated

    def draw_on_frame_with_results(
        self, bgr_frame: np.ndarray, all_keypoints: List[Dict]
    ) -> np.ndarray:
        """Draw keypoints on frame without re-running pose estimation."""
        annotated = bgr_frame.copy()
        h, w = annotated.shape[:2]

        # Draw connections first (behind dots)
        import cv2
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
        self.pose.close()
        log.info("pose_estimator_closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
