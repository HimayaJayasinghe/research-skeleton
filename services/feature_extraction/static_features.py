"""
services/feature_extraction/static_features.py
Compute scale-invariant static skeletal features:
  - Limb lengths (12 raw + 12 normalized = 24)
  - Joint angles (8)
  - Body proportions (6)
  - Derived features (2)
  Total: ~42 static features
"""
import numpy as np
import structlog
from typing import Dict, List, Optional

log = structlog.get_logger()


class StaticFeatureExtractor:
    """Extracts scale-invariant biometric features from skeleton keypoints."""

    # ── Limb definitions: (start_landmark, end_landmark) ──────────────────────
    LIMB_DEFINITIONS = {
        "left_upper_arm": ("left_shoulder", "left_elbow"),
        "left_forearm": ("left_elbow", "left_wrist"),
        "right_upper_arm": ("right_shoulder", "right_elbow"),
        "right_forearm": ("right_elbow", "right_wrist"),
        "shoulder_width": ("left_shoulder", "right_shoulder"),
        "left_torso": ("left_shoulder", "left_hip"),
        "right_torso": ("right_shoulder", "right_hip"),
        "hip_width": ("left_hip", "right_hip"),
        "left_thigh": ("left_hip", "left_knee"),
        "left_shin": ("left_knee", "left_ankle"),
        "right_thigh": ("right_hip", "right_knee"),
        "right_shin": ("right_knee", "right_ankle"),
    }

    # ── Angle definitions: (point_a, vertex, point_b) ────────────────────────
    ANGLE_DEFINITIONS = {
        "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
        "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
        "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
        "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
        "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
        "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    }

    # Consistent feature ordering for vector conversion
    _feature_order: Optional[List[str]] = None

    @staticmethod
    def _get_coords(keypoints: Dict[str, Dict], name: str) -> np.ndarray:
        """Extract (x, y, z) coordinates from a named keypoint."""
        kp = keypoints[name]
        return np.array([kp["x"], kp["y"], kp["z"]], dtype=np.float64)

    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """3D Euclidean distance."""
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def _angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
        """Angle at vertex between vectors (vertex→a) and (vertex→b) in degrees."""
        v1 = a - vertex
        v2 = b - vertex
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    def compute_limb_lengths(self, kps: Dict[str, Dict]) -> Dict[str, float]:
        """Compute raw limb lengths. Returns only those that are visible."""
        lengths = {}
        for name, (start, end) in self.LIMB_DEFINITIONS.items():
            if start in kps and end in kps:
                p1 = self._get_coords(kps, start)
                p2 = self._get_coords(kps, end)
                lengths[name] = self._distance(p1, p2)
        return lengths

    def compute_joint_angles(self, kps: Dict[str, Dict]) -> Dict[str, float]:
        """Compute all 8 joint angles in degrees."""
        angles = {}
        for name, (a, v, b) in self.ANGLE_DEFINITIONS.items():
            pa = self._get_coords(kps, a)
            pv = self._get_coords(kps, v)
            pb = self._get_coords(kps, b)
            angles[name] = self._angle(pa, pv, pb)
        return angles

    def compute_torso_length(self, kps: Dict[str, Dict]) -> float:
        """Average of left and right torso side lengths."""
        left = self._distance(
            self._get_coords(kps, "left_shoulder"),
            self._get_coords(kps, "left_hip"),
        )
        right = self._distance(
            self._get_coords(kps, "right_shoulder"),
            self._get_coords(kps, "right_hip"),
        )
        return (left + right) / 2.0

    def compute_body_proportions(
        self, kps: Dict[str, Dict], limbs: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute 6 scale-invariant body proportion ratios."""
        torso = self.compute_torso_length(kps)
        if torso < 1e-6:
            return {}

        left_leg = limbs["left_thigh"] + limbs["left_shin"]
        right_leg = limbs["right_thigh"] + limbs["right_shin"]
        avg_leg = (left_leg + right_leg) / 2.0

        left_arm = limbs["left_upper_arm"] + limbs["left_forearm"]
        right_arm = limbs["right_upper_arm"] + limbs["right_forearm"]
        avg_arm = (left_arm + right_arm) / 2.0

        return {
            "torso_to_leg_ratio": torso / (avg_leg + 1e-8),
            "arm_to_torso_ratio": avg_arm / (torso + 1e-8),
            "shoulder_to_hip_ratio": limbs["shoulder_width"] / (limbs["hip_width"] + 1e-8),
            "left_right_arm_symmetry": left_arm / (right_arm + 1e-8),
            "left_right_leg_symmetry": left_leg / (right_leg + 1e-8),
            "upper_to_lower_body_ratio": (torso + avg_arm) / (avg_leg + 1e-8),
        }

    def normalize_lengths(
        self, limbs: Dict[str, float], torso: float
    ) -> Dict[str, float]:
        """Normalize all limb lengths by torso length for scale invariance."""
        if torso < 1e-6:
            return {f"{k}_norm": 0.0 for k in limbs}
        return {f"{k}_norm": v / torso for k, v in limbs.items()}

    def extract_all(self, kps: Dict[str, Dict]) -> Optional[Dict[str, float]]:
        """Extract the complete 42-dimensional static feature dictionary.

        Returns None if extraction fails (missing keypoints, etc.).
        """
        try:
            # Best-effort extraction: if a limb is missing, we use 0.0 instead of failing
            def get_len(l_map, key, default=0.0):
                return l_map.get(key, default)

            limbs = self.compute_limb_lengths(kps)          # 12
            angles = self.compute_joint_angles(kps)          # 8
            
            # Robust proportions
            torso = self.compute_torso_length(kps)
            proportions = self.compute_body_proportions(kps, limbs)  # 6
            normalized = self.normalize_lengths(limbs, torso)  # 12

            features = {}
            for k in self.LIMB_DEFINITIONS: features[k] = limbs.get(k, 0.0)
            for k in self.ANGLE_DEFINITIONS: features[k] = angles.get(k, 0.0)
            features.update(proportions)
            features.update(normalized)

            # Derived features
            avg_leg = (get_len(limbs, "left_thigh") + get_len(limbs, "left_shin") + 
                       get_len(limbs, "right_thigh") + get_len(limbs, "right_shin")) / 2.0
            features["torso_length"] = torso
            features["height_estimate"] = torso + avg_leg

            # Fill missing required proportions with defaults
            for k in ["torso_to_leg_ratio", "arm_to_torso_ratio", "shoulder_to_hip_ratio"]:
                if k not in features: features[k] = 1.0

            return features  # 42 features total

        except Exception as e:
            log.warning("static_feature_extraction_best_effort_failed", error=str(e))
            return None

    def to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict → ordered numpy vector (consistent ordering)."""
        if self._feature_order is None:
            self._feature_order = sorted(features.keys())
        return np.array(
            [features.get(k, 0.0) for k in self._feature_order], dtype=np.float64
        )

    def get_feature_names(self) -> List[str]:
        """Return the ordered feature names matching to_vector() output."""
        if self._feature_order is None:
            # Generate from a dummy extraction to establish order
            return sorted(self.LIMB_DEFINITIONS.keys())
        return list(self._feature_order)

    @staticmethod
    def smooth_features(
        current: Dict[str, float],
        previous: Optional[Dict[str, float]],
        alpha: float = 0.3,
    ) -> Dict[str, float]:
        """Exponential moving average smoothing across frames."""
        if previous is None:
            return current
        smoothed = {}
        for k in current:
            if k in previous:
                smoothed[k] = alpha * current[k] + (1 - alpha) * previous[k]
            else:
                smoothed[k] = current[k]
        return smoothed
