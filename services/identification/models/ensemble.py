"""
services/identification/models/ensemble.py
Ensemble identifier combining SVM (static features) and LSTM (gait sequences).

Fusion strategy:
  final_confidence = svm_weight * svm_conf + lstm_weight * lstm_conf
  predicted_user = argmax(final_confidence) across all candidates
"""
import numpy as np
import structlog
from typing import Dict, Any, Optional, List
from .svm_model import SkeletonSVM
from .lstm_model import SkeletonLSTM

log = structlog.get_logger()


class EnsembleIdentifier:
    """Fuses SVM (static) and LSTM (gait) predictions for robust identification."""

    def __init__(
        self,
        svm_weight: float = 0.5,
        lstm_weight: float = 0.5,
        confidence_threshold: float = 0.65,
    ):
        self.svm = SkeletonSVM()
        self.lstm = SkeletonLSTM()
        self.svm_weight = svm_weight
        self.lstm_weight = lstm_weight
        self.confidence_threshold = confidence_threshold

    @property
    def svm_ready(self) -> bool:
        return self.svm.is_trained

    @property
    def lstm_ready(self) -> bool:
        return self.lstm.is_trained

    @property
    def is_trained(self) -> bool:
        return self.svm_ready  # SVM is minimum requirement

    def predict(
        self,
        static_features: Optional[np.ndarray] = None,
        gait_sequence: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Combined prediction using available models.

        Args:
            static_features: (n_features,) for SVM — e.g. (42,)
            gait_sequence: (seq_len, n_angles) for LSTM — e.g. (30, 8)
            top_k: Number of top candidates to return

        Returns:
            Dict with ensemble prediction results
        """
        svm_result = None
        lstm_result = None

        # ── SVM prediction (static features) ─────────────────────────────────
        if static_features is not None and self.svm_ready:
            try:
                svm_result = self.svm.predict(static_features, top_k=top_k)
            except Exception as e:
                log.warning("svm_prediction_failed", error=str(e))

        # ── LSTM prediction (gait sequence) ───────────────────────────────────
        if gait_sequence is not None and self.lstm_ready:
            try:
                lstm_result = self.lstm.predict(gait_sequence, top_k=top_k)
            except Exception as e:
                log.warning("lstm_prediction_failed", error=str(e))

        # ── Fusion ────────────────────────────────────────────────────────────
        return self._fuse(svm_result, lstm_result, top_k)

    def _fuse(
        self,
        svm_result: Optional[Dict],
        lstm_result: Optional[Dict],
        top_k: int,
    ) -> Dict[str, Any]:
        """Weighted fusion of SVM and LSTM predictions."""

        # Case 1: Both models available
        if svm_result and lstm_result:
            return self._weighted_fusion(svm_result, lstm_result, top_k)

        # Case 2: SVM only
        if svm_result:
            best = svm_result["top_k"][0]
            is_known = best["confidence"] >= self.confidence_threshold
            return {
                "predicted_user": best["user_id"] if is_known else "unknown",
                "confidence": best["confidence"],
                "is_known": is_known,
                "method": "svm_only",
                "svm_prediction": svm_result,
                "lstm_prediction": None,
                "top_k": svm_result["top_k"][:top_k],
            }

        # Case 3: LSTM only
        if lstm_result:
            best = lstm_result["top_k"][0]
            is_known = best["confidence"] >= self.confidence_threshold
            return {
                "predicted_user": best["user_id"] if is_known else "unknown",
                "confidence": best["confidence"],
                "is_known": is_known,
                "method": "lstm_only",
                "svm_prediction": None,
                "lstm_prediction": lstm_result,
                "top_k": lstm_result["top_k"][:top_k],
            }

        # Case 4: No model available
        return {
            "predicted_user": "unknown",
            "confidence": 0.0,
            "is_known": False,
            "method": "none",
            "svm_prediction": None,
            "lstm_prediction": None,
            "top_k": [],
        }

    def _weighted_fusion(
        self,
        svm_result: Dict,
        lstm_result: Dict,
        top_k: int,
    ) -> Dict[str, Any]:
        """Combine SVM and LSTM scores with weighted averaging."""
        # Collect all candidate user IDs from both models
        all_users = set()
        svm_scores = {}
        lstm_scores = {}

        for c in svm_result.get("top_k", []):
            uid = c["user_id"]
            all_users.add(uid)
            svm_scores[uid] = c["confidence"]

        for c in lstm_result.get("top_k", []):
            uid = c["user_id"]
            all_users.add(uid)
            lstm_scores[uid] = c["confidence"]

        # Compute fused scores
        fused = []
        for uid in all_users:
            s = svm_scores.get(uid, 0.0)
            l = lstm_scores.get(uid, 0.0)
            combined = self.svm_weight * s + self.lstm_weight * l
            fused.append({
                "user_id": uid,
                "confidence": combined,
                "svm_score": s,
                "lstm_score": l,
            })

        # Sort by combined confidence
        fused.sort(key=lambda x: x["confidence"], reverse=True)
        fused = fused[:top_k]

        best = fused[0] if fused else {"user_id": "unknown", "confidence": 0.0}
        is_known = best["confidence"] >= self.confidence_threshold

        return {
            "predicted_user": best["user_id"] if is_known else "unknown",
            "confidence": best["confidence"],
            "is_known": is_known,
            "method": "ensemble",
            "svm_prediction": svm_result,
            "lstm_prediction": lstm_result,
            "top_k": fused,
        }

    def save(self, directory: str):
        """Save both models."""
        if self.svm_ready:
            self.svm.save(directory)
        if self.lstm_ready:
            self.lstm.save(directory)
        log.info("ensemble_saved", path=directory)

    def load(self, directory: str):
        """Load both models (graceful if one is missing)."""
        from pathlib import Path
        path = Path(directory)

        if (path / "svm_model.pkl").exists():
            self.svm.load(directory)

        if (path / "lstm_model.pt").exists():
            self.lstm.load(directory)

        log.info(
            "ensemble_loaded",
            svm_ready=self.svm_ready,
            lstm_ready=self.lstm_ready,
        )
