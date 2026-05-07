"""
services/identification/predictor.py
Real-time prediction pipeline using the ensemble model.
"""
import numpy as np
import time
import structlog
from pathlib import Path
from typing import Dict, Any, Optional

from .models.ensemble import EnsembleIdentifier

log = structlog.get_logger()


class Predictor:
    """Real-time identification using the loaded ensemble model."""

    def __init__(
        self,
        model_dir: str = "./models",
        svm_weight: float = 0.5,
        lstm_weight: float = 0.5,
        confidence_threshold: float = 0.65,
    ):
        self.model_dir = model_dir
        self.ensemble = EnsembleIdentifier(
            svm_weight=svm_weight,
            lstm_weight=lstm_weight,
            confidence_threshold=confidence_threshold,
        )
        self._loaded = False

    def load_models(self) -> bool:
        """Load trained models from disk."""
        path = Path(self.model_dir)
        if not path.exists():
            log.warning("model_dir_not_found", path=str(path))
            return False

        try:
            self.ensemble.load(self.model_dir)
            self._loaded = self.ensemble.is_trained
            log.info(
                "models_loaded",
                svm=self.ensemble.svm_ready,
                lstm=self.ensemble.lstm_ready,
            )
            return self._loaded
        except Exception as e:
            log.error("model_load_failed", error=str(e))
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.ensemble.is_trained

    def identify(
        self,
        static_features: Optional[np.ndarray] = None,
        gait_sequence: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run identification.

        Args:
            static_features: (42,) static feature vector
            gait_sequence: (30, 8) angle time series for LSTM

        Returns:
            Identification result dict
        """
        t_start = time.perf_counter()

        if not self.is_ready:
            return {
                "predicted_user": "unknown",
                "confidence": 0.0,
                "is_known": False,
                "method": "none",
                "error": "Models not loaded",
                "latency_ms": 0.0,
            }

        result = self.ensemble.predict(
            static_features=static_features,
            gait_sequence=gait_sequence,
            top_k=top_k,
        )

        latency = (time.perf_counter() - t_start) * 1000
        result["latency_ms"] = round(latency, 2)

        return result

    def update_weights(self, svm_weight: float, lstm_weight: float):
        """Update ensemble fusion weights."""
        self.ensemble.svm_weight = svm_weight
        self.ensemble.lstm_weight = lstm_weight
        log.info("weights_updated", svm=svm_weight, lstm=lstm_weight)

    def update_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.ensemble.confidence_threshold = threshold
        log.info("threshold_updated", threshold=threshold)
