"""
services/identification/trainer.py
Training pipeline: loads data from MongoDB, trains SVM + LSTM, saves models.
"""
import numpy as np
import structlog
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .models.svm_model import SkeletonSVM
from .models.lstm_model import SkeletonLSTM

log = structlog.get_logger()


class ModelTrainer:
    """Orchestrates training of SVM and LSTM models."""

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    async def train_svm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Train the SVM on static features.

        Args:
            X: (n_samples, 42) static feature matrix
            y: (n_samples,) user ID labels
        """
        if len(X) == 0 or len(y) == 0:
            return {"success": False, "message": "No training data available"}

        # Require at least 2 classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return {
                "success": False,
                "message": f"Need at least 2 users, got {len(unique_classes)}",
            }

        svm = SkeletonSVM(kernel="rbf", C=10.0, gamma="scale")
        metrics = svm.train(X, y, feature_names=feature_names)
        svm.version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        svm.save(self.model_dir)

        return {
            "success": True,
            "model_type": "svm",
            "version": svm.version,
            "metrics": metrics,
        }

    async def train_lstm(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> Dict[str, Any]:
        """Train the LSTM on gait sequences.

        Args:
            sequences: (n_samples, seq_len, n_angles) — e.g. (500, 30, 8)
            labels: (n_samples,) user ID labels
        """
        if len(sequences) == 0 or len(labels) == 0:
            return {"success": False, "message": "No gait sequence data available"}

        unique_classes = np.unique(labels)
        if len(unique_classes) < 2:
            return {
                "success": False,
                "message": f"Need at least 2 users, got {len(unique_classes)}",
            }

        # Need enough samples for stratified split
        from collections import Counter
        counts = Counter(labels.tolist())
        min_count = min(counts.values())
        if min_count < 3:
            return {
                "success": False,
                "message": f"Need at least 3 gait samples per user, min is {min_count}",
            }

        lstm = SkeletonLSTM(
            input_size=sequences.shape[2],
            hidden_size=128,
            num_layers=2,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
        )
        metrics = lstm.train(sequences, labels)
        lstm.version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        lstm.save(self.model_dir)

        return {
            "success": True,
            "model_type": "lstm",
            "version": lstm.version,
            "metrics": metrics,
        }

    async def train_ensemble(
        self,
        static_X: np.ndarray,
        static_y: np.ndarray,
        gait_sequences: Optional[np.ndarray] = None,
        gait_labels: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Train both SVM and LSTM models."""
        results = {}

        # Train SVM
        svm_result = await self.train_svm(static_X, static_y, feature_names)
        results["svm"] = svm_result

        # Train LSTM (if gait data available)
        if gait_sequences is not None and gait_labels is not None and len(gait_sequences) > 0:
            lstm_result = await self.train_lstm(gait_sequences, gait_labels)
            results["lstm"] = lstm_result
        else:
            results["lstm"] = {"success": False, "message": "No gait data provided"}

        results["success"] = results["svm"].get("success", False)
        return results
