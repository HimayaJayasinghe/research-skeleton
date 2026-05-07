"""
services/identification/models/svm_model.py
SVM classifier for skeleton-based person identification using static features.

Architecture:
  Input → StandardScaler → SVM (RBF kernel, probability=True) → user_id + confidence
"""
import numpy as np
import joblib
import structlog
from pathlib import Path
from typing import Dict, List, Optional, Any
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

log = structlog.get_logger()


class SkeletonSVM:
    """SVM classifier for skeleton-based person identification."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 10.0,
        gamma: str = "scale",
    ):
        self.svm = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            class_weight="balanced",
            decision_function_shape="ovr",
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.version = "0.0.0"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Train the SVM model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (user IDs as strings)
            feature_names: Optional ordered feature names
            cv_folds: Number of cross-validation folds

        Returns:
            Dict of training metrics
        """
        self.feature_names = feature_names
        n_samples, n_features = X.shape

        # Encode string labels → integers
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)

        log.info(
            "svm_training_started",
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
        )

        # Scale features to zero mean, unit variance
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation
        cv_folds_actual = min(cv_folds, min(np.bincount(y_encoded)))
        cv_folds_actual = max(cv_folds_actual, 2)
        cv = StratifiedKFold(n_splits=cv_folds_actual, shuffle=True, random_state=42)

        cv_scores = cross_val_score(
            self.svm, X_scaled, y_encoded, cv=cv, scoring="f1_macro"
        )

        # Train on full data
        self.svm.fit(X_scaled, y_encoded)
        self.is_trained = True

        # Evaluate on training data
        y_pred = self.svm.predict(X_scaled)
        report = classification_report(
            y_encoded, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
        )
        cm = confusion_matrix(y_encoded, y_pred).tolist()

        metrics = {
            "cv_f1_mean": float(np.mean(cv_scores)),
            "cv_f1_std": float(np.std(cv_scores)),
            "train_accuracy": float(report["accuracy"]),
            "train_f1_macro": float(report["macro avg"]["f1-score"]),
            "train_f1_weighted": float(report["weighted avg"]["f1-score"]),
            "num_classes": n_classes,
            "num_samples": n_samples,
            "num_features": n_features,
            "confusion_matrix": cm,
            "class_names": list(self.label_encoder.classes_),
            "per_class": {
                name: {
                    "precision": report[name]["precision"],
                    "recall": report[name]["recall"],
                    "f1": report[name]["f1-score"],
                    "support": report[name]["support"],
                }
                for name in self.label_encoder.classes_
            },
        }

        log.info(
            "svm_training_complete",
            accuracy=metrics["train_accuracy"],
            f1=metrics["train_f1_macro"],
            cv_f1=metrics["cv_f1_mean"],
        )
        return metrics

    def predict(
        self,
        feature_vector: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Predict identity from a static feature vector.

        Args:
            feature_vector: (n_features,) array
            top_k: Number of top candidates to return

        Returns:
            Dict with prediction results
        """
        if not self.is_trained:
            raise RuntimeError("SVM model not trained yet")

        X = feature_vector.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get class probabilities
        probs = self.svm.predict_proba(X_scaled)[0]

        # Sort by confidence descending
        sorted_idx = np.argsort(probs)[::-1]
        candidates = [
            {
                "user_id": str(self.label_encoder.classes_[i]),
                "confidence": float(probs[i]),
            }
            for i in sorted_idx[:top_k]
        ]

        best = candidates[0]
        return {
            "predicted_user": best["user_id"],
            "confidence": best["confidence"],
            "top_k": candidates,
        }

    def save(self, directory: str):
        """Persist model to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.svm, path / "svm_model.pkl")
        joblib.dump(self.scaler, path / "svm_scaler.pkl")
        joblib.dump(self.label_encoder, path / "svm_label_encoder.pkl")
        if self.feature_names:
            joblib.dump(self.feature_names, path / "svm_feature_names.pkl")
        log.info("svm_model_saved", path=str(path))

    def load(self, directory: str):
        """Load model from disk."""
        path = Path(directory)
        self.svm = joblib.load(path / "svm_model.pkl")
        self.scaler = joblib.load(path / "svm_scaler.pkl")
        self.label_encoder = joblib.load(path / "svm_label_encoder.pkl")
        feat_path = path / "svm_feature_names.pkl"
        if feat_path.exists():
            self.feature_names = joblib.load(feat_path)
        self.is_trained = True
        log.info("svm_model_loaded", path=str(path))
