"""
services/identification/models/lstm_model.py
LSTM model for gait-based person identification using temporal joint angle sequences.

Architecture:
  Input (seq_len, n_angles) → LSTM(hidden=128, layers=2, bidirectional)
      → FC(256→128→n_classes) → Softmax → user_id + confidence
"""
import numpy as np
import structlog
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

log = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class GaitSequenceDataset(Dataset):
    """PyTorch dataset for gait angle sequences."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: (n_samples, seq_len, n_features)
            labels: (n_samples,) encoded integer labels
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  LSTM NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMNetwork(nn.Module):
    """Bidirectional LSTM for gait pattern classification."""

    def __init__(
        self,
        input_size: int = 8,       # Number of joint angles per frame
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        fc_input = hidden_size * self.directions
        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, num_classes)
        """
        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)

        # Use the last hidden state from both directions
        if self.bidirectional:
            # h_n shape: (num_layers * directions, batch, hidden_size)
            forward_last = h_n[-2]   # Last forward layer
            backward_last = h_n[-1]  # Last backward layer
            h_combined = torch.cat([forward_last, backward_last], dim=1)
        else:
            h_combined = h_n[-1]

        logits = self.classifier(h_combined)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
#  LSTM WRAPPER (train / predict / save / load)
# ═══════════════════════════════════════════════════════════════════════════════

class SkeletonLSTM:
    """LSTM-based gait identification model wrapper."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.model: Optional[LSTMNetwork] = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.version = "0.0.0"

    def _build_model(self, num_classes: int):
        """Instantiate the LSTM network."""
        self.model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes,
            dropout=self.dropout,
        ).to(self.device)

    def train(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        val_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the LSTM model.

        Args:
            sequences: (n_samples, seq_len, n_angles) — e.g. (500, 30, 8)
            labels: (n_samples,) — user IDs as strings
            val_split: Fraction held for validation

        Returns:
            Dict of training metrics
        """
        n_samples = len(labels)
        seq_len = sequences.shape[1]
        n_features = sequences.shape[2]
        self.input_size = n_features

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        n_classes = len(self.label_encoder.classes_)

        log.info(
            "lstm_training_started",
            n_samples=n_samples,
            seq_len=seq_len,
            n_features=n_features,
            n_classes=n_classes,
            device=str(self.device),
        )

        # Normalize each feature dimension across all timesteps
        original_shape = sequences.shape
        flat = sequences.reshape(-1, n_features)
        flat_scaled = self.scaler.fit_transform(flat)
        sequences_scaled = flat_scaled.reshape(original_shape)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            sequences_scaled, y_encoded,
            test_size=val_split, stratify=y_encoded, random_state=42,
        )

        train_ds = GaitSequenceDataset(X_train, y_train)
        val_ds = GaitSequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Build model
        self._build_model(n_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )

        best_val_f1 = 0.0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(self.epochs):
            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # ── Validate ──────────────────────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            all_preds, all_true = [], []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1).detach().cpu().tolist()
                    all_preds.extend(preds)
                    all_true.extend(batch_y.cpu().tolist())

            val_loss /= len(val_loader)
            val_f1 = f1_score(all_true, all_preds, average="macro")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

            scheduler.step(val_loss)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                log.info(
                    "lstm_epoch",
                    epoch=epoch + 1,
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    val_f1=f"{val_f1:.4f}",
                )

        # Load best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_trained = True

        # Final evaluation
        self.model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                preds = logits.argmax(dim=1).detach().cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(batch_y.tolist())

        report = classification_report(
            all_true, all_preds,
            target_names=self.label_encoder.classes_,
            output_dict=True,
        )

        metrics = {
            "best_val_f1": float(best_val_f1),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "val_accuracy": float(report["accuracy"]),
            "val_f1_macro": float(report["macro avg"]["f1-score"]),
            "num_classes": n_classes,
            "num_samples": n_samples,
            "epochs_trained": self.epochs,
            "class_names": list(self.label_encoder.classes_),
            "history": history,
        }

        log.info(
            "lstm_training_complete",
            val_f1=metrics["best_val_f1"],
            val_accuracy=metrics["val_accuracy"],
        )
        return metrics

    def predict(
        self,
        sequence: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Predict identity from a gait angle sequence.

        Args:
            sequence: (seq_len, n_angles) single sequence — e.g. (30, 8)
            top_k: Number of top candidates

        Returns:
            Dict with prediction results
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("LSTM model not trained yet")

        self.model.eval()

        # Scale
        flat = sequence.reshape(-1, sequence.shape[-1])
        flat_scaled = self.scaler.transform(flat)
        seq_scaled = flat_scaled.reshape(1, *sequence.shape)

        x = torch.FloatTensor(seq_scaled).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]

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
        """Save LSTM model and preprocessing to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save(self.model.state_dict(), path / "lstm_model.pt")

            # Save model config for reconstruction
            import json
            config = {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_classes": len(self.label_encoder.classes_),
                "dropout": self.dropout,
            }
            with open(path / "lstm_config.json", "w") as f:
                json.dump(config, f)

        import joblib
        joblib.dump(self.label_encoder, path / "lstm_label_encoder.pkl")
        joblib.dump(self.scaler, path / "lstm_scaler.pkl")
        log.info("lstm_model_saved", path=str(path))

    def load(self, directory: str):
        """Load LSTM model and preprocessing from disk."""
        import json
        import joblib

        path = Path(directory)

        # Load config
        with open(path / "lstm_config.json", "r") as f:
            config = json.load(f)

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config.get("dropout", 0.3)

        # Rebuild and load weights
        self._build_model(config["num_classes"])
        self.model.load_state_dict(
            torch.load(path / "lstm_model.pt", map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.label_encoder = joblib.load(path / "lstm_label_encoder.pkl")
        self.scaler = joblib.load(path / "lstm_scaler.pkl")
        self.is_trained = True
        log.info("lstm_model_loaded", path=str(path))
