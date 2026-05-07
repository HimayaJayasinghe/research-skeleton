"""
scripts/train_model.py
CLI script to train SVM + LSTM models from enrolled data in MongoDB.

Usage:
    python scripts/train_model.py --type ensemble --epochs 100
"""
import sys
import asyncio
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from database.connection import MongoDB
from database.crud import FeatureProfileCRUD, ModelCRUD
from database.schemas import TrainedModelRecord
from services.identification.trainer import ModelTrainer


async def train(model_type: str = "ensemble", epochs: int = 100):
    """Train models using all enrolled user data from MongoDB."""
    # Connect
    await MongoDB.connect(settings.mongodb_uri, settings.mongodb_db)

    print(f"\n{'='*60}")
    print(f"  MODEL TRAINING")
    print(f"  Type: {model_type} | Epochs: {epochs}")
    print(f"{'='*60}\n")

    # Load training data
    data = await FeatureProfileCRUD.get_training_data()

    static_X = data["static_X"]
    static_y = data["static_y"]
    gait_X = data["gait_X"]
    gait_y = data["gait_y"]

    print(f"  Static samples: {len(static_X)}")
    print(f"  Gait samples:   {len(gait_X)}")

    if len(static_X) == 0:
        print("\n  ❌ No training data! Enroll users first.")
        await MongoDB.close()
        return

    import numpy as np
    unique_users = np.unique(static_y)
    print(f"  Unique users:   {len(unique_users)} → {list(unique_users)}")

    if len(unique_users) < 2:
        print("\n  ❌ Need at least 2 users to train. Enroll more users.")
        await MongoDB.close()
        return

    trainer = ModelTrainer(model_dir=settings.model_dir)

    # Train
    print(f"\n  Training {model_type}...")
    print(f"  {'─'*50}")

    if model_type in ("svm", "ensemble"):
        print("\n  [SVM] Training on static features...")
        svm_result = await trainer.train_svm(static_X, static_y)
        if svm_result["success"]:
            metrics = svm_result["metrics"]
            print(f"  [SVM] ✅ Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  [SVM] ✅ F1 (macro): {metrics['train_f1_macro']:.4f}")
            print(f"  [SVM] ✅ CV F1: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")

            record = TrainedModelRecord(
                model_type="svm",
                version=svm_result["version"],
                num_classes=metrics["num_classes"],
                accuracy=metrics["train_accuracy"],
                f1_score=metrics["train_f1_macro"],
                model_path=settings.model_dir,
                is_active=True,
                metrics=metrics,
            )
            await ModelCRUD.save_record(record)
        else:
            print(f"  [SVM] ❌ {svm_result.get('message', 'Failed')}")

    if model_type in ("lstm", "ensemble") and len(gait_X) > 0:
        print("\n  [LSTM] Training on gait sequences...")
        try:
            lstm_result = await trainer.train_lstm(gait_X, gait_y, epochs=epochs)
            if lstm_result["success"]:
                metrics = lstm_result["metrics"]
                print(f"  [LSTM] ✅ Val Accuracy: {metrics.get('val_accuracy', 0):.4f}")
                print(f"  [LSTM] ✅ Best Val F1: {metrics['best_val_f1']:.4f}")

                record = TrainedModelRecord(
                    model_type="lstm",
                    version=lstm_result["version"],
                    num_classes=metrics["num_classes"],
                    accuracy=metrics.get("val_accuracy", 0),
                    f1_score=metrics["best_val_f1"],
                    model_path=settings.model_dir,
                    is_active=True,
                    metrics=metrics,
                )
                await ModelCRUD.save_record(record)
            else:
                print(f"  [LSTM] ❌ {lstm_result.get('message', 'Failed')}")
        except Exception as e:
            print(f"  [LSTM] ❌ Error: {e}")
    elif len(gait_X) == 0:
        print("\n  [LSTM] ⚠️  No gait data available. Skipping LSTM training.")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Models saved to: {settings.model_dir}")
    print(f"{'='*60}\n")

    await MongoDB.close()


def main():
    parser = argparse.ArgumentParser(description="Train identification models")
    parser.add_argument(
        "--type", type=str, default="ensemble",
        choices=["svm", "lstm", "ensemble"],
        help="Model type to train",
    )
    parser.add_argument("--epochs", type=int, default=100, help="LSTM training epochs")
    args = parser.parse_args()

    asyncio.run(train(args.type, args.epochs))


if __name__ == "__main__":
    main()
