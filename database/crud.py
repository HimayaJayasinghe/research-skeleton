"""
database/crud.py
CRUD operations for all MongoDB collections.
"""
import numpy as np
import structlog
from datetime import datetime
from typing import List, Optional, Dict, Any
from .connection import MongoDB
from .schemas import UserInDB, FeatureProfileInDB, IdentificationLog, TrainedModelRecord

log = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
#  USER CRUD
# ═══════════════════════════════════════════════════════════════════════════════

class UserCRUD:
    """CRUD operations for the 'users' collection."""

    @staticmethod
    def _col():
        return MongoDB.get_collection("users")

    @classmethod
    async def create(cls, user: UserInDB) -> str:
        doc = user.model_dump()
        await cls._col().insert_one(doc)
        log.info("user_created", user_id=user.user_id, name=user.name)
        return user.user_id

    @classmethod
    async def get_by_id(cls, user_id: str) -> Optional[Dict]:
        return await cls._col().find_one({"user_id": user_id}, {"_id": 0})

    @classmethod
    async def get_by_name(cls, name: str) -> Optional[Dict]:
        return await cls._col().find_one({"name": name}, {"_id": 0})

    @classmethod
    async def list_all(cls) -> List[Dict]:
        cursor = cls._col().find({}, {"_id": 0})
        return await cursor.to_list(length=100)

    @classmethod
    async def update_enrollment_status(
        cls, user_id: str, status: str, frames: int = 0
    ):
        await cls._col().update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "enrollment_status": status,
                    "enrollment_frames_count": frames,
                    "updated_at": datetime.utcnow(),
                }
            },
        )

    @classmethod
    async def delete(cls, user_id: str) -> bool:
        result = await cls._col().delete_one({"user_id": user_id})
        if result.deleted_count > 0:
            log.info("user_deleted", user_id=user_id)
            return True
        return False

    @classmethod
    async def count(cls) -> int:
        return await cls._col().count_documents({})

    @classmethod
    async def clear_all(cls):
        """Reset the collection (danger!)."""
        await cls._col().delete_many({})
        log.info("users_cleared")


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE PROFILE CRUD
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureProfileCRUD:
    """CRUD operations for the 'feature_profiles' collection."""

    @staticmethod
    def _col():
        return MongoDB.get_collection("feature_profiles")

    @classmethod
    async def upsert(
        cls,
        user_id: str,
        static_vector: List[float],
        gait_sequence: Optional[List[List[float]]] = None,
    ):
        """Add a feature sample and update running statistics."""
        existing = await cls._col().find_one({"user_id": user_id})

        if existing is None:
            doc = {
                "user_id": user_id,
                "static_features": {
                    "mean_vector": static_vector,
                    "std_vector": [0.0] * len(static_vector),
                    "samples": [static_vector],
                },
                "gait_features": {
                    "samples": [gait_sequence] if gait_sequence else [],
                },
                "sample_count": 1,
                "last_updated": datetime.utcnow(),
                "version": 1,
            }
            await cls._col().insert_one(doc)
        else:
            # Append sample and recompute statistics
            static_samples = existing["static_features"].get("samples", [])
            static_samples.append(static_vector)

            static_arr = np.array(static_samples)
            static_mean = static_arr.mean(axis=0).tolist()
            static_std = static_arr.std(axis=0).tolist()

            update = {
                "static_features.mean_vector": static_mean,
                "static_features.std_vector": static_std,
                "static_features.samples": static_samples,
                "sample_count": len(static_samples),
                "last_updated": datetime.utcnow(),
            }

            if gait_sequence:
                gait_samples = existing["gait_features"].get("samples", [])
                gait_samples.append(gait_sequence)
                update["gait_features.samples"] = gait_samples

            await cls._col().update_one(
                {"user_id": user_id},
                {"$set": update, "$inc": {"version": 1}},
            )

    @classmethod
    async def get_by_user(cls, user_id: str) -> Optional[Dict]:
        return await cls._col().find_one({"user_id": user_id}, {"_id": 0})

    @classmethod
    async def get_all_profiles(cls) -> List[Dict]:
        cursor = cls._col().find({}, {"_id": 0})
        return await cursor.to_list(length=100)

    @classmethod
    async def get_training_data(cls) -> Dict[str, Any]:
        """Get all feature data formatted for model training."""
        profiles = await cls.get_all_profiles()

        static_X, static_y = [], []
        gait_X, gait_y = [], []

        for profile in profiles:
            uid = profile["user_id"]

            # Static features: use all samples
            for sample in profile["static_features"].get("samples", []):
                if sample:
                    static_X.append(sample)
                    static_y.append(uid)

            # Gait features: use all samples
            for sample in profile["gait_features"].get("samples", []):
                if sample:
                    gait_X.append(sample)
                    gait_y.append(uid)

        return {
            "static_X": np.array(static_X) if static_X else np.array([]),
            "static_y": np.array(static_y) if static_y else np.array([]),
            "gait_X": np.array(gait_X) if gait_X else np.array([]),
            "gait_y": np.array(gait_y) if gait_y else np.array([]),
        }

    @classmethod
    async def delete_by_user(cls, user_id: str) -> bool:
        result = await cls._col().delete_one({"user_id": user_id})
        return result.deleted_count > 0

    @classmethod
    async def clear_all(cls):
        """Reset the collection (danger!)."""
        await cls._col().delete_many({})
        log.info("feature_profiles_cleared")


# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTIFICATION LOG CRUD
# ═══════════════════════════════════════════════════════════════════════════════

class IdentificationLogCRUD:
    """CRUD operations for the 'identification_logs' collection."""

    @staticmethod
    def _col():
        return MongoDB.get_collection("identification_logs")

    @classmethod
    async def log_identification(cls, log_entry: IdentificationLog):
        await cls._col().insert_one(log_entry.model_dump())

    @classmethod
    async def get_recent(cls, limit: int = 50) -> List[Dict]:
        cursor = cls._col().find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return await cursor.to_list(length=limit)

    @classmethod
    async def get_all(cls) -> List[Dict]:
        cursor = cls._col().find({}, {"_id": 0}).sort("timestamp", 1)
        return await cursor.to_list(length=10000)

    @classmethod
    async def get_stats(cls) -> Dict[str, Any]:
        """Get identification statistics."""
        total = await cls._col().count_documents({})
        known = await cls._col().count_documents({"predicted_user_id": {"$ne": None}})

        # Average confidence
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_confidence": {"$avg": "$confidence"},
                "avg_latency": {"$avg": "$latency_ms"},
            }}
        ]
        agg = await cls._col().aggregate(pipeline).to_list(1)
        stats = agg[0] if agg else {"avg_confidence": 0, "avg_latency": 0}

        return {
            "total_identifications": total,
            "known_identifications": known,
            "unknown_identifications": total - known,
            "avg_confidence": stats.get("avg_confidence", 0),
            "avg_latency_ms": stats.get("avg_latency", 0),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL CRUD
# ═══════════════════════════════════════════════════════════════════════════════

class ModelCRUD:
    """CRUD operations for the 'trained_models' collection."""

    @staticmethod
    def _col():
        return MongoDB.get_collection("trained_models")

    @classmethod
    async def save_record(cls, record: TrainedModelRecord):
        """Save a model training record."""
        # Deactivate previous active models of the same type
        await cls._col().update_many(
            {"model_type": record.model_type, "is_active": True},
            {"$set": {"is_active": False}},
        )
        await cls._col().insert_one(record.model_dump())
        log.info("model_record_saved", model_type=record.model_type, version=record.version)

    @classmethod
    async def get_active(cls, model_type: str) -> Optional[Dict]:
        return await cls._col().find_one(
            {"model_type": model_type, "is_active": True},
            {"_id": 0},
        )

    @classmethod
    async def list_all(cls) -> List[Dict]:
        cursor = cls._col().find({}, {"_id": 0}).sort("trained_at", -1)
        return await cursor.to_list(length=50)
