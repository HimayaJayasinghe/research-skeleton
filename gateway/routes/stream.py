"""
gateway/routes/stream.py
WebSocket endpoint for real-time video streaming pipeline.

Flow:
  Client sends base64 JPEG frame via WebSocket
    -> Video Processing (preprocess)
    -> Pose Estimation (keypoints)
    -> Feature Extraction (static + gait)
    -> Identification (ensemble predict)
    -> Send result JSON back to client
"""
import asyncio
import json
import time
from typing import Dict, Optional

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings
from database.crud import FeatureProfileCRUD, IdentificationLogCRUD, UserCRUD
from database.schemas import IdentificationLog
from services.feature_extraction.gait_features import GaitFeatureExtractor
from services.feature_extraction.static_features import StaticFeatureExtractor
from services.identification.predictor import Predictor
from services.pose_estimation.estimator import PoseEstimator
from services.video_processing.processor import VideoProcessor

log = structlog.get_logger()

router = APIRouter(tags=["WebSocket Stream"])
DISPLAY_CONFIDENCE_THRESHOLD = 0.60


class StreamPipeline:
    """Per-connection pipeline instances."""

    def __init__(self, predictor: Predictor):
        self.processor = VideoProcessor()
        self.pose = PoseEstimator(
            model_complexity=settings.mediapipe_model_complexity,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self.static_ext = StaticFeatureExtractor()
        self.gait_ext = GaitFeatureExtractor(
            window_size=settings.lstm_sequence_length,
            fps=30.0,
        )
        self.predictor = predictor
        self.prev_features: Optional[Dict] = None
        self.frame_count = 0
        self.user_name_map: Dict[str, str] = {}
        self._last_user_cache_refresh = 0.0
        # Performance: skip heavy identification on some frames
        self._identify_every_n = 2  # run prediction every Nth frame
        self._last_identification: Dict = {
            "predicted_user": "unknown",
            "confidence": 0.0,
            "is_known": False,
            "method": "none",
            "top_k": [],
        }

    async def _refresh_user_name_map(self):
        """Refresh user id -> name cache periodically to avoid per-frame DB reads."""
        now = time.time()
        if now - self._last_user_cache_refresh < 5:
            return

        users = await UserCRUD.list_all()
        self.user_name_map = {u["user_id"]: u["name"] for u in users}
        self._last_user_cache_refresh = now

    async def process_frame(
        self,
        frame_b64: str,
        mode: str = "identify",
        user_id: Optional[str] = None,
    ) -> Dict:
        """Full pipeline for one frame."""
        t_start = time.perf_counter()
        self.frame_count += 1

        frame_bgr = VideoProcessor.base64_to_frame(frame_b64)
        if frame_bgr is None:
            return {
                "detected": False,
                "body_visible": False,
                "features_ok": False,
                "frame": self.frame_count,
                "mode": mode,
                "status_msg": "Invalid frame received",
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        rgb = self.processor.preprocess_frame(frame_bgr)
        all_kps = self.pose.estimate(rgb)

        if all_kps is None:
            return {
                "detected": False,
                "body_visible": False,
                "features_ok": False,
                "frame": self.frame_count,
                "mode": mode,
                "status_msg": "No person detected",
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        keypoints = [
            {"x": kp["x"], "y": kp["y"], "visibility": kp["visibility"]}
            for kp in all_kps
        ]

        body_kps = self.pose.get_body_keypoints(all_kps)
        if body_kps is None:
            return {
                "detected": True,
                "body_visible": False,
                "features_ok": False,
                "frame": self.frame_count,
                "mode": mode,
                "keypoints": keypoints,
                "status_msg": "Please step back until the full body is visible",
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        raw_features = self.static_ext.extract_all(body_kps)
        if raw_features is None:
            return {
                "detected": True,
                "body_visible": True,
                "features_ok": False,
                "frame": self.frame_count,
                "mode": mode,
                "keypoints": keypoints,
                "status_msg": "Detection too noisy, hold still",
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
            }

        features = StaticFeatureExtractor.smooth_features(
            raw_features,
            self.prev_features,
            alpha=0.3,
        )
        self.prev_features = features
        static_vector = self.static_ext.to_vector(features)

        angles = self.static_ext.compute_joint_angles(body_kps)
        self.gait_ext.add_frame(body_kps, angles)
        gait_ready = self.gait_ext.is_ready()
        gait_sequence = self.gait_ext.get_sequence_matrix() if gait_ready else None

        # Only run identification every N frames to reduce latency
        should_identify = (self.frame_count % self._identify_every_n == 0)

        if should_identify:
            identification = self.predictor.identify(
                static_features=static_vector,
                gait_sequence=gait_sequence,
            )
            self._last_identification = identification
        else:
            identification = self._last_identification

        await self._refresh_user_name_map()

        raw_user_id = identification.get("predicted_user", "unknown")
        confidence = float(identification.get("confidence", 0.0))
        confidence_ok = confidence >= DISPLAY_CONFIDENCE_THRESHOLD
        predicted_name = self.user_name_map.get(raw_user_id)

        is_known_for_display = confidence_ok and raw_user_id != "unknown" and predicted_name is not None
        display_user = predicted_name if is_known_for_display else "unknown"

        top_candidates = []
        for c in identification.get("top_k", [])[:3]:
            uid = c.get("user_id", "")
            cand_name = self.user_name_map.get(uid, "Unknown")
            top_candidates.append(
                {
                    **c,
                    "user": cand_name,
                }
            )

        result_extra: Dict[str, object] = {}
        if mode == "enroll" and user_id:
            try:
                await FeatureProfileCRUD.upsert(
                    user_id=user_id,
                    static_vector=static_vector.tolist(),
                    gait_sequence=gait_sequence.tolist() if gait_sequence is not None else None,
                )

                profile = await FeatureProfileCRUD.get_by_user(user_id)
                count = profile["sample_count"] if profile else 0
                status = "completed" if count >= settings.min_enrollment_frames else "in_progress"
                await UserCRUD.update_enrollment_status(user_id, status, count)

                result_extra = {
                    "frames_collected": count,
                    "enrollment_status": status,
                    "progress": min(count / settings.min_enrollment_frames * 100, 100),
                }
            except Exception as exc:
                log.error("auto_enroll_failed", error=str(exc))

        latency = time.perf_counter() - t_start
        # Fire-and-forget: log to DB without blocking the pipeline response
        if mode == "identify" and should_identify:
            try:
                log_entry = IdentificationLog(
                    predicted_user_id=display_user,
                    confidence=confidence,
                    svm_confidence=float(
                        identification.get("svm_prediction", {}).get("confidence", 0)
                    )
                    if identification.get("svm_prediction")
                    else 0.0,
                    lstm_confidence=float(
                        identification.get("lstm_prediction", {}).get("confidence", 0)
                    )
                    if identification.get("lstm_prediction")
                    else 0.0,
                    feature_vector=static_vector.tolist(),
                    model_version=identification.get("method", "none"),
                    latency_ms=round(latency * 1000, 2),
                )
                asyncio.create_task(IdentificationLogCRUD.log_identification(log_entry))
            except Exception as exc:
                log.error("stream_log_failed", error=str(exc))

        return {
            "detected": True,
            "body_visible": True,
            "features_ok": True,
            "frame": self.frame_count,
            "mode": mode,
            "keypoints": keypoints,
            "status_msg": "Good alignment! Scanning...",
            "num_features": len(features),
            "static_features": static_vector.tolist(),
            "gait_ready": gait_ready,
            "gait_buffer": self.gait_ext.buffer_length(),
            "identification": {
                "user": display_user,
                "confidence": round(confidence, 4),
                "is_known": is_known_for_display,
                "method": identification.get("method", "none"),
                "top_k": top_candidates,
            },
            "latency_ms": round(latency * 1000, 2),
            **result_extra,
        }

    def cleanup(self):
        self.pose.close()


# Global predictor reference (set from gateway main)
_predictor: Optional[Predictor] = None


def set_predictor(p: Predictor):
    global _predictor
    _predictor = p


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing."""
    await websocket.accept()
    log.info("websocket_connected")

    if _predictor is None:
        await websocket.send_json({"error": "Predictor not initialized"})
        await websocket.close()
        return

    pipeline = StreamPipeline(_predictor)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                msg = json.loads(data)
                frame_b64 = msg.get("frame", "")
                mode = msg.get("mode", "identify")
                user_id = msg.get("user_id")
            except json.JSONDecodeError:
                frame_b64 = data
                mode = "identify"
                user_id = None

            if not frame_b64:
                continue

            result = await pipeline.process_frame(frame_b64, mode=mode, user_id=user_id)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        log.info("websocket_disconnected")
    except Exception as exc:
        log.error("websocket_error", error=str(exc))
    finally:
        pipeline.cleanup()
