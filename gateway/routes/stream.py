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
from concurrent.futures import ThreadPoolExecutor
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


# Shared thread pool for CPU-bound tasks (MediaPipe, LSTM)
_cpu_executor = ThreadPoolExecutor(max_workers=2)


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
        # Performance: run prediction every Nth frame, cache last result
        self._identify_every_n = 3
        self._last_identification: Dict = {
            "predicted_user": "unknown",
            "confidence": 0.0,
            "is_known": False,
            "method": "none",
            "top_k": [],
        }
        # DB write throttle — log at most once every 2 seconds
        self._last_db_log_time = 0.0

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
        # Run MediaPipe in thread pool — avoids blocking the async event loop
        loop = asyncio.get_event_loop()
        all_kps = await loop.run_in_executor(_cpu_executor, self.pose.estimate, rgb)

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
        # Skip identification entirely during enrollment — not needed and saves ~50ms
        should_identify = (
            mode != "enroll" and self.frame_count % self._identify_every_n == 0
        )

        if should_identify:
            # Run ML inference in thread pool — avoids blocking the async event loop
            loop = asyncio.get_event_loop()
            identification = await loop.run_in_executor(
                _cpu_executor,
                lambda: self.predictor.identify(
                    static_features=static_vector,
                    gait_sequence=gait_sequence,
                )
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
        # Fire-and-forget DB log — throttled to max 1 write per 2 seconds
        now_ts = time.time()
        if mode == "identify" and should_identify and (now_ts - self._last_db_log_time) >= 2.0:
            self._last_db_log_time = now_ts
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


# ── IP Camera Server-Side Streaming ──────────────────────────────────────────

def _open_rtsp(rtsp_url: str, max_retries: int = 3):
    """Try to open an RTSP stream with OpenCV. Returns cap or None."""
    import cv2
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            log.info("rtsp_opened", url=rtsp_url, attempt=attempt + 1)
            return cap
        cap.release()
        log.warning("rtsp_retry", attempt=attempt + 1, url=rtsp_url)
        time.sleep(1)
    return None


def _discover_rtsp(host: str, user: str, password: str) -> Optional[str]:
    """Try common Hikvision/ONVIF RTSP paths and return the first one that works."""
    import cv2
    paths = [
        "/stream1",
        "/h264/ch1/main/av_stream",
        "/h264/ch01/main/av_stream",
        "/live/ch00_0",
        "/live/main",
        "/Streaming/Channels/101",
        "/cam/realmonitor?channel=1&subtype=0",
    ]
    for path in paths:
        url = f"rtsp://{user}:{password}@{host}:554{path}"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = cap.isOpened()
        cap.release()
        if ok:
            log.info("rtsp_discovered", url=url)
            return url
        log.debug("rtsp_path_failed", path=path)
    return None


@router.websocket("/ws/ip-stream")
async def websocket_ip_stream(websocket: WebSocket):
    """Server-side IP camera streaming endpoint.

    Flow:
      Server opens RTSP → reads frame → processes through pipeline
      → sends result JSON + annotated JPEG frame to browser.
      Browser never needs direct camera access.
    """
    await websocket.accept()
    log.info("ip_stream_connected")

    if _predictor is None:
        await websocket.send_json({"error": "Predictor not initialized"})
        await websocket.close()
        return

    # Check for runtime RTSP URL override from the client (cmd: set_rtsp)
    try:
        first_msg_raw = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
        first_msg = json.loads(first_msg_raw)
        if first_msg.get("cmd") == "set_rtsp" and first_msg.get("url"):
            rtsp_url = first_msg["url"].strip()
            log.info("rtsp_override_from_client", url=rtsp_url)
    except (asyncio.TimeoutError, json.JSONDecodeError):
        pass  # No override message — use .env value

    # Resolve RTSP URL (fall back to auto-discovery if still empty)
    if not rtsp_url:
        rtsp_url = settings.ip_camera_rtsp_url.strip()
    if not rtsp_url and settings.ip_camera_host:
        await websocket.send_json({
            "status": "discovering",
            "msg": f"Auto-discovering RTSP path on {settings.ip_camera_host}..."
        })
        loop = asyncio.get_event_loop()
        rtsp_url = await loop.run_in_executor(
            _cpu_executor,
            _discover_rtsp,
            settings.ip_camera_host,
            settings.ip_camera_user,
            settings.ip_camera_pass,
        )

    if not rtsp_url:
        await websocket.send_json({
            "error": "No RTSP URL configured and auto-discovery failed. "
                     "Set IP_CAMERA_RTSP_URL in .env or enter it in the dashboard"
        })
        await websocket.close()
        return

    # Open camera in executor (blocking)
    loop = asyncio.get_event_loop()
    cap = await loop.run_in_executor(_cpu_executor, _open_rtsp, rtsp_url)

    if cap is None:
        await websocket.send_json({
            "error": f"Cannot connect to IP camera at {rtsp_url}. "
                     "Check network, credentials, and RTSP path."
        })
        await websocket.close()
        return

    await websocket.send_json({
        "status": "connected",
        "msg": f"IP camera connected: {rtsp_url}",
        "rtsp_url": rtsp_url,
    })

    import cv2, base64 as b64mod
    pipeline = StreamPipeline(_predictor)
    consecutive_failures = 0

    try:
        while True:
            # Read frame from RTSP in thread pool (blocking I/O)
            def _read_frame():
                ret, frame = cap.read()
                return ret, frame

            ret, frame_bgr = await loop.run_in_executor(_cpu_executor, _read_frame)

            if not ret or frame_bgr is None:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    await websocket.send_json({
                        "error": "IP camera stream lost. Too many read failures."
                    })
                    break
                await asyncio.sleep(0.05)
                continue

            consecutive_failures = 0

            # Encode frame to base64 JPEG (same format as webcam pipeline expects)
            def _encode_frame(bgr):
                _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
                return b64mod.b64encode(buf).decode("utf-8")

            frame_b64 = await loop.run_in_executor(
                _cpu_executor, _encode_frame, frame_bgr
            )

            # Run the full pipeline
            result = await pipeline.process_frame(frame_b64, mode="identify")

            # Attach the raw camera frame as base64 so the browser can display it
            result["camera_frame"] = frame_b64

            await websocket.send_json(result)

            # Small yield to keep event loop responsive
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        log.info("ip_stream_disconnected")
    except Exception as exc:
        log.error("ip_stream_error", error=str(exc))
    finally:
        cap.release()
        pipeline.cleanup()
        log.info("ip_stream_closed")

