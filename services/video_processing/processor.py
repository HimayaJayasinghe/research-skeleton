"""
services/video_processing/processor.py
Captures and preprocesses webcam/IP camera frames for the pose estimation pipeline.
Supports:
  - Local webcam via camera index (int)
  - Phone/IP camera via URL string (e.g., http://192.168.1.5:8080/video)
"""
import cv2
import numpy as np
import base64
import time
import structlog
from typing import Optional, Tuple, Union

log = structlog.get_logger()


class VideoProcessor:
    """Handles webcam/IP camera capture and frame preprocessing."""

    def __init__(
        self,
        camera_index: int = 0,
        camera_url: Optional[str] = None,
        target_size: Tuple[int, int] = (640, 480),
        skip_frames: int = 1,
    ):
        self.camera_index = camera_index
        self.camera_url = camera_url  # IP camera URL (overrides camera_index)
        self.target_size = target_size
        self.skip_frames = skip_frames
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self._is_ip_camera = camera_url is not None and camera_url.strip() != ""
        self._consecutive_failures = 0
        self._max_failures = 30  # Reconnect after this many consecutive read failures

    def _get_capture_source(self) -> Union[int, str]:
        """Return the capture source — URL for IP camera, index for webcam."""
        if self._is_ip_camera:
            return self.camera_url
        return self.camera_index

    def start_capture(self) -> bool:
        """Initialize the camera capture (webcam or IP camera)."""
        source = self._get_capture_source()

        if self._is_ip_camera:
            log.info("connecting_to_ip_camera", url=self.camera_url)
            # For IP cameras, retry connection a few times
            for attempt in range(3):
                self.cap = cv2.VideoCapture(source)
                if self.cap.isOpened():
                    break
                log.warning("ip_camera_retry", attempt=attempt + 1)
                time.sleep(2)
            if not self.cap or not self.cap.isOpened():
                log.error("ip_camera_open_failed", url=self.camera_url)
                raise RuntimeError(
                    f"Cannot connect to IP camera at {self.camera_url}\n"
                    f"Make sure:\n"
                    f"  1. Your phone and PC are on the same WiFi network\n"
                    f"  2. The IP Webcam app is running and streaming\n"
                    f"  3. The URL is correct (e.g., http://192.168.1.5:8080/video)"
                )
            log.info("ip_camera_connected", url=self.camera_url)
        else:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                log.error("camera_open_failed", index=self.camera_index)
                raise RuntimeError(f"Cannot open camera {self.camera_index}")

            # Optimal settings for local webcam
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        source_type = "ip_camera" if self._is_ip_camera else "webcam"
        log.info("camera_started", type=source_type, width=actual_w, height=actual_h)
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera (webcam or IP camera)."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            self._consecutive_failures += 1
            # For IP cameras, try to reconnect after too many failures
            if self._is_ip_camera and self._consecutive_failures >= self._max_failures:
                log.warning("ip_camera_reconnecting", failures=self._consecutive_failures)
                self._reconnect()
            return None
        self._consecutive_failures = 0
        self.frame_count += 1
        return frame

    def _reconnect(self):
        """Attempt to reconnect to the IP camera."""
        self._consecutive_failures = 0
        if self.cap is not None:
            self.cap.release()
        time.sleep(1)
        source = self._get_capture_source()
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            log.info("ip_camera_reconnected")
        else:
            log.error("ip_camera_reconnect_failed")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for pose estimation.

        Steps:
        1. Resize to target dimensions
        2. Convert BGR to RGB (MediaPipe expects RGB)
        3. Normalize brightness using CLAHE
        """
        # Resize
        processed = cv2.resize(
            frame, self.target_size, interpolation=cv2.INTER_LINEAR
        )
        # Convert BGR → RGB
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # Brightness normalization using CLAHE on L channel
        lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return processed

    def should_process(self) -> bool:
        """Determine if current frame should be processed (frame skipping)."""
        return self.frame_count % self.skip_frames == 0

    @staticmethod
    def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
        """Encode frame as base64 JPEG for network transmission."""
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def base64_to_frame(b64_string: str) -> np.ndarray:
        """Decode base64 JPEG back to numpy array (BGR)."""
        img_bytes = base64.b64decode(b64_string)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    @staticmethod
    def frame_to_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
        """Encode frame as JPEG bytes."""
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return buffer.tobytes()

    @staticmethod
    def bytes_to_frame(data: bytes) -> np.ndarray:
        """Decode JPEG bytes to numpy array (BGR)."""
        np_arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def release(self):
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            log.info("camera_released")

    def __enter__(self):
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
