"""
scripts/run_identification.py
CLI script for real-time identification from webcam.
Runs the full pipeline: Video → Pose → Features → Identification.

Usage:
    python scripts/run_identification.py
"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from services.video_processing.processor import VideoProcessor
from services.pose_estimation.estimator import PoseEstimator
from services.feature_extraction.static_features import StaticFeatureExtractor
from services.feature_extraction.gait_features import GaitFeatureExtractor
from services.identification.predictor import Predictor


def main():
    """Run real-time identification from webcam."""
    print(f"\n{'='*60}")
    print(f"  SKELETON-BASED PERSON IDENTIFICATION")
    print(f"  Press 'q' to quit")
    print(f"{'='*60}\n")

    # Initialize
    video = VideoProcessor(
        camera_index=settings.camera_index,
        target_size=(settings.frame_width, settings.frame_height),
    )
    pose = PoseEstimator(
        model_complexity=settings.mediapipe_model_complexity,
        min_detection_confidence=settings.min_detection_confidence,
    )
    static_ext = StaticFeatureExtractor()
    gait_ext = GaitFeatureExtractor(
        window_size=settings.lstm_sequence_length,
        fps=settings.target_fps,
    )
    predictor = Predictor(
        model_dir=settings.model_dir,
        svm_weight=settings.svm_weight,
        lstm_weight=settings.lstm_weight,
        confidence_threshold=settings.confidence_threshold,
    )

    # Load models
    if predictor.load_models():
        print(f"  ✅ Models loaded (SVM: {predictor.ensemble.svm_ready}, LSTM: {predictor.ensemble.lstm_ready})")
    else:
        print(f"  ⚠️  No trained models found. Running in feature-display mode.")

    prev_features = None
    frame_times = []

    with video:
        while True:
            t_start = time.perf_counter()

            frame = video.read_frame()
            if frame is None:
                break

            rgb = video.preprocess_frame(frame)
            all_kps = pose.estimate(rgb)

            display = frame.copy()
            id_text = "No person detected"
            conf_text = ""
            color = (0, 0, 255)

            if all_kps is not None:
                # Draw skeleton
                display = pose.draw_on_frame_with_results(display, all_kps)

                body_kps = pose.get_body_keypoints(all_kps)
                if body_kps is not None:
                    raw_features = static_ext.extract_all(body_kps)
                    if raw_features is not None:
                        features = StaticFeatureExtractor.smooth_features(
                            raw_features, prev_features, alpha=0.3
                        )
                        prev_features = features
                        static_vector = static_ext.to_vector(features)

                        # Gait
                        angles = static_ext.compute_joint_angles(body_kps)
                        gait_ext.add_frame(body_kps, angles)
                        gait_seq = gait_ext.get_sequence_matrix()

                        # Identify
                        if predictor.is_ready:
                            result = predictor.identify(
                                static_features=static_vector,
                                gait_sequence=gait_seq,
                            )
                            user = result.get("predicted_user", "unknown")
                            conf = result.get("confidence", 0)
                            is_known = result.get("is_known", False)
                            method = result.get("method", "")

                            if is_known:
                                id_text = f"{user}"
                                conf_text = f"{conf*100:.1f}% ({method})"
                                color = (0, 255, 0)
                            else:
                                id_text = f"Unknown ({conf*100:.1f}%)"
                                conf_text = method
                                color = (0, 165, 255)
                        else:
                            id_text = f"Features: {len(features)} | Gait: {gait_ext.buffer_length()}/30"
                            color = (255, 200, 0)

            # Draw info overlay
            cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
            cv2.putText(display, id_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, conf_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # FPS
            t_end = time.perf_counter()
            frame_ms = (t_end - t_start) * 1000
            frame_times.append(frame_ms)
            fps = 1000.0 / frame_ms if frame_ms > 0 else 0
            cv2.putText(
                display,
                f"FPS: {fps:.0f} | Latency: {frame_ms:.0f}ms",
                (display.shape[1] - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1,
            )

            cv2.imshow("Skeleton Identification", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    pose.close()

    # Print summary
    if frame_times:
        print(f"\n{'='*60}")
        print(f"  Session Summary")
        print(f"  Avg latency:  {np.mean(frame_times):.1f}ms")
        print(f"  Avg FPS:      {1000.0 / np.mean(frame_times):.1f}")
        print(f"  P95 latency:  {np.percentile(frame_times, 95):.1f}ms")
        print(f"  Total frames: {len(frame_times)}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
