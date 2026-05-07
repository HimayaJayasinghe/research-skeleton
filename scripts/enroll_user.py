"""
scripts/enroll_user.py
CLI script to enroll a user via webcam.
Captures skeleton features and stores them in MongoDB for model training.

Usage:
    python scripts/enroll_user.py --name "John Doe" --duration 60
"""
import sys
import time
import argparse
import asyncio
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from database.connection import MongoDB
from database.crud import UserCRUD, FeatureProfileCRUD
from database.schemas import UserInDB
from services.video_processing.processor import VideoProcessor
from services.pose_estimation.estimator import PoseEstimator
from services.feature_extraction.static_features import StaticFeatureExtractor
from services.feature_extraction.gait_features import GaitFeatureExtractor


async def enroll(name: str, duration: int = 60, camera: int = 0, camera_url: str = None):
    """Enroll a user by capturing skeleton features from webcam or IP camera."""
    import cv2

    # Connect to DB
    await MongoDB.connect(settings.mongodb_uri, settings.mongodb_db)

    # Check if user exists
    existing = await UserCRUD.get_by_name(name)
    if existing:
        user_id = existing["user_id"]
        print(f"[INFO] User '{name}' already exists (ID: {user_id}). Adding more samples.")
    else:
        user = UserInDB(name=name)
        user_id = await UserCRUD.create(user)
        print(f"[INFO] Created user '{name}' (ID: {user_id})")

    await UserCRUD.update_enrollment_status(user_id, "in_progress")

    # Initialize components — use IP camera URL if provided
    video = VideoProcessor(
        camera_index=camera,
        camera_url=camera_url,
        target_size=(640, 480),
    )
    # Lower detection confidence for more frames captured (default 0.5 was too strict)
    pose = PoseEstimator(
        model_complexity=settings.mediapipe_model_complexity,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    static_ext = StaticFeatureExtractor()
    gait_ext = GaitFeatureExtractor(window_size=30, fps=30.0)

    # Determine camera source for display
    cam_source = camera_url if camera_url else f"Webcam #{camera}"

    print(f"\n{'='*60}")
    print(f"  ENROLLMENT: {name}")
    print(f"  Camera: {cam_source}")
    print(f"  Duration: {duration}s  |  Min frames: {settings.min_enrollment_frames}")
    print(f"  Walk naturally in front of the camera.")
    print(f"  Press 'q' to stop early.")
    print(f"{'='*60}\n")

    if camera_url:
        print(f"  📱 Using phone camera at: {camera_url}")
        print(f"  Connecting...\n")

    frames_collected = 0
    prev_features = None
    start_time = time.time()
    frame_idx = 0

    with video:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                print(f"\n[INFO] Time limit reached ({duration}s)")
                break

            frame = video.read_frame()
            if frame is None:
                continue

            frame_idx += 1

            rgb = video.preprocess_frame(frame)
            all_kps = pose.estimate(rgb)

            if all_kps is not None:
                body_kps = pose.get_body_keypoints(all_kps)
                if body_kps is not None:
                    raw_features = static_ext.extract_all(body_kps)
                    if raw_features is not None:
                        # Smooth
                        features = StaticFeatureExtractor.smooth_features(
                            raw_features, prev_features, alpha=0.3
                        )
                        prev_features = features
                        static_vector = static_ext.to_vector(features).tolist()

                        # Gait
                        angles = static_ext.compute_joint_angles(body_kps)
                        gait_ext.add_frame(body_kps, angles)

                        # Gait Sequence
                        gait_seq = None
                        seq_matrix = gait_ext.get_sequence_matrix()
                        if seq_matrix is not None:
                            gait_seq = seq_matrix.tolist()

                        # Store in MongoDB
                        await FeatureProfileCRUD.upsert(
                            user_id=user_id,
                            static_vector=static_vector,
                            gait_sequence=gait_seq,
                        )
                        frames_collected += 1

                        # Progress
                        pct = min(frames_collected / settings.min_enrollment_frames * 100, 100)
                        remaining = duration - elapsed
                        print(
                            f"\r  Frames: {frames_collected} | "
                            f"Progress: {pct:.0f}% | "
                            f"Time: {remaining:.0f}s remaining  ",
                            end="",
                        )

            # Display — use cached keypoints to AVOID re-running pose estimation
            annotated = pose.draw_skeleton(frame, rgb, cached_keypoints=all_kps)
            cv2.putText(
                annotated,
                f"Enrolling: {name} | Frames: {frames_collected}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("Enrollment", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Stopped by user.")
                break

    cv2.destroyAllWindows()
    pose.close()

    # Update status
    status = "completed" if frames_collected >= settings.min_enrollment_frames else "pending"
    await UserCRUD.update_enrollment_status(user_id, status, frames_collected)

    print(f"\n{'='*60}")
    print(f"  Enrollment {'COMPLETE ✅' if status == 'completed' else 'INCOMPLETE ⚠️'}")
    print(f"  User: {name}")
    print(f"  Frames: {frames_collected}")
    print(f"  Status: {status}")
    print(f"{'='*60}\n")

    await MongoDB.close()


def main():
    parser = argparse.ArgumentParser(description="Enroll a user via webcam or phone camera")
    parser.add_argument("--name", type=str, required=True, help="User's full name")
    parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (for local webcam)")
    parser.add_argument(
        "--camera-url", type=str, default=None,
        help="IP camera URL (e.g., http://192.168.1.5:8080/video). "
             "Use this to stream from your phone camera via IP Webcam app."
    )
    args = parser.parse_args()

    asyncio.run(enroll(args.name, args.duration, args.camera, args.camera_url))


if __name__ == "__main__":
    main()
