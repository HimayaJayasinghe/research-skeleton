"""
scripts/load_kinect_dataset.py
Loads the Kinect-ReID dataset, processes RGB images through MediaPipe,
extracts skeleton features, and stores them in MongoDB for model training.

Dataset structure (per frame):
  - frameN_RGB.jpg          : Color image (640x480)
  - frameN_Depth.jpg        : Depth image
  - frameN_PlayerIdxs.png   : Player segmentation mask (pixel values = player ID)
  - frameN_skeletons.bin    : Kinect skeleton binary (not used — we use MediaPipe)

Strategy:
  1. Load RGB images sequentially (frame0 → frame999)
  2. Run MediaPipe Pose estimation to extract skeleton keypoints
  3. Extract 42-dim static features from each valid frame
  4. Accumulate gait sequences (sliding window of joint angles)
  5. Use PlayerIdx images to detect which person is visible
     (different pixel values = different people)
  6. Assign person IDs and store features in MongoDB

Usage:
    python scripts/load_kinect_dataset.py --data-dir data/kinect-reid/dataset
    python scripts/load_kinect_dataset.py --data-dir data/kinect-reid/dataset --dry-run
"""
import sys
import os
import cv2
import asyncio
import argparse
import numpy as np
import structlog
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from database.connection import MongoDB
from database.crud import UserCRUD, FeatureProfileCRUD
from database.schemas import UserInDB
from services.pose_estimation.estimator import PoseEstimator
from services.feature_extraction.static_features import StaticFeatureExtractor
from services.feature_extraction.gait_features import GaitFeatureExtractor

log = structlog.get_logger()


def identify_person_from_playeridx(player_idx_path: str) -> int:
    """Read PlayerIdx image and determine which player ID is most prominent.
    
    The PlayerIdx image has pixel values where:
      0 = background
      1,2,3... = different tracked players
    
    Returns the dominant player index (1-based), or 0 if no player detected.
    """
    img = cv2.imread(player_idx_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    
    # Count non-zero pixels per player index
    unique_vals = np.unique(img)
    non_zero = unique_vals[unique_vals > 0]
    
    if len(non_zero) == 0:
        return 0
    
    # Return the player ID with the most pixels
    best_id = 0
    best_count = 0
    for val in non_zero:
        count = np.sum(img == val)
        if count > best_count:
            best_count = count
            best_id = int(val)
    
    return best_id


def detect_person_transitions(data_dir: str, max_frames: int = 1000) -> dict:
    """Scan all PlayerIdx images to determine person segments.
    
    Returns a dict mapping frame_number → player_id for all valid frames.
    Also detects person transitions (when a new person enters the scene).
    """
    frame_to_player = {}
    
    for i in range(max_frames):
        pidx_path = os.path.join(data_dir, f"frame{i}_PlayerIdxs.png")
        rgb_path = os.path.join(data_dir, f"frame{i}_RGB.jpg")
        
        if not os.path.exists(rgb_path):
            continue
        
        if os.path.exists(pidx_path):
            player_id = identify_person_from_playeridx(pidx_path)
        else:
            player_id = 0
            
        frame_to_player[i] = player_id
    
    return frame_to_player


def segment_persons_by_appearance(frame_to_player: dict) -> dict:
    """Group frames into person segments.
    
    Uses the player index data plus temporal proximity to assign
    person identities. When a new player appears after a gap or
    with a different player index, it's treated as a new person.
    
    Returns: dict mapping person_id (str) → list of frame numbers
    """
    if not frame_to_player:
        return {}
    
    # Sort frames
    sorted_frames = sorted(frame_to_player.keys())
    
    # Find segments: groups of consecutive frames with the same player
    segments = []
    current_segment = {"start": sorted_frames[0], "frames": [], "player_id": 0}
    
    for i, frame_num in enumerate(sorted_frames):
        player_id = frame_to_player[frame_num]
        
        if player_id == 0:
            # No player in this frame
            if current_segment["frames"]:
                segments.append(current_segment)
                current_segment = {"start": frame_num, "frames": [], "player_id": 0}
            continue
        
        # Check if this is a continuation or a new person
        if (not current_segment["frames"] or 
            current_segment["player_id"] == 0 or
            (current_segment["player_id"] == player_id and 
             frame_num - current_segment["frames"][-1] <= 5)):
            # Same segment
            current_segment["frames"].append(frame_num)
            current_segment["player_id"] = player_id
        else:
            # New segment
            if current_segment["frames"]:
                segments.append(current_segment)
            current_segment = {"start": frame_num, "frames": [frame_num], "player_id": player_id}
    
    if current_segment["frames"]:
        segments.append(current_segment)
    
    # Assign person IDs to segments
    # Multiple segments with the same player_id within the same Kinect session
    # may belong to the same person, but we'll treat temporal gaps as different appearances
    person_segments = {}
    person_counter = 0
    
    for seg in segments:
        if len(seg["frames"]) < 10:
            # Skip very short segments (noise)
            continue
        person_counter += 1
        person_id = f"person_{person_counter:03d}"
        person_segments[person_id] = seg["frames"]
    
    return person_segments


async def process_dataset(
    data_dir: str,
    dry_run: bool = False,
    max_frames: int = 1000,
    min_segment_frames: int = 20,
):
    """Process the Kinect-ReID dataset through our pipeline."""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return
    
    print(f"\n{'='*65}")
    print(f"  KINECT-REID DATASET LOADER")
    print(f"  Source: {data_dir}")
    print(f"  Mode:  {'DRY RUN (no DB writes)' if dry_run else 'LIVE (writing to MongoDB)'}")
    print(f"{'='*65}")
    
    # ── Step 1: Scan dataset for person segments ──────────────────────
    print(f"\n  [1/4] Scanning PlayerIdx images for person segments...")
    frame_to_player = detect_person_transitions(str(data_path), max_frames)
    total_frames_with_player = sum(1 for v in frame_to_player.values() if v > 0)
    print(f"        Found {len(frame_to_player)} total frames, {total_frames_with_player} with detected players")
    
    person_segments = segment_persons_by_appearance(frame_to_player)
    
    if not person_segments:
        # Fallback: If PlayerIdx doesn't give useful segmentation,
        # split the dataset into equal chunks as different "persons"
        print(f"        ⚠️  PlayerIdx segmentation gave few results. Using frame-range fallback.")
        
        # Look at all RGB files
        rgb_files = sorted(
            [f for f in os.listdir(str(data_path)) if f.endswith("_RGB.jpg")],
            key=lambda x: int(x.split("_")[0].replace("frame", ""))
        )
        all_frame_nums = [int(f.split("_")[0].replace("frame", "")) for f in rgb_files]
        all_frame_nums.sort()
        
        # Split into chunks of ~200 frames, treating each as a different person
        # (In a real scenario, you'd manually label or use appearance-based clustering)
        chunk_size = 200
        person_segments = {}
        for idx, start in enumerate(range(0, len(all_frame_nums), chunk_size)):
            chunk = all_frame_nums[start:start + chunk_size]
            if len(chunk) >= min_segment_frames:
                person_segments[f"person_{idx+1:03d}"] = chunk
    
    print(f"        Identified {len(person_segments)} person segments:")
    for pid, frames in person_segments.items():
        print(f"          {pid}: frames {frames[0]}–{frames[-1]} ({len(frames)} frames)")
    
    # ── Step 2: Connect to MongoDB (if not dry run) ───────────────────
    if not dry_run:
        print(f"\n  [2/4] Connecting to MongoDB...")
        await MongoDB.connect(settings.mongodb_uri, settings.mongodb_db)
        print(f"        ✅ Connected")
        
        # Clear existing data to avoid mixing 2D/3D gait formats
        print(f"        🧹 Cleaning database...")
        await FeatureProfileCRUD.clear_all()
        await UserCRUD.clear_all()
    else:
        print(f"\n  [2/4] Dry run — skipping MongoDB connection")
    
    # ── Step 3: Process frames through MediaPipe pipeline ─────────────
    print(f"\n  [3/4] Processing frames through MediaPipe → Feature Extraction...")
    
    pose = PoseEstimator(model_complexity=1)
    static_ext = StaticFeatureExtractor()
    
    all_results = {}  # person_id → {static_features: [...], gait_features: [...]}
    
    for person_id, frame_nums in person_segments.items():
        print(f"\n        Processing {person_id} ({len(frame_nums)} frames)...")
        
        gait_ext = GaitFeatureExtractor(window_size=30, fps=30.0)
        person_static = []
        person_gait = []
        prev_features = None
        valid_count = 0
        
        for i, frame_num in enumerate(frame_nums):
            rgb_path = os.path.join(str(data_path), f"frame{frame_num}_RGB.jpg")
            
            if not os.path.exists(rgb_path):
                continue
            
            frame = cv2.imread(rgb_path)
            if frame is None:
                continue
            
            # Convert BGR → RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Pose estimation
            all_kps = pose.estimate(rgb)
            if all_kps is None:
                continue
            
            body_kps = pose.get_body_keypoints(all_kps)
            if body_kps is None:
                continue
            
            # Static features
            raw_features = static_ext.extract_all(body_kps)
            if raw_features is None:
                continue
            
            # Smooth features
            features = StaticFeatureExtractor.smooth_features(
                raw_features, prev_features, alpha=0.3
            )
            prev_features = features
            static_vector = static_ext.to_vector(features).tolist()
            person_static.append(static_vector)
            
            # Gait accumulation
            angles = static_ext.compute_joint_angles(body_kps)
            gait_ext.add_frame(body_kps, angles)
            
            # GET RAW SEQUENCE FOR LSTM (3D data)
            sequence_matrix = gait_ext.get_sequence_matrix()
            if sequence_matrix is not None:
                person_gait.append(sequence_matrix.tolist())
            
            valid_count += 1
            
            # Progress
            if (i + 1) % 50 == 0 or i == len(frame_nums) - 1:
                print(f"          Frame {i+1}/{len(frame_nums)}: {valid_count} valid skeleton detections")
        
        all_results[person_id] = {
            "static_features": person_static,
            "gait_features": person_gait,
            "valid_frames": valid_count,
            "total_frames": len(frame_nums),
        }
        
        print(f"        ✅ {person_id}: {valid_count}/{len(frame_nums)} valid frames, "
              f"{len(person_static)} static vectors, {len(person_gait)} gait sequences")
    
    pose.close()
    
    # ── Step 4: Store in MongoDB ──────────────────────────────────────
    print(f"\n  [4/4] {'Storing' if not dry_run else 'Would store'} features in MongoDB...")
    
    stored_users = 0
    total_features = 0
    
    for person_id, data in all_results.items():
        if data["valid_frames"] < 10:
            print(f"        ⚠️  Skipping {person_id}: only {data['valid_frames']} valid frames")
            continue
        
        if dry_run:
            print(f"        [DRY] Would create user '{person_id}' with {len(data['static_features'])} feature vectors")
            stored_users += 1
            total_features += len(data["static_features"])
            continue
        
        # Create user in MongoDB
        user = UserInDB(
            name=person_id,
            metadata={"source": "kinect-reid", "dataset_frames": data["total_frames"]},
        )
        user_id = await UserCRUD.create(user)
        
        # Store feature vectors
        for i, static_vec in enumerate(data["static_features"]):
            gait_seq = None
            if data["gait_features"]:
                # Pair with gait sequence if available
                # Since gait requires 30 frames, it starts later than static features
                # We align them by index roughly or just store them
                g_idx = i
                if g_idx < len(data["gait_features"]):
                    gait_seq = data["gait_features"][g_idx]
            
            await FeatureProfileCRUD.upsert(
                user_id=user_id,
                static_vector=static_vec,
                gait_sequence=gait_seq,
            )
        
        # Update enrollment status
        count = len(data["static_features"])
        status = "completed" if count >= settings.min_enrollment_frames else "in_progress"
        await UserCRUD.update_enrollment_status(user_id, status, count)
        
        stored_users += 1
        total_features += count
        print(f"        ✅ Stored {person_id} (ID: {user_id}) — {count} feature vectors, status: {status}")
    
    if not dry_run:
        await MongoDB.close()
    
    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  DATASET LOADING COMPLETE {'(DRY RUN)' if dry_run else ''}")
    print(f"  ")
    print(f"  Persons processed:    {stored_users}")
    print(f"  Total feature vectors:{total_features}")
    print(f"  ")
    if not dry_run:
        print(f"  Next step: Train models")
        print(f"    python scripts/train_model.py --type ensemble")
    else:
        print(f"  Re-run without --dry-run to store in MongoDB:")
        print(f"    python scripts/load_kinect_dataset.py --data-dir {data_dir}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Load Kinect-ReID dataset")
    parser.add_argument(
        "--data-dir", type=str,
        default="data/kinect-reid/dataset",
        help="Path to the dataset folder containing frame*_RGB.jpg files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process images but don't write to MongoDB",
    )
    parser.add_argument(
        "--max-frames", type=int, default=1000,
        help="Maximum number of frames to process",
    )
    args = parser.parse_args()
    
    asyncio.run(process_dataset(
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        max_frames=args.max_frames,
    ))


if __name__ == "__main__":
    main()
