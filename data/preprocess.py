"""
Preprocessing pipeline for EdgeSight fall detection.

Extracts pose keypoints using MediaPipe, creates overlapping windows,
and saves processed data as numpy arrays.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque

import numpy as np
import cv2
from tqdm import tqdm

# Suppress MediaPipe logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

try:
    import mediapipe as mp
    from mediapipe.tasks.python.components.containers import Landmark
except ImportError:
    print("[ERROR] MediaPipe not installed. Run: pip install mediapipe")
    sys.exit(1)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    window_size: int = 30          # Frames per clip
    stride: int = 15               # Stride between windows
    keypoint_count: int = 17       # MediaPipe pose landmarks
    features_per_keypoint: int = 3  # x, y, confidence
    target_fps: int = 10           # Target frames per second
    confidence_threshold: float = 0.5  # Minimum keypoint confidence


class PoseExtractor:
    """Extracts pose keypoints from video using MediaPipe."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize pose extractor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balanced accuracy/speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe pose landmark names (17 keypoints used)
        self.keypoint_indices = [
            0,   # nose
            11,  # left_shoulder
            12,  # right_shoulder
            13,  # left_elbow
            14,  # right_elbow
            15,  # left_wrist
            16,  # right_wrist
            23,  # left_hip
            24,  # right_hip
            25,  # left_knee
            26,  # right_knee
            27,  # left_ankle
            28,  # right_ankle
            31,  # left_foot_index
            32,  # right_foot_index
        ]

    def extract_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose keypoints from a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            Array of shape (51,) with [x1, y1, c1, x2, y2, c2, ...]
            or None if no pose detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Extract 17 keypoints
        keypoints = []
        landmarks = results.pose_landmarks.landmark

        for idx in self.keypoint_indices:
            landmark = landmarks[idx]
            keypoints.extend([
                landmark.x,      # Normalized x [0, 1]
                landmark.y,      # Normalized y [0, 1]
                landmark.visibility if hasattr(landmark, 'visibility') else landmark.presence
            ])

        return np.array(keypoints, dtype=np.float32)

    def extract_from_video(
        self,
        video_path: Path,
        label: int
    ) -> Tuple[List[np.ndarray], int]:
        """Extract pose sequences from entire video.

        Args:
            video_path: Path to video file
            label: 1 for fall, 0 for non-fall

        Returns:
            Tuple of (list of keypoint arrays, frame count)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARNING] Could not open video: {video_path}")
            return [], 0

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame skip for target FPS
        if fps > 0:
            frame_skip = max(1, int(fps / self.config.target_fps))
        else:
            frame_skip = 1

        keypoints_list = []
        frame_count = 0
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames to match target FPS
            if frame_count % frame_skip != 0:
                continue

            # Extract pose
            keypoints = self.extract_from_frame(frame)
            if keypoints is not None:
                keypoints_list.append(keypoints)
                frames_processed += 1

        cap.release()

        return keypoints_list, frames_processed


class VideoWindowing:
    """Creates overlapping windows from pose sequences."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize windowing.

        Args:
            config: Preprocessing configuration
        """
        self.config = config

    def create_windows(
        self,
        keypoints_list: List[np.ndarray],
        label: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create overlapping windows from keypoint sequence.

        Args:
            keypoints_list: List of keypoint arrays per frame
            label: 1 for fall, 0 for non-fall

        Returns:
            Tuple of (X_windows, y_labels)
            X_windows shape: (num_windows, 30, 51)
            y_labels shape: (num_windows,)
        """
        if len(keypoints_list) < self.config.window_size:
            # Pad with zeros if sequence too short
            while len(keypoints_list) < self.config.window_size:
                keypoints_list.append(np.zeros(51, dtype=np.float32))

        windows = []
        labels = []

        # Create overlapping windows
        for i in range(0, len(keypoints_list) - self.config.window_size + 1, self.config.stride):
            window = np.stack(keypoints_list[i:i + self.config.window_size])
            windows.append(window)
            labels.append(label)

        if not windows:
            # Single window if stride too large
            window = np.stack(keypoints_list[:self.config.window_size])
            windows.append(window)
            labels.append(label)

        return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


class Preprocessor:
    """Main preprocessing pipeline."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration (uses defaults if None)
        """
        self.config = config or PreprocessingConfig()
        self.pose_extractor = PoseExtractor(self.config)
        self.windowing = VideoWindowing(self.config)

    def process_video(self, video_path: Path, label: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Process a single video file.

        Args:
            video_path: Path to video file
            label: 1 for fall, 0 for non-fall

        Returns:
            Tuple of (X_windows, y_labels) or None if processing failed
        """
        keypoints_list, frames_processed = self.pose_extractor.extract_from_video(video_path, label)

        if len(keypoints_list) < 5:  # Minimum frames for a valid clip
            return None

        X, y = self.windowing.create_windows(keypoints_list, label)
        return X, y

    def process_directory(
        self,
        data_dir: Path,
        output_dir: Path,
        is_fall_func=None
    ) -> Dict[str, int]:
        """Process all videos in a directory.

        Args:
            data_dir: Root directory containing videos
            output_dir: Directory to save processed files
            is_fall_func: Function to determine if video is a fall (path -> bool)

        Returns:
            Statistics dictionary
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Bolt Performance Optimization:
        # Changed video_exts from set to tuple for faster .endswith() checking inside the hot loop.
        video_exts = ('.avi', '.mp4', '.mov', '.mkv', '.mpg', '.mpeg', '.webm')
        all_X = []
        all_y = []
        stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'total_windows': 0,
            'fall_windows': 0,
            'non_fall_windows': 0
        }

        # Find all video files
        video_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                # Optimized .endswith() to accept the tuple directly, eliminating generator overhead.
                if file.lower().endswith(video_exts):
                    video_files.append(Path(root) / file)

        print(f"[INFO] Found {len(video_files)} video files in {data_dir}")

        # Process each video
        for video_path in tqdm(video_files, desc=f"Processing {data_dir.name}"):
            stats['total_videos'] += 1

            # Determine label
            if is_fall_func:
                label = 1 if is_fall_func(video_path) else 0
            else:
                # Default: check filename/path for fall indicators
                path_str = str(video_path).lower()
                label = 1 if any(x in path_str for x in ['fall', 'chute']) else 0

            result = self.process_video(video_path, label)

            if result is not None:
                X, y = result
                all_X.append(X)
                all_y.append(y)
                stats['processed_videos'] += 1
                stats['total_windows'] += len(y)
                stats['fall_windows'] += int(y.sum())
                stats['non_fall_windows'] += len(y) - int(y.sum())
            else:
                stats['failed_videos'] += 1

        if not all_X:
            print(f"[WARNING] No valid windows created from {data_dir}")
            return stats

        # Concatenate all windows
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        # Save to disk
        dataset_name = data_dir.name
        np.save(output_dir / f"X_{dataset_name}.npy", X_combined)
        np.save(output_dir / f"y_{dataset_name}.npy", y_combined)

        print(f"[INFO] Saved {dataset_name}: X.shape={X_combined.shape}, y.shape={y_combined.shape}")

        return stats

    def print_final_stats(self, all_stats: List[Dict[str, int]]) -> None:
        """Print combined statistics from all datasets."""
        print("\n" + "="*60)
        print("PREPROCESSING STATISTICS")
        print("="*60)

        total_videos = sum(s['total_videos'] for s in all_stats)
        processed = sum(s['processed_videos'] for s in all_stats)
        failed = sum(s['failed_videos'] for s in all_stats)
        total_windows = sum(s['total_windows'] for s in all_stats)
        fall_windows = sum(s['fall_windows'] for s in all_stats)
        non_fall_windows = sum(s['non_fall_windows'] for s in all_stats)

        print(f"Total videos found: {total_videos}")
        print(f"Successfully processed: {processed}")
        print(f"Failed to process: {failed}")
        print(f"Success rate: {100*processed/total_videos:.1f}%" if total_videos > 0 else "N/A")

        print(f"\nWindow statistics:")
        print(f"  Total windows: {total_windows}")
        print(f"  Fall windows: {fall_windows}")
        print(f"  Non-fall windows: {non_fall_windows}")

        if total_windows > 0:
            fall_ratio = fall_windows / total_windows
            print(f"  Fall ratio: {fall_ratio:.2%}")

            if fall_ratio < 0.15:
                print(f"  [WARNING] Low fall ratio - class imbalance detected")
            elif fall_ratio > 0.60:
                print(f"  [WARNING] High fall ratio - check labeling")

        print("="*60)


def is_fall_urfall(video_path: Path) -> bool:
    """Determine if UR Fall video is a fall based on path."""
    path_str = str(video_path).lower()
    return 'fall' in path_str or 'chute' in path_str


def is_fall_le2i(video_path: Path) -> bool:
    """Determine if Le2i video is a fall based on filename."""
    # Le2i uses numbered files where specific ranges are falls
    # This is a heuristic - may need adjustment based on actual dataset
    filename = video_path.stem.lower()
    return 'fall' in filename or 'chute' in filename


def main():
    """Main entry point for preprocessing."""
    parser = argparse.ArgumentParser(description="EdgeSight preprocessing pipeline")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Directory containing raw datasets")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Directory to save processed data")
    parser.add_argument("--window-size", type=int, default=30,
                        help="Frames per window")
    parser.add_argument("--stride", type=int, default=15,
                        help="Stride between windows")
    args = parser.parse_args()

    print("="*60)
    print("EdgeSight Preprocessing Pipeline")
    print("="*60)

    config = PreprocessingConfig(
        window_size=args.window_size,
        stride=args.stride
    )
    preprocessor = Preprocessor(config)

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    # Process UR Fall dataset
    ur_dir = raw_dir / "urfall"
    if ur_dir.exists():
        print(f"\n[Phase 1] Processing UR Fall dataset...")
        stats = preprocessor.process_directory(ur_dir, output_dir, is_fall_urfall)
        all_stats.append(stats)
    else:
        print(f"[WARNING] UR Fall dataset not found at {ur_dir}")

    # Process Le2i dataset
    le2i_dir = raw_dir / "le2i"
    if le2i_dir.exists():
        print(f"\n[Phase 2] Processing Le2i dataset...")
        stats = preprocessor.process_directory(le2i_dir, output_dir, is_fall_le2i)
        all_stats.append(stats)
    else:
        print(f"[WARNING] Le2i dataset not found at {le2i_dir}")

    # Print statistics
    preprocessor.print_final_stats(all_stats)

    # Combine datasets if both present
    if (output_dir / "X_urfall.npy").exists() and (output_dir / "X_le2i.npy").exists():
        print("\n[Phase 3] Combining datasets...")
        X_ur = np.load(output_dir / "X_urfall.npy")
        y_ur = np.load(output_dir / "y_urfall.npy")
        X_le = np.load(output_dir / "X_le2i.npy")
        y_le = np.load(output_dir / "y_le2i.npy")

        X_combined = np.concatenate([X_ur, X_le], axis=0)
        y_combined = np.concatenate([y_ur, y_le], axis=0)

        np.save(output_dir / "X.npy", X_combined)
        np.save(output_dir / "y.npy", y_combined)

        print(f"[SUCCESS] Combined dataset saved: X.shape={X_combined.shape}, y.shape={y_combined.shape}")

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nOutput files in {output_dir}:")
    for f in output_dir.iterdir():
        if f.suffix == '.npy':
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
