"""
Dataset downloader for EdgeSight fall detection project.

Downloads UR Fall Detection Dataset and Le2i Fall Detection Dataset.
Organizes into data/raw/ directory with proper structure.
"""

import os
import sys
import shutil
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import requests
from tqdm import tqdm


@dataclass
class DatasetStats:
    """Statistics for a downloaded dataset."""
    name: str
    total_clips: int
    fall_clips: int
    non_fall_clips: int
    avg_clip_length_seconds: float
    total_size_gb: float


class DatasetDownloader:
    """Downloader for fall detection datasets."""

    def __init__(self, root_dir: str = "data/raw"):
        """Initialize downloader.

        Args:
            root_dir: Root directory for raw datasets
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def download_ur_fall(self) -> Path:
        """Download UR Fall Detection Dataset.

        Note: UR Fall requires manual download due to academic registration.
        Creates instructions file if dataset not present.

        Returns:
            Path to dataset directory
        """
        ur_dir = self.root_dir / "urfall"
        ur_dir.mkdir(exist_ok=True)

        # Check if already downloaded
        if any(ur_dir.iterdir()):
            print(f"[INFO] UR Fall dataset found at {ur_dir}")
            return ur_dir

        # UR Fall requires manual download - create instructions
        instructions = ur_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        instructions.write_text("""
UR Fall Detection Dataset - Manual Download Required
====================================================

The UR Fall Detection Dataset requires registration and manual download.

1. Visit: http://fenix.ur.edu.pl/mkepski/ds/uf.html
2. Fill out the registration form (academic use)
3. Download the dataset (approximately 3.5 GB)
4. Extract to: data/raw/urfall/

Expected structure after extraction:
    urfall/
    ├── falls/          # Fall videos (.avi files)
    ├── adls/           # Activities of daily living (non-fall videos)
    └── labels/         # Optional annotation files

After downloading, run preprocess.py to extract pose features.
""")
        print(f"[INFO] UR Fall dataset requires manual download")
        print(f"[INFO] Instructions saved to: {instructions}")
        print(f"[INFO] Visit: http://fenix.ur.edu.pl/mkepski/ds/uf.html")

        return ur_dir

    def download_le2i_fall(self) -> Path:
        """Download Le2i Fall Detection Dataset.

        Uses KaggleHub for automatic download if available,
        otherwise provides manual instructions.

        Returns:
            Path to dataset directory
        """
        le2i_dir = self.root_dir / "le2i"
        le2i_dir.mkdir(exist_ok=True)

        # Check if already downloaded
        if any(le2i_dir.iterdir()):
            print(f"[INFO] Le2i dataset found at {le2i_dir}")
            return le2i_dir

        # Create instructions for Kaggle download
        instructions = le2i_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        instructions.write_text("""
Le2i Fall Detection Dataset - Download Options
===============================================

Option 1: Kaggle (Recommended)
-------------------------------
1. Install kagglehub: pip install kagglehub
2. Run in Python:
   import kagglehub
   path = kagglehub.dataset_download("muhammadwaseem18/le2i-fall-dataset")
   # Copy contents to: data/raw/le2i/

Option 2: Kaggle Website
------------------------
1. Visit: https://www.kaggle.com/datasets/muhammadwaseem18/le2i-fall-dataset
2. Click "Download" (requires Kaggle account)
3. Extract to: data/raw/le2i/

Expected structure:
    le2i/
    ├── Coffee_room_01/     # Fall and non-fall videos
    ├── Coffee_room_02/
    ├── Home_01/
    ├── Home_02/
    └── ...

After downloading, run preprocess.py to extract pose features.
""")
        print(f"[INFO] Le2i dataset download instructions created")
        print(f"[INFO] Instructions saved to: {instructions}")

        # Try automatic download via kagglehub
        try:
            import kagglehub
            print("[INFO] Attempting automatic download via kagglehub...")
            path = kagglehub.dataset_download("muhammadwaseem18/le2i-fall-dataset")
            print(f"[INFO] Downloaded to: {path}")

            # Move to expected location
            if Path(path).is_dir():
                for item in Path(path).iterdir():
                    dest = le2i_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
                print(f"[SUCCESS] Le2i dataset organized at {le2i_dir}")
        except ImportError:
            print("[INFO] kagglehub not installed, manual download required")
        except Exception as e:
            print(f"[WARNING] Automatic download failed: {e}")
            print(f"[INFO] Please follow manual instructions in: {instructions}")

        return le2i_dir

    def count_videos(self, directory: Path) -> Tuple[int, int, int, float]:
        """Count videos in dataset directory.

        Args:
            directory: Dataset root directory

        Returns:
            Tuple of (total, fall_count, non_fall_count, avg_duration)
        """
        # Bolt Performance Optimization:
        # Changed video_exts from set to tuple for faster .endswith() checking inside the hot loop.
        video_exts = ('.avi', '.mp4', '.mov', '.mkv', '.mpg', '.mpeg')
        total = 0
        fall_count = 0
        non_fall_count = 0
        durations = []

        for root, _, files in os.walk(directory):
            for file in files:
                # Optimized .endswith() to accept the tuple directly, eliminating generator overhead.
                if file.lower().endswith(video_exts):
                    total += 1
                    file_lower = file.lower()
                    # Heuristic classification based on filename/path
                    if any(x in file_lower for x in ['fall', 'chute']):
                        fall_count += 1
                    elif any(x in file_lower for x in ['adl', 'normal', 'daily', 'walking', 'sitting']):
                        non_fall_count += 1

        # Estimate average duration (will be updated during preprocessing)
        avg_duration = 5.0  # Placeholder

        return total, fall_count, non_fall_count, avg_duration

    def print_statistics(self) -> None:
        """Print statistics for all downloaded datasets."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)

        stats_list = []

        # UR Fall stats
        ur_dir = self.root_dir / "urfall"
        if ur_dir.exists():
            total, falls, non_falls, avg_dur = self.count_videos(ur_dir)
            stats = DatasetStats(
                name="UR Fall",
                total_clips=total,
                fall_clips=falls,
                non_fall_clips=non_falls,
                avg_clip_length_seconds=avg_dur,
                total_size_gb=0.0  # Calculated below
            )
            stats_list.append(stats)

        # Le2i stats
        le2i_dir = self.root_dir / "le2i"
        if le2i_dir.exists():
            total, falls, non_falls, avg_dur = self.count_videos(le2i_dir)
            stats = DatasetStats(
                name="Le2i Fall",
                total_clips=total,
                fall_clips=falls,
                non_fall_clips=non_falls,
                avg_clip_length_seconds=avg_dur,
                total_size_gb=0.0
            )
            stats_list.append(stats)

        if not stats_list:
            print("[WARNING] No datasets found. Please run download first.")
            return

        total_clips = 0
        total_falls = 0
        total_non_falls = 0

        for stats in stats_list:
            print(f"\n{stats.name}:")
            print(f"  Total clips: {stats.total_clips}")
            print(f"  Fall clips: {stats.fall_clips}")
            print(f"  Non-fall clips: {stats.non_fall_clips}")
            if stats.total_clips > 0:
                fall_ratio = stats.fall_clips / stats.total_clips
                print(f"  Fall ratio: {fall_ratio:.2%}")
            print(f"  Avg clip length: ~{stats.avg_clip_length_seconds:.1f}s")
            total_clips += stats.total_clips
            total_falls += stats.fall_clips
            total_non_falls += stats.non_fall_clips

        print("\n" + "-"*60)
        print("COMBINED STATISTICS:")
        print(f"  Total clips: {total_clips}")
        print(f"  Total falls: {total_falls}")
        print(f"  Total non-falls: {total_non_falls}")
        if total_clips > 0:
            combined_ratio = total_falls / total_clips
            print(f"  Combined fall ratio: {combined_ratio:.2%}")
            if combined_ratio < 0.20:
                print(f"  [WARNING] Low fall ratio - consider data augmentation")
            elif combined_ratio > 0.50:
                print(f"  [WARNING] High fall ratio - check dataset balance")
        print("="*60)


def main():
    """Main entry point for dataset download."""
    print("="*60)
    print("EdgeSight Dataset Downloader")
    print("="*60)

    downloader = DatasetDownloader()

    print("\n[Phase 1] Downloading UR Fall Detection Dataset...")
    downloader.download_ur_fall()

    print("\n[Phase 2] Downloading Le2i Fall Detection Dataset...")
    downloader.download_le2i_fall()

    print("\n[Phase 3] Generating statistics...")
    downloader.print_statistics()

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Complete any manual downloads (see DOWNLOAD_INSTRUCTIONS.txt files)")
    print("2. Run: python data/preprocess.py")
    print("3. Check output for X.npy and y.npy files")


if __name__ == "__main__":
    main()
