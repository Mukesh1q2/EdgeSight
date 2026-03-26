"""
Synthetic Test Data Generator for EdgeSight

Generates simulated pose sequences for testing without real fall datasets.
Useful for:
- Quick integration testing
- CI/CD pipelines without large data downloads
- Unit testing with controlled inputs
- Benchmarking without I/O bottlenecks

Author: EdgeSight Team
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class PoseConfig:
    """Configuration for synthetic pose generation."""
    num_keypoints: int = 15  # Number of body keypoints
    fps: int = 30  # Frames per second
    sequence_length: int = 16  # Frames per sequence
    image_width: int = 640
    image_height: int = 480


class PoseKeypointSimulator:
    """
    Simulates human pose keypoints for fall detection testing.
    
    Models a simplified skeleton with 15 keypoints:
    0: Nose
    1-2: Eyes
    3-4: Shoulders
    5-6: Elbows
    7-8: Wrists
    9: Chest
    10-11: Hips
    12-13: Knees
    14: Center (average position)
    """
    
    def __init__(self, config: PoseConfig = None):
        self.config = config or PoseConfig()
        
        # Skeleton connections (parent-child relationships)
        self.skeleton = {
            0: [],  # Nose (root)
            1: [0],  # Left eye -> nose
            2: [0],  # Right eye -> nose
            3: [0],  # Left shoulder -> nose
            4: [0],  # Right shoulder -> nose
            5: [3],  # Left elbow -> left shoulder
            6: [4],  # Right elbow -> right shoulder
            7: [5],  # Left wrist -> left elbow
            8: [6],  # Right wrist -> right elbow
            9: [3, 4],  # Chest -> shoulders
            10: [9],  # Left hip -> chest
            11: [9],  # Right hip -> chest
            12: [10],  # Left knee -> left hip
            13: [11],  # Right knee -> right hip
            14: [10, 11],  # Center -> hips
        }
        
    def generate_standing_pose(self, center_x: float = 0.5, center_y: float = 0.7) -> np.ndarray:
        """
        Generate keypoints for a standing person.
        
        Args:
            center_x: Normalized x position (0-1)
            center_y: Normalized y position (0-1)
            
        Returns:
            Array of shape (15, 3) with [x, y, confidence] for each keypoint
        """
        keypoints = np.zeros((self.config.num_keypoints, 3), dtype=np.float32)
        
        # Set center position
        cx, cy = center_x, center_y
        
        # Nose
        keypoints[0] = [cx, cy - 0.25, 0.95]
        
        # Eyes
        keypoints[1] = [cx - 0.03, cy - 0.23, 0.9]
        keypoints[2] = [cx + 0.03, cy - 0.23, 0.9]
        
        # Shoulders
        keypoints[3] = [cx - 0.08, cy - 0.15, 0.95]
        keypoints[4] = [cx + 0.08, cy - 0.15, 0.95]
        
        # Elbows
        keypoints[5] = [cx - 0.12, cy - 0.05, 0.9]
        keypoints[6] = [cx + 0.12, cy - 0.05, 0.9]
        
        # Wrists
        keypoints[7] = [cx - 0.15, cy + 0.02, 0.85]
        keypoints[8] = [cx + 0.15, cy + 0.02, 0.85]
        
        # Chest
        keypoints[9] = [cx, cy - 0.08, 0.9]
        
        # Hips
        keypoints[10] = [cx - 0.06, cy + 0.05, 0.9]
        keypoints[11] = [cx + 0.06, cy + 0.05, 0.9]
        
        # Knees
        keypoints[12] = [cx - 0.06, cy + 0.18, 0.85]
        keypoints[13] = [cx + 0.06, cy + 0.18, 0.85]
        
        # Center (average of hips)
        keypoints[14] = [cx, cy + 0.05, 0.9]
        
        return keypoints
    
    def generate_falling_pose(self, center_x: float = 0.5, 
                            progress: float = 0.5) -> np.ndarray:
        """
        Generate keypoints for a person in the process of falling.
        
        Args:
            center_x: Normalized x position (0-1)
            progress: Fall progress from 0 (standing) to 1 (fallen)
            
        Returns:
            Array of shape (15, 3) with [x, y, confidence]
        """
        standing = self.generate_standing_pose(center_x, 0.7)
        
        # Create fallen pose (horizontal, lower in frame)
        fallen = standing.copy()
        
        # Lower all points as fall progresses
        vertical_shift = progress * 0.3
        fallen[:, 1] += vertical_shift
        
        # Rotate to horizontal as fall progresses
        # Shoulders and hips get closer in height
        shoulder_y = 0.6 + vertical_shift
        hip_y = 0.65 + vertical_shift
        
        fallen[3, 1] = shoulder_y - 0.1 * progress  # Left shoulder
        fallen[4, 1] = shoulder_y - 0.1 * progress  # Right shoulder
        fallen[10, 1] = hip_y + 0.1 * progress  # Left hip
        fallen[11, 1] = hip_y + 0.1 * progress  # Right hip
        
        # Arms flail slightly
        fallen[5, 0] += progress * 0.1  # Left elbow
        fallen[6, 0] -= progress * 0.1  # Right elbow
        fallen[7, 0] += progress * 0.15  # Left wrist
        fallen[8, 0] -= progress * 0.15  # Right wrist
        
        # Lower confidence during chaotic motion
        confidence_drop = progress * 0.1
        fallen[:, 2] -= confidence_drop
        fallen[:, 2] = np.clip(fallen[:, 2], 0.5, 1.0)
        
        return fallen
    
    def generate_walking_pose(self, center_x: float = 0.5, 
                             frame: int = 0) -> np.ndarray:
        """
        Generate keypoints for a walking person.
        
        Args:
            center_x: Normalized x position (0-1)
            frame: Frame number for walking cycle animation
            
        Returns:
            Array of shape (15, 3) with [x, y, confidence]
        """
        base = self.generate_standing_pose(center_x, 0.7)
        
        # Walking cycle (2 seconds at 30fps = 60 frames)
        cycle = (frame % 60) / 60.0
        
        # Leg swing
        leg_swing = np.sin(cycle * 2 * np.pi) * 0.05
        
        # Alternate leg movement
        base[12, 0] += leg_swing  # Left knee
        base[12, 1] += abs(leg_swing) * 0.5  # Slight dip
        base[13, 0] -= leg_swing  # Right knee
        
        # Slight body sway
        base[:, 0] += np.sin(cycle * 2 * np.pi) * 0.01
        
        return base
    
    def add_noise(self, keypoints: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """Add random noise to simulate detection uncertainty."""
        noisy = keypoints.copy()
        noisy[:, :2] += np.random.normal(0, noise_level, (self.config.num_keypoints, 2))
        noisy[:, 2] += np.random.normal(0, noise_level * 0.5, self.config.num_keypoints)
        noisy[:, 2] = np.clip(noisy[:, 2], 0.0, 1.0)
        return noisy


class SyntheticDatasetGenerator:
    """Generates complete synthetic datasets for training/testing."""
    
    def __init__(self, config: PoseConfig = None, seed: int = 42):
        self.config = config or PoseConfig()
        self.simulator = PoseKeypointSimulator(self.config)
        np.random.seed(seed)
        
    def generate_fall_sequence(self) -> Tuple[np.ndarray, int]:
        """
        Generate a complete fall event sequence.
        
        Returns:
            Tuple of (sequence array, label)
            sequence: (seq_len, 45) array of flattened keypoints
            label: 1 (fall)
        """
        sequence = []
        center_x = np.random.uniform(0.3, 0.7)
        
        # Pre-fall: standing (3-5 frames)
        pre_fall_frames = np.random.randint(3, 6)
        for _ in range(pre_fall_frames):
            pose = self.simulator.generate_standing_pose(center_x)
            pose = self.simulator.add_noise(pose, noise_level=0.01)
            sequence.append(pose.flatten())
        
        # Falling: transition (8-12 frames)
        fall_frames = self.config.sequence_length - pre_fall_frames
        for i in range(fall_frames):
            progress = i / fall_frames
            pose = self.simulator.generate_falling_pose(center_x, progress)
            pose = self.simulator.add_noise(pose, noise_level=0.03)
            sequence.append(pose.flatten())
        
        return np.array(sequence, dtype=np.float32), 1  # Fall label
    
    def generate_normal_sequence(self, activity: str = 'standing') -> Tuple[np.ndarray, int]:
        """
        Generate a normal activity sequence (no fall).
        
        Args:
            activity: 'standing', 'walking', or 'sitting'
            
        Returns:
            Tuple of (sequence array, label)
            label: 0 (no fall)
        """
        sequence = []
        center_x = np.random.uniform(0.3, 0.7)
        
        for frame in range(self.config.sequence_length):
            if activity == 'standing':
                pose = self.simulator.generate_standing_pose(center_x)
                noise = 0.01
            elif activity == 'walking':
                pose = self.simulator.generate_walking_pose(center_x, frame)
                noise = 0.02
            else:  # sitting
                # Similar to standing but lower
                pose = self.simulator.generate_standing_pose(center_x, 0.5)
                noise = 0.01
            
            pose = self.simulator.add_noise(pose, noise_level=noise)
            sequence.append(pose.flatten())
        
        return np.array(sequence, dtype=np.float32), 0  # No fall label
    
    def generate_dataset(self, 
                        num_samples: int = 1000,
                        fall_ratio: float = 0.3,
                        output_dir: str = "data/processed") -> Dict[str, str]:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_samples: Total number of sequences to generate
            fall_ratio: Ratio of fall samples (0.0-1.0)
            output_dir: Directory to save the dataset
            
        Returns:
            Dictionary with paths to generated files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        sequences = []
        labels = []
        metadata = []
        
        num_falls = int(num_samples * fall_ratio)
        num_normal = num_samples - num_falls
        
        print(f"Generating {num_falls} fall sequences...")
        for i in range(num_falls):
            seq, label = self.generate_fall_sequence()
            sequences.append(seq)
            labels.append(label)
            metadata.append({
                'id': i,
                'type': 'fall',
                'label': label
            })
        
        print(f"Generating {num_normal} normal sequences...")
        activities = ['standing', 'walking', 'sitting']
        for i in range(num_normal):
            activity = np.random.choice(activities)
            seq, label = self.generate_normal_sequence(activity)
            sequences.append(seq)
            labels.append(label)
            metadata.append({
                'id': num_falls + i,
                'type': activity,
                'label': label
            })
        
        # Shuffle
        indices = np.random.permutation(num_samples)
        sequences = [sequences[i] for i in indices]
        labels = [labels[i] for i in indices]
        metadata = [metadata[i] for i in indices]
        
        # Save as numpy arrays
        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        
        data_path = os.path.join(output_dir, "synthetic_sequences.npy")
        labels_path = os.path.join(output_dir, "synthetic_labels.npy")
        meta_path = os.path.join(output_dir, "synthetic_metadata.json")
        
        np.save(data_path, X)
        np.save(labels_path, y)
        
        with open(meta_path, 'w') as f:
            json.dump({
                'num_samples': num_samples,
                'fall_ratio': fall_ratio,
                'sequence_length': self.config.sequence_length,
                'num_keypoints': self.config.num_keypoints,
                'samples': metadata
            }, f, indent=2)
        
        print(f"\n✓ Dataset saved to {output_dir}")
        print(f"  - Data: {X.shape} -> {data_path}")
        print(f"  - Labels: {y.shape} -> {labels_path}")
        print(f"  - Metadata: {meta_path}")
        print(f"\nDataset statistics:")
        print(f"  - Falls: {num_falls} ({num_falls/num_samples*100:.1f}%)")
        print(f"  - Normal: {num_normal} ({num_normal/num_samples*100:.1f}%)")
        
        return {
            'data': data_path,
            'labels': labels_path,
            'metadata': meta_path
        }
    
    def generate_single_sample_visualization(self, output_path: str = "synthetic_sample.png"):
        """Generate a visualization of sample poses for documentation."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Synthetic Pose Samples')
        
        # Standing progression
        for i, ax in enumerate(axes[0]):
            pose = self.simulator.generate_standing_pose(0.5 + i * 0.05)
            self._plot_pose(ax, pose, f"Standing {i+1}")
        
        # Falling progression
        for i, ax in enumerate(axes[1]):
            progress = i / 3
            pose = self.simulator.generate_falling_pose(0.5, progress)
            self._plot_pose(ax, pose, f"Falling ({int(progress*100)}%)")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Sample visualization saved to {output_path}")
    
    def _plot_pose(self, ax, pose: np.ndarray, title: str):
        """Plot a single pose on a matplotlib axis."""
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Inverted Y for image coordinates
        ax.set_aspect('equal')
        ax.set_title(title)
        
        # Plot keypoints
        for i, (x, y, conf) in enumerate(pose):
            color = plt.cm.viridis(conf)
            ax.plot(x, y, 'o', markersize=8, color=color, alpha=0.8)
        
        # Draw skeleton
        skeleton = self.simulator.skeleton
        for child, parents in skeleton.items():
            for parent in parents:
                x = [pose[parent, 0], pose[child, 0]]
                y = [pose[parent, 1], pose[child, 1]]
                ax.plot(x, y, 'b-', alpha=0.5, linewidth=2)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic test data for EdgeSight'
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=100,
        help='Number of sequences to generate (default: 100)'
    )
    parser.add_argument(
        '-r', '--fall-ratio',
        type=float,
        default=0.3,
        help='Ratio of fall samples (0.0-1.0, default: 0.3)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='data/processed',
        help='Output directory (default: data/processed)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate sample visualization'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EdgeSight Synthetic Data Generator")
    print("=" * 60)
    
    generator = SyntheticDatasetGenerator(seed=args.seed)
    
    # Generate dataset
    paths = generator.generate_dataset(
        num_samples=args.num_samples,
        fall_ratio=args.fall_ratio,
        output_dir=args.output
    )
    
    # Generate visualization if requested
    if args.visualize:
        viz_path = os.path.join(args.output, 'synthetic_samples.png')
        generator.generate_single_sample_visualization(viz_path)
    
    print("\n" + "=" * 60)
    print("Done! Use this data for testing without real datasets.")
    print("=" * 60)


if __name__ == '__main__':
    main()
