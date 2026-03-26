"""
PyTorch Dataset class for EdgeSight fall detection.

Provides train/val/test splits with stratification for class balance.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


class FallDetectionDataset(Dataset):
    """PyTorch Dataset for fall detection pose sequences.

    Loads preprocessed X.npy and y.npy files containing:
    - X: (N, 30, 51) - N clips of 30 frames with 51 keypoint features
    - y: (N,) - binary labels (1=fall, 0=non-fall)
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        transform: Optional[Callable] = None
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing X.npy and y.npy
            split: One of 'train', 'val', 'test', 'all'
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducible splits
            transform: Optional transform to apply to samples
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Validate split ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        # Load data
        self.X, self.y = self._load_data()

        # Create stratified split
        self.indices = self._create_split(
            train_ratio, val_ratio, test_ratio, random_seed
        )

        print(f"[{split}] Dataset: {len(self.indices)} samples")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load X.npy and y.npy from data directory.

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        x_path = self.data_dir / "X.npy"
        y_path = self.data_dir / "y.npy"

        if not x_path.exists() or not y_path.exists():
            # Try loading individual dataset files
            x_files = list(self.data_dir.glob("X_*.npy"))
            y_files = list(self.data_dir.glob("y_*.npy"))

            if not x_files:
                raise FileNotFoundError(
                    f"No data files found in {self.data_dir}. "
                    "Run preprocess.py first."
                )

            # Combine all available datasets
            X_list = []
            y_list = []
            for x_file in x_files:
                dataset_name = x_file.stem.replace("X_", "")
                y_file = self.data_dir / f"y_{dataset_name}.npy"

                if y_file.exists():
                    X_list.append(np.load(x_file))
                    y_list.append(np.load(y_file))
                    print(f"[INFO] Loaded {dataset_name}: {X_list[-1].shape}")

            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)

            # Save combined for future use
            np.save(x_path, X)
            np.save(y_path, y)
        else:
            X = np.load(x_path)
            y = np.load(y_path)

        # Validate shapes
        assert X.ndim == 3, f"Expected X to be 3D, got shape {X.shape}"
        assert y.ndim == 1, f"Expected y to be 1D, got shape {y.shape}"
        assert X.shape[0] == y.shape[0], \
            f"X and y must have same length: {X.shape[0]} vs {y.shape[0]}"
        assert X.shape[1] == 30, f"Expected 30 frames per clip, got {X.shape[1]}"
        assert X.shape[2] == 51, f"Expected 51 features per frame, got {X.shape[2]}"

        print(f"[INFO] Loaded data: X.shape={X.shape}, y.shape={y.shape}")

        # Print class distribution
        fall_count = int(y.sum())
        non_fall_count = len(y) - fall_count
        print(f"[INFO] Class distribution: {fall_count} falls, {non_fall_count} non-falls")

        return X, y

    def _create_split(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: int
    ) -> np.ndarray:
        """Create stratified train/val/test split.

        Args:
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion
            random_seed: Random seed for reproducibility

        Returns:
            Array of indices for the requested split
        """
        all_indices = np.arange(len(self.y))

        if self.split == "all":
            return all_indices

        # First split: train vs (val + test)
        val_test_ratio = val_ratio + test_ratio
        train_idx, temp_idx = train_test_split(
            all_indices,
            test_size=val_test_ratio,
            stratify=self.y,
            random_state=random_seed
        )

        # Second split: val vs test
        temp_labels = self.y[temp_idx]
        val_ratio_adjusted = val_ratio / val_test_ratio
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_labels,
            random_state=random_seed
        )

        if self.split == "train":
            return train_idx
        elif self.split == "val":
            return val_idx
        elif self.split == "test":
            return test_idx
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Index within this split

        Returns:
            Tuple of (features, label) as torch tensors
        """
        real_idx = self.indices[idx]
        x = self.X[real_idx]
        y = self.y[real_idx]

        # Convert to torch tensors
        x_tensor = torch.from_numpy(x).float()  # Shape: (30, 51)
        y_tensor = torch.tensor(y, dtype=torch.float32)  # Shape: scalar

        if self.transform:
            x_tensor = self.transform(x_tensor)

        return x_tensor, y_tensor

    def get_class_counts(self) -> Tuple[int, int]:
        """Get fall and non-fall counts for this split.

        Returns:
            Tuple of (fall_count, non_fall_count)
        """
        labels = self.y[self.indices]
        fall_count = int(labels.sum())
        non_fall_count = len(labels) - fall_count
        return fall_count, non_fall_count

    def get_pos_weight(self) -> float:
        """Calculate positive weight for BCE loss to handle imbalance.

        Returns:
            Weight factor for positive class
        """
        fall_count, non_fall_count = self.get_class_counts()
        if fall_count == 0:
            return 1.0
        return non_fall_count / fall_count


def create_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 32,
    num_workers: int = 0,
    random_seed: int = 42,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders.

    Args:
        data_dir: Directory containing preprocessed .npy files
        batch_size: Batch size for all splits
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducibility
        train_transform: Optional transform for training data
        val_transform: Optional transform for validation/test data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = FallDetectionDataset(
        data_dir=data_dir,
        split="train",
        random_seed=random_seed,
        transform=train_transform
    )

    val_dataset = FallDetectionDataset(
        data_dir=data_dir,
        split="val",
        random_seed=random_seed,
        transform=val_transform
    )

    test_dataset = FallDetectionDataset(
        data_dir=data_dir,
        split="test",
        random_seed=random_seed,
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Ensure consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Print summary
    print("\n" + "="*60)
    print("DATALOADER SUMMARY")
    print("="*60)
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"Batch size: {batch_size}")

    # Class distribution
    train_falls, train_non = train_dataset.get_class_counts()
    val_falls, val_non = val_dataset.get_class_counts()
    test_falls, test_non = test_dataset.get_class_counts()

    print(f"\nClass distribution:")
    print(f"  Train: {train_falls} falls, {train_non} non-falls")
    print(f"  Val:   {val_falls} falls, {val_non} non-falls")
    print(f"  Test:  {test_falls} falls, {test_non} non-falls")

    pos_weight = train_dataset.get_pos_weight()
    print(f"\nPositive weight (for BCE loss): {pos_weight:.2f}")
    print("="*60)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser(description="Test FallDetectionDataset")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("Testing FallDetectionDataset...")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # Test a batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  X shape: {batch_x.shape}")  # Should be (batch_size, 30, 51)
    print(f"  y shape: {batch_y.shape}")  # Should be (batch_size,)
    print(f"  X dtype: {batch_x.dtype}")
    print(f"  y dtype: {batch_y.dtype}")

    print("\nDataset test complete!")
