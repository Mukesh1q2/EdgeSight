"""
Training script for EdgeSight FallNet model.

Trains the fall detection model with early stopping, learning rate scheduling,
and logging to TensorBoard and/or Weights & Biases.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import FallNet, count_parameters
from data.dataset import create_dataloaders


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10  # Early stopping patience
    lr_patience: int = 5  # LR scheduler patience
    lr_factor: float = 0.5  # LR reduction factor
    device: str = "auto"  # auto, cuda, cpu
    num_workers: int = 0
    seed: int = 42
    data_dir: str = "data/processed"
    checkpoint_dir: str = "model/checkpoints"
    log_dir: str = "model/logs"
    use_wandb: bool = False
    wandb_project: str = "edgesight"
    target_f1: float = 0.90  # Target validation F1 before export
    resume_from: Optional[str] = None


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: TrainingConfig) -> torch.device:
    """Get device for training.

    Args:
        config: Training configuration

    Returns:
        torch.device
    """
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Binary predictions
    y_pred_binary = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(np.mean(y_pred_binary == y_true)),
        "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
    }

    # AUC-ROC (only if both classes present)
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = 0.0

    return metrics


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation metric (higher is better)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def train_epoch(
    model: FallNet,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch.

    Args:
        model: FallNet model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device for training
        epoch: Current epoch number

    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item() * batch_x.size(0)
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(outputs.detach().cpu().numpy())

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute metrics
    avg_loss = total_loss / len(train_loader.dataset)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    metrics = compute_metrics(all_labels, (all_probs >= 0.5).astype(int), all_probs)
    metrics["loss"] = avg_loss

    return avg_loss, metrics


def validate(
    model: FallNet,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Validate model.

    Args:
        model: FallNet model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device for validation

    Returns:
        Tuple of (avg_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    metrics = compute_metrics(all_labels, (all_probs >= 0.5).astype(int), all_probs)
    metrics["loss"] = avg_loss

    return avg_loss, metrics


def save_checkpoint(
    model: FallNet,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Path
) -> None:
    """Save model checkpoint.

    Args:
        model: FallNet model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoint_path)


def train(config: TrainingConfig) -> None:
    """Main training function.

    Args:
        config: Training configuration
    """
    print("="*60)
    print("EdgeSight Model Training")
    print("="*60)

    # Setup
    set_seed(config.seed)
    device = get_device(config)
    print(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    print("\n[Phase 1] Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        random_seed=config.seed
    )

    # Model
    print("\n[Phase 2] Creating model...")
    model = FallNet()
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Loss with class weighting
    pos_weight = train_loader.dataset.get_pos_weight()
    print(f"Using pos_weight={pos_weight:.2f} for class imbalance")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Monitor F1 score
        patience=config.lr_patience,
        factor=config.lr_factor,
        verbose=True
    )

    # Logging
    writer = SummaryWriter(log_dir=log_dir)
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=config.wandb_project, config=asdict(config))
        wandb.watch(model, log="all")

    # Training loop
    print("\n[Phase 3] Training...")
    early_stopping = EarlyStopping(patience=config.patience)
    best_val_f1 = 0.0
    best_epoch = 0
    start_epoch = 1

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_auc": [],
    }

    # Resume from checkpoint
    if config.resume_from:
        resume_path = Path(config.resume_from)
        if resume_path.exists():
            print(f"\n[INFO] Resuming training from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            start_epoch = checkpoint.get("epoch", 0) + 1
            if "metrics" in checkpoint and "f1" in checkpoint["metrics"]:
                best_val_f1 = checkpoint["metrics"]["f1"]
            best_epoch = start_epoch - 1
            print(f"  Loaded epoch {start_epoch - 1} with best F1: {best_val_f1:.4f}")
        else:
            print(f"\n[WARNING] Checkpoint not found: {resume_path}. Starting from scratch.")

    for epoch in range(start_epoch, config.epochs + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_metrics["f1"])

        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/val_f1", val_metrics["f1"], epoch)
        writer.add_scalar("Metrics/val_auc", val_metrics["auc"], epoch)
        writer.add_scalar("Metrics/val_precision", val_metrics["precision"], epoch)
        writer.add_scalar("Metrics/val_recall", val_metrics["recall"], epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
            })

        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])

        # Progress output
        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                checkpoint_dir / "best_model.pt"
            )
            print(f"  [SAVED] New best model (F1: {best_val_f1:.4f})")

        # Early stopping
        if early_stopping(val_metrics["f1"]):
            print(f"\n[Early Stopping] No improvement for {config.patience} epochs")
            break

    # Cleanup
    writer.close()
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"TensorBoard logs: {log_dir}")

    # Save history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Target check
    if best_val_f1 >= config.target_f1:
        print(f"\n[SUCCESS] Target F1 ({config.target_f1}) achieved!")
        print("Ready for ONNX export. Run: python model/export_onnx.py")
    else:
        print(f"\n[WARNING] Target F1 ({config.target_f1}) not achieved ({best_val_f1:.4f})")
        print("Consider: more epochs, larger LSTM (512 hidden), or data augmentation")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train EdgeSight FallNet model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        data_dir=args.data_dir,
        use_wandb=args.use_wandb,
        seed=args.seed,
        resume_from=args.resume,
    )

    train(config)


if __name__ == "__main__":
    main()
