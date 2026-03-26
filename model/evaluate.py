"""
Evaluation script for EdgeSight FallNet model.

Computes comprehensive metrics (accuracy, precision, recall, F1, AUC-ROC)
and generates confusion matrix and ROC curve visualizations.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import FallNet
from data.dataset import FallDetectionDataset, create_dataloaders


def load_checkpoint(model: FallNet, checkpoint_path: Path, device: torch.device) -> Dict:
    """Load model from checkpoint.

    Args:
        model: FallNet model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def evaluate_model(
    model: FallNet,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on test data.

    Args:
        model: FallNet model
        test_loader: Test data loader
        device: Device for inference
        threshold: Classification threshold

    Returns:
        Tuple of (true_labels, predicted_labels, predicted_probabilities)
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_labels.extend(batch_y.numpy())
            all_probs.extend(outputs.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    return y_true, y_pred, y_prob


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # AUC-ROC (only if both classes present)
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    class_names: List[str] = ["Non-Fall", "Fall"]
) -> None:
    """Plot and save confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save figure
        class_names: Class names for labels
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"}
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Fall Detection")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] Confusion matrix to: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    auc_score: float
) -> None:
    """Plot and save ROC curve.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
        auc_score: AUC score for annotation
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fall Detection")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] ROC curve to: {save_path}")


def plot_metrics_comparison(
    metrics: Dict[str, float],
    save_path: Path
) -> None:
    """Plot metrics comparison bar chart.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save figure
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc"],
    ]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metric_names)))
    bars = plt.bar(metric_names, values, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    plt.ylim([0.0, 1.05])
    plt.ylabel("Score")
    plt.title("EdgeSight Model Performance Metrics")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] Metrics comparison to: {save_path}")


def evaluate(
    checkpoint_path: Path,
    data_dir: str = "data/processed",
    batch_size: int = 32,
    device: str = "auto",
    results_dir: str = "model/results",
    threshold: float = 0.5
) -> Dict[str, float]:
    """Main evaluation function.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory with processed data
        batch_size: Batch size for evaluation
        device: Device for inference
        results_dir: Directory to save results
        threshold: Classification threshold

    Returns:
        Dictionary of computed metrics
    """
    print("="*60)
    print("EdgeSight Model Evaluation")
    print("="*60)

    # Setup
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else (device if device != "auto" else "cpu"))
    print(f"Using device: {device}")

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[Phase 1] Loading test data...")
    _, _, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0
    )

    # Load model
    print("\n[Phase 2] Loading model...")
    model = FallNet()
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    if "metrics" in checkpoint:
        print(f"Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation metrics at checkpoint: {checkpoint['metrics']}")

    # Evaluate
    print("\n[Phase 3] Running evaluation...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, threshold)

    # Compute metrics
    print("\n[Phase 4] Computing metrics...")
    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    # Print results
    print("\n" + "="*60)
    print("TEST SET METRICS")
    print("="*60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"AUC-ROC:     {metrics['auc']:.4f}")
    print("="*60)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Non-Fall", "Fall"],
        digits=4
    ))

    # Generate visualizations
    print("\n[Phase 5] Generating visualizations...")
    plot_confusion_matrix(y_true, y_pred, results_path / "confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, results_path / "roc_curve.png", metrics["auc"])
    plot_metrics_comparison(metrics, results_path / "metrics_comparison.png")

    # Save metrics to file
    metrics_file = results_path / "test_metrics.json"
    import json
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVED] Metrics to: {metrics_file}")

    # Target check
    target_f1 = 0.88
    if metrics["f1"] >= target_f1:
        print(f"\n[SUCCESS] Target F1 ({target_f1}) achieved: {metrics['f1']:.4f}")
    else:
        print(f"\n[WARNING] Target F1 ({target_f1}) not achieved: {metrics['f1']:.4f}")

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate EdgeSight FallNet model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--results-dir", type=str, default="model/results")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint),
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
        results_dir=args.results_dir,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
