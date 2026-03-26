"""
INT8 Quantization script for EdgeSight FallNet ONNX model.

Applies post-training static quantization using ONNX Runtime.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import FallNet

import torch


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    calibration_samples: int = 100
    quant_format: str = "QLinearOps"  # or "QDQ"
    activation_type: str = "QInt8"    # or "QUInt8"
    weight_type: str = "QInt8"        # or "QUInt8"
    optimize_model: bool = True


class CalibrationDataProvider(CalibrationDataReader):
    """Provides calibration data for static quantization."""

    def __init__(
        self,
        model: FallNet,
        num_samples: int = 100,
        input_shape: tuple = (1, 30, 51),
        data_path: Optional[Path] = None
    ):
        """Initialize calibration data provider.

        Args:
            model: PyTorch model to generate representative data
            num_samples: Number of calibration samples
            input_shape: Shape of input tensor
            data_path: Optional path to real data (uses synthetic if None)
        """
        self.model = model
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.data_path = data_path
        self.current_index = 0

        # Generate or load calibration data
        if data_path and data_path.exists():
            self.data = self._load_real_data()
        else:
            self.data = self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> List[np.ndarray]:
        """Generate synthetic calibration data.

        Generates representative samples using the model's expected
        input distribution (random normal for simplicity).

        Returns:
            List of input tensors
        """
        print(f"Generating {self.num_samples} synthetic calibration samples...")
        data = []
        for _ in range(self.num_samples):
            # Generate random sample with similar stats to real data
            sample = np.random.randn(*self.input_shape).astype(np.float32) * 0.5 + 0.5
            sample = np.clip(sample, 0, 1)  # Keypoints are normalized [0, 1]
            data.append({"pose_sequence": sample})
        return data

    def _load_real_data(self) -> List[np.ndarray]:
        """Load real calibration data from processed dataset.

        Returns:
            List of input tensors
        """
        print(f"Loading {self.num_samples} real calibration samples from {self.data_path}...")
        X = np.load(self.data_path)

        # Randomly sample calibration data
        indices = np.random.choice(len(X), min(self.num_samples, len(X)), replace=False)
        data = []
        for idx in indices:
            sample = X[idx:idx+1].astype(np.float32)  # Keep batch dim
            data.append({"pose_sequence": sample})

        return data

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Get next calibration sample.

        Returns:
            Next sample or None if exhausted
        """
        if self.current_index < len(self.data):
            sample = self.data[self.current_index]
            self.current_index += 1
            return sample
        return None

    def rewind(self) -> None:
        """Reset to beginning of calibration data."""
        self.current_index = 0


def quantize_model(
    input_model_path: Path,
    output_model_path: Path,
    calibration_data_reader: CalibrationDataReader,
    config: QuantizationConfig
) -> Dict[str, any]:
    """Quantize ONNX model to INT8.

    Args:
        input_model_path: Path to FP32 ONNX model
        output_model_path: Path to save INT8 ONNX model
        calibration_data_reader: Provider of calibration data
        config: Quantization configuration

    Returns:
        Dictionary with quantization metadata
    """
    print("="*60)
    print("INT8 Quantization")
    print("="*60)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Map config types to QuantType
    activation_quant_type = QuantType.QInt8 if config.activation_type == "QInt8" else QuantType.QUInt8
    weight_quant_type = QuantType.QInt8 if config.weight_type == "QInt8" else QuantType.QUInt8

    print(f"\n[Phase 1] Running static quantization...")
    print(f"  Calibration samples: {config.calibration_samples}")
    print(f"  Activation type: {config.activation_type}")
    print(f"  Weight type: {config.weight_type}")

    try:
        quantize_static(
            model_input=str(input_model_path),
            model_output=str(output_model_path),
            calibration_data_reader=calibration_data_reader,
            quant_format=onnxruntime.quantization.QuantFormat.QLinearOps,
            activation_type=activation_quant_type,
            weight_type=weight_quant_type,
            optimize_model=config.optimize_model
        )

        print(f"[SUCCESS] Quantized model saved to: {output_model_path}")

        # Calculate size reduction
        fp32_size = input_model_path.stat().st_size / (1024 * 1024)
        int8_size = output_model_path.stat().st_size / (1024 * 1024)
        reduction = (1 - int8_size / fp32_size) * 100

        print(f"\n[Phase 2] Size comparison:")
        print(f"  FP32 model: {fp32_size:.2f} MB")
        print(f"  INT8 model: {int8_size:.2f} MB")
        print(f"  Reduction:  {reduction:.1f}%")

        return {
            "input_path": str(input_model_path),
            "output_path": str(output_model_path),
            "fp32_size_mb": fp32_size,
            "int8_size_mb": int8_size,
            "size_reduction_percent": reduction,
            "success": True
        }

    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}")
        return {
            "input_path": str(input_model_path),
            "output_path": str(output_model_path),
            "success": False,
            "error": str(e)
        }


def validate_quantized_model(
    fp32_path: Path,
    int8_path: Path,
    num_samples: int = 100,
    tolerance: float = 0.02
) -> Dict[str, float]:
    """Validate quantized model accuracy.

    Args:
        fp32_path: Path to FP32 model
        int8_path: Path to INT8 model
        num_samples: Number of validation samples
        tolerance: Maximum acceptable accuracy drop

    Returns:
        Dictionary with validation results
    """
    print(f"\n[Phase 3] Validating quantized model...")

    # Create sessions
    fp32_session = ort.InferenceSession(
        str(fp32_path),
        providers=["CPUExecutionProvider"]
    )
    int8_session = ort.InferenceSession(
        str(int8_path),
        providers=["CPUExecutionProvider"]
    )

    fp32_input_name = fp32_session.get_inputs()[0].name
    int8_input_name = int8_session.get_inputs()[0].name

    # Test on random samples
    max_diff = 0.0
    total_diff = 0.0

    for _ in range(num_samples):
        # Random input (normalized keypoints)
        x = np.random.randn(1, 30, 51).astype(np.float32) * 0.5 + 0.5
        x = np.clip(x, 0, 1)

        # Run inference
        fp32_out = fp32_session.run(None, {fp32_input_name: x})[0]
        int8_out = int8_session.run(None, {int8_input_name: x})[0]

        # Compare
        diff = np.abs(fp32_out - int8_out).mean()
        total_diff += diff
        max_diff = max(max_diff, np.abs(fp32_out - int8_out).max())

    avg_diff = total_diff / num_samples

    print(f"  Average output difference: {avg_diff:.6f}")
    print(f"  Maximum output difference: {max_diff:.6f}")

    # Accuracy comparison would require labeled test data
    # For now, we check output difference
    passed = max_diff < tolerance

    if passed:
        print(f"[SUCCESS] Validation passed (max diff < {tolerance})")
    else:
        print(f"[WARNING] High output difference detected")
        print(f"  This may or may not affect accuracy significantly")

    return {
        "avg_output_diff": avg_diff,
        "max_output_diff": max_diff,
        "tolerance": tolerance,
        "passed": passed
    }


def compare_accuracy(
    fp32_path: Path,
    int8_path: Path,
    data_path: Path,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compare accuracy of FP32 and INT8 models on labeled data.

    Args:
        fp32_path: Path to FP32 model
        int8_path: Path to INT8 model
        data_path: Path to X.npy and y.npy files
        threshold: Classification threshold

    Returns:
        Dictionary with accuracy metrics
    """
    print(f"\n[Phase 4] Comparing accuracy on labeled data...")

    # Load data
    X = np.load(data_path / "X.npy")
    y = np.load(data_path / "y.npy")

    # Sample subset for speed
    sample_size = min(1000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]

    # Create sessions
    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    fp32_input_name = fp32_session.get_inputs()[0].name
    int8_input_name = int8_session.get_inputs()[0].name

    # Run inference
    fp32_preds = []
    int8_preds = []

    for i in range(len(X_sample)):
        x = X_sample[i:i+1].astype(np.float32)
        fp32_out = fp32_session.run(None, {fp32_input_name: x})[0][0]
        int8_out = int8_session.run(None, {int8_input_name: x})[0][0]

        fp32_preds.append(1 if fp32_out >= threshold else 0)
        int8_preds.append(1 if int8_out >= threshold else 0)

    fp32_preds = np.array(fp32_preds)
    int8_preds = np.array(int8_preds)

    # Calculate accuracy
    fp32_acc = (fp32_preds == y_sample).mean()
    int8_acc = (int8_preds == y_sample).mean()
    acc_drop = fp32_acc - int8_acc

    print(f"\nAccuracy Comparison:")
    print(f"  FP32 accuracy: {fp32_acc:.4f}")
    print(f"  INT8 accuracy: {int8_acc:.4f}")
    print(f"  Accuracy drop: {acc_drop:.4f} ({acc_drop*100:.2f}%)")

    max_drop = 0.02  # 2% max drop
    if acc_drop <= max_drop:
        print(f"[SUCCESS] Accuracy drop within tolerance (< {max_drop*100}%)")
    else:
        print(f"[WARNING] Accuracy drop exceeds {max_drop*100}%")

    return {
        "fp32_accuracy": fp32_acc,
        "int8_accuracy": int8_acc,
        "accuracy_drop": acc_drop,
        "sample_size": sample_size,
        "passed": acc_drop <= max_drop
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quantize FallNet to INT8")
    parser.add_argument(
        "--input",
        type=str,
        default="model/exported/fallnet_fp32.onnx",
        help="Path to FP32 ONNX model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/exported/fallnet_int8.onnx",
        help="Path to save INT8 ONNX model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with processed data for calibration"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--skip-accuracy-check",
        action="store_true",
        help="Skip accuracy comparison (use if data not available)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    data_dir = Path(args.data_dir)

    if not input_path.exists():
        print(f"[ERROR] FP32 model not found: {input_path}")
        print("Export a model first: python model/export_onnx.py")
        return

    # Load model for calibration data generation
    print("Loading PyTorch model for calibration...")
    model = FallNet()
    model.eval()

    # Create calibration data provider
    x_path = data_dir / "X.npy"
    if x_path.exists():
        cal_data = CalibrationDataProvider(
            model=model,
            num_samples=args.calibration_samples,
            data_path=x_path
        )
    else:
        print(f"[WARNING] Real data not found, using synthetic calibration data")
        cal_data = CalibrationDataProvider(
            model=model,
            num_samples=args.calibration_samples
        )

    # Quantize
    config = QuantizationConfig(calibration_samples=args.calibration_samples)
    result = quantize_model(input_path, output_path, cal_data, config)

    if not result["success"]:
        return

    # Validate
    val_result = validate_quantized_model(input_path, output_path)

    # Compare accuracy if data available
    if not args.skip_accuracy_check and x_path.exists():
        acc_result = compare_accuracy(input_path, output_path, data_dir)
    else:
        acc_result = {"passed": None, "note": "Skipped (no data available)"}

    # Summary
    print("\n" + "="*60)
    print("QUANTIZATION SUMMARY")
    print("="*60)
    print(f"FP32 model: {result['input_path']}")
    print(f"INT8 model: {result['output_path']}")
    print(f"Size reduction: {result['size_reduction_percent']:.1f}%")
    print(f"Max output diff: {val_result['max_output_diff']:.6f}")
    if 'fp32_accuracy' in acc_result:
        print(f"Accuracy drop: {acc_result['accuracy_drop']*100:.2f}%")
        print(f"Accuracy check passed: {acc_result['passed']}")
    print("="*60)

    print("\nNext step: Run benchmarks")
    print("  python scripts/run_benchmark.py")


if __name__ == "__main__":
    main()
