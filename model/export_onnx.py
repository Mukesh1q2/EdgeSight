"""
ONNX export script for EdgeSight FallNet model.

Exports the trained PyTorch model to ONNX format with validation.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import onnx
import onnxruntime as ort

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import FallNet


def export_to_onnx(
    model: FallNet,
    output_path: Path,
    input_shape: Tuple[int, int, int] = (1, 30, 51),
    opset_version: int = 17,
    validate: bool = True,
    num_validation_samples: int = 10,
    tolerance: float = 0.01,
    model_metadata: Dict[str, str] = None
) -> Dict[str, any]:
    """Export PyTorch model to ONNX format.

    Args:
        model: Trained FallNet model
        output_path: Path to save ONNX model
        input_shape: Shape of input tensor (batch, seq_len, features)
        opset_version: ONNX opset version
        validate: Whether to validate export
        num_validation_samples: Number of samples for validation
        tolerance: Maximum allowed difference between PyTorch and ONNX outputs

    Returns:
        Dictionary with export metadata
    """
    print("="*60)
    print("ONNX Export")
    print("="*60)

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Define input/output names
    input_names = ["pose_sequence"]
    output_names = ["fall_probability"]

    # Dynamic axes for batch size
    dynamic_axes = {
        "pose_sequence": {0: "batch_size"},
        "fall_probability": {0: "batch_size"}
    }

    # Export
    print(f"\n[Phase 1] Exporting to ONNX...")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")
    print(f"  Dynamic axes: {dynamic_axes}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    print(f"[SUCCESS] Model exported to: {output_path}")

    # Inject metadata
    if model_metadata:
        print(f"  Injecting metadata...")
        onnx_model = onnx.load(output_path)
        
        # Set doc string
        doc_parts = []
        for k, v in model_metadata.items():
            meta = onnx_model.metadata_props.add()
            meta.key = k
            meta.value = str(v)
            doc_parts.append(f"{k}: {v}")
            
        onnx_model.doc_string = " | ".join(doc_parts)
        onnx.save(onnx_model, output_path)
        print("  [SUCCESS] Metadata injected")

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")

    # Validate ONNX model
    print(f"\n[Phase 2] Validating ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] ONNX model validation passed")

    # Validate outputs match
    metadata = {
        "export_path": str(output_path),
        "file_size_mb": file_size_mb,
        "opset_version": opset_version,
        "input_shape": input_shape,
        "dynamic_axes": dynamic_axes,
        "validation_passed": False,
        "max_diff": None,
    }

    if validate:
        print(f"\n[Phase 3] Validating outputs against PyTorch...")
        max_diff = validate_export(
            model, output_path, num_samples=num_validation_samples
        )
        metadata["max_diff"] = max_diff
        metadata["validation_passed"] = max_diff < tolerance

        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Tolerance: {tolerance:.6f}")

        if max_diff < tolerance:
            print("[SUCCESS] Output validation passed")
        else:
            print(f"[WARNING] Output validation failed (diff > {tolerance})")
            print("  Model may still work correctly - investigate further")

    return metadata


def validate_export(
    model: FallNet,
    onnx_path: Path,
    num_samples: int = 10,
    batch_sizes: list = [1, 8, 16]
) -> float:
    """Validate ONNX export by comparing outputs with PyTorch.

    Args:
        model: Original PyTorch model
        onnx_path: Path to ONNX model
        num_samples: Number of random samples to test
        batch_sizes: List of batch sizes to test

    Returns:
        Maximum absolute difference between PyTorch and ONNX outputs
    """
    model.eval()

    # Create ONNX Runtime session
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    max_diff = 0.0

    for batch_size in batch_sizes:
        for _ in range(num_samples // len(batch_sizes)):
            # Random input
            x = torch.randn(batch_size, 30, 51)

            # PyTorch output
            with torch.no_grad():
                pytorch_out = model(x).numpy()

            # ONNX output
            onnx_out = session.run(None, {input_name: x.numpy()})[0]

            # Compare
            diff = np.abs(pytorch_out - onnx_out).max()
            max_diff = max(max_diff, diff)

    return float(max_diff)


def inspect_onnx_model(onnx_path: Path) -> None:
    """Print information about an ONNX model.

    Args:
        onnx_path: Path to ONNX model
    """
    print("\n" + "="*60)
    print("ONNX Model Inspection")
    print("="*60)

    # Load and check
    model = onnx.load(onnx_path)

    # Model info
    print(f"\nIR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Domain: {model.domain}")
    print(f"Model Version: {model.model_version}")

    # Graph info
    graph = model.graph
    print(f"\nGraph:")
    print(f"  Name: {graph.name}")
    print(f"  Inputs:")
    for input in graph.input:
        print(f"    - {input.name}: {input.type}")
    print(f"  Outputs:")
    for output in graph.output:
        print(f"    - {output.name}: {output.type}")

    # Nodes summary
    op_types = {}
    for node in graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    print(f"\nOperations ({len(graph.node)} total):")
    for op_type, count in sorted(op_types.items(), key=lambda x: -x[1]):
        print(f"  {op_type}: {count}")

    # Parameters
    param_count = sum(
        np.prod([dim.dim_value for dim in init.dims])
        for init in graph.initializer
    )
    print(f"\nTotal parameters: {param_count:,}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export FallNet to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/exported/fallnet_fp32.onnx",
        help="Output path for ONNX model"
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export")
    parser.add_argument("--validate", action="store_true", default=True, help="Validate export")
    parser.add_argument("--inspect", action="store_true", help="Inspect ONNX model after export")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Train a model first: python model/train.py")
        return

    # Load model
    print("Loading model from checkpoint...")
    model = FallNet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model_metadata = {"epoch": str(checkpoint.get('epoch', 'unknown'))}
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print(f"Validation metrics: {metrics}")
        for k, v in metrics.items():
            model_metadata[f"val_{k}"] = f"{v:.4f}"

    import datetime
    model_metadata["export_time"] = datetime.datetime.now().isoformat()

    # Export
    metadata = export_to_onnx(
        model=model,
        output_path=output_path,
        input_shape=(args.batch_size, 30, 51),
        opset_version=args.opset,
        validate=args.validate,
        model_metadata=model_metadata
    )

    # Inspect
    if args.inspect:
        inspect_onnx_model(output_path)

    # Summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"Output: {metadata['export_path']}")
    print(f"File size: {metadata['file_size_mb']:.2f} MB")
    print(f"Opset version: {metadata['opset_version']}")
    print(f"Validation passed: {metadata['validation_passed']}")
    if metadata['max_diff'] is not None:
        print(f"Max output diff: {metadata['max_diff']:.6f}")

    print("\nNext step: Run quantization")
    print("  python model/quantize.py")


if __name__ == "__main__":
    main()
