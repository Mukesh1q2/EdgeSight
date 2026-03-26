"""
Python ONNX Runtime inference baseline for EdgeSight.

Provides a simple Python interface for running fall detection inference
using ONNX Runtime - used as the baseline for C++ performance comparison.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Union, Optional, Dict
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort


@dataclass
class InferenceResult:
    """Result from a single inference."""
    probability: float
    latency_ms: float


class PythonFallDetector:
    """Python ONNX Runtime inference wrapper.

    Mirrors the C++ FallDetector API for fair comparison.
    """

    def __init__(
        self,
        model_path: str,
        use_int8: bool = False,
        intra_op_num_threads: int = 4
    ):
        """Initialize Python ONNX Runtime inference.

        Args:
            model_path: Path to ONNX model file
            use_int8: Whether to use INT8 quantization (requires int8 model)
            intra_op_num_threads: Number of threads for intra-op parallelism
        """
        self.model_path = model_path
        self.use_int8 = use_int8

        # Check model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = intra_op_num_threads

        # Execution providers
        providers = ["CPUExecutionProvider"]

        # Load session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Validate input shape
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[1] != 30 or input_shape[2] != 51:
            # Some models have dynamic batch, check other dims
            if len(input_shape) >= 3:
                # Try to get from type info
                pass

        self.last_latency_ms = 0.0

    def predict(self, pose_sequence: List[List[float]]) -> float:
        """Run inference on a single pose sequence.

        Args:
            pose_sequence: 30 frames × 51 features

        Returns:
            Fall probability in [0, 1]
        """
        # Convert to numpy array
        x = np.array(pose_sequence, dtype=np.float32).reshape(1, 30, 51)

        # Run inference with timing
        start = time.perf_counter()
        outputs = self.session.run(
            None,
            {self.input_name: x}
        )
        end = time.perf_counter()

        self.last_latency_ms = (end - start) * 1000

        # Return probability
        prob = float(outputs[0][0])
        return max(0.0, min(1.0, prob))

    def predict_batch(
        self,
        pose_sequences: List[List[List[float]]]
    ) -> List[float]:
        """Run batch inference.

        Args:
            pose_sequences: List of (30, 51) sequences

        Returns:
            List of fall probabilities
        """
        # Convert to numpy array
        batch_size = len(pose_sequences)
        x = np.array(pose_sequences, dtype=np.float32).reshape(batch_size, 30, 51)

        # Run inference with timing
        start = time.perf_counter()
        outputs = self.session.run(
            None,
            {self.input_name: x}
        )
        end = time.perf_counter()

        self.last_latency_ms = (end - start) * 1000

        # Return probabilities
        probs = outputs[0].flatten().tolist()
        return [max(0.0, min(1.0, p)) for p in probs]

    def get_last_latency_ms(self) -> float:
        """Get latency of last inference."""
        return self.last_latency_ms


def benchmark_configuration(
    model_path: str,
    num_runs: int = 1000,
    warmup_runs: int = 100,
    batch_size: int = 1,
    use_int8: bool = False
) -> Dict:
    """Benchmark a specific configuration.

    Args:
        model_path: Path to ONNX model
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (discarded)
        batch_size: Batch size for inference
        use_int8: Whether using INT8 model

    Returns:
        Dictionary with benchmark results
    """
    detector = PythonFallDetector(model_path, use_int8=use_int8)

    # Generate random test data
    np.random.seed(42)

    # Warmup
    for _ in range(warmup_runs):
        if batch_size == 1:
            x = np.random.randn(30, 51).astype(np.float32)
            detector.predict(x.tolist())
        else:
            x = np.random.randn(batch_size, 30, 51).astype(np.float32)
            detector.predict_batch(x.tolist())

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        if batch_size == 1:
            x = np.random.randn(30, 51).astype(np.float32)
            detector.predict(x.tolist())
        else:
            x = np.random.randn(batch_size, 30, 51).astype(np.float32)
            detector.predict_batch(x.tolist())
        latencies.append(detector.get_last_latency_ms())

    # Statistics
    latencies = np.array(latencies)
    results = {
        "model_path": model_path,
        "use_int8": use_int8,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "throughput_fps": 1000.0 / float(np.mean(latencies)) * batch_size,
        "latencies": latencies.tolist(),
    }

    return results


def main():
    """Test the Python inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Python ONNX Runtime inference")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--int8", action="store_true", help="Model is INT8 quantized")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    args = parser.parse_args()

    # Load detector
    print(f"Loading model: {args.model}")
    detector = PythonFallDetector(args.model, use_int8=args.int8)

    # Test single inference
    print("\nTesting single inference...")
    x = np.random.randn(30, 51).astype(np.float32) * 0.5 + 0.5
    x = np.clip(x, 0, 1)
    prob = detector.predict(x.tolist())
    print(f"  Probability: {prob:.4f}")
    print(f"  Latency: {detector.get_last_latency_ms():.2f} ms")

    # Test batch inference
    print("\nTesting batch inference (batch=8)...")
    batch = [x.tolist() for _ in range(8)]
    probs = detector.predict_batch(batch)
    print(f"  Probabilities: {[f'{p:.4f}' for p in probs]}")
    print(f"  Latency: {detector.get_last_latency_ms():.2f} ms")
    print(f"  Per-item latency: {detector.get_last_latency_ms() / 8:.2f} ms")

    # Benchmark
    if args.benchmark:
        print(f"\nRunning benchmark ({args.runs} runs)...")
        results = benchmark_configuration(
            args.model,
            num_runs=args.runs,
            use_int8=args.int8
        )
        print(f"  Mean latency: {results['mean_ms']:.2f} ms")
        print(f"  P95 latency: {results['p95_ms']:.2f} ms")
        print(f"  P99 latency: {results['p99_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_fps']:.1f} clips/sec")


if __name__ == "__main__":
    main()
