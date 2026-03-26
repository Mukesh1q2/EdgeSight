"""
Benchmark suite for EdgeSight.

Compares inference performance across:
- Python + FP32 ONNX Runtime
- Python + INT8 ONNX Runtime
- C++ engine (via compiled binary or ctypes)

Generates latency statistics and saves raw timings for report generation.
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.python_infer import PythonFallDetector, benchmark_configuration


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    num_warmup: int = 100
    num_runs: int = 1000
    batch_size: int = 1
    random_seed: int = 42


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    throughput_clips_per_sec: float
    raw_latencies: List[float]

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding raw latencies for summary)."""
        return {
            "name": self.name,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "std_ms": self.std_ms,
            "throughput_clips_per_sec": self.throughput_clips_per_sec,
        }


class BenchmarkSuite:
    """Orchestrates benchmarking across configurations."""

    def __init__(
        self,
        fp32_model_path: str,
        int8_model_path: Optional[str] = None,
        cpp_binary_path: Optional[str] = None,
        config: Optional[BenchmarkConfig] = None
    ):
        """Initialize benchmark suite.

        Args:
            fp32_model_path: Path to FP32 ONNX model
            int8_model_path: Path to INT8 ONNX model (optional)
            cpp_binary_path: Path to C++ benchmark binary (optional)
            config: Benchmark configuration
        """
        self.fp32_model_path = fp32_model_path
        self.int8_model_path = int8_model_path
        self.cpp_binary_path = cpp_binary_path
        self.config = config or BenchmarkConfig()

        np.random.seed(self.config.random_seed)

    def benchmark_python_fp32(self) -> BenchmarkResult:
        """Benchmark Python FP32 inference."""
        print("\n[1/3] Benchmarking Python + FP32...")

        results = benchmark_configuration(
            self.fp32_model_path,
            num_runs=self.config.num_runs,
            warmup_runs=self.config.num_warmup,
            batch_size=self.config.batch_size,
            use_int8=False
        )

        return BenchmarkResult(
            name="Python + FP32",
            mean_ms=results["mean_ms"],
            median_ms=results["median_ms"],
            p95_ms=results["p95_ms"],
            p99_ms=results["p99_ms"],
            std_ms=results["std_ms"],
            throughput_clips_per_sec=results["throughput_fps"],
            raw_latencies=results["latencies"]
        )

    def benchmark_python_int8(self) -> Optional[BenchmarkResult]:
        """Benchmark Python INT8 inference."""
        if not self.int8_model_path or not Path(self.int8_model_path).exists():
            print("\n[2/3] Skipping Python + INT8 (model not found)")
            return None

        print("\n[2/3] Benchmarking Python + INT8...")

        results = benchmark_configuration(
            self.int8_model_path,
            num_runs=self.config.num_runs,
            warmup_runs=self.config.num_warmup,
            batch_size=self.config.batch_size,
            use_int8=True
        )

        return BenchmarkResult(
            name="Python + INT8",
            mean_ms=results["mean_ms"],
            median_ms=results["median_ms"],
            p95_ms=results["p95_ms"],
            p99_ms=results["p99_ms"],
            std_ms=results["std_ms"],
            throughput_clips_per_sec=results["throughput_fps"],
            raw_latencies=results["latencies"]
        )

    def benchmark_cpp(self) -> Optional[BenchmarkResult]:
        """Benchmark C++ engine inference."""
        if not self.cpp_binary_path or not Path(self.cpp_binary_path).exists():
            print("\n[3/3] Skipping C++ engine (binary not found)")
            print(f"  Expected at: {self.cpp_binary_path}")
            print("  Build with: cmake --build build --config Release")
            return None

        print("\n[3/3] Benchmarking C++ Engine...")

        # Run C++ benchmark binary
        cmd = [
            self.cpp_binary_path,
            "--model", self.fp32_model_path,
            "--runs", str(self.config.num_runs),
            "--warmup", str(self.config.num_warmup)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output (expected JSON on last line)
            lines = result.stdout.strip().split('\n')
            json_line = lines[-1]
            cpp_results = json.loads(json_line)

            return BenchmarkResult(
                name="C++ FP32",
                mean_ms=cpp_results["mean_ms"],
                median_ms=cpp_results["median_ms"],
                p95_ms=cpp_results["p95_ms"],
                p99_ms=cpp_results["p99_ms"],
                std_ms=cpp_results["std_ms"],
                throughput_clips_per_sec=cpp_results["throughput_clips_per_sec"],
                raw_latencies=cpp_results.get("latencies", [])
            )

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] C++ benchmark failed: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] Failed to parse C++ benchmark output: {e}")
            return None

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks and return results."""
        results = []

        # Python FP32
        results.append(self.benchmark_python_fp32())

        # Python INT8
        int8_result = self.benchmark_python_int8()
        if int8_result:
            results.append(int8_result)

        # C++
        cpp_result = self.benchmark_cpp()
        if cpp_result:
            results.append(cpp_result)

        return results


def compute_speedup(
    baseline: BenchmarkResult,
    target: BenchmarkResult
) -> Dict[str, float]:
    """Compute speedup ratios."""
    return {
        "mean_speedup": baseline.mean_ms / target.mean_ms,
        "p95_speedup": baseline.p95_ms / target.p95_ms,
        "throughput_speedup": target.throughput_clips_per_sec / baseline.throughput_clips_per_sec,
    }


def print_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark results as formatted table."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)

    # Header
    print(f"{'Configuration':<20} {'Mean':<12} {'Median':<12} {'P95':<12} {'P99':<12} {'Throughput':<15}")
    print("-"*80)

    # Results
    for r in results:
        print(
            f"{r.name:<20} "
            f"{r.mean_ms:>10.2f}ms "
            f"{r.median_ms:>10.2f}ms "
            f"{r.p95_ms:>10.2f}ms "
            f"{r.p99_ms:>10.2f}ms "
            f"{r.throughput_clips_per_sec:>10.1f} clips/s"
        )

    # Speedup comparison (if C++ available)
    cpp_results = [r for r in results if "C++" in r.name]
    python_results = [r for r in results if "Python" in r.name]

    if cpp_results and python_results:
        print("\n" + "-"*80)
        print("SPEEDUP COMPARISON (C++ vs Python)")
        print("-"*80)

        baseline = python_results[0]  # Python FP32 as baseline
        for cpp_r in cpp_results:
            speedups = compute_speedup(baseline, cpp_r)
            print(f"\n{cpp_r.name} vs {baseline.name}:")
            print(f"  Mean latency: {speedups['mean_speedup']:.2f}x faster")
            print(f"  P95 latency: {speedups['p95_speedup']:.2f}x faster")
            print(f"  Throughput: {speedups['throughput_speedup']:.2f}x higher")

            # Check against target
            target_speedup = 2.5
            if speedups['mean_speedup'] >= target_speedup:
                print(f"  [PASS] Exceeds target speedup ({target_speedup}x)")
            else:
                print(f"  [WARN] Below target speedup ({target_speedup}x)")

    print("="*80)


def save_results(
    results: List[BenchmarkResult],
    output_path: Path
) -> None:
    """Save benchmark results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": [r.to_dict() for r in results],
        "raw_timings": {
            r.name: r.raw_latencies for r in results
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n[SAVED] Results to: {output_path}")


def find_cpp_binary() -> Optional[str]:
    """Auto-detect cpp_benchmark binary from common locations."""
    possible_paths = [
        "build/bin/cpp_benchmark.exe",
        "build/bin/cpp_benchmark",
        "build/Release/cpp_benchmark.exe",
        "build/Debug/cpp_benchmark.exe",
        "inference/engine/build/cpp_benchmark.exe",
        "inference/engine/build/cpp_benchmark",
        "cmake-build-release/cpp_benchmark.exe",
        "cmake-build-debug/cpp_benchmark.exe",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    # Try to find in PATH
    try:
        import shutil
        cpp_in_path = shutil.which("cpp_benchmark")
        if cpp_in_path:
            return cpp_in_path
    except:
        pass
    
    return None


def main():
    """Main entry point."""
    import argparse

    # Auto-detect C++ binary
    default_cpp_binary = find_cpp_binary()
    
    parser = argparse.ArgumentParser(description="EdgeSight Benchmark Suite")
    parser.add_argument("--fp32-model", type=str,
                        default="model/exported/fallnet_fp32.onnx")
    parser.add_argument("--int8-model", type=str,
                        default="model/exported/fallnet_int8.onnx")
    parser.add_argument("--cpp-binary", type=str,
                        default=default_cpp_binary or "build/bin/cpp_benchmark.exe",
                        help="Path to C++ benchmark binary (auto-detected if available)")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output", type=str,
                        default="inference/benchmark/results/timings.json")
    parser.add_argument("--compare-all", action="store_true",
                        help="Generate comparison report including C++ results")
    args = parser.parse_args()

    print("="*80)
    print("EdgeSight Benchmark Suite")
    print("="*80)

    # Configuration
    config = BenchmarkConfig(
        num_warmup=args.warmup,
        num_runs=args.runs
    )

    # Create benchmark suite
    suite = BenchmarkSuite(
        fp32_model_path=args.fp32_model,
        int8_model_path=args.int8_model if Path(args.int8_model).exists() else None,
        cpp_binary_path=args.cpp_binary if Path(args.cpp_binary).exists() else None,
        config=config
    )

    # Run benchmarks
    results = suite.run_all()

    if not results:
        print("\n[ERROR] No benchmarks completed successfully")
        return 1

    # Print results
    print_results(results)

    # Save results
    save_results(results, Path(args.output))

    print("\nNext: Generate report")
    print("  python inference/benchmark/report_generator.py")
    
    # Auto-generate comparison report if --compare-all flag used
    if args.compare_all and results:
        try:
            import subprocess
            report_cmd = [
                sys.executable,
                "inference/benchmark/report_generator.py",
                "--timings", args.output,
                "--output", "inference/benchmark/results/benchmark_report.pdf"
            ]
            print("\n[REPORT] Generating comparison PDF...")
            subprocess.run(report_cmd, check=True)
        except Exception as e:
            print(f"[WARN] Failed to auto-generate report: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
