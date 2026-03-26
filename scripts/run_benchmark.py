"""
One-click benchmark runner for EdgeSight.

Runs the full benchmark suite and generates the PDF report.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print(f"[SUCCESS] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description}")
        print(f"Error: {e}")
        return False


def main():
    """Main entry point."""
    print("="*60)
    print("EdgeSight - One-Click Benchmark Runner")
    print("="*60)

    # Check for models
    fp32_model = Path("model/exported/fallnet_fp32.onnx")
    int8_model = Path("model/exported/fallnet_int8.onnx")

    if not fp32_model.exists():
        print(f"\n[ERROR] FP32 model not found: {fp32_model}")
        print("Export model first: python model/export_onnx.py")
        return 1

    print(f"\n[OK] Found FP32 model: {fp32_model}")
    if int8_model.exists():
        print(f"[OK] Found INT8 model: {int8_model}")
    else:
        print(f"[WARN] INT8 model not found: {int8_model}")
        print("Quantize model for INT8 comparison: python model/quantize.py")

    # Create output directory
    output_dir = Path("inference/benchmark/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    timings_file = output_dir / "timings.json"

    benchmark_cmd = [
        sys.executable, "-m", "inference.benchmark.benchmark",
        "--fp32-model", str(fp32_model),
        "--output", str(timings_file),
        "--runs", "1000",
        "--warmup", "100"
    ]

    if int8_model.exists():
        benchmark_cmd.extend(["--int8-model", str(int8_model)])

    if not run_command(benchmark_cmd, "Running Benchmarks"):
        print("\n[ERROR] Benchmark failed")
        return 1

    # Check timings file exists
    if not timings_file.exists():
        print(f"\n[ERROR] Benchmark output not found: {timings_file}")
        return 1

    # Generate report
    report_file = output_dir / "EdgeSight_Benchmark_Report.pdf"

    report_cmd = [
        sys.executable, "-m", "inference.benchmark.report_generator",
        "--input", str(timings_file),
        "--output", str(report_file)
    ]

    if not run_command(report_cmd, "Generating PDF Report"):
        print("\n[WARNING] Report generation failed, but benchmarks completed")
        return 1

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - Timings: {timings_file}")
    print(f"  - Report:  {report_file}")

    if report_file.exists():
        size_kb = report_file.stat().st_size / 1024
        print(f"  - Size:    {size_kb:.1f} KB")

    print("\nNext steps:")
    print("  - Open the PDF report to view results")
    print("  - Check C++ speedup vs Python baseline")
    print("  - Verify targets: < 20ms (FP32), < 12ms (INT8)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
