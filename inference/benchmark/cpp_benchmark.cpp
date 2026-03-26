/**
 * @file cpp_benchmark.cpp
 * @brief Standalone C++ benchmark executable
 *
 * This binary is called by the Python benchmark suite to measure
 * C++ inference performance. Outputs JSON results to stdout.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cstring>

#include "fall_detector.h"

using namespace edgesight;

// Generate random pose sequence (normalized keypoints)
std::vector<std::vector<float>> generate_sequence(int seq_len, int features, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::vector<float>> sequence;

    for (int i = 0; i < seq_len; ++i) {
        std::vector<float> frame;
        for (int j = 0; j < features; ++j) {
            frame.push_back(dist(rng));
        }
        sequence.push_back(frame);
    }

    return sequence;
}

struct BenchmarkResult {
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double std_ms = 0.0;
    double throughput_clips_per_sec = 0.0;
};

// Simple statistics
double compute_mean(const std::vector<double>& data) {
    double sum = 0.0;
    for (double x : data) sum += x;
    return sum / data.size();
}

double compute_std(const std::vector<double>& data, double mean) {
    double sum_sq = 0.0;
    for (double x : data) {
        double diff = x - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / data.size());
}

double compute_percentile(std::vector<double> sorted_data, double p) {
    if (sorted_data.empty()) return 0.0;
    std::sort(sorted_data.begin(), sorted_data.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (sorted_data.size() - 1));
    return sorted_data[idx];
}

BenchmarkResult run_benchmark(
    FallDetector& detector,
    int num_warmup,
    int num_runs,
    std::mt19937& rng
) {
    const int seq_len = 30;
    const int features = 51;

    // Warmup
    for (int i = 0; i < num_warmup; ++i) {
        auto seq = generate_sequence(seq_len, features, rng);
        detector.predict(seq);
    }

    // Benchmark
    std::vector<double> latencies;
    latencies.reserve(num_runs);

    for (int i = 0; i < num_runs; ++i) {
        auto seq = generate_sequence(seq_len, features, rng);
        detector.predict(seq);
        latencies.push_back(detector.get_last_latency_ms());
    }

    // Compute statistics
    BenchmarkResult result;
    result.mean_ms = compute_mean(latencies);
    result.std_ms = compute_std(latencies, result.mean_ms);
    result.median_ms = compute_percentile(latencies, 50.0);
    result.p95_ms = compute_percentile(latencies, 95.0);
    result.p99_ms = compute_percentile(latencies, 99.0);
    result.throughput_clips_per_sec = 1000.0 / result.mean_ms;

    return result;
}

void print_json_result(const BenchmarkResult& result) {
    std::cout << "{\n";
    std::cout << "  \"mean_ms\": " << result.mean_ms << ",\n";
    std::cout << "  \"median_ms\": " << result.median_ms << ",\n";
    std::cout << "  \"p95_ms\": " << result.p95_ms << ",\n";
    std::cout << "  \"p99_ms\": " << result.p99_ms << ",\n";
    std::cout << "  \"std_ms\": " << result.std_ms << ",\n";
    std::cout << "  \"throughput_clips_per_sec\": " << result.throughput_clips_per_sec << "\n";
    std::cout << "}\n";
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  --model <path>    Path to ONNX model (required)\n";
    std::cerr << "  --runs <n>        Number of benchmark runs (default: 1000)\n";
    std::cerr << "  --warmup <n>      Number of warmup runs (default: 100)\n";
    std::cerr << "  --help            Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string model_path;
    int num_runs = 1000;
    int num_warmup = 100;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[i + 1];
            ++i;
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = std::atoi(argv[i + 1]);
            ++i;
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            num_warmup = std::atoi(argv[i + 1]);
            ++i;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        print_usage(argv[0]);
        return 1;
    }

    // Initialize random number generator
    std::mt19937 rng(42);  // Fixed seed for reproducibility

    try {
        // Load detector
        std::cerr << "Loading model: " << model_path << std::endl;
        FallDetector detector(model_path);
        std::cerr << "Model loaded successfully\n";

        // Run benchmark
        std::cerr << "Running benchmark: " << num_warmup << " warmup, " << num_runs << " runs\n";
        auto result = run_benchmark(detector, num_warmup, num_runs, rng);

        // Output results as JSON (last line)
        print_json_result(result);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
