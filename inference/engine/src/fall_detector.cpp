/**
 * @file fall_detector.cpp
 * @brief Implementation of FallDetector C++ inference engine
 */

#include "fall_detector.h"

#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace edgesight {

FallDetector::FallDetector(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "FallDetector"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

    // Initialize session options
    init_session_options();

    // Load model
    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
    } catch (const Ort::Exception& e) {
        throw ModelLoadException(std::string("Failed to load ONNX model: ") + e.what());
    }

    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;

    // Input node
    Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(0, allocator);
    input_names_.push_back(input_name.get());
    input_name.release();  // Keep the string alive

    // Output node
    Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(0, allocator);
    output_names_.push_back(output_name.get());
    output_name.release();  // Keep the string alive
}

FallDetector::~FallDetector() {
    // Smart pointers handle cleanup automatically
    // session_ will be destroyed first
}

void FallDetector::init_session_options() {
    // Enable all graph optimizations
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Set threading options for optimal latency
    // Using 4 threads for intra-op parallelism (good balance for most CPUs)
    session_options_.SetIntraOpNumThreads(4);

    // Enable sequential execution mode for deterministic latency
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
}

void FallDetector::validate_input(
    const std::vector<std::vector<float>>& pose_sequence) const {

    if (pose_sequence.size() != static_cast<size_t>(seq_length_)) {
        throw std::invalid_argument(
            "Input must have exactly " + std::to_string(seq_length_) +
            " frames, got " + std::to_string(pose_sequence.size()));
    }

    for (size_t i = 0; i < pose_sequence.size(); ++i) {
        if (pose_sequence[i].size() != static_cast<size_t>(input_dim_)) {
            throw std::invalid_argument(
                "Frame " + std::to_string(i) + " must have exactly " +
                std::to_string(input_dim_) + " features, got " +
                std::to_string(pose_sequence[i].size()));
        }
    }
}

std::vector<float> FallDetector::flatten_sequence(
    const std::vector<std::vector<float>>& pose_sequence) const {

    std::vector<float> flattened;
    flattened.reserve(seq_length_ * input_dim_);

    for (const auto& frame : pose_sequence) {
        flattened.insert(flattened.end(), frame.begin(), frame.end());
    }

    return flattened;
}

float FallDetector::predict(const std::vector<std::vector<float>>& pose_sequence) {
    // Validate input
    validate_input(pose_sequence);

    // Flatten input
    std::vector<float> flattened = flatten_sequence(pose_sequence);

    // Run inference
    std::vector<float> output = run_inference(flattened, 1);

    // Sigmoid already applied in model, just clamp to [0, 1]
    float prob = output[0];
    prob = std::max(0.0f, std::min(1.0f, prob));

    return prob;
}

std::vector<float> FallDetector::predict_batch(
    const std::vector<std::vector<std::vector<float>>>& batch) {

    if (batch.empty()) {
        return {};
    }

    // Validate all inputs
    for (const auto& sequence : batch) {
        validate_input(sequence);
    }

    // Flatten all sequences
    std::vector<float> flattened;
    flattened.reserve(batch.size() * seq_length_ * input_dim_);

    for (const auto& sequence : batch) {
        std::vector<float> seq_flat = flatten_sequence(sequence);
        flattened.insert(flattened.end(), seq_flat.begin(), seq_flat.end());
    }

    // Run batch inference
    return run_inference(flattened, batch.size());
}

std::vector<float> FallDetector::run_inference(
    const std::vector<float>& flattened_input,
    size_t batch_size) {

    // Create input tensor
    // Shape: (batch_size, seq_length_, input_dim_)
    std::vector<int64_t> input_shape = {
        static_cast<int64_t>(batch_size),
        seq_length_,
        input_dim_
    };

    // Lock for thread-safe session access
    std::lock_guard<std::mutex> lock(session_mutex_);

    // Measure latency
    auto start = std::chrono::high_resolution_clock::now();

    try {
        // Create input tensor (no copy, uses our data directly)
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(flattened_input.data()),  // const_cast needed for API
            flattened_input.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Run inference
        std::vector<Ort::Value> output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,  // Number of inputs
            output_names_.data(),
            1   // Number of outputs
        );

        // Record latency
        auto end = std::chrono::high_resolution_clock::now();
        last_latency_ms_ = std::chrono::duration<double, std::milli>(end - start).count();

        // Extract output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> results(output_data, output_data + output_count);

        // Clamp to valid probability range
        for (float& prob : results) {
            prob = std::max(0.0f, std::min(1.0f, prob));
        }

        return results;

    } catch (const Ort::Exception& e) {
        throw InferenceException(std::string("ONNX Runtime error: ") + e.what());
    }
}

} // namespace edgesight
