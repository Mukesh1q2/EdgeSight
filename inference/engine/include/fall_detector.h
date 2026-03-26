/**
 * @file fall_detector.h
 * @brief C++ inference engine for fall detection using ONNX Runtime
 *
 * Provides a thread-safe, RAII-managed wrapper around ONNX Runtime
 * for running fall detection inference on pose sequences.
 */

#ifndef FALL_DETECTOR_H
#define FALL_DETECTOR_H

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>

#include <onnxruntime_cxx_api.h>

namespace edgesight {

/**
 * @brief Fall detection inference engine
 *
 * Thread-safe wrapper around ONNX Runtime for fall detection inference.
 * Uses RAII for memory management and std::mutex for thread-safety.
 */
class FallDetector {
public:
    /**
     * @brief Construct FallDetector from ONNX model path
     *
     * @param model_path Path to ONNX model file
     * @throws std::runtime_error if model loading fails
     */
    explicit FallDetector(const std::string& model_path);

    /**
     * @brief Destructor - automatically cleans up ONNX resources
     */
    ~FallDetector();

    // Disable copy (heavy resource) but allow move
    FallDetector(const FallDetector&) = delete;
    FallDetector& operator=(const FallDetector&) = delete;
    FallDetector(FallDetector&&) noexcept = default;
    FallDetector& operator=(FallDetector&&) noexcept = default;

    /**
     * @brief Run inference on a single pose sequence
     *
     * Input format: 30 frames × 17 keypoints × 3 features (x, y, confidence)
     * Flattened to 30 vectors of 51 floats each.
     *
     * @param pose_sequence Vector of 30 frames, each frame is 51 floats
     * @return Fall probability in range [0, 1]
     * @throws std::invalid_argument if input dimensions are wrong
     * @throws std::runtime_error if inference fails
     */
    float predict(const std::vector<std::vector<float>>& pose_sequence);

    /**
     * @brief Run batch inference on multiple pose sequences
     *
     * More efficient than calling predict() multiple times.
     *
     * @param batch Vector of pose sequences, each is (30, 51)
     * @return Vector of fall probabilities, one per sequence
     * @throws std::invalid_argument if input dimensions are wrong
     * @throws std::runtime_error if inference fails
     */
    std::vector<float> predict_batch(
        const std::vector<std::vector<std::vector<float>>>& batch);

    /**
     * @brief Get latency of last inference in milliseconds
     *
     * @return Last inference latency in ms, or 0.0 if no inference run
     */
    double get_last_latency_ms() const { return last_latency_ms_; }

    /**
     * @brief Get model input dimensions
     *
     * @return Pair of (sequence_length, features_per_frame)
     */
    std::pair<int, int> get_input_dims() const { return {seq_length_, input_dim_}; }

    /**
     * @brief Check if model was loaded successfully
     *
     * @return true if ready for inference
     */
    bool is_ready() const { return session_ != nullptr; }

private:
    // ONNX Runtime environment (singleton per process)
    Ort::Env env_;

    // Session options for optimization
    Ort::SessionOptions session_options_;

    // Inference session - protected by mutex for thread-safety
    std::unique_ptr<Ort::Session> session_;

    // Memory info for tensor creation
    Ort::MemoryInfo memory_info_;

    // Input/output node names (must stay alive for session calls)
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Thread safety for session calls
    mutable std::mutex session_mutex_;

    // Latency tracking
    double last_latency_ms_ = 0.0;

    // Model dimensions (extracted from model or defaults)
    static constexpr int seq_length_ = 30;
    static constexpr int input_dim_ = 51;

    /**
     * @brief Initialize session options with optimizations
     */
    void init_session_options();

    /**
     * @brief Validate input dimensions
     *
     * @param pose_sequence Input data to validate
     * @throws std::invalid_argument if dimensions wrong
     */
    void validate_input(const std::vector<std::vector<float>>& pose_sequence) const;

    /**
     * @brief Flatten 2D pose sequence to 1D array
     *
     * @param pose_sequence 2D input: (30, 51)
     * @return Flattened 1D vector: (1530,)
     */
    std::vector<float> flatten_sequence(
        const std::vector<std::vector<float>>& pose_sequence) const;

    /**
     * @brief Run inference on flattened input
     *
     * @param flattened_input Flattened input data
     * @param batch_size Number of sequences in batch
     * @return Output probabilities
     */
    std::vector<float> run_inference(
        const std::vector<float>& flattened_input,
        size_t batch_size);
};

/**
 * @brief Exception thrown when model loading fails
 */
class ModelLoadException : public std::runtime_error {
public:
    explicit ModelLoadException(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief Exception thrown when inference fails
 */
class InferenceException : public std::runtime_error {
public:
    explicit InferenceException(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace edgesight

#endif // FALL_DETECTOR_H
