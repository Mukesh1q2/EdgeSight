/**
 * @file processing_thread.h
 * @brief Worker thread for real-time inference pipeline
 *
 * Runs in a separate QThread to handle:
 *   - Webcam frame capture (OpenCV)
 *   - MediaPipe pose estimation
 *   - Fall detection inference (C++ engine)
 *   - Result emission to main GUI thread
 */

#ifndef PROCESSING_THREAD_H
#define PROCESSING_THREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <vector>
#include <queue>
#include <memory>
#include <atomic>

#include <opencv2/core.hpp>

#include "fall_detector.h"
#include "alert_manager.h"

namespace edgesight {

/**
 * @brief Processing result structure
 */
struct ProcessingResult {
    cv::Mat frame;                    // Annotated frame for display
    float fall_probability = 0.0f;  // Current fall probability
    bool pose_detected = false;       // Whether pose was found
    double inference_latency_ms = 0.0;
    int fps = 0;
};

/**
 * @brief Worker thread for real-time inference
 *
 * Runs the full pipeline: capture → pose → buffer → infer → emit
 */
class ProcessingThread : public QThread {
    Q_OBJECT

public:
    /**
     * @brief Construct processing thread
     * @param model_path Path to ONNX model
     * @param camera_index Camera device index (0 = default)
     * @param alert_manager Optional alert manager for notifications
     */
    explicit ProcessingThread(
        const std::string& model_path,
        int camera_index = 0,
        AlertManager* alert_manager = nullptr,
        QObject* parent = nullptr
    );

    ~ProcessingThread();

    // Thread control
    void stop();
    bool isRunning() const { return running_.load(); }

    // Settings
    void setDetectionThreshold(float threshold);
    void setModelPath(const std::string& path);
    void setCameraIndex(int index);
    void setVideoFile(const std::string& path);  // Fallback to video file
    
    // Camera retry configuration
    void setMaxRetries(int retries);
    void setRetryDelayMs(int delay_ms);

    // Get current settings
    float detectionThreshold() const { return detection_threshold_.load(); }

protected:
    void run() override;

signals:
    /**
     * @brief New processing result available
     */
    void resultReady(const ProcessingResult& result);

    /**
     * @brief Fall detected alert
     */
    void fallDetected(float probability);

    /**
     * @brief Error occurred
     */
    void error(const QString& message);

    /**
     * @brief Status update
     */
    void statusUpdate(const QString& status);

private:
    // Configuration
    std::string model_path_;
    int camera_index_ = 0;
    AlertManager* alert_manager_ = nullptr;

    // Thread state
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<float> detection_threshold_{0.75f};

    // Inference engine
    std::unique_ptr<FallDetector> detector_;

    // Frame buffer (circular buffer of 30 frames)
    static constexpr int BUFFER_SIZE = 30;
    std::queue<std::vector<float>> pose_buffer_;
    QMutex buffer_mutex_;

    // Consecutive fall frames counter
    int consecutive_fall_frames_ = 0;
    static constexpr int FALL_FRAME_THRESHOLD = 3;
    
    // Camera retry configuration
    static constexpr int MAX_RETRIES = 3;
    static constexpr int RETRY_DELAY_MS = 1000;
    int max_retries_ = MAX_RETRIES;
    int retry_delay_ms_ = RETRY_DELAY_MS;
    int current_retry_ = 0;
    
    // Video file fallback (when camera fails)
    std::string video_file_path_;

    // MediaPipe (simulated with OpenCV for now)
    // In production, link against MediaPipe C++ API
    bool extractPoseKeypoints(const cv::Mat& frame, std::vector<float>& keypoints);

    // Pad 45 features to 51
    std::vector<float> padFeatures(const std::vector<float>& features45);

    // Run inference on buffered poses
    float runInference();
};

} // namespace edgesight

#endif // PROCESSING_THREAD_H
