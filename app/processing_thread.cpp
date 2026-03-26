/**
 * @file processing_thread.cpp
 * @brief Implementation of processing thread
 */

#include "processing_thread.h"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <filesystem>

// MediaPipe includes would go here in production
// For now, we simulate pose detection

namespace edgesight {

ProcessingThread::ProcessingThread(
    const std::string& model_path,
    int camera_index,
    AlertManager* alert_manager,
    QObject* parent
) : QThread(parent),
    model_path_(model_path),
    camera_index_(camera_index),
    alert_manager_(alert_manager) {
}

ProcessingThread::~ProcessingThread() {
    stop();
    wait();
}

void ProcessingThread::stop() {
    stop_requested_.store(true);
    running_.store(false);
}

void ProcessingThread::setDetectionThreshold(float threshold) {
    detection_threshold_.store(threshold);
}

void ProcessingThread::setModelPath(const std::string& path) {
    QMutexLocker locker(&buffer_mutex_);
    model_path_ = path;
    // Reinitialize detector on next run
    detector_.reset();
}

void ProcessingThread::setCameraIndex(int index) {
    camera_index_ = index;
    video_file_path_.clear();  // Clear any video file fallback
}

void ProcessingThread::setVideoFile(const std::string& path) {
    video_file_path_ = path;
    camera_index_ = -1;  // Mark as using video file
}

void ProcessingThread::setMaxRetries(int retries) {
    max_retries_ = std::max(0, retries);
}

void ProcessingThread::setRetryDelayMs(int delay_ms) {
    retry_delay_ms_ = std::max(100, delay_ms);
}

bool ProcessingThread::extractPoseKeypoints(const cv::Mat& frame, std::vector<float>& keypoints) {
    // In production, this would use MediaPipe Pose
    // For now, we simulate pose detection with simple heuristics
    
    // Convert to grayscale for simple motion detection
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }
    
    // Simple person detection (look for largest blob)
    // This is a placeholder - real implementation uses MediaPipe
    
    // Generate simulated keypoints based on frame center
    float cx = frame.cols / 2.0f;
    float cy = frame.rows / 2.0f;
    
    keypoints.clear();
    
    // Simulate 15 keypoints (x, y, confidence)
    // This is just a simulation - real implementation uses MediaPipe
    for (int i = 0; i < 15; ++i) {
        float x = (cx + (i % 5 - 2) * 30) / frame.cols;  // Normalized [0, 1]
        float y = (cy + (i / 5 - 1) * 40) / frame.rows;  // Normalized [0, 1]
        float conf = 0.8f + static_cast<float>(rand()) / RAND_MAX * 0.2f;
        
        keypoints.push_back(x);
        keypoints.push_back(y);
        keypoints.push_back(conf);
    }
    
    return true;
}

std::vector<float> ProcessingThread::padFeatures(const std::vector<float>& features45) {
    // Pad from 45 to 51 features by adding 6 zeros at the end
    std::vector<float> result = features45;
    result.resize(51, 0.0f);
    return result;
}

float ProcessingThread::runInference() {
    if (!detector_ || !detector_->is_ready()) {
        return 0.0f;
    }
    
    QMutexLocker locker(&buffer_mutex_);
    
    if (pose_buffer_.size() < BUFFER_SIZE) {
        return 0.0f;  // Not enough frames yet
    }
    
    // Copy buffer to vector
    std::vector<std::vector<float>> sequence;
    std::queue<std::vector<float>> temp_buffer = pose_buffer_;
    while (!temp_buffer.empty()) {
        sequence.push_back(temp_buffer.front());
        temp_buffer.pop();
    }
    
    locker.unlock();
    
    // Run inference
    try {
        return detector_->predict(sequence);
    } catch (const std::exception& e) {
        emit error(QString("Inference error: ") + e.what());
        return 0.0f;
    }
}

void ProcessingThread::run() {
    running_.store(true);
    current_retry_ = 0;
    
    cv::VideoCapture cap;
    bool using_camera = video_file_path_.empty();
    
    // Try to open video source with retry logic
    if (using_camera) {
        emit statusUpdate("Initializing camera...");
        
        // Retry loop for camera with exponential backoff
        while (current_retry_ <= max_retries_) {
            cap.open(camera_index_);
            if (cap.isOpened()) {
                emit statusUpdate(QString("Camera opened (attempt %1)").arg(current_retry_ + 1));
                break;
            }
            
            current_retry_++;
            if (current_retry_ <= max_retries_) {
                int delay = retry_delay_ms_ * current_retry_;  // Exponential backoff
                emit statusUpdate(QString("Camera failed, retrying in %1ms (attempt %2/%3)")
                    .arg(delay).arg(current_retry_).arg(max_retries_));
                msleep(delay);
            }
        }
        
        // Camera failed after all retries
        if (!cap.isOpened()) {
            emit error(QString("Failed to open camera after %1 attempts. Check:\n"
                              "  - Camera is connected\n"
                              "  - No other app is using it\n"
                              "  - Index %2 is valid").arg(max_retries_ + 1).arg(camera_index_));
            running_.store(false);
            return;
        }
    } else {
        // Open video file fallback
        emit statusUpdate(QString("Opening video file: %1").arg(QString::fromStdString(video_file_path_)));
        cap.open(video_file_path_);
        
        if (!cap.isOpened()) {
            emit error(QString("Failed to open video file: %1").arg(QString::fromStdString(video_file_path_)));
            running_.store(false);
            return;
        }
        
        emit statusUpdate("Video file opened successfully");
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    // Check if model file exists
    if (!std::filesystem::exists(model_path_)) {
        QString err_msg = QString("Model file not found: ") + QString::fromStdString(model_path_);
        emit error(err_msg);
        emit statusUpdate("Please export a model using export_onnx.py");
        running_.store(false);
        return;
    }

    // Initialize detector
    emit statusUpdate("Loading fall detection model...");
    try {
        detector_ = std::make_unique<FallDetector>(model_path_);
        emit statusUpdate("Model loaded successfully");
    } catch (const std::exception& e) {
        emit error(QString("Failed to load model: ") + e.what());
        running_.store(false);
        return;
    }
    
    // FPS calculation
    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    int fps = 0;
    
    // Main loop
    cv::Mat frame;
    while (!stop_requested_.load()) {
        // Capture frame
        if (!cap.read(frame)) {
            emit error("Failed to capture frame");
            break;
        }
        
        // Calculate FPS
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        if (elapsed >= 1000) {
            fps = frame_count;
            frame_count = 0;
            last_time = now;
        }
        
        // Extract pose keypoints (every N frames to save CPU)
        std::vector<float> keypoints45;
        bool pose_detected = extractPoseKeypoints(frame, keypoints45);
        
        if (pose_detected) {
            // Pad to 51 features
            std::vector<float> keypoints51 = padFeatures(keypoints45);
            
            // Add to buffer
            {
                QMutexLocker locker(&buffer_mutex_);
                pose_buffer_.push(keypoints51);
                if (pose_buffer_.size() > BUFFER_SIZE) {
                    pose_buffer_.pop();
                }
            }
        }
        
        // Run inference if buffer is full
        float probability = 0.0f;
        double latency_ms = 0.0;
        
        {
            QMutexLocker locker(&buffer_mutex_);
            if (pose_buffer_.size() == BUFFER_SIZE) {
                locker.unlock();
                
                probability = runInference();
                latency_ms = detector_->get_last_latency_ms();
            }
        }
        
        // Check for fall
        float threshold = detection_threshold_.load();
        if (probability > threshold) {
            consecutive_fall_frames_++;
            
            if (consecutive_fall_frames_ >= FALL_FRAME_THRESHOLD) {
                emit fallDetected(probability);
                
                if (alert_manager_) {
                    alert_manager_->triggerAlert(probability);
                }
                
                // Draw alert on frame
                cv::putText(frame, "FALL DETECTED!", cv::Point(50, 50),
                           cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
            }
        } else {
            consecutive_fall_frames_ = 0;
        }
        
        // Draw pose overlay (simulated)
        if (pose_detected) {
            // Draw skeleton lines
            cv::line(frame, cv::Point(320, 100), cv::Point(320, 200),
                    cv::Scalar(0, 255, 0), 2);  // Spine
            cv::line(frame, cv::Point(320, 200), cv::Point(280, 300),
                    cv::Scalar(0, 255, 0), 2);  // Left leg
            cv::line(frame, cv::Point(320, 200), cv::Point(360, 300),
                    cv::Scalar(0, 255, 0), 2);  // Right leg
        }
        
        // Draw probability
        std::string prob_text = "Fall: " + std::to_string(int(probability * 100)) + "%";
        cv::Scalar color = probability > threshold ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::putText(frame, prob_text, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        
        // Draw FPS and latency
        std::string fps_text = "FPS: " + std::to_string(fps);
        cv::putText(frame, fps_text, cv::Point(10, frame.rows - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        
        std::string latency_text = "Latency: " + std::to_string(int(latency_ms)) + "ms";
        cv::putText(frame, latency_text, cv::Point(frame.cols - 150, frame.rows - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        
        // Emit result
        ProcessingResult result;
        result.frame = frame.clone();
        result.fall_probability = probability;
        result.pose_detected = pose_detected;
        result.inference_latency_ms = latency_ms;
        result.fps = fps;
        
        emit resultReady(result);
        
        // Small delay to prevent 100% CPU usage
        msleep(5);
    }
    
    cap.release();
    running_.store(false);
    emit statusUpdate("Processing stopped");
}

} // namespace edgesight
