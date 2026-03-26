/**
 * @file config.h
 * @brief Application-wide configuration constants
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace edgesight {
namespace config {

// Model paths
constexpr const char* DEFAULT_FP32_MODEL = "model/exported/fallnet_fp32.onnx";
constexpr const char* DEFAULT_INT8_MODEL = "model/exported/fallnet_int8.onnx";

// Inference settings
constexpr float DEFAULT_DETECTION_THRESHOLD = 0.75f;
constexpr int ALERT_CONSECUTIVE_FRAMES = 3;  // Frames above threshold to trigger alert
constexpr int ALERT_COOLDOWN_SECONDS = 30;   // Minimum time between alerts

// UI settings
constexpr int WEBCAM_WIDTH = 640;
constexpr int WEBCAM_HEIGHT = 480;
constexpr int TARGET_FPS = 30;
constexpr int POSE_DETECTION_INTERVAL_MS = 100;  // MediaPipe pose every 100ms

// Chart settings
constexpr int HISTORY_BUFFER_SECONDS = 60;
constexpr int CHART_UPDATE_INTERVAL_MS = 100;

// Alert settings
constexpr bool DEFAULT_ALERTS_ENABLED = true;
constexpr const char* DEFAULT_ALERT_SOUND = "default";

// MediaPipe pose keypoint indices (17 keypoints used)
// 0: nose, 11-12: shoulders, 13-14: elbows, 15-16: wrists,
// 23-24: hips, 25-26: knees, 27-28: ankles, 31-32: feet
constexpr int POSE_KEYPOINT_INDICES[] = {
    0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32
};
constexpr int NUM_KEYPOINTS = 15;  // 15 keypoints x 3 (x, y, conf) = 45 features
// Note: We pad to 51 for model compatibility (fill remaining with zeros)

// SMTP settings (load from config file in production)
struct SmtpConfig {
    std::string server;
    int port = 587;
    std::string username;
    std::string password;
    std::string from_address;
    std::string to_address;
};

// Twilio settings (load from config file in production)
struct TwilioConfig {
    std::string account_sid;
    std::string auth_token;
    std::string from_number;
    std::string to_number;
};

} // namespace config
} // namespace edgesight

#endif // CONFIG_H
