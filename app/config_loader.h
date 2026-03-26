/**
 * @file config_loader.h
 * @brief Application configuration management with JSON support
 */

#ifndef EDGESIGHT_CONFIG_LOADER_H
#define EDGESIGHT_CONFIG_LOADER_H

#include <string>
#include <map>
#include <mutex>

namespace edgesight {

/**
 * @brief Configuration structure for application settings
 */
struct AppConfig {
    // Camera settings
    int camera_index = 0;
    int camera_width = 640;
    int camera_height = 480;
    int camera_fps = 30;

    // Detection settings
    float detection_threshold = 0.85f;
    int fall_frame_threshold = 3;
    int buffer_size = 16;

    // Model settings
    std::string model_path = "model/exported/fall_detection.onnx";

    // Alert settings
    std::string smtp_server = "smtp.gmail.com";
    int smtp_port = 587;
    std::string alert_email;
    std::string sender_email;
    std::string sms_gateway;

    // UI settings
    bool auto_start = false;
    bool save_alerts = true;
    std::string log_directory = "app/logs";

    // Performance settings
    bool use_gpu = true;
    int inference_threads = 1;
};

/**
 * @brief Load configuration from JSON file
 * @param path Path to config file
 * @return Loaded configuration with defaults for missing values
 */
AppConfig loadConfig(const std::string& path);

/**
 * @brief Save configuration to JSON file
 * @param path Path to config file
 * @param config Configuration to save
 * @return true on success
 */
bool saveConfig(const std::string& path, const AppConfig& config);

/**
 * @brief Get default configuration
 * @return Default configuration
 */
AppConfig getDefaultConfig();

/**
 * @brief Global configuration singleton
 */
class ConfigManager {
public:
    static ConfigManager& instance();

    /**
     * @brief Load configuration from file
     * @param path Path to config file
     * @return true if loaded successfully
     */
    bool load(const std::string& path);

    /**
     * @brief Save current configuration to file
     * @param path Path to save to (uses loaded path if empty)
     * @return true on success
     */
    bool save(const std::string& path = "");

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    AppConfig get() const;

    /**
     * @brief Set configuration
     * @param config New configuration
     */
    void set(const AppConfig& config);

    /**
     * @brief Get string value from config
     * @param key Setting key
     * @param default_val Default if not found
     * @return Setting value
     */
    std::string getString(const std::string& key, const std::string& default_val = "") const;

    /**
     * @brief Get integer value from config
     * @param key Setting key
     * @param default_val Default if not found
     * @return Setting value
     */
    int getInt(const std::string& key, int default_val = 0) const;

    /**
     * @brief Get float value from config
     * @param key Setting key
     * @param default_val Default if not found
     * @return Setting value
     */
    float getFloat(const std::string& key, float default_val = 0.0f) const;

    /**
     * @brief Get boolean value from config
     * @param key Setting key
     * @param default_val Default if not found
     * @return Setting value
     */
    bool getBool(const std::string& key, bool default_val = false) const;

    /**
     * @brief Set string value
     * @param key Setting key
     * @param value Setting value
     */
    void setString(const std::string& key, const std::string& value);

    /**
     * @brief Set integer value
     * @param key Setting key
     * @param value Setting value
     */
    void setInt(const std::string& key, int value);

    /**
     * @brief Set float value
     * @param key Setting key
     * @param value Setting value
     */
    void setFloat(const std::string& key, float value);

    /**
     * @brief Set boolean value
     * @param key Setting key
     * @param value Setting value
     */
    void setBool(const std::string& key, bool value);

private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    mutable std::mutex mutex_;
    AppConfig config_;
    std::string config_path_;
    std::map<std::string, std::string> raw_values_;

    void syncToRaw();
    void syncFromRaw();
};

} // namespace edgesight

#endif // EDGESIGHT_CONFIG_LOADER_H
