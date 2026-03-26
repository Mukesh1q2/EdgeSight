/**
 * @file config_loader.cpp
 * @brief Implementation of configuration loader
 */

#include "config_loader.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>

// For a production build, you would link with nlohmann/json or similar
// This is a simple manual JSON parser for the current implementation

namespace edgesight {

// Simple JSON utilities (stub - production uses nlohmann/json)
namespace json_utils {
    // Trim whitespace
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }

    // Extract value from "key": value line
    bool extractValue(const std::string& line, std::string& key, std::string& value) {
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) return false;

        key = trim(line.substr(0, colon_pos));
        // Remove quotes from key
        if (key.size() >= 2 && key.front() == '"' && key.back() == '"') {
            key = key.substr(1, key.size() - 2);
        }

        value = trim(line.substr(colon_pos + 1));
        // Remove trailing comma
        if (!value.empty() && value.back() == ',') {
            value.pop_back();
            value = trim(value);
        }
        // Remove quotes from value
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }

        return true;
    }
} // namespace json_utils

AppConfig getDefaultConfig() {
    return AppConfig{};
}

AppConfig loadConfig(const std::string& path) {
    AppConfig config = getDefaultConfig();

    // If config file doesn't exist, return defaults
    if (!std::filesystem::exists(path)) {
        return config;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        return config;
    }

    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(file, line)) {
        std::string key, value;
        if (json_utils::extractValue(line, key, value)) {
            values[key] = value;
        }
    }

    // Parse values into config (production would use proper JSON)
    auto it = values.find("camera_index");
    if (it != values.end()) config.camera_index = std::stoi(it->second);

    it = values.find("camera_width");
    if (it != values.end()) config.camera_width = std::stoi(it->second);

    it = values.find("camera_height");
    if (it != values.end()) config.camera_height = std::stoi(it->second);

    it = values.find("camera_fps");
    if (it != values.end()) config.camera_fps = std::stoi(it->second);

    it = values.find("detection_threshold");
    if (it != values.end()) config.detection_threshold = std::stof(it->second);

    it = values.find("fall_frame_threshold");
    if (it != values.end()) config.fall_frame_threshold = std::stoi(it->second);

    it = values.find("buffer_size");
    if (it != values.end()) config.buffer_size = std::stoi(it->second);

    it = values.find("model_path");
    if (it != values.end()) config.model_path = it->second;

    it = values.find("smtp_server");
    if (it != values.end()) config.smtp_server = it->second;

    it = values.find("smtp_port");
    if (it != values.end()) config.smtp_port = std::stoi(it->second);

    it = values.find("alert_email");
    if (it != values.end()) config.alert_email = it->second;

    it = values.find("sender_email");
    if (it != values.end()) config.sender_email = it->second;

    it = values.find("sms_gateway");
    if (it != values.end()) config.sms_gateway = it->second;

    it = values.find("auto_start");
    if (it != values.end()) config.auto_start = (it->second == "true");

    it = values.find("save_alerts");
    if (it != values.end()) config.save_alerts = (it->second == "true");

    it = values.find("log_directory");
    if (it != values.end()) config.log_directory = it->second;

    it = values.find("use_gpu");
    if (it != values.end()) config.use_gpu = (it->second == "true");

    it = values.find("inference_threads");
    if (it != values.end()) config.inference_threads = std::stoi(it->second);

    return config;
}

bool saveConfig(const std::string& path, const AppConfig& config) {
    // Ensure directory exists
    std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << "{\n";
    file << "    \"camera_index\": " << config.camera_index << ",\n";
    file << "    \"camera_width\": " << config.camera_width << ",\n";
    file << "    \"camera_height\": " << config.camera_height << ",\n";
    file << "    \"camera_fps\": " << config.camera_fps << ",\n";
    file << "    \"detection_threshold\": " << config.detection_threshold << ",\n";
    file << "    \"fall_frame_threshold\": " << config.fall_frame_threshold << ",\n";
    file << "    \"buffer_size\": " << config.buffer_size << ",\n";
    file << "    \"model_path\": \"" << config.model_path << "\",\n";
    file << "    \"smtp_server\": \"" << config.smtp_server << "\",\n";
    file << "    \"smtp_port\": " << config.smtp_port << ",\n";
    file << "    \"alert_email\": \"" << config.alert_email << "\",\n";
    file << "    \"sender_email\": \"" << config.sender_email << "\",\n";
    file << "    \"sms_gateway\": \"" << config.sms_gateway << "\",\n";
    file << "    \"auto_start\": " << (config.auto_start ? "true" : "false") << ",\n";
    file << "    \"save_alerts\": " << (config.save_alerts ? "true" : "false") << ",\n";
    file << "    \"log_directory\": \"" << config.log_directory << "\",\n";
    file << "    \"use_gpu\": " << (config.use_gpu ? "true" : "false") << ",\n";
    file << "    \"inference_threads\": " << config.inference_threads << "\n";
    file << "}\n";

    return file.good();
}

// Singleton implementation
ConfigManager& ConfigManager::instance() {
    static ConfigManager instance;
    return instance;
}

bool ConfigManager::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = loadConfig(path);
    config_path_ = path;
    syncToRaw();
    return true;
}

bool ConfigManager::save(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string save_path = path.empty() ? config_path_ : path;
    if (save_path.empty()) {
        return false;
    }
    return saveConfig(save_path, config_);
}

AppConfig ConfigManager::get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void ConfigManager::set(const AppConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    syncToRaw();
}

std::string ConfigManager::getString(const std::string& key, const std::string& default_val) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = raw_values_.find(key);
    return (it != raw_values_.end()) ? it->second : default_val;
}

int ConfigManager::getInt(const std::string& key, int default_val) const {
    std::string val = getString(key);
    if (val.empty()) return default_val;
    try {
        return std::stoi(val);
    } catch (...) {
        return default_val;
    }
}

float ConfigManager::getFloat(const std::string& key, float default_val) const {
    std::string val = getString(key);
    if (val.empty()) return default_val;
    try {
        return std::stof(val);
    } catch (...) {
        return default_val;
    }
}

bool ConfigManager::getBool(const std::string& key, bool default_val) const {
    std::string val = getString(key);
    if (val == "true") return true;
    if (val == "false") return false;
    return default_val;
}

void ConfigManager::setString(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    raw_values_[key] = value;
    syncFromRaw();
}

void ConfigManager::setInt(const std::string& key, int value) {
    setString(key, std::to_string(value));
}

void ConfigManager::setFloat(const std::string& key, float value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << value;
    setString(key, ss.str());
}

void ConfigManager::setBool(const std::string& key, bool value) {
    setString(key, value ? "true" : "false");
}

void ConfigManager::syncToRaw() {
    raw_values_["camera_index"] = std::to_string(config_.camera_index);
    raw_values_["camera_width"] = std::to_string(config_.camera_width);
    raw_values_["camera_height"] = std::to_string(config_.camera_height);
    raw_values_["camera_fps"] = std::to_string(config_.camera_fps);
    raw_values_["detection_threshold"] = std::to_string(config_.detection_threshold);
    raw_values_["fall_frame_threshold"] = std::to_string(config_.fall_frame_threshold);
    raw_values_["buffer_size"] = std::to_string(config_.buffer_size);
    raw_values_["model_path"] = config_.model_path;
    raw_values_["smtp_server"] = config_.smtp_server;
    raw_values_["smtp_port"] = std::to_string(config_.smtp_port);
    raw_values_["alert_email"] = config_.alert_email;
    raw_values_["sender_email"] = config_.sender_email;
    raw_values_["sms_gateway"] = config_.sms_gateway;
    raw_values_["auto_start"] = config_.auto_start ? "true" : "false";
    raw_values_["save_alerts"] = config_.save_alerts ? "true" : "false";
    raw_values_["log_directory"] = config_.log_directory;
    raw_values_["use_gpu"] = config_.use_gpu ? "true" : "false";
    raw_values_["inference_threads"] = std::to_string(config_.inference_threads);
}

void ConfigManager::syncFromRaw() {
    auto it = raw_values_.find("camera_index");
    if (it != raw_values_.end()) config_.camera_index = std::stoi(it->second);

    it = raw_values_.find("camera_width");
    if (it != raw_values_.end()) config_.camera_width = std::stoi(it->second);

    it = raw_values_.find("camera_height");
    if (it != raw_values_.end()) config_.camera_height = std::stoi(it->second);

    it = raw_values_.find("camera_fps");
    if (it != raw_values_.end()) config_.camera_fps = std::stoi(it->second);

    it = raw_values_.find("detection_threshold");
    if (it != raw_values_.end()) config_.detection_threshold = std::stof(it->second);

    it = raw_values_.find("fall_frame_threshold");
    if (it != raw_values_.end()) config_.fall_frame_threshold = std::stoi(it->second);

    it = raw_values_.find("buffer_size");
    if (it != raw_values_.end()) config_.buffer_size = std::stoi(it->second);

    it = raw_values_.find("model_path");
    if (it != raw_values_.end()) config_.model_path = it->second;

    it = raw_values_.find("smtp_server");
    if (it != raw_values_.end()) config_.smtp_server = it->second;

    it = raw_values_.find("smtp_port");
    if (it != raw_values_.end()) config_.smtp_port = std::stoi(it->second);

    it = raw_values_.find("alert_email");
    if (it != raw_values_.end()) config_.alert_email = it->second;

    it = raw_values_.find("sender_email");
    if (it != raw_values_.end()) config_.sender_email = it->second;

    it = raw_values_.find("sms_gateway");
    if (it != raw_values_.end()) config_.sms_gateway = it->second;

    it = raw_values_.find("auto_start");
    if (it != raw_values_.end()) config_.auto_start = (it->second == "true");

    it = raw_values_.find("save_alerts");
    if (it != raw_values_.end()) config_.save_alerts = (it->second == "true");

    it = raw_values_.find("log_directory");
    if (it != raw_values_.end()) config_.log_directory = it->second;

    it = raw_values_.find("use_gpu");
    if (it != raw_values_.end()) config_.use_gpu = (it->second == "true");

    it = raw_values_.find("inference_threads");
    if (it != raw_values_.end()) config_.inference_threads = std::stoi(it->second);
}

} // namespace edgesight
