/**
 * @file alert_manager.cpp
 * @brief Implementation of alert manager
 */

#include "alert_manager.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace edgesight {

AlertManager::AlertManager() {
    // Ensure logs directory exists
    std::filesystem::create_directories("app/logs");
}

AlertManager::~AlertManager() {
    // Wait for pending async operations
    for (auto& future : pending_operations_) {
        if (future.valid()) {
            future.wait();
        }
    }
}

void AlertManager::setEnabled(bool enabled) {
    enabled_ = enabled;
}

void AlertManager::setCooldown(int seconds) {
    cooldown_seconds_ = seconds;
}

bool AlertManager::triggerAlert(float probability) {
    if (!enabled_) {
        return false;
    }

    // Check cooldown
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_alert_time_).count();
    
    if (elapsed < cooldown_seconds_) {
        return false;  // Still on cooldown
    }

    // Update last alert time
    last_alert_time_ = now;

    // Play beep (synchronous, immediate feedback)
    playBeep();

    // Create alert info
    AlertInfo info;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    info.timestamp = oss.str();
    info.probability = probability;
    
    std::ostringstream msg;
    msg << "FALL DETECTED - Confidence: " << std::fixed << std::setprecision(1) 
        << (probability * 100.0f) << "%";
    info.message = msg.str();

    // Log to CSV
    logAlert(info);

    // Add to history
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        alert_history_.push_front(info);
        if (alert_history_.size() > 100) {
            alert_history_.pop_back();
        }
    }

    // Send async notifications
    cleanupPendingOperations();
    
    pending_operations_.push_back(
        std::async(std::launch::async, [this, info]() { sendEmailAsync(info); })
    );
    
    pending_operations_.push_back(
        std::async(std::launch::async, [this, info]() { sendSmsAsync(info); })
    );

    return true;
}

void AlertManager::playBeep() {
#ifdef _WIN32
    // Play system beep
    MessageBeep(MB_ICONEXCLAMATION);
    
    // Also play a custom sound if available
    // PlaySound(TEXT("alert.wav"), NULL, SND_FILENAME | SND_ASYNC);
#else
    std::cout << '\a' << std::flush;  // Terminal bell
#endif
}

void AlertManager::logAlert(const AlertInfo& info) {
    std::string log_path = "app/logs/alerts.csv";
    
    bool file_exists = std::filesystem::exists(log_path);
    
    std::ofstream file(log_path, std::ios::app);
    if (file.is_open()) {
        // Write header if new file
        if (!file_exists) {
            file << "timestamp,probability,message\n";
        }
        
        // Write alert
        file << info.timestamp << ","
             << std::fixed << std::setprecision(4) << info.probability << ","
             << "\"" << info.message << "\"\n";
    }
}

std::vector<AlertInfo> AlertManager::getRecentAlerts(size_t count) const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    std::vector<AlertInfo> result;
    size_t num = std::min(count, alert_history_.size());
    
    for (size_t i = 0; i < num; ++i) {
        result.push_back(alert_history_[i]);
    }
    
    return result;
}

void AlertManager::setSmtpConfig(const std::string& server, int port,
                                  const std::string& username, const std::string& password,
                                  const std::string& from, const std::string& to) {
    smtp_server_ = server;
    smtp_port_ = port;
    smtp_username_ = username;
    smtp_password_ = password;
    smtp_from_ = from;
    smtp_to_ = to;
}

void AlertManager::setTwilioConfig(const std::string& account_sid,
                                   const std::string& auth_token,
                                   const std::string& from,
                                   const std::string& to) {
    twilio_account_sid_ = account_sid;
    twilio_auth_token_ = auth_token;
    twilio_from_ = from;
    twilio_to_ = to;
}

void AlertManager::sendEmailAsync(const AlertInfo& info) {
    if (smtp_server_.empty() || smtp_to_.empty()) {
        return;  // Not configured
    }
    
    bool success = sendEmail(info);
    if (!success) {
        std::cerr << "[AlertManager] Failed to send email alert\n";
    }
}

void AlertManager::sendSmsAsync(const AlertInfo& info) {
    if (twilio_account_sid_.empty() || twilio_to_.empty()) {
        return;  // Not configured
    }
    
    bool success = sendSms(info);
    if (!success) {
        std::cerr << "[AlertManager] Failed to send SMS alert\n";
    }
}

bool AlertManager::sendEmail(const AlertInfo& info) {
#ifdef ALERTS_ENABLED
    if (smtp_server_.empty()) return false;
    
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    
    // Build email payload
    std::string payload = "From: " + smtp_from_ + "\r\n" +
                          "To: " + smtp_to_ + "\r\n" +
                          "Subject: EdgeSight Alert: Fall Detected\r\n" +
                          "\r\n" +
                          info.message + "\r\n" +
                          "Timestamp: " + info.timestamp + "\r\n";
    
    // Configure CURL for SMTP
    std::string url = "smtp://" + smtp_server_ + ":" + std::to_string(smtp_port_);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USE_SSL, (long)CURLUSESSL_ALL);
    curl_easy_setopt(curl, CURLOPT_USERNAME, smtp_username_.c_str());
    curl_easy_setopt(curl, CURLOPT_PASSWORD, smtp_password_.c_str());
    curl_easy_setopt(curl, CURLOPT_MAIL_FROM, smtp_from_.c_str());
    
    struct curl_slist* recipients = curl_slist_append(nullptr, smtp_to_.c_str());
    curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);
    
    // Set payload
    curl_easy_setopt(curl, CURLOPT_READDATA, &payload);
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
    
    CURLcode res = curl_easy_perform(curl);
    
    curl_slist_free_all(recipients);
    curl_easy_cleanup(curl);
    
    return (res == CURLE_OK);
#else
    std::cout << "[AlertManager] Email alert (disabled in build): " << info.message << "\n";
    return true;
#endif
}

bool AlertManager::sendSms(const AlertInfo& info) {
#ifdef ALERTS_ENABLED
    if (twilio_account_sid_.empty()) return false;
    
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    
    // Build Twilio API URL
    std::string url = "https://api.twilio.com/2010-04-01/Accounts/" + 
                      twilio_account_sid_ + "/Messages.json";
    
    // Build POST data
    std::string post_data = "From=" + twilio_from_ +
                            "&To=" + twilio_to_ +
                            "&Body=" + curl_easy_escape(curl, info.message.c_str(), info.message.length());
    
    // Set credentials
    std::string auth = twilio_account_sid_ + ":" + twilio_auth_token_;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
    curl_easy_setopt(curl, CURLOPT_USERNAME, twilio_account_sid_.c_str());
    curl_easy_setopt(curl, CURLOPT_PASSWORD, twilio_auth_token_.c_str());
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
    
    CURLcode res = curl_easy_perform(curl);
    
    curl_easy_cleanup(curl);
    
    return (res == CURLE_OK);
#else
    std::cout << "[AlertManager] SMS alert (disabled in build): " << info.message << "\n";
    return true;
#endif
}

void AlertManager::cleanupPendingOperations() {
    // Remove completed futures
    pending_operations_.erase(
        std::remove_if(pending_operations_.begin(), pending_operations_.end(),
            [](const std::future<void>& f) {
                return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            }),
        pending_operations_.end()
    );
}

size_t AlertManager::curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    // We don't need to store the response
    return size * nmemb;
}

} // namespace edgesight
