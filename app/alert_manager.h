/**
 * @file alert_manager.h
 * @brief Alert manager for SMS and email notifications
 */

#ifndef ALERT_MANAGER_H
#define ALERT_MANAGER_H

#include <string>
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <chrono>

#ifdef ALERTS_ENABLED
#include <curl/curl.h>
#else
// Stub types when CURL not available
typedef void CURL;
#endif

namespace edgesight {

/**
 * @brief Alert information structure
 */
struct AlertInfo {
    std::string timestamp;
    float probability;
    std::string message;
};

/**
 * @brief Alert manager for sending notifications
 *
 * Handles SMS (via Twilio) and email (via SMTP) alerts
 * asynchronously to avoid blocking the UI.
 */
class AlertManager {
public:
    /**
     * @brief Construct alert manager
     */
    AlertManager();

    /**
     * @brief Destructor
     */
    ~AlertManager();

    // Disable copy
    AlertManager(const AlertManager&) = delete;
    AlertManager& operator=(const AlertManager&) = delete;

    /**
     * @brief Enable or disable alerts
     */
    void setEnabled(bool enabled);

    /**
     * @brief Check if alerts are enabled
     */
    bool isEnabled() const { return enabled_; }

    /**
     * @brief Set cooldown period between alerts
     * @param seconds Minimum seconds between alerts
     */
    void setCooldown(int seconds);

    /**
     * @brief Trigger a fall alert
     * @param probability Fall detection probability
     * @return true if alert was triggered, false if on cooldown
     */
    bool triggerAlert(float probability);

    /**
     * @brief Play system beep
     */
    void playBeep();

    /**
     * @brief Log alert to CSV file
     */
    void logAlert(const AlertInfo& info);

    /**
     * @brief Get recent alert history
     */
    std::vector<AlertInfo> getRecentAlerts(size_t count = 10) const;

    // Configuration setters
    void setSmtpConfig(const std::string& server, int port,
                       const std::string& username, const std::string& password,
                       const std::string& from, const std::string& to);

    void setTwilioConfig(const std::string& account_sid,
                         const std::string& auth_token,
                         const std::string& from,
                         const std::string& to);

private:
    bool enabled_ = true;
    int cooldown_seconds_ = 30;
    
    std::chrono::steady_clock::time_point last_alert_time_;
    
    // Alert history
    mutable std::mutex history_mutex_;
    std::deque<AlertInfo> alert_history_;
    
    // SMTP settings
    std::string smtp_server_;
    int smtp_port_ = 587;
    std::string smtp_username_;
    std::string smtp_password_;
    std::string smtp_from_;
    std::string smtp_to_;
    
    // Twilio settings
    std::string twilio_account_sid_;
    std::string twilio_auth_token_;
    std::string twilio_from_;
    std::string twilio_to_;

    // Async operations
    std::vector<std::future<void>> pending_operations_;

    // Send methods (async)
    void sendEmailAsync(const AlertInfo& info);
    void sendSmsAsync(const AlertInfo& info);
    
    // Synchronous send methods
    bool sendEmail(const AlertInfo& info);
    bool sendSms(const AlertInfo& info);

    // Cleanup completed async operations
    void cleanupPendingOperations();

    // CURL callback for HTTP requests
    static size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
};

} // namespace edgesight

#endif // ALERT_MANAGER_H
