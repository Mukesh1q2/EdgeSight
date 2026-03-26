/**
 * @file mainwindow.h
 * @brief Main application window
 *
 * Split-view UI with:
 *   - Left panel: Live webcam feed with overlays
 *   - Right panel: Controls, gauge, chart, alert log
 */

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QLabel>
#include <QProgressBar>
#include <QSlider>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
#include <QListWidget>
#include <QTextEdit>
#include <QGroupBox>

#include <opencv2/core.hpp>

#include "processing_thread.h"
#include "alert_manager.h"

namespace edgesight {

/**
 * @brief Main application window
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    // Processing thread slots
    void onResultReady(const ProcessingResult& result);
    void onFallDetected(float probability);
    void onProcessingError(const QString& message);
    void onStatusUpdate(const QString& status);

    // UI control slots
    void onStartClicked();
    void onStopClicked();
    void onThresholdChanged(int value);
    void onModelChanged(int index);
    void onCameraChanged(int index);
    void onAlertsToggled(bool enabled);

    // Chart update
    void updateChart();
    
    // Keyboard shortcuts setup
    void setupKeyboardShortcuts();
    
    // Screenshot action
    void saveScreenshot();
    
    // Input validation
    bool validateCameraIndex(int index);
    bool validateThreshold(float threshold);
    bool validateEmail(const QString& email);
    bool validateInputs();

private:
    void setupUI();
    void createLeftPanel();  // Webcam view
    void createRightPanel(); // Controls and info
    void updateProbabilityGauge(float probability);
    void updateRiskIndicator(float probability);
    void addAlertToLog(const QString& timestamp, float probability);

    // Convert OpenCV Mat to QPixmap
    QPixmap matToPixmap(const cv::Mat& mat);

    // Processing
    std::unique_ptr<ProcessingThread> processing_thread_;
    std::unique_ptr<AlertManager> alert_manager_;

    // UI components
    QWidget* left_panel_ = nullptr;
    QWidget* right_panel_ = nullptr;

    QLabel* video_label_ = nullptr;        // Webcam display
    QLabel* fps_label_ = nullptr;          // FPS counter
    QLabel* latency_label_ = nullptr;      // Inference latency
    QLabel* status_label_ = nullptr;       // Status bar

    QProgressBar* probability_gauge_ = nullptr;  // Fall probability gauge
    QLabel* risk_label_ = nullptr;             // Risk level indicator
    QLabel* chart_label_ = nullptr;              // Simple chart display

    QSlider* threshold_slider_ = nullptr;
    QLabel* threshold_value_label_ = nullptr;
    QComboBox* model_selector_ = nullptr;
    QComboBox* camera_selector_ = nullptr;
    QCheckBox* alerts_checkbox_ = nullptr;

    QPushButton* start_button_ = nullptr;
    QPushButton* stop_button_ = nullptr;

    QListWidget* alert_log_ = nullptr;
    QTextEdit* log_output_ = nullptr;

    // Chart data (rolling buffer of probabilities)
    std::vector<float> probability_history_;
    QTimer* chart_timer_ = nullptr;
    
    // Frame drop detection
    std::chrono::steady_clock::time_point last_frame_time_;
    int dropped_frame_count_ = 0;
    QLabel* frame_drop_indicator_ = nullptr;

    // Current threshold
    float current_threshold_ = 0.75f;
};

} // namespace edgesight

#endif // MAINWINDOW_H
