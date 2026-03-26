/**
 * @file mainwindow.cpp
 * @brief Implementation of main application window
 */

#include "mainwindow.h"

#include <chrono>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <QPainter>
#include <QCloseEvent>
#include <QDateTime>
#include <QShortcut>
#include <QKeySequence>
#include <QStandardPaths>
#include <QFileDialog>
#include <QRegularExpression>

#include <opencv2/imgproc.hpp>

namespace edgesight {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
    
    setWindowTitle("EdgeSight - Real-time Fall Detection");
    resize(1200, 700);
    
    // Initialize alert manager
    alert_manager_ = std::make_unique<AlertManager>();
    
    // Setup UI
    setupUI();
    
    // Chart timer
    chart_timer_ = new QTimer(this);
    connect(chart_timer_, &QTimer::timeout, this, &MainWindow::updateChart);
    chart_timer_->start(100);  // Update chart every 100ms
    
    // Setup keyboard shortcuts
    setupKeyboardShortcuts();
}

MainWindow::~MainWindow() {
    if (processing_thread_) {
        processing_thread_->stop();
        processing_thread_->wait();
    }
}

void MainWindow::setupUI() {
    // Central widget with horizontal layout
    QWidget* central_widget = new QWidget(this);
    setCentralWidget(central_widget);
    
    QHBoxLayout* main_layout = new QHBoxLayout(central_widget);
    main_layout->setSpacing(10);
    main_layout->setContentsMargins(10, 10, 10, 10);
    
    // Create panels
    createLeftPanel();
    createRightPanel();
    
    main_layout->addWidget(left_panel_, 3);  // 60% width
    main_layout->addWidget(right_panel_, 2); // 40% width
}

void MainWindow::createLeftPanel() {
    left_panel_ = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(left_panel_);
    layout->setSpacing(5);
    
    // Video display
    QGroupBox* video_group = new QGroupBox("Live Feed", left_panel_);
    QVBoxLayout* video_layout = new QVBoxLayout(video_group);
    
    video_label_ = new QLabel("Click Start to begin", video_group);
    video_label_->setMinimumSize(640, 480);
    video_label_->setAlignment(Qt::AlignCenter);
    video_label_->setStyleSheet("background-color: #0a0e1a; color: #dfe2f3; font-size: 16px; border-radius: 12px;");
    video_layout->addWidget(video_label_);
    
    // Info bar under video
    QHBoxLayout* info_layout = new QHBoxLayout();
    fps_label_ = new QLabel("FPS: -", video_group);
    latency_label_ = new QLabel("Latency: -ms", video_group);
    status_label_ = new QLabel("Status: Stopped", video_group);
    
    info_layout->addWidget(fps_label_);
    info_layout->addWidget(latency_label_);
    info_layout->addStretch();
    info_layout->addWidget(status_label_);
    
    video_layout->addLayout(info_layout);
    layout->addWidget(video_group);
}

void MainWindow::createRightPanel() {
    right_panel_ = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(right_panel_);
    layout->setSpacing(10);
    
    // Probability gauge
    QGroupBox* gauge_group = new QGroupBox("Fall Probability", right_panel_);
    QVBoxLayout* gauge_layout = new QVBoxLayout(gauge_group);
    
    probability_gauge_ = new QProgressBar(gauge_group);
    probability_gauge_->setRange(0, 100);
    probability_gauge_->setValue(0);
    probability_gauge_->setTextVisible(true);
    probability_gauge_->setFormat("%p%");
    probability_gauge_->setStyleSheet(
        "QProgressBar { border: none; border-radius: 6px; background-color: #313442; text-align: center; color: #dfe2f3; }"
        "QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00bfa5, stop:1 #00d4ff); border-radius: 6px; }"
    );
    gauge_layout->addWidget(probability_gauge_);
    
    risk_label_ = new QLabel("Risk: LOW", gauge_group);
    risk_label_->setAlignment(Qt::AlignCenter);
    risk_label_->setStyleSheet("font-weight: bold; color: #44ddc1; font-size: 14px;");
    gauge_layout->addWidget(risk_label_);
    
    layout->addWidget(gauge_group);
    
    // Chart display (simplified - just a placeholder for real chart)
    QGroupBox* chart_group = new QGroupBox("Probability History (60s)", right_panel_);
    QVBoxLayout* chart_layout = new QVBoxLayout(chart_group);
    
    chart_label_ = new QLabel("Chart will appear when running", chart_group);
    chart_label_->setMinimumHeight(100);
    chart_label_->setAlignment(Qt::AlignCenter);
    chart_label_->setStyleSheet("background-color: #171b28; border-radius: 8px; color: #bbc9cf;");
    chart_layout->addWidget(chart_label_);
    
    layout->addWidget(chart_group);
    
    // Controls
    QGroupBox* controls_group = new QGroupBox("Settings", right_panel_);
    QGridLayout* controls_layout = new QGridLayout(controls_group);
    
    // Threshold slider
    controls_layout->addWidget(new QLabel("Threshold:"), 0, 0);
    threshold_slider_ = new QSlider(Qt::Horizontal, controls_group);
    threshold_slider_->setRange(50, 95);
    threshold_slider_->setValue(75);
    connect(threshold_slider_, &QSlider::valueChanged, this, &MainWindow::onThresholdChanged);
    controls_layout->addWidget(threshold_slider_, 0, 1);
    
    threshold_value_label_ = new QLabel("75%", controls_group);
    controls_layout->addWidget(threshold_value_label_, 0, 2);
    
    // Model selector
    controls_layout->addWidget(new QLabel("Model:"), 1, 0);
    model_selector_ = new QComboBox(controls_group);
    model_selector_->addItem("FP32 (Default)", QString("model/exported/fallnet_fp32.onnx"));
    model_selector_->addItem("INT8 (Fast)", QString("model/exported/fallnet_int8.onnx"));
    connect(model_selector_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onModelChanged);
    controls_layout->addWidget(model_selector_, 1, 1, 1, 2);
    
    // Camera selector
    controls_layout->addWidget(new QLabel("Camera:"), 2, 0);
    camera_selector_ = new QComboBox(controls_group);
    camera_selector_->addItem("Camera 0", 0);
    camera_selector_->addItem("Camera 1", 1);
    connect(camera_selector_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onCameraChanged);
    controls_layout->addWidget(camera_selector_, 2, 1, 1, 2);
    
    // Alerts toggle
    alerts_checkbox_ = new QCheckBox("Enable Alerts", controls_group);
    alerts_checkbox_->setChecked(true);
    connect(alerts_checkbox_, &QCheckBox::toggled, this, &MainWindow::onAlertsToggled);
    controls_layout->addWidget(alerts_checkbox_, 3, 0, 1, 3);
    
    layout->addWidget(controls_group);
    
    // Control buttons
    QHBoxLayout* button_layout = new QHBoxLayout();
    
    start_button_ = new QPushButton("Start", right_panel_);
    start_button_->setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #a8e8ff, stop:1 #00d4ff); color: #003642; font-weight: bold; border-radius: 8px; padding: 8px 20px;");
    connect(start_button_, &QPushButton::clicked, this, &MainWindow::onStartClicked);
    button_layout->addWidget(start_button_);
    
    stop_button_ = new QPushButton("Stop", right_panel_);
    stop_button_->setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffb3b3, stop:1 #ff1744); color: #ffffff; font-weight: bold; border-radius: 8px; padding: 8px 20px;");
    stop_button_->setEnabled(false);
    connect(stop_button_, &QPushButton::clicked, this, &MainWindow::onStopClicked);
    button_layout->addWidget(stop_button_);
    
    layout->addLayout(button_layout);
    
    // Alert log
    QGroupBox* log_group = new QGroupBox("Alert Log", right_panel_);
    QVBoxLayout* log_layout = new QVBoxLayout(log_group);
    
    alert_log_ = new QListWidget(log_group);
    alert_log_->setMaximumHeight(120);
    log_layout->addWidget(alert_log_);
    
    layout->addWidget(log_group);
    
    // Stretch to fill remaining space
    layout->addStretch();
}

void MainWindow::onStartClicked() {
    if (processing_thread_ && processing_thread_->isRunning()) {
        return;
    }
    
    // Validate inputs before starting
    if (!validateInputs()) {
        return;
    }
    
    // Get selected model path
    QString model_path = model_selector_->currentData().toString();
    int camera_index = camera_selector_->currentData().toInt();
    
    // Create and start processing thread
    processing_thread_ = std::make_unique<ProcessingThread>(
        model_path.toStdString(),
        camera_index,
        alert_manager_.get(),
        this
    );
    
    processing_thread_->setDetectionThreshold(current_threshold_);
    
    connect(processing_thread_.get(), &ProcessingThread::resultReady,
            this, &MainWindow::onResultReady, Qt::QueuedConnection);
    connect(processing_thread_.get(), &ProcessingThread::fallDetected,
            this, &MainWindow::onFallDetected, Qt::QueuedConnection);
    connect(processing_thread_.get(), &ProcessingThread::error,
            this, &MainWindow::onProcessingError, Qt::QueuedConnection);
    connect(processing_thread_.get(), &ProcessingThread::statusUpdate,
            this, &MainWindow::onStatusUpdate, Qt::QueuedConnection);
    
    processing_thread_->start();
    
    start_button_->setEnabled(false);
    stop_button_->setEnabled(true);
    status_label_->setText("Status: Running");
}

void MainWindow::onStopClicked() {
    if (processing_thread_) {
        processing_thread_->stop();
        processing_thread_->wait();
        processing_thread_.reset();
    }
    
    start_button_->setEnabled(true);
    stop_button_->setEnabled(false);
    status_label_->setText("Status: Stopped");
    video_label_->setText("Click Start to begin");
}

void MainWindow::onThresholdChanged(int value) {
    // Validate threshold range
    if (value < 50 || value > 99) {
        return;  // Invalid threshold
    }
    
    current_threshold_ = value / 100.0f;
    threshold_value_label_->setText(QString::number(value) + "%");
    
    if (processing_thread_) {
        processing_thread_->setDetectionThreshold(current_threshold_);
    }
}

void MainWindow::onModelChanged(int index) {
    if (processing_thread_) {
        QString model_path = model_selector_->itemData(index).toString();
        processing_thread_->setModelPath(model_path.toStdString());
    }
}

void MainWindow::onCameraChanged(int index) {
    if (!processing_thread_ || !processing_thread_->isRunning()) {
        return;
    }
    
    // Restart with new camera
    onStopClicked();
    onStartClicked();
}

void MainWindow::onAlertsToggled(bool enabled) {
    if (alert_manager_) {
        alert_manager_->setEnabled(enabled);
    }
}

void MainWindow::onResultReady(const ProcessingResult& result) {
    // Frame drop detection
    auto now = std::chrono::steady_clock::now();
    if (last_frame_time_.time_since_epoch().count() > 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame_time_).count();
        // Expected ~100ms per frame at 10fps from processing thread
        if (elapsed > 150) {  // Frame took too long
            dropped_frame_count_++;
        }
    }
    last_frame_time_ = now;
    
    // Update video display
    QPixmap pixmap = matToPixmap(result.frame);
    video_label_->setPixmap(pixmap.scaled(video_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    
    // Update info
    fps_label_->setText(QString("FPS: %1").arg(result.fps));
    latency_label_->setText(QString("Latency: %1ms").arg(int(result.inference_latency_ms)));
    
    // Update gauge
    updateProbabilityGauge(result.fall_probability);
    updateRiskIndicator(result.fall_probability);
    
    // Add to history
    probability_history_.push_back(result.fall_probability);
    if (probability_history_.size() > 600) {  // 60 seconds at 10fps
        probability_history_.erase(probability_history_.begin());
    }
}

void MainWindow::onFallDetected(float probability) {
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    addAlertToLog(timestamp, probability);
}

void MainWindow::onProcessingError(const QString& message) {
    QMessageBox::critical(this, "Error", message);
    onStopClicked();
}

void MainWindow::onStatusUpdate(const QString& status) {
    status_label_->setText("Status: " + status);
}

void MainWindow::updateProbabilityGauge(float probability) {
    int percent = int(probability * 100);
    probability_gauge_->setValue(percent);
    
    // Update color based on threshold
    QString color = percent > (current_threshold_ * 100)
        ? "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffb3b3, stop:1 #ff1744)"
        : "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00bfa5, stop:1 #00d4ff)";
    probability_gauge_->setStyleSheet(
        QString("QProgressBar { border: none; border-radius: 6px; background-color: #313442; text-align: center; color: #dfe2f3; }"
               "QProgressBar::chunk { background-color: %1; border-radius: 6px; }").arg(color)
    );
}

void MainWindow::updateRiskIndicator(float probability) {
    if (probability < 0.3f) {
        risk_label_->setText("Risk: LOW");
        risk_label_->setStyleSheet("font-weight: bold; color: #44ddc1; font-size: 14px;");
    } else if (probability < current_threshold_) {
        risk_label_->setText("Risk: MEDIUM");
        risk_label_->setStyleSheet("font-weight: bold; color: #ffcc80; font-size: 14px;");
    } else {
        risk_label_->setText("Risk: HIGH");
        risk_label_->setStyleSheet("font-weight: bold; color: #ffb3b3; font-size: 14px;");
    }
}

void MainWindow::addAlertToLog(const QString& timestamp, float probability) {
    QString message = QString("[%1] FALL detected (confidence: %2%)")
                       .arg(timestamp)
                       .arg(int(probability * 100));
    alert_log_->addItem(message);
    alert_log_->scrollToBottom();
}

void MainWindow::updateChart() {
    // Simple chart drawing
    if (probability_history_.empty()) {
        return;
    }
    
    // Create a simple pixmap chart
    QPixmap chart(400, 100);
    chart.fill(QColor(23, 27, 40));  // #171b28 surface-low
    
    QPainter painter(&chart);
    painter.setPen(QPen(QColor(0, 212, 255), 2));  // #00d4ff primary
    
    int width = chart.width();
    int height = chart.height();
    int num_points = static_cast<int>(probability_history_.size());
    
    if (num_points > 1) {
        float x_step = static_cast<float>(width) / (num_points - 1);
        
        for (int i = 1; i < num_points; ++i) {
            int x1 = int((i - 1) * x_step);
            int y1 = height - int(probability_history_[i - 1] * height);
            int x2 = int(i * x_step);
            int y2 = height - int(probability_history_[i] * height);
            
            painter.drawLine(x1, y1, x2, y2);
        }
    }
    
    painter.end();
    chart_label_->setPixmap(chart);
}

QPixmap MainWindow::matToPixmap(const cv::Mat& mat) {
    if (mat.empty()) {
        return QPixmap();
    }
    
    // Convert BGR to RGB
    cv::Mat rgb_mat;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_GRAY2RGB);
    } else {
        rgb_mat = mat;
    }
    
    // Create QImage
    QImage image(rgb_mat.data, rgb_mat.cols, rgb_mat.rows, rgb_mat.step, QImage::Format_RGB888);
    
    return QPixmap::fromImage(image.copy());
}

void MainWindow::closeEvent(QCloseEvent* event) {
    onStopClicked();
    event->accept();
}

void MainWindow::setupKeyboardShortcuts() {
    // Ctrl+R: Toggle detection (Start/Stop)
    auto* toggle_shortcut = new QShortcut(QKeySequence("Ctrl+R"), this);
    connect(toggle_shortcut, &QShortcut::activated, this, [this]() {
        if (stop_button_->isEnabled()) {
            onStopClicked();
        } else {
            onStartClicked();
        }
    });

    // Ctrl+S: Save screenshot
    auto* screenshot_shortcut = new QShortcut(QKeySequence("Ctrl+S"), this);
    connect(screenshot_shortcut, &QShortcut::activated, this, &MainWindow::saveScreenshot);

    // Ctrl+Q: Quit
    auto* quit_shortcut = new QShortcut(QKeySequence("Ctrl+Q"), this);
    connect(quit_shortcut, &QShortcut::activated, this, &QWidget::close);

    // Space: Toggle threshold panel focus (when not in text input)
    auto* space_shortcut = new QShortcut(QKeySequence("Space"), this);
    connect(space_shortcut, &QShortcut::activated, this, [this]() {
        // Toggle focus between slider and other controls
        if (threshold_slider_->hasFocus()) {
            start_button_->setFocus();
        } else {
            threshold_slider_->setFocus();
        }
    });
}

void MainWindow::saveScreenshot() {
    // Get the current video frame
    const QPixmap* screenshot = video_label_->pixmap();
    if (!screenshot || screenshot->isNull()) {
        QMessageBox::warning(this, "Screenshot", "No frame available to save");
        return;
    }

    // Generate filename with timestamp
    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    QString default_filename = QString("edgesight_capture_%1.png").arg(timestamp);

    // Get pictures folder
    QString pictures_dir = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    QString default_path = pictures_dir + "/" + default_filename;

    // Show save dialog
    QString filename = QFileDialog::getSaveFileName(
        this,
        "Save Screenshot",
        default_path,
        "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
    );

    if (filename.isEmpty()) {
        return;
    }

    // Save the screenshot
    if (screenshot->save(filename)) {
        status_label_->setText(QString("Screenshot saved: %1").arg(filename));
    } else {
        QMessageBox::warning(this, "Screenshot", "Failed to save screenshot");
    }
}

bool MainWindow::validateCameraIndex(int index) {
    // Check if camera index is in reasonable range
    if (index < 0 || index > 10) {
        QMessageBox::warning(this, "Validation Error",
                           QString("Camera index %1 is out of valid range (0-10)").arg(index));
        return false;
    }
    return true;
}

bool MainWindow::validateThreshold(float threshold) {
    // Threshold should be between 0.5 and 0.99
    if (threshold < 0.5f || threshold > 0.99f) {
        QMessageBox::warning(this, "Validation Error",
                           QString("Threshold must be between 50%% and 99%% (got %1%%)")
                           .arg(int(threshold * 100)));
        return false;
    }
    return true;
}

bool MainWindow::validateEmail(const QString& email) {
    // Simple email validation regex
    QRegularExpression email_regex(
        R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        QRegularExpression::CaseInsensitiveOption
    );
    
    if (!email_regex.match(email).hasMatch()) {
        QMessageBox::warning(this, "Validation Error",
                           QString("Invalid email address: %1").arg(email));
        return false;
    }
    return true;
}

bool MainWindow::validateInputs() {
    // Validate camera index
    int camera_index = camera_selector_->currentData().toInt();
    if (!validateCameraIndex(camera_index)) {
        return false;
    }
    
    // Validate threshold
    if (!validateThreshold(current_threshold_)) {
        return false;
    }
    
    // Validate alert email if alerts are enabled
    if (alerts_checkbox_->isChecked() && alert_manager_) {
        // In a real implementation, you'd get the email from settings
        // For now, we just check that alerts are configured
        if (!alert_manager_->isEnabled()) {
            QMessageBox::information(this, "Alerts",
                "Alerts are enabled but not configured. Configure SMTP settings in Settings panel.");
        }
    }
    
    return true;
}

} // namespace edgesight
