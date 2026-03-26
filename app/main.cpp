/**
 * @file main.cpp
 * @brief Application entry point
 */

#include <QApplication>
#include <QStyleFactory>

#include "mainwindow.h"
#include "logger.h"

using namespace edgesight;

int main(int argc, char* argv[]) {
    // Initialize logging system
    initLogging("app/logs", LogLevel::INFO);
    qInstallMessageHandler(qtMessageHandler);
    LOG_INFO("EdgeSight application starting...");
    
    // Set application attributes
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
    
    // Create application
    QApplication app(argc, argv);
    
    // Application info
    app.setApplicationName("EdgeSight");
    app.setOrganizationName("EdgeSight Project");
    app.setApplicationDisplayName("EdgeSight - Fall Detection");
    
    LOG_INFO("Application initialized");
    
    // Set modern style
    app.setStyle(QStyleFactory::create("Fusion"));
    
    // Aegis Vision Dark Theme — matches web dashboard design system
    QPalette dark_palette;
    dark_palette.setColor(QPalette::Window, QColor(15, 19, 31));           // #0f131f surface-dim
    dark_palette.setColor(QPalette::WindowText, QColor(223, 226, 243));    // #dfe2f3 on-surface
    dark_palette.setColor(QPalette::Base, QColor(10, 14, 26));             // #0a0e1a surface-lowest
    dark_palette.setColor(QPalette::AlternateBase, QColor(23, 27, 40));    // #171b28 surface-low
    dark_palette.setColor(QPalette::ToolTipBase, QColor(38, 42, 55));      // #262a37 surface-high
    dark_palette.setColor(QPalette::ToolTipText, QColor(223, 226, 243));   // #dfe2f3
    dark_palette.setColor(QPalette::Text, QColor(223, 226, 243));          // #dfe2f3
    dark_palette.setColor(QPalette::Button, QColor(27, 31, 44));           // #1b1f2c surface
    dark_palette.setColor(QPalette::ButtonText, QColor(168, 232, 255));    // #a8e8ff primary-soft
    dark_palette.setColor(QPalette::BrightText, QColor(255, 23, 68));      // #ff1744 tertiary
    dark_palette.setColor(QPalette::Highlight, QColor(0, 212, 255));       // #00d4ff primary
    dark_palette.setColor(QPalette::HighlightedText, QColor(0, 54, 66));   // #003642 on-primary
    dark_palette.setColor(QPalette::Link, QColor(60, 215, 255));           // #3cd7ff primary-dim
    dark_palette.setColor(QPalette::Disabled, QPalette::Text, QColor(133, 147, 152));  // #859398 outline
    dark_palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(133, 147, 152));
    app.setPalette(dark_palette);
    
    // Global stylesheet — Aegis Vision dark glassmorphism
    app.setStyleSheet(
        "QMainWindow { background-color: #0f131f; }"
        "QGroupBox { "
        "  background-color: #171b28; "
        "  border: 1px solid rgba(60, 73, 78, 0.15); "
        "  border-radius: 12px; "
        "  margin-top: 8px; "
        "  padding-top: 20px; "
        "  color: #bbc9cf; "
        "  font-family: 'Segoe UI', sans-serif; "
        "  font-weight: 500; "
        "}"
        "QGroupBox::title { "
        "  subcontrol-origin: margin; "
        "  subcontrol-position: top left; "
        "  padding: 2px 10px; "
        "  color: #bbc9cf; "
        "  font-size: 11px; "
        "  letter-spacing: 1px; "
        "  text-transform: uppercase; "
        "}"
        "QPushButton { "
        "  border-radius: 8px; "
        "  padding: 8px 20px; "
        "  font-weight: 600; "
        "  font-size: 13px; "
        "}"
        "QProgressBar { "
        "  border: none; "
        "  border-radius: 6px; "
        "  background-color: #313442; "
        "  text-align: center; "
        "  color: #dfe2f3; "
        "}"
        "QProgressBar::chunk { "
        "  border-radius: 6px; "
        "}"
        "QSlider::groove:horizontal { "
        "  height: 4px; "
        "  background: #313442; "
        "  border-radius: 2px; "
        "}"
        "QSlider::handle:horizontal { "
        "  width: 16px; "
        "  height: 16px; "
        "  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #a8e8ff, stop:1 #00d4ff); "
        "  border-radius: 8px; "
        "  margin: -6px 0; "
        "}"
        "QComboBox { "
        "  background-color: #313442; "
        "  border: 1px solid rgba(60, 73, 78, 0.3); "
        "  border-radius: 8px; "
        "  padding: 4px 8px; "
        "  color: #dfe2f3; "
        "}"
        "QListWidget { "
        "  background-color: #0a0e1a; "
        "  border: none; "
        "  border-radius: 8px; "
        "  color: #dfe2f3; "
        "}"
        "QLabel { color: #dfe2f3; }"
    );
    
    // Create and show main window
    MainWindow window;
    window.show();
    
    // Run event loop
    return app.exec();
}
