<div align="center">
  <h1>🛡️ EdgeSight</h1>
  <p><strong>Real-Time AI Fall Detection for Edge Devices</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" />
    <img src="https://img.shields.io/badge/ONNX-Runtime-orange?logo=onnx" />
    <img src="https://img.shields.io/badge/MediaPipe-Pose-green?logo=google" />
    <img src="https://img.shields.io/badge/FastAPI-0.100+-teal?logo=fastapi" />
    <img src="https://img.shields.io/badge/C++-17-blue?logo=cplusplus" />
  </p>
  <p>LSTM-based fall detection using pose estimation, optimized for Qualcomm edge hardware.<br/>
  Features a futuristic glassmorphism web dashboard and a native desktop client.</p>
</div>

---

## ✨ Features

| Component | Description |
|-----------|-------------|
| **🧠 AI Engine** | ONNX-optimized LSTM model processing 16-frame pose sequences at <2ms latency |
| **📹 Pose Estimation** | MediaPipe real-time skeleton tracking (15 keypoints × 3 features) |
| **🌐 Web Dashboard** | Dark glassmorphism UI with animated gauge, rolling chart, WebSocket streaming |
| **🖥️ Desktop Client** | Python/tkinter app with MJPEG video feed and matching Aegis Vision theme |
| **🚨 Strict Reality** | Hardware-aware startup. The system yields fatal errors if the camera feed disconnects. |
| **🔔 Alert System** | Configurable threshold with timestamped fall detection alerts |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Mukesh1q2/EdgeSight.git
cd EdgeSight

# Install required dependencies
pip install fastapi uvicorn opencv-python onnxruntime numpy websockets

# Optional: real camera support (highly recommended)
pip install mediapipe

# Optional: Python Desktop Client support
pip install Pillow requests
```

### Option 1: Web Dashboard

```bash
# Start the FastAPI server
python fastapi_server.py

# Open your browser and navigate to:
# http://127.0.0.1:5000
```

### Option 2: Desktop Client

```bash
# Ensure the server is running first
python fastapi_server.py

# In a new terminal window, run the desktop client
python desktop_client.py
```

## 🏗️ Architecture

```text
┌─────────────┐     WebSocket/REST      ┌──────────────────┐
│  Web Browser│◄───────────────────────►│                  │
│ (Dashboard) │      MJPEG Feed         │  FastAPI Server  │
└─────────────┘                         │   (Port 5000)    │
                                        │                  │
┌─────────────┐      REST + MJPEG       │  ┌────────────┐  │
│ Desktop GUI │◄───────────────────────►│  │ ONNX Model │  │
│ (Tkinter)   │                         │  └────────────┘  │
└─────────────┘                         │  ┌────────────┐  │
                                        │  │ MediaPipe  │  │
┌─────────────┐                         │  │   Pose     │  │
│ C++ Native  │   (local inference)     │  └────────────┘  │
│ Application │                         │  ┌────────────┐  │
└─────────────┘                         │  │  Camera /  │  │
                                        │  │ Simulation │  │
                                        │  └────────────┘  │
                                        └──────────────────┘
```

## 📊 Performance Profiling

| Metric | Target Value | Measured Value |
|--------|--------------|----------------|
| End-to-End Latency | < 50ms | 25-35ms |
| Inference Latency | < 5ms | ~1.5ms (CPU) |
| Frame Rate (Processing) | > 20 FPS | 23-30 FPS |
| WebSocket Update Rate | 10 Hz | 10 Hz |
| Sequence Window | 16 frames | 16 frames |
| Pose Features | 51 variables | 51 (17 kpts × 3) |

## 🛠️ Project Structure

```bash
EdgeSight/
├── app/                  # C++ Qt Desktop application source
├── data/                 # Video processing & data collection scripts
├── inference/            # Model inference engine (C++ & Python)
├── models/               # Pre-trained ONNX models
├── web/                  # Web dashboard (HTML/CSS/JS)
├── fastapi_server.py     # Main web server & inference runner
├── desktop_client.py     # Native Python GUI client
└── README.md             # This file
```

---
*Built with Reactivity, Intelligence, and Design in mind.*
