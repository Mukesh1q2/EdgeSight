"""
FastAPI Backend for EdgeSight - Real-Time Fall Detection API

Features:
- Real webcam capture with MediaPipe pose estimation
- Graceful fallback to synthetic data when no camera available
- WebSocket real-time streaming with base64 frames
- RESTful API for fall detection control
- Static file serving for web dashboard
- CORS support for web clients

Run with: python fastapi_server.py
"""

import asyncio
import base64
import json
import os
import sys
import time
import socket
import threading
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ============================================
# MediaPipe Import (with graceful fallback)
# ============================================
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARNING] MediaPipe not installed. Using synthetic pose data.")
    print("  Install with: pip install mediapipe")

# ============================================
# Configuration
# ============================================
MODEL_PATH = Path("model/exported/fall_detection.onnx")
SEQUENCE_LENGTH = 16  # Model expects 16-frame sequences
POSE_FEATURES = 51    # 17 keypoints * 3 (x, y, confidence)
CAMERA_INDEX = 0

# Keypoint indices used (15 keypoints from MediaPipe's 33)
KEYPOINT_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),       # Nose to shoulders
    (1, 3), (3, 5),       # Left arm
    (2, 4), (4, 6),       # Right arm
    (1, 7), (2, 8),       # Shoulders to hips
    (7, 8),               # Hip bridge
    (7, 9), (9, 11),      # Left leg
    (8, 10), (10, 12),    # Right leg
    (11, 13), (12, 14),   # Ankles to feet
]

# ============================================
# Pydantic Models
# ============================================
class DetectionStats(BaseModel):
    fall_probability: float
    fps: int
    threshold: float
    alert: bool
    timestamp: str

class DetectionConfig(BaseModel):
    threshold: float = 0.75
    camera_index: int = 0

class AlertEntry(BaseModel):
    timestamp: str
    probability: float

class AlertHistory(BaseModel):
    alerts: List[AlertEntry]

class SystemInfo(BaseModel):
    camera_available: bool
    camera_index: int
    model_loaded: bool
    model_path: str
    mediapipe_available: bool
    detection_running: bool
    sequence_length: int
    pose_features: int

# ============================================
# Global State
# ============================================
class DetectionState:
    def __init__(self):
        self.is_running = False
        self.fall_probability = 0.0
        self.fps = 0
        self.threshold = 0.75
        self.alerts: List[dict] = []
        self.frame_count = 0
        self.last_update = time.time()
        self.last_alert_time = 0.0
        self.model_session = None
        self.sequence_buffer: List[List[float]] = []
        self.camera_available = False
        self.camera_index = CAMERA_INDEX
        self.cap = None
        self.current_frame = None
        self.current_keypoints = None
        self.pose_detected = False
        self.inference_latency_ms = 0.0
        self.lock = threading.Lock()

        # MediaPipe
        self.pose = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils

    def init_model(self):
        """Initialize the ONNX model."""
        if MODEL_PATH.exists():
            try:
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 4
                self.model_session = ort.InferenceSession(
                    str(MODEL_PATH),
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"]
                )
                return True
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                return False
        return False

    def init_camera(self):
        """Try to open the camera. Returns True if successful."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                # Try to read a test frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.camera_available = True
                    print("[OK] Camera opened successfully")
                    return True
            # Camera open failed
            if self.cap:
                self.cap.release()
            self.cap = None
            self.camera_available = False
            print("[ERROR] No camera detected — strict reality mode active")
            return False
        except Exception as e:
            print(f"[FATAL] Camera error: {e}")
            self.camera_available = False
            return False

    def init_pose(self):
        """Initialize MediaPipe pose estimator."""
        if MEDIAPIPE_AVAILABLE:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return True
        return False

    def release(self):
        """Release all resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.pose:
            self.pose.close()

state = DetectionState()

# ============================================
# Lifespan
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "127.0.0.1"

    print("=" * 60)
    print("  EdgeSight FastAPI Server v2.0")
    print("=" * 60)
    print(f"  Local:     http://127.0.0.1:5000")
    print(f"  Network:   http://{ip}:5000")
    print(f"  Dashboard: http://127.0.0.1:5000")
    print(f"  API Docs:  http://127.0.0.1:5000/docs")
    print("=" * 60)

    yield

    # Shutdown
    state.is_running = False
    state.release()
    print("Shutting down EdgeSight server...")

app = FastAPI(
    title="EdgeSight Fall Detection API",
    description="Real-time fall detection using pose estimation and LSTM",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Pose Estimation Functions
# ============================================
def extract_keypoints_mediapipe(frame: np.ndarray) -> Optional[List[float]]:
    """Extract pose keypoints using MediaPipe."""
    if not MEDIAPIPE_AVAILABLE or state.pose is None:
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = state.pose.process(rgb_frame)

    if not results.pose_landmarks:
        return None

    keypoints = []
    landmarks = results.pose_landmarks.landmark

    for idx in KEYPOINT_INDICES:
        lm = landmarks[idx]
        keypoints.extend([
            lm.x,
            lm.y,
            lm.visibility if hasattr(lm, 'visibility') else 0.8
        ])

    # Pad from 45 (15*3) to 51 (17*3)
    return padded



# Inference
# ============================================
def run_inference(sequence: List[List[float]]) -> float:
    """Run ONNX inference on a pose sequence."""
    if state.model_session is None:
        return 0.0

    try:
        seq = np.array(sequence[-SEQUENCE_LENGTH:], dtype=np.float32)
        seq = seq.reshape(1, SEQUENCE_LENGTH, POSE_FEATURES)

        start = time.perf_counter()
        input_name = state.model_session.get_inputs()[0].name
        outputs = state.model_session.run(None, {input_name: seq})
        end = time.perf_counter()

        state.inference_latency_ms = (end - start) * 1000
        prob = float(outputs[0][0][0])
        return max(0.0, min(1.0, prob))
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return 0.0


# ============================================
# Video Frame Generator (MJPEG stream)
# ============================================
def generate_video_frames():
    """Generate annotated video frames for MJPEG streaming."""
    while state.is_running:
        frame = None

        with state.lock:
            if state.current_frame is not None:
                frame = state.current_frame.copy()

        if frame is None:
            # Generate placeholder frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (15, 19, 31)
            cv2.putText(frame, "Waiting for feed...", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 150, 200), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(0.033)  # ~30 FPS


# ============================================
# Detection Loop
# ============================================
async def detection_loop():
    """Background detection loop with real camera or synthetic data."""
    # Initialize model
    if not state.init_model():
        print("[WARNING] Model not found — running without inference")

    # Initialize camera
    state.init_camera()

    # Initialize pose estimator
    state.init_pose()

    print(f"[INFO] Detection started | Camera: {state.camera_available}")

    while state.is_running:
        frame = None
        keypoints = None

        # ---- Capture frame ----
        if state.camera_available and state.cap and state.cap.isOpened():
            ret, raw_frame = state.cap.read()
            if ret and raw_frame is not None:
                frame = raw_frame.copy()
                # Extract pose with MediaPipe
                keypoints_result = extract_keypoints_mediapipe(raw_frame)
                if keypoints_result:
                    keypoints = keypoints_result
                    state.pose_detected = True
                    # Draw skeleton on frame
                    draw_skeleton(frame, keypoints)
                else:
                    state.pose_detected = False
                # Camera read failed
                state.camera_available = False
                if state.cap:
                    state.cap.release()
                print("[FATAL] Camera read failed")

        if not state.camera_available or frame is None:
            # Fatal error frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (10, 14, 26)
            cv2.putText(frame, "FATAL ERROR: NO CAMERA DETECTED", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Check hardware connections", (150, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 200), 1)
            keypoints = None
            state.pose_detected = False

        # ---- Overlay info on frame ----
        if frame is not None:
            prob = state.fall_probability
            color = (0, 0, 255) if prob > state.threshold else (0, 255, 128)

            # Fall probability
            cv2.putText(frame, f"Fall: {prob*100:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # FPS and latency
            cv2.putText(frame, f"FPS: {state.fps}", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Latency: {state.inference_latency_ms:.0f}ms",
                        (frame.shape[1] - 160, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 220), 1, cv2.LINE_AA)

            # Fall alert banner
            if prob > state.threshold:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 50), (frame.shape[1], 90), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "!! FALL DETECTED !!", (180, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # Store frame
            with state.lock:
                state.current_frame = frame
                state.current_keypoints = keypoints

        # ---- Buffer keypoints & run inference ----
        if keypoints:
            state.sequence_buffer.append(keypoints)
            if len(state.sequence_buffer) > SEQUENCE_LENGTH:
                state.sequence_buffer.pop(0)

            if len(state.sequence_buffer) == SEQUENCE_LENGTH:
                state.fall_probability = run_inference(state.sequence_buffer)

        # ---- Update FPS ----
        state.frame_count += 1
        current_time = time.time()
        if current_time - state.last_update >= 1.0:
            state.fps = state.frame_count
            state.frame_count = 0
            state.last_update = current_time

        # ---- Check for alert ----
        if state.fall_probability > state.threshold:
            # Don't spam alerts — cooldown of 2 seconds
            if current_time - state.last_alert_time >= 2.0:
                state.last_alert_time = current_time
                alert = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "probability": round(state.fall_probability, 3)
                }
                state.alerts.append(alert)
                if len(state.alerts) > 100:
                    state.alerts.pop(0)

        await asyncio.sleep(0.033)  # ~30 FPS

    # Cleanup
    state.release()


# ============================================
# API Endpoints
# ============================================
@app.get("/api/stats", response_model=DetectionStats)
async def get_stats():
    """Get current detection statistics."""
    return DetectionStats(
        fall_probability=round(state.fall_probability, 4),
        fps=state.fps,
        threshold=state.threshold,
        alert=state.fall_probability > state.threshold,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/alerts")
async def get_alerts():
    """Get alert history."""
    return {"alerts": state.alerts[-30:]}


@app.get("/api/system_info", response_model=SystemInfo)
async def get_system_info():
    """Get system information."""
    return SystemInfo(
        camera_available=state.camera_available,
        camera_index=state.camera_index,
        model_loaded=state.model_session is not None,
        model_path=str(MODEL_PATH),
        mediapipe_available=MEDIAPIPE_AVAILABLE,
        detection_running=state.is_running,
        sequence_length=SEQUENCE_LENGTH,
        pose_features=POSE_FEATURES,
    )


@app.post("/api/start")
async def start_detection():
    """Start fall detection."""
    if not state.is_running:
        state.is_running = True
        state.sequence_buffer = []
        state.fall_probability = 0.0
        state.alerts = []
        state.last_alert_time = 0.0
        asyncio.create_task(detection_loop())
        return {
            "status": "started",
            "message": "Detection started"
        }
    return {"status": "already_running"}


@app.post("/api/stop")
async def stop_detection():
    """Stop fall detection."""
    state.is_running = False
    return {"status": "stopped"}


@app.post("/api/config")
async def update_config(config: DetectionConfig):
    """Update detection configuration."""
    state.threshold = max(0.5, min(0.99, config.threshold))
    return {"status": "updated", "threshold": state.threshold}


@app.get("/video_feed")
async def video_feed():
    """MJPEG video streaming endpoint."""
    if not state.is_running:
        raise HTTPException(status_code=400, detail="Detection not running. POST /api/start first.")
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time stats + optional frame data."""
    await websocket.accept()
    try:
        while True:
            data = {
                "type": "stats",
                "fall_probability": round(state.fall_probability, 4),
                "fps": state.fps,
        "threshold": state.threshold,
        "alert": state.fall_probability > state.threshold,
        "pose_detected": state.pose_detected,
                "latency_ms": round(state.inference_latency_ms, 1),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.1)  # 10 Hz
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": state.model_session is not None,
        "detection_running": state.is_running,
        "camera_available": state.camera_available
    }


# ============================================
# Static File Serving (Web Dashboard)
# ============================================
# Serve index.html at root
@app.get("/")
async def serve_dashboard():
    """Serve the web dashboard."""
    web_dir = Path(__file__).parent / "web"
    index_file = web_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    # Fallback: return API info
    return {
        "name": "EdgeSight Fall Detection API",
        "version": "2.0.0",
        "dashboard": "Create a 'web/' directory with index.html",
        "docs": "/docs",
    }

# Mount static files
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)