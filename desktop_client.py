"""
EdgeSight Desktop Client — Python/Tkinter
Connects to the FastAPI server and displays the same
real-time fall detection dashboard in a native window.

Run: python desktop_client.py [--server http://127.0.0.1:5000]
Requires: pip install requests Pillow
"""

import sys
import math
import time
import json
import io
import argparse
import threading
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("[ERROR] tkinter not available")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARNING] Pillow not installed. Video feed will not display.")
    print("  Install with: pip install Pillow")

# ============================================
# Configuration
# ============================================
DEFAULT_SERVER = "http://127.0.0.1:5000"

# Aegis Vision color palette
COLORS = {
    "bg_lowest": "#0a0e1a",
    "bg_dim": "#0f131f",
    "bg_low": "#171b28",
    "bg": "#1b1f2c",
    "bg_high": "#262a37",
    "bg_highest": "#313442",
    "primary": "#00d4ff",
    "primary_soft": "#a8e8ff",
    "primary_dark": "#004e5f",
    "secondary": "#00bfa5",
    "secondary_soft": "#44ddc1",
    "tertiary": "#ff1744",
    "tertiary_soft": "#ffb3b3",
    "warning": "#ff9800",
    "warning_soft": "#ffcc80",
    "text": "#dfe2f3",
    "text_dim": "#bbc9cf",
    "outline": "#859398",
    "outline_variant": "#3c494e",
}


# ============================================
# API Client
# ============================================
class EdgeSightAPI:
    """Simple API client for the FastAPI backend."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def _get(self, endpoint: str):
        try:
            req = Request(f"{self.server_url}{endpoint}")
            resp = urlopen(req, timeout=3)
            return json.loads(resp.read().decode())
        except Exception:
            return None

    def _post(self, endpoint: str, data=None):
        try:
            body = json.dumps(data).encode() if data else b""
            req = Request(
                f"{self.server_url}{endpoint}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=5)
            return json.loads(resp.read().decode())
        except Exception:
            return None

    def health(self):
        return self._get("/health")

    def stats(self):
        return self._get("/api/stats")

    def system_info(self):
        return self._get("/api/system_info")

    def start(self):
        return self._post("/api/start")

    def stop(self):
        return self._post("/api/stop")

    def set_threshold(self, value: float):
        return self._post("/api/config", {"threshold": value})

    def video_feed_url(self):
        return f"{self.server_url}/video_feed"


# ============================================
# Desktop Dashboard Application
# ============================================
class EdgeSightDesktop(tk.Tk):
    """Main desktop application window."""

    def __init__(self, server_url: str):
        super().__init__()

        self.api = EdgeSightAPI(server_url)
        self.is_running = False
        self.probability = 0.0
        self.fps = 0
        self.latency_ms = 0.0
        self.threshold = 0.75
        self.prob_history = []
        self.alerts = []
        self.video_stream = None
        self.video_thread = None

        # Window setup
        self.title("EdgeSight — Fall Detection Desktop")
        self.configure(bg=COLORS["bg_dim"])
        self.geometry("1280x780")
        self.minsize(1024, 600)

        # Override close behavior
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._setup_styles()
        self._build_ui()
        self._start_polling()

    # ============================================
    # Styles
    # ============================================
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # General
        style.configure(".", background=COLORS["bg_dim"], foreground=COLORS["text"],
                         font=("Segoe UI", 10))

        # Frames
        style.configure("Card.TFrame", background=COLORS["bg_low"])
        style.configure("Dark.TFrame", background=COLORS["bg_dim"])

        # Labels
        style.configure("Title.TLabel", background=COLORS["bg_dim"],
                         foreground=COLORS["primary_soft"],
                         font=("Segoe UI", 18, "bold"))
        style.configure("Logo.TLabel", background=COLORS["bg_dim"],
                         foreground=COLORS["text"],
                         font=("Segoe UI", 14, "bold"))
        style.configure("Chip.TLabel", background=COLORS["bg_highest"],
                         foreground=COLORS["text_dim"],
                         font=("Segoe UI", 9))
        style.configure("Value.TLabel", background=COLORS["bg_low"],
                         foreground=COLORS["primary_soft"],
                         font=("Segoe UI", 28, "bold"))
        style.configure("Unit.TLabel", background=COLORS["bg_low"],
                         foreground=COLORS["text_dim"],
                         font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=COLORS["bg_low"],
                         foreground=COLORS["text_dim"],
                         font=("Segoe UI", 9))
        style.configure("Risk.Low.TLabel", background=COLORS["bg_low"],
                         foreground=COLORS["secondary_soft"],
                         font=("Segoe UI", 12, "bold"))
        style.configure("Risk.Medium.TLabel", background=COLORS["bg_low"],
                         foreground=COLORS["warning_soft"],
                         font=("Segoe UI", 12, "bold"))
        style.configure("Risk.High.TLabel", background=COLORS["bg_low"],
                         foreground=COLORS["tertiary_soft"],
                         font=("Segoe UI", 12, "bold"))

        # Scale (slider)
        style.configure("Cyan.Horizontal.TScale",
                         background=COLORS["bg_low"],
                         troughcolor=COLORS["bg_highest"])

    # ============================================
    # UI Construction
    # ============================================
    def _build_ui(self):
        # Top bar
        self._build_topbar()

        # Main content area
        content = ttk.Frame(self, style="Dark.TFrame")
        content.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 0))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        # Left panel — video
        self._build_video_panel(content)

        # Right panel — telemetry
        self._build_telemetry_panel(content)

        # Bottom bar
        self._build_bottombar()

    def _build_topbar(self):
        bar = ttk.Frame(self, style="Dark.TFrame")
        bar.pack(fill=tk.X, padx=16, pady=(10, 0))

        # Logo
        logo = ttk.Label(bar, text="● EDGESIGHT", style="Logo.TLabel")
        logo.pack(side=tk.LEFT)

        # Status chips (right side)
        chips = ttk.Frame(bar, style="Dark.TFrame")
        chips.pack(side=tk.RIGHT)

        self.chip_cam = ttk.Label(chips, text="● Camera", style="Chip.TLabel")
        self.chip_cam.pack(side=tk.LEFT, padx=4)

        self.chip_model = ttk.Label(chips, text="● Model", style="Chip.TLabel")
        self.chip_model.pack(side=tk.LEFT, padx=4)

        self.lbl_fps = ttk.Label(chips, text="FPS 0", style="Chip.TLabel")
        self.lbl_fps.pack(side=tk.LEFT, padx=4)

        self.lbl_latency = ttk.Label(chips, text="Latency 0ms", style="Chip.TLabel")
        self.lbl_latency.pack(side=tk.LEFT, padx=4)

    def _build_video_panel(self, parent):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=8)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        # Video canvas
        self.video_canvas = tk.Canvas(
            frame, bg=COLORS["bg_lowest"],
            highlightthickness=0, bd=0
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Placeholder text
        self.video_canvas.create_text(
            320, 240,
            text="Click START to begin detection",
            fill=COLORS["outline"],
            font=("Segoe UI", 14),
            tags="placeholder"
        )

    def _build_telemetry_panel(self, parent):
        frame = ttk.Frame(parent, style="Dark.TFrame")
        frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=8)

        # Gauge card
        gauge_card = ttk.Frame(frame, style="Card.TFrame")
        gauge_card.pack(fill=tk.X, pady=(0, 8))

        # Gauge canvas (circular ring)
        self.gauge_canvas = tk.Canvas(
            gauge_card, width=180, height=180,
            bg=COLORS["bg_low"], highlightthickness=0
        )
        self.gauge_canvas.pack(pady=12)

        self.lbl_gauge_value = ttk.Label(gauge_card, text="0.0%", style="Value.TLabel")
        self.lbl_gauge_value.pack()
        ttk.Label(gauge_card, text="FALL RISK", style="Unit.TLabel").pack(pady=(0, 8))

        # Risk badge
        self.lbl_risk = ttk.Label(gauge_card, text="LOW", style="Risk.Low.TLabel")
        self.lbl_risk.pack(pady=(0, 12))

        # Threshold slider
        thresh_card = ttk.Frame(frame, style="Card.TFrame")
        thresh_card.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(thresh_card, text="DETECTION THRESHOLD", style="Section.TLabel").pack(
            anchor=tk.W, padx=12, pady=(8, 0))

        slider_row = ttk.Frame(thresh_card, style="Card.TFrame")
        slider_row.pack(fill=tk.X, padx=12, pady=8)

        self.threshold_var = tk.IntVar(value=75)
        self.threshold_slider = ttk.Scale(
            slider_row, from_=50, to=95,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            command=self._on_threshold_change,
            style="Cyan.Horizontal.TScale",
        )
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lbl_threshold = ttk.Label(slider_row, text="75%", style="Section.TLabel")
        self.lbl_threshold.pack(side=tk.RIGHT, padx=(8, 0))

        # Chart canvas
        chart_card = ttk.Frame(frame, style="Card.TFrame")
        chart_card.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(chart_card, text="PROBABILITY HISTORY (60s)", style="Section.TLabel").pack(
            anchor=tk.W, padx=12, pady=(8, 0))

        self.chart_canvas = tk.Canvas(
            chart_card, height=100,
            bg=COLORS["bg_low"], highlightthickness=0
        )
        self.chart_canvas.pack(fill=tk.X, padx=12, pady=(4, 12))

        # Alert log
        alert_card = ttk.Frame(frame, style="Card.TFrame")
        alert_card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(alert_card, text="ALERT LOG", style="Section.TLabel").pack(
            anchor=tk.W, padx=12, pady=(8, 0))

        self.alert_listbox = tk.Listbox(
            alert_card,
            bg=COLORS["bg_lowest"],
            fg=COLORS["tertiary_soft"],
            selectbackground=COLORS["bg_high"],
            font=("Consolas", 9),
            bd=0, highlightthickness=0, relief=tk.FLAT,
        )
        self.alert_listbox.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))
        self.alert_listbox.insert(tk.END, "  No alerts yet")

    def _build_bottombar(self):
        bar = ttk.Frame(self, style="Dark.TFrame")
        bar.pack(fill=tk.X, padx=16, pady=10)

        # Start/Stop buttons
        btn_frame = tk.Frame(bar, bg=COLORS["bg_dim"])
        btn_frame.pack(side=tk.LEFT)

        self.btn_start = tk.Button(
            btn_frame, text="▶ START",
            bg=COLORS["primary"], fg=COLORS["primary_dark"],
            activebackground=COLORS["primary_soft"],
            font=("Segoe UI", 11, "bold"),
            bd=0, padx=20, pady=6, cursor="hand2",
            command=self._on_start,
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = tk.Button(
            btn_frame, text="⏹ STOP",
            bg=COLORS["tertiary"], fg="#ffffff",
            activebackground=COLORS["tertiary_soft"],
            font=("Segoe UI", 11, "bold"),
            bd=0, padx=20, pady=6, cursor="hand2",
            state=tk.DISABLED,
            command=self._on_stop,
        )
        self.btn_stop.pack(side=tk.LEFT)

        # Version label
        ttk.Label(bar, text="EdgeSight Desktop v2.0", style="Chip.TLabel").pack(side=tk.RIGHT)

    # ============================================
    # Actions
    # ============================================
    def _on_start(self):
        result = self.api.start()
        if result:
            self.is_running = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.prob_history.clear()
            self.alert_listbox.delete(0, tk.END)

            # Start video feed thread
            if PIL_AVAILABLE:
                self._start_video_feed()

            # Fetch system info
            info = self.api.system_info()
            if info:
                self._update_chips(info)

    def _on_stop(self):
        self.api.stop()
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        # Stop video thread
        if self.video_stream:
            try:
                self.video_stream.close()
            except Exception:
                pass
            self.video_stream = None

        # Reset
        self.probability = 0.0
        self._draw_gauge(0.0)
        self.lbl_gauge_value.config(text="0.0%")
        self.lbl_risk.config(text="LOW", style="Risk.Low.TLabel")

    def _on_threshold_change(self, value):
        val = int(float(value))
        self.threshold = val / 100.0
        self.lbl_threshold.config(text=f"{val}%")
        # Send to server (debounced — only on release)
        self.after(300, lambda: self.api.set_threshold(self.threshold))

    def on_close(self):
        self.is_running = False
        try:
            self.api.stop()
        except Exception:
            pass
        self.destroy()

    # ============================================
    # Video Feed (MJPEG)
    # ============================================
    def _start_video_feed(self):
        def stream_video():
            try:
                url = self.api.video_feed_url()
                stream = urlopen(url, timeout=10)
                buffer = b""
                while self.is_running:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    # Find JPEG frame
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9")
                    if start != -1 and end != -1:
                        jpg = buffer[start:end + 2]
                        buffer = buffer[end + 2:]
                        img = Image.open(io.BytesIO(jpg))
                        self.after(0, self._update_video_frame, img)
            except Exception as e:
                print(f"[Video] Stream error: {e}")

        self.video_thread = threading.Thread(target=stream_video, daemon=True)
        self.video_thread.start()

    def _update_video_frame(self, pil_image):
        """Update video canvas with a new frame from the MJPEG stream."""
        try:
            canvas_w = self.video_canvas.winfo_width()
            canvas_h = self.video_canvas.winfo_height()
            if canvas_w <= 1 or canvas_h <= 1:
                return

            # Scale image to fit canvas
            img_ratio = pil_image.width / pil_image.height
            canvas_ratio = canvas_w / canvas_h
            if img_ratio > canvas_ratio:
                new_w = canvas_w
                new_h = int(canvas_w / img_ratio)
            else:
                new_h = canvas_h
                new_w = int(canvas_h * img_ratio)

            resized = pil_image.resize((new_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)

            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_w // 2, canvas_h // 2,
                image=photo, anchor=tk.CENTER
            )
            self._current_photo = photo  # Keep reference
        except Exception:
            pass

    # ============================================
    # Polling Loop
    # ============================================
    def _start_polling(self):
        self._poll()

    def _poll(self):
        if self.is_running:
            stats = self.api.stats()
            if stats:
                self.probability = stats.get("fall_probability", 0.0)
                self.fps = stats.get("fps", 0)
                self.threshold = stats.get("threshold", 0.75)

                # Update UI
                self._draw_gauge(self.probability)
                self.lbl_gauge_value.config(
                    text=f"{self.probability * 100:.1f}%"
                )
                self.lbl_fps.config(text=f"FPS {self.fps}")
                self.lbl_latency.config(text=f"Latency {int(self.latency_ms)}ms")

                # Risk badge
                if self.probability < 0.3:
                    self.lbl_risk.config(text="LOW", style="Risk.Low.TLabel")
                elif self.probability < self.threshold:
                    self.lbl_risk.config(text="MEDIUM", style="Risk.Medium.TLabel")
                else:
                    self.lbl_risk.config(text="HIGH", style="Risk.High.TLabel")

                # Chart
                self.prob_history.append(self.probability)
                if len(self.prob_history) > 600:
                    self.prob_history.pop(0)
                self._draw_chart()

                # Alert
                if stats.get("alert", False):
                    ts = datetime.now().strftime("%H:%M:%S")
                    alert_text = f"[{ts}] FALL DETECTED ({self.probability*100:.0f}%)"
                    if not self.alerts or self.alerts[-1] != alert_text:
                        self.alerts.append(alert_text)
                        self.alert_listbox.insert(0, alert_text)
                        if self.alert_listbox.size() > 20:
                            self.alert_listbox.delete(20, tk.END)

        self.after(100, self._poll)  # 10Hz

    # ============================================
    # Gauge Drawing
    # ============================================
    def _draw_gauge(self, probability: float):
        c = self.gauge_canvas
        c.delete("all")
        cx, cy, r = 90, 90, 75
        width = 10

        # Background ring
        c.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=COLORS["bg_highest"], width=width
        )

        # Progress arc
        extent = -probability * 360
        if probability < 0.3:
            color = COLORS["primary"]
        elif probability < 0.75:
            color = COLORS["warning"]
        else:
            color = COLORS["tertiary"]

        c.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=90, extent=extent,
            outline=color, width=width, style=tk.ARC
        )

    # ============================================
    # Chart Drawing
    # ============================================
    def _draw_chart(self):
        c = self.chart_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1 or h <= 1:
            return

        data = self.prob_history
        max_pts = 600

        # Grid lines
        for i in range(5):
            y = (h / 4) * i
            c.create_line(0, y, w, y, fill=COLORS["outline_variant"], width=1)

        if len(data) < 2:
            return

        x_step = w / (max_pts - 1)

        # Line
        points = []
        for i, val in enumerate(data):
            x = (max_pts - len(data) + i) * x_step
            y = h - (val * h)
            points.extend([x, y])

        if len(points) >= 4:
            c.create_line(points, fill=COLORS["primary"], width=2, smooth=True)

        # Threshold line
        thresh_y = h - (self.threshold * h)
        c.create_line(0, thresh_y, w, thresh_y, fill=COLORS["tertiary"],
                       width=1, dash=(4, 4))
        c.create_text(w - 60, thresh_y - 8, text=f"Threshold {int(self.threshold*100)}%",
                       fill=COLORS["tertiary_soft"], font=("Segoe UI", 7))

    # ============================================
    # UI Updates
    # ============================================
    def _update_chips(self, info: dict):
        cam_color = COLORS["secondary"] if info.get("camera_available") else COLORS["outline"]
        model_color = COLORS["secondary"] if info.get("model_loaded") else COLORS["tertiary"]
        self.chip_cam.config(text=f"● Camera", foreground=cam_color)
        self.chip_model.config(text=f"● Model", foreground=model_color)


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser(description="EdgeSight Desktop Client")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"Server URL (default: {DEFAULT_SERVER})")
    args = parser.parse_args()

    # Check server health
    api = EdgeSightAPI(args.server)
    health = api.health()
    if not health:
        print(f"[ERROR] Cannot connect to server at {args.server}")
        print("  Make sure the server is running: python fastapi_server.py")
        print(f"  Then run: python desktop_client.py --server {args.server}")
        sys.exit(1)

    print(f"[OK] Connected to EdgeSight server at {args.server}")
    print(f"  Model loaded: {health.get('model_loaded', False)}")
    print(f"  Camera available: {health.get('camera_available', False)}")

    if not health.get("camera_available", False):
        print("\n[FATAL] No camera detected. EdgeSight strictly requires a camera.")
        print("  Simulation mode has been removed. Please attach a camera.")

    app = EdgeSightDesktop(args.server)
    
    # Disable start if no camera
    if not health.get("camera_available", False):
        app.btn_start.config(state=tk.DISABLED)
        app.video_canvas.delete("all")
        app.video_canvas.create_text(
            320, 240,
            text="FATAL: NO CAMERA DETECTED",
            fill=COLORS["tertiary"],
            font=("Segoe UI", 16, "bold"),
            tags="error"
        )
    
    app.mainloop()


if __name__ == "__main__":
    main()
