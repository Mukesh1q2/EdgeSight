/**
 * EdgeSight Dashboard — Client-Side Application
 * Real-time WebSocket connection, animated gauge, rolling chart, alert log
 */

// ============================================
// Configuration
// ============================================
const API_BASE = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws`;

// ============================================
// State
// ============================================
let ws = null;
let isRunning = false;
let probabilityHistory = [];
const MAX_HISTORY = 600; // 60 seconds at 10Hz
let chartCtx = null;
let gaugeCircumference = 2 * Math.PI * 85; // r=85 from SVG

// ============================================
// DOM Elements
// ============================================
const els = {
    videoFeed: document.getElementById('video-feed'),
    videoPlaceholder: document.getElementById('video-placeholder'),
    gaugeProgress: document.getElementById('gauge-progress'),
    gaugeValue: document.getElementById('gauge-value'),
    gaugeCard: document.getElementById('gauge-card'),
    riskLevel: document.getElementById('risk-level'),
    fpsValue: document.getElementById('fps-value'),
    latencyValue: document.getElementById('latency-value'),
    chipCamera: document.getElementById('chip-camera'),
    chipModel: document.getElementById('chip-model'),
    thresholdSlider: document.getElementById('threshold-slider'),
    thresholdValue: document.getElementById('threshold-value'),
    probChart: document.getElementById('prob-chart'),
    alertList: document.getElementById('alert-list'),
    alertCount: document.getElementById('alert-count'),
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    simBanner: document.getElementById('sim-banner'),
    livePulse: document.getElementById('live-pulse'),
};

// ============================================
// API Functions
// ============================================
async function apiPost(endpoint) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
        return await res.json();
    } catch (err) {
        console.error(`API error (${endpoint}):`, err);
        return null;
    }
}

async function apiGet(endpoint) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`);
        return await res.json();
    } catch (err) {
        console.error(`API error (${endpoint}):`, err);
        return null;
    }
}

// ============================================
// Start / Stop Detection
// ============================================
async function startDetection() {
    const result = await apiPost('/api/start');
    if (result && result.status !== 'error') {
        isRunning = true;
        els.btnStart.disabled = true;
        els.btnStop.disabled = false;

        // Show video feed
        els.videoPlaceholder.style.display = 'none';
        els.videoFeed.style.display = 'block';
        els.videoFeed.src = `${API_BASE}/video_feed?t=${Date.now()}`;

        // Show live pulse
        els.livePulse.style.display = 'flex';

        // Connect WebSocket
        connectWebSocket();

        // Fetch system info
        updateSystemInfo();

        // Clear history
        probabilityHistory = [];
        clearAlerts();
    }
}

async function stopDetection() {
    const result = await apiPost('/api/stop');
    if (result) {
        isRunning = false;
        els.btnStart.disabled = false;
        els.btnStop.disabled = true;

        // Hide video feed
        els.videoFeed.style.display = 'none';
        els.videoFeed.src = '';
        els.videoPlaceholder.style.display = 'flex';

        // Hide live pulse
        els.livePulse.style.display = 'none';

        // Disconnect WebSocket
        if (ws) {
            ws.close();
            ws = null;
        }

        // Reset gauge
        updateGauge(0);
        updateRiskBadge(0);
    }
}

// ============================================
// WebSocket Connection
// ============================================
function connectWebSocket() {
    if (ws) {
        ws.close();
    }

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('[WS] Connected');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleStatsUpdate(data);
        } catch (err) {
            console.error('[WS] Parse error:', err);
        }
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected');
        if (isRunning) {
            // Attempt reconnect after 2s
            setTimeout(connectWebSocket, 2000);
        }
    };

    ws.onerror = (err) => {
        console.error('[WS] Error:', err);
    };
}

// ============================================
// Handle Real-Time Stats
// ============================================
function handleStatsUpdate(data) {
    const prob = data.fall_probability || 0;
    const fps = data.fps || 0;
    const latency = data.latency_ms || 0;
    const threshold = data.threshold || 0.75;
    const isAlert = data.alert || false;

    // Update gauge
    updateGauge(prob);

    // Update risk badge
    updateRiskBadge(prob, threshold);

    // Update telemetry chips
    els.fpsValue.textContent = fps;
    els.latencyValue.textContent = `${Math.round(latency)}ms`;

    // Update chart
    probabilityHistory.push(prob);
    if (probabilityHistory.length > MAX_HISTORY) {
        probabilityHistory.shift();
    }
    drawChart();

    // Handle alerts
    if (isAlert) {
        const timestamp = data.timestamp || new Date().toLocaleTimeString();
        addAlert(timestamp, prob);
    }
}

// ============================================
// Gauge Animation
// ============================================
function updateGauge(probability) {
    const percent = probability * 100;
    const offset = gaugeCircumference - (gaugeCircumference * probability);

    els.gaugeProgress.setAttribute('stroke-dashoffset', Math.max(0, offset));
    els.gaugeValue.textContent = percent.toFixed(1);

    // Dynamic color
    if (percent < 30) {
        els.gaugeValue.style.color = '#a8e8ff'; // Cyan
    } else if (percent < 75) {
        els.gaugeValue.style.color = '#ffcc80'; // Orange
    } else {
        els.gaugeValue.style.color = '#ffb3b3'; // Red
    }
}

// ============================================
// Risk Badge
// ============================================
function updateRiskBadge(probability, threshold = 0.75) {
    let level, risk;

    if (probability < 0.3) {
        level = 'LOW';
        risk = 'low';
    } else if (probability < threshold) {
        level = 'MEDIUM';
        risk = 'medium';
    } else {
        level = 'HIGH';
        risk = 'high';
    }

    els.riskLevel.textContent = level;
    els.riskLevel.setAttribute('data-risk', risk);
}

// ============================================
// Chart Drawing
// ============================================
function initChart() {
    const canvas = els.probChart;
    chartCtx = canvas.getContext('2d');

    // Set canvas resolution
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * 2;
    canvas.height = rect.height * 2;
    chartCtx.scale(2, 2);
}

function drawChart() {
    if (!chartCtx) return;

    const canvas = els.probChart;
    const w = canvas.width / 2;
    const h = canvas.height / 2;
    const data = probabilityHistory;

    // Clear
    chartCtx.clearRect(0, 0, w, h);

    // Background
    chartCtx.fillStyle = 'rgba(23, 27, 40, 0.5)';
    chartCtx.fillRect(0, 0, w, h);

    // Grid lines
    chartCtx.strokeStyle = 'rgba(60, 73, 78, 0.2)';
    chartCtx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = (h / 4) * i;
        chartCtx.beginPath();
        chartCtx.moveTo(0, y);
        chartCtx.lineTo(w, y);
        chartCtx.stroke();
    }

    if (data.length < 2) return;

    const xStep = w / (MAX_HISTORY - 1);

    // Glow effect (under the line)
    const gradient = chartCtx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, 'rgba(0, 212, 255, 0.15)');
    gradient.addColorStop(1, 'rgba(0, 212, 255, 0)');

    chartCtx.beginPath();
    chartCtx.moveTo((MAX_HISTORY - data.length) * xStep, h);
    for (let i = 0; i < data.length; i++) {
        const x = (MAX_HISTORY - data.length + i) * xStep;
        const y = h - (data[i] * h);
        if (i === 0) chartCtx.lineTo(x, y);
        else chartCtx.lineTo(x, y);
    }
    chartCtx.lineTo((MAX_HISTORY - 1) * xStep, h);
    chartCtx.closePath();
    chartCtx.fillStyle = gradient;
    chartCtx.fill();

    // Line
    chartCtx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = (MAX_HISTORY - data.length + i) * xStep;
        const y = h - (data[i] * h);
        if (i === 0) chartCtx.moveTo(x, y);
        else chartCtx.lineTo(x, y);
    }
    chartCtx.strokeStyle = '#00d4ff';
    chartCtx.lineWidth = 2;
    chartCtx.lineJoin = 'round';
    chartCtx.lineCap = 'round';
    chartCtx.shadowColor = '#00d4ff';
    chartCtx.shadowBlur = 8;
    chartCtx.stroke();
    chartCtx.shadowBlur = 0;

    // Threshold line
    const thresholdY = h - (parseFloat(els.thresholdSlider.value) / 100) * h;
    chartCtx.beginPath();
    chartCtx.moveTo(0, thresholdY);
    chartCtx.lineTo(w, thresholdY);
    chartCtx.strokeStyle = 'rgba(255, 23, 68, 0.4)';
    chartCtx.lineWidth = 1;
    chartCtx.setLineDash([4, 4]);
    chartCtx.stroke();
    chartCtx.setLineDash([]);

    // Threshold label
    chartCtx.fillStyle = 'rgba(255, 23, 68, 0.6)';
    chartCtx.font = '9px Space Grotesk, sans-serif';
    chartCtx.fillText(`Threshold ${els.thresholdSlider.value}%`, w - 85, thresholdY - 4);
}

// ============================================
// Alert Log
// ============================================
let lastAlertTime = 0;

function addAlert(timestamp, probability) {
    const now = Date.now();
    if (now - lastAlertTime < 3000) return; // 3s cooldown
    lastAlertTime = now;

    // Remove empty message
    const emptyMsg = els.alertList.querySelector('.alert-empty');
    if (emptyMsg) emptyMsg.remove();

    const li = document.createElement('li');
    li.className = 'alert-item';
    li.innerHTML = `
        <span class="alert-time">[${timestamp}]</span>
        <span>FALL DETECTED (${(probability * 100).toFixed(0)}%)</span>
    `;

    els.alertList.insertBefore(li, els.alertList.firstChild);

    // Keep max 20
    while (els.alertList.children.length > 20) {
        els.alertList.removeChild(els.alertList.lastChild);
    }

    // Update count
    els.alertCount.textContent = els.alertList.children.length;
    els.alertCount.style.background = 'var(--tertiary-container)';
    els.alertCount.style.color = 'var(--tertiary-soft)';
}

function clearAlerts() {
    els.alertList.innerHTML = '<li class="alert-empty">No alerts yet</li>';
    els.alertCount.textContent = '0';
    els.alertCount.style.background = '';
    els.alertCount.style.color = '';
}

// ============================================
// System Info
// ============================================
async function updateSystemInfo() {
    const info = await apiGet('/api/system_info');
    if (!info) return;

    // Camera chip
    const camDot = els.chipCamera.querySelector('.status-dot');
    if (info.camera_available) {
        camDot.className = 'status-dot status-dot--active';
    } else {
        camDot.className = 'status-dot status-dot--inactive';
    }

    // Model chip
    const modelDot = els.chipModel.querySelector('.status-dot');
    if (info.model_loaded) {
        modelDot.className = 'status-dot status-dot--active';
    } else {
        modelDot.className = 'status-dot status-dot--error';
    }

    // Strict Reality Check
    if (!info.camera_available) {
        els.btnStart.disabled = true;
        els.btnStart.style.opacity = '0.5';
        els.btnStart.title = "No camera detected";
        const placeholderIcon = els.videoPlaceholder.querySelector('#video-placeholder-icon');
        const placeholderText = els.videoPlaceholder.querySelector('#video-placeholder-text');
        
        if (placeholderIcon) placeholderIcon.textContent = "⚠️";
        if (placeholderText) placeholderText.innerHTML = "<strong style='color:#ff1744'>FATAL: NO CAMERA DETECTED</strong><br>Check hardware connection to start.";
    }
}

// ============================================
// Threshold Slider
// ============================================
els.thresholdSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    els.thresholdValue.textContent = `${value}%`;
});

els.thresholdSlider.addEventListener('change', async (e) => {
    const value = parseInt(e.target.value) / 100;
    await fetch(`${API_BASE}/api/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: value }),
    });
});

// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    updateGauge(0);
    updateRiskBadge(0);

    // Check server health on load
    apiGet('/health').then((data) => {
        if (data) {
            const modelDot = els.chipModel.querySelector('.status-dot');
            modelDot.className = data.model_loaded
                ? 'status-dot status-dot--active'
                : 'status-dot status-dot--inactive';
        }
    });

    // Handle window resize for chart
    window.addEventListener('resize', () => {
        initChart();
        drawChart();
    });
});

// Make functions globally available
window.startDetection = startDetection;
window.stopDetection = stopDetection;
