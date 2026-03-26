# EdgeSight - Fall Detection System
# Multi-stage build for optimal image size

# ============================================
# Stage 1: Python runtime for inference
# ============================================
FROM python:3.10-slim as python-base

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart websockets

# Copy Python application code
COPY model/ model/
COPY inference/ inference/
COPY data/ data/
COPY scripts/ scripts/

# ============================================
# Stage 2: Final image
# ============================================
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=python-base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=python-base /app /app

# Create directories
RUN mkdir -p model/exported model/checkpoints app/logs data/processed

# Copy web server
COPY web_server.py .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/stats || exit 1

# Run web server
CMD ["python", "web_server.py"]