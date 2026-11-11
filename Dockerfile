# Multi-stage build for minimal production image
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt


FROM python:3.12-slim AS runtime

WORKDIR /app

# Install OpenCV runtime dependencies (minimal)
# Note: libgl1-mesa-glx is deprecated in Ubuntu 26.04+; use libgl1 instead
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy pre-built wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Copy application code
COPY backend/ ./backend/
COPY models/ ./models/

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check endpoint
EXPOSE 8000

# Gunicorn with Uvicorn workers - optimized for ONNX Runtime
# Note: binds to 0.0.0.0 within the container so nginx (in a separate
# container) can proxy to it via the Docker network. The port is NOT
# published to the host (no `ports:` mapping in docker-compose.yml),
# so external traffic reaches the API only through nginx over HTTPS.
CMD ["gunicorn", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--worker-tmp-dir", "/dev/shm", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "backend.app.main:app"]