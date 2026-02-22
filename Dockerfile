# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)
#
# ALICE Dockerfile - CPU/CUDA variant
#
# Multi-stage build for minimal runtime image.
# For AMD ROCm, use Dockerfile.rocm instead.
#
# Build:
#   docker build -t alice:latest .                          # CPU-only
#   docker build --build-arg GPU=cuda -t alice:cuda .       # NVIDIA CUDA
#
# Run:
#   docker run -p 8080:8080 -v alice-models:/data/models alice:latest
#   docker run --gpus all -p 8080:8080 -v alice-models:/data/models alice:cuda

ARG GPU=cpu
ARG PYTHON_VERSION=3.11

# =============================================================================
# Stage 1: Builder - install dependencies
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

ARG GPU

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install PyTorch based on GPU argument
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$GPU" = "cuda" ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124; \
    else \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime - minimal image
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG GPU

LABEL org.opencontainers.image.title="ALICE"
LABEL org.opencontainers.image.description="Artificial Latent Image Composition Engine - Remote Stable Diffusion Service"
LABEL org.opencontainers.image.url="https://github.com/SyntheticAutonomicMind/ALICE"
LABEL org.opencontainers.image.source="https://github.com/SyntheticAutonomicMind/ALICE"
LABEL org.opencontainers.image.licenses="GPL-3.0-only"

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r alice && useradd -r -g alice -d /app -s /sbin/nologin alice

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create application directory structure
WORKDIR /app
RUN mkdir -p /data/models /data/images /data/logs /data/data /data/auth /config

# Copy application code
COPY src/ /app/src/
COPY web/ /app/web/
COPY requirements.txt /app/

# Copy default config (user can override via volume mount)
COPY docker/config.docker.yaml /config/config.yaml

# Set ownership
RUN chown -R alice:alice /app /data /config

# Environment
ENV ALICE_CONFIG=/config/config.yaml
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# GPU-specific environment
# CUDA: GPU access managed by --gpus flag at runtime
# CPU: No special environment needed

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8080/livez || exit 1

# Expose port
EXPOSE 8080

# Switch to non-root user
USER alice

# Entry point
CMD ["python", "-m", "src.main"]
