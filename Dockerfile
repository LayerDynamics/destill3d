# Destill3D Dockerfile
# Multi-stage build with optional GPU support

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir --prefix=/install .

# ─── Stage 2: Runtime (CPU) ──────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd -m -s /bin/bash destill3d
USER destill3d

# Create data directories
RUN mkdir -p /home/destill3d/.destill3d/models \
    && mkdir -p /home/destill3d/.destill3d/data

# Set environment
ENV DESTILL3D_DATA_DIR=/home/destill3d/.destill3d
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["python", "-m", "destill3d"]
CMD ["--help"]

# ─── Stage 3: Runtime (GPU) ──────────────────────────────────────────────────

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime-gpu

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Install GPU-specific packages
RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    faiss-gpu

# Create non-root user
RUN useradd -m -s /bin/bash destill3d
USER destill3d

RUN mkdir -p /home/destill3d/.destill3d/models \
    && mkdir -p /home/destill3d/.destill3d/data

ENV DESTILL3D_DATA_DIR=/home/destill3d/.destill3d
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all

ENTRYPOINT ["python3", "-m", "destill3d"]
CMD ["--help"]
