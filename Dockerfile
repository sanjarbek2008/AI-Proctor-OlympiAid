# =============================================================================
#  AI OlympiAid — Production Dockerfile
#  Base: pytorch/pytorch CUDA image (ships with torch/torchvision pre-built)
#  Runtime: FastAPI + Uvicorn on port 8000
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS builder

# System packages needed at build time (libGL for OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the dependency manifest first (layer-cache friendly)
COPY requirements.txt .

# Install into an isolated prefix so we can copy cleanly to the runtime stage.
# torch / torchvision are already present in the base image, so we skip them
# here to avoid a 3 GB redundant download.
RUN pip install --no-cache-dir --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        $(grep -v -E "^(torch|torchvision)$" requirements.txt)


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS runtime

LABEL maintainer="AI OlympiAid Team"
LABEL description="AI Proctoring Service – FastAPI + CUDA"

# Minimal runtime OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Pull installed packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# ── Copy application code ─────────────────────────────────────────────────────
COPY app/              ./app/
COPY requirements.txt  ./

# ── Copy model / task files shipped with the repo ─────────────────────────────
# These are large binary assets required at inference time.
# If they are pulled from remote storage instead, remove these lines
# and handle download in an entrypoint script or init container.
COPY yolov8n.pt                    ./
COPY face_landmarker.task          ./
COPY blaze_face_short_range.tflite ./

# ── Environment ───────────────────────────────────────────────────────────────
# PYTHONUNBUFFERED → log output is not buffered (critical for container logs)
# PYTHONDONTWRITEBYTECODE → skip .pyc files (saves space)
# PYTHONPATH → makes `from app.xxx import yyy` work from /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Tell PyTorch / CUDA to use GPU 0 by default
    CUDA_VISIBLE_DEVICES=0 \
    # Suppress tokenizer parallelism warnings from HuggingFace
    TOKENIZERS_PARALLELISM=false \
    # MediaPipe / OpenCV: no display needed
    DISPLAY=""

# Expose the FastAPI port
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# • app.main:app  → Python package path (app/ dir contains __init__.py)
# • --workers 1   → single worker; GPU models are not fork-safe
# • --loop uvloop → faster async event loop (installed via uvicorn[standard])
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "info"]
