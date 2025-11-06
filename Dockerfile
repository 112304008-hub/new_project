# syntax=docker/dockerfile:1.7-labs
# Allow swapping in a prebuilt base image that already has dependencies installed
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Build metadata args (can be overridden at build time)
ARG APP_GIT_SHA=UNKNOWN
ARG APP_BUILD_TIME=UNKNOWN

# Minimal image: rely on manylinux wheels (no apt layer to avoid blocked mirrors)
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    APP_GIT_SHA=${APP_GIT_SHA} \
    APP_BUILD_TIME=${APP_BUILD_TIME} \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy requirements and install first to benefit from Docker layer caching
ARG SKIP_PIP_INSTALL=false
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv <<'EOF'
set -e
if [ "${SKIP_PIP_INSTALL}" = "true" ]; then
    echo "[deps] Skipping dependency installation (SKIP_PIP_INSTALL=true)"
else
    echo "[deps] Installing Python dependencies via uv..."
    python -m pip install --upgrade pip setuptools wheel
    pip install uv
    uv pip install --system -r requirements.txt
fi
EOF

# Copy application code (exclude data/models here to keep layer cache stable)
# - root Python files and runtime scripts
COPY *.py /app/
# - HTML template and static assets
COPY template2.html /app/
COPY static/ /app/static/
# - optional utility scripts (kept for convenience)
COPY scripts/ /app/scripts/

EXPOSE 8000

# Simplified single-line healthcheck: success if HTTP 200
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=2).status==200 else 1)" 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
