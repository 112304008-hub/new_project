# syntax=docker/dockerfile:1.7-labs
FROM python:3.11-slim

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
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
# Install uv (fast Python package installer) and use its cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy application code (exclude data/models here to keep layer cache stable)
# - root Python files and runtime scripts
COPY *.py /app/
# - HTML template and static assets
COPY template2.html /app/
COPY static/ /app/static/
# - optional utility scripts (kept for convenience)
COPY scripts/ /app/scripts/

# Copy data and models in separate layers so changes don't invalidate app code cache
COPY data/ /app/data/
COPY models/ /app/models/

EXPOSE 8000

# Simplified single-line healthcheck: success if HTTP 200
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=2).status==200 else 1)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
