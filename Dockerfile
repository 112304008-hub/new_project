FROM python:3.11-slim

# Build metadata args (can be overridden at build time)
ARG APP_GIT_SHA=UNKNOWN
ARG APP_BUILD_TIME=UNKNOWN

# Minimal image: rely on manylinux wheels (no apt layer to avoid blocked mirrors)
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    APP_GIT_SHA=${APP_GIT_SHA} \
    APP_BUILD_TIME=${APP_BUILD_TIME}

WORKDIR /app

# Copy requirements and install first to benefit from Docker layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

EXPOSE 8000

# Simplified single-line healthcheck: success if HTTP 200
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=2).status==200 else 1)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
