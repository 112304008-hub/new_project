FROM python:3.11-slim

# Minimal image: rely on manylinux wheels (no apt layer to avoid blocked mirrors)
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy requirements and install first to benefit from Docker layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

EXPOSE 8000

# Lightweight healthcheck using stdlib (avoids curl dependency)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python - <<'PY' || exit 1
import urllib.request, sys
try:
    urllib.request.urlopen('http://localhost:8000', timeout=2)
except Exception as e:
    sys.exit(1)
PY

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
