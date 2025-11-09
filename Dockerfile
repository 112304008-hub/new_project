# syntax=docker/dockerfile:1.7-labs
# Base image can be overridden (e.g., pytorch/pytorch:2.4.1-cpu or prebuilt deps image)

# 1) Allow overriding base image (CPU/GPU or a pre-baked deps image)
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
FROM ${BASE_IMAGE}

# Metadata injected by CI/CD (optional)
ARG APP_GIT_SHA=UNKNOWN
ARG APP_BUILD_TIME=UNKNOWN

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_GIT_SHA=${APP_GIT_SHA} \
    APP_BUILD_TIME=${APP_BUILD_TIME} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_LEVEL=INFO

WORKDIR /app

# 2) Optional dependency install (skip when BASE_IMAGE already contains deps)
ARG SKIP_PIP_INSTALL=false
ARG TORCH_FILTER=true
COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/uv <<'EOF'
set -e
if [ "${SKIP_PIP_INSTALL}" = "true" ]; then
    echo "[deps] Skipping dependency installation (SKIP_PIP_INSTALL=true)"
else
    echo "[deps] Installing Python dependencies via uv..."
    python -m pip install --upgrade pip setuptools wheel
    pip install --no-cache-dir uv
    if [ "${TORCH_FILTER}" = "true" ]; then
        echo "[deps] Generating requirements.notorch.txt (filtering torch/torchvision/torchaudio)..."
        python - <<'PY'
import re, sys, io
with io.open('requirements.txt', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
out = []
for line in lines:
    s = line.strip()
    if not s or s.startswith('#'):
        out.append(line)
        continue
    if re.match(r'(?i)^(torch|torchvision|torchaudio)\b', s):
        print(f"[deps] Skipping preinstalled package line: {line}")
        continue
    out.append(line)
with io.open('requirements.notorch.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
PY
        uv pip install --system -r requirements.notorch.txt
    else
        uv pip install --system -r requirements.txt
    fi
fi
EOF

# 3) Copy only runtime essentials (avoid bundling experimental scripts)
COPY main.py /app/main.py
COPY stock.py /app/stock.py
COPY template2.html /app/
COPY static/ /app/static/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=2).status==200 else 1)"

# 顯式指定 log level 讓啟動 & access logs 出現在 docker logs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]