# syntax=docker/dockerfile:1.7-labs
# Base image can be overridden to use a prebuilt PyTorch image from Docker Hub, e.g. pytorch/pytorch:2.4.1-cpu

### ???孵???1嚗?閮勗??冽?摰?BASE_IMAGE嚗?閮剔 pytorch/pytorch:2.4.1-cpu嚗??build-arg 閬神??CUDA ??
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

### ?? 靽??航蕭皞舀?metadata嚗憛恬?
ARG APP_GIT_SHA=UNKNOWN
ARG APP_BUILD_TIME=UNKNOWN

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_GIT_SHA=${APP_GIT_SHA} \
    APP_BUILD_TIME=${APP_BUILD_TIME} \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

### ???孵???2嚗??SKIP_PIP_INSTALL ??TORCH_FILTER嚗?雿雿輻??torch ??撱?base image ???? torch
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

### ? 銴ˊ?蝔?蝔?蝣潸???鞈?嚗???Docker layer 敹怠?嚗?
COPY *.py /app/
COPY template2.html /app/
COPY static/ /app/static/
COPY scripts/ /app/scripts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]