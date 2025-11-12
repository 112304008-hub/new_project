<div align="center">

# 📈 股價之神 - AI 股票預測系統

基於 FastAPI + 機器學習的股票短期預測服務  
提供資料建置、模型推論、自動更新、批次處理等完整功能

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[快速開始](#-快速開始) • [API 文件](#api-快速參考) • [部署指南](#-docker-部署) • [完整文檔](docs/README.md)

</div>

---

## ✨ 核心功能

### 🎯 預測服務
- **即時預測**：支援隨機森林 (RF) 與邏輯回歸 (LR) 兩種模型
- **多股票支援**：美股、台股等多市場股票預測
- **互動介面**：提供網頁版抽籤預測介面

### 📊 資料管理
- **自動建置**：Yahoo Finance 自動抓取歷史資料
- **特徵工程**：50+ 技術指標與滯後特徵
- **批次處理**：支援 S&P 500、Nasdaq-100 等指數批次建置

### ⚙️ 自動化
- 全域 5 分鐘自動更新（內建）：服務啟動後，每 5 分鐘掃描 `data/` 內現有 `*_short_term_with_lag3.csv` 以受控併發更新；可在 `main.py` 的 `GLOBAL_UPDATE_INTERVAL_MIN` 與 `GLOBAL_UPDATE_CONCURRENCY` 調整。
- 批次建置：提供 `/api/bulk_build_start`、`/api/bulk_build_status`、`/api/bulk_build_stop`。

### 🔍 監控與診斷
- 健康檢查：`/health`
- 診斷工具：`/api/diagnostics`、`/api/stattests`、`/api/lag_stats`、`/api/series`、`/api/latest_features`

> 附註：本服務僅使用「個股 CSV」（`data/{symbol}_short_term_with_lag3.csv`），不再依賴聚合檔；多數資料/統計端點需帶 `symbol` 參數。服務預設會自動執行「全域 5 分鐘更新」；如需外部排程，也可改用批次 API。專案已移除 `/api/auto/*` 端點與註冊檔。


## 1) 啟動服務（開發模式）

# 建立虛擬環境並安裝依賴（如尚未安裝）
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 啟動 FastAPI（熱重載）
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

開啟瀏覽器：http://localhost:8000

---

## 2) 兩個最常用的 API

1. 建置單一股票 CSV（若不存在會自動用 yfinance 下載並產生特徵）

```powershell
# 產生或更新 AAPL 的特徵 CSV
Invoke-RestMethod -Method GET -Uri "http://localhost:8000/api/build_symbol?symbol=AAPL"
```

2. 取得預測結果（簡潔版）

```powershell
# 使用隨機森林模型 (rf) 預測 AAPL
Invoke-RestMethod -Method GET -Uri "http://localhost:8000/api/predict?symbol=AAPL&model=rf"
```

更多：

```powershell
# 抽籤格式（含 threshold 與信心度）
Invoke-RestMethod -Method GET -Uri "http://localhost:8000/api/draw?symbol=AAPL&model=rf"

# 列出目前有資料的所有 symbols
Invoke-RestMethod -Method GET -Uri "http://localhost:8000/api/list_symbols"
```


---

## 3) 本機快速測試（不經由 HTTP）

使用 `scripts/dev/run_predict.py` 直接呼叫 `stock.predict()`：

```powershell
python -m scripts.dev.run_predict --symbol AAPL --model rf
```

---

## 🧭 正式部署（Production）

提供兩種方式：
1) 只有後端（直接聽 APP_PORT，預設 8001，最快測起來）

```powershell
# 最短可跑版本（保證可用；可直接複製貼上）
docker build -f C:\Users\runyu\OneDrive\桌面\new-project\Dockerfile -t new_project:latest C:\Users\runyu\OneDrive\桌面\new-project
Set-Location C:\Users\runyu\OneDrive\桌面\new-project\infra\compose
docker compose -f docker-compose.prod.yml up -d web
$port = if ($env:APP_PORT) { $env:APP_PORT } else { 8001 }
for ($i=0; $i -lt 8; $i++) { try { Invoke-WebRequest -Uri "http://localhost:$port/health" -TimeoutSec 3; break } catch { Start-Sleep -Seconds 2 } }

# 或者一鍵腳本（從任何目錄都可）：
powershell -File .\infra\compose\run_web.ps1

### 容器名稱衝突處理

若看到錯誤：`Error when allocating new name: Conflict. The container name "..." is already in use`，表示有舊容器尚未移除。

現行 compose 未再強制固定 `container_name`，常見的自動名稱可能為：`new-project-web-1`、`new-project-caddy-1` 或 `compose-web-1` 等（依你專案資料夾與 Docker Desktop 版本而異）。

請勿在 PowerShell 直接貼上 Markdown 的三個反引號 ``` （那只是 README 格式），只要貼指令本身即可。

快速清理方式（自動偵測並移除所有名稱含 new_project 或 newproject 的容器）：
```powershell
# 列出相關容器
docker ps -a --format "table {{.ID}}\t{{.Names}}\t{{.Status}}" | Select-String new_project
docker ps -a --format "table {{.ID}}\t{{.Names}}\t{{.Status}}" | Select-String newproject

# 停止並移除（動態）
docker ps -a --format "{{.Names}}" | Select-String new_project | ForEach-Object { docker stop $_.Line 2>$null; docker rm $_.Line 2>$null }
docker ps -a --format "{{.Names}}" | Select-String newproject   | ForEach-Object { docker stop $_.Line 2>$null; docker rm $_.Line 2>$null }

# 重新啟動 web（僅後端）
docker compose -f docker-compose.prod.yml up -d web
```

或使用提供的腳本：
```powershell
Set-Location infra/compose
./cleanup_containers.ps1
docker compose -f docker-compose.prod.yml up -d web
```

若只想確認名稱：
```powershell
docker ps -a --format "{{.Names}}" | Select-String new_project
docker ps -a --format "{{.Names}}" | Select-String newproject
```
```

2) 含 Caddy 反向代理與 HTTPS（80/443）

最簡用法（一鍵腳本，從任何目錄都可執行）：

```powershell
powershell -File .\infra\compose\run_all.ps1  # 會建置映像，啟動 web+caddy，並做健康檢查
# 可選參數：-Domain your-domain.example -AcmeEmail you@example.com -ApiKey your-secret -AppPort 8001
```

手動（可選）：

```powershell
# 在專案根目錄建置映像檔
docker build -t new_project:latest .

# 設定必要環境變數（或改用 .env）
$env:DOMAIN = "your-domain.example"
$env:ACME_EMAIL = "you@example.com"
# 若要保護 /api/*：
# $env:API_KEY = "your-secret-key"

# 啟動所有服務（web + caddy）
Set-Location infra/compose
docker compose -f docker-compose.prod.yml up -d

# 檢查健康（走 Caddy 80/443，不受 APP_PORT 影響）
Invoke-WebRequest -Uri "http://$env:DOMAIN/health"

# 觀察日誌與停止
docker compose -f docker-compose.prod.yml logs -f web
docker compose -f docker-compose.prod.yml logs -f caddy
docker compose -f docker-compose.prod.yml down
```
```

說明：
- 正式 compose 會將本機 `data/` 與 `models/` 掛載為容器內的 `/app/data` 與 `/app/models`（bind mount），重啟不會遺失資料。
- 更新程式：重新 `docker build -t new_project:latest .` 後，再 `docker compose -f docker-compose.prod.yml up -d` 即可滾更。
- 若要使用外部排程取代內建全域更新，可關閉 `ENABLE_GLOBAL_UPDATER` 並定期呼叫 `/api/bulk_build_start`。

## 🚀 CI/CD（GitHub Actions）

（簡化）目前不使用 GHCR 發佈映像，相關指引已移除。若未來需要，可再加入 CI 工作流與 Registry 配置說明。
## 🚢 Docker 建置與執行（開發/測試）

Dockerfile 預設使用基底映像：`pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`

建置前請先顯式拉取基底映像（避免網路或 mirror 造成的拉取異常）：

```powershell
docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
```

接著建置與執行：

```powershell
# 建置（預設 GPU Runtime 基底；會自動安裝 requirements.txt，並略過 torch/torchvision/torchaudio）
docker build -t new_project:dev .

# 執行（容器內固定使用 8000；若主機 8000 已被占用，改用 8001 或其他）
docker run --rm -p 8001:8000 --name stock-ai new_project:dev

# 檢查健康
Invoke-WebRequest -Uri "http://localhost:8001/health"

# 追蹤日誌
docker logs -f stock-ai

# 停止容器
docker stop stock-ai

### 常見問題（Troubleshooting）

若出現錯誤：`Bind for 0.0.0.0:8000 failed: port is already allocated`

表示主機的 8000 埠已被其他程式或容器占用，處理方式：

1) 直接改用其他主機埠（最簡單）：

```powershell
docker run --rm -p 8001:8000 --name stock-ai new_project:dev
Invoke-WebRequest -Uri "http://localhost:8001/health"
```

2) 找出並停止占用 8000 的容器：

```powershell
docker ps --filter "publish=8000" --format "table {{.ID}}\t{{.Names}}\t{{.Ports}}"
# 停止該容器
docker stop <容器ID或名稱>
```

3) 若是本機程式佔用（非容器），查詢 PID 並結束：

```powershell
netstat -ano | Select-String ":8000"
taskkill /PID <PID> /F
```

#### 健康檢查連線被關閉 / 8001 連不上

檢查 `docker ps` 的埠對映；若顯示 `0.0.0.0:8080->8000/tcp`，代表主機埠其實是 8080，你需改用：
```powershell
Invoke-WebRequest -Uri "http://localhost:8080/health"
```
常見原因：
- 你先前設定了 `$env:APP_PORT = 8080`，後面健檢仍打 8001。
- 重新 up 之前忘記關閉舊容器，使你混淆目前使用的主機埠。

#### 想暫時停用全域自動更新（GLOBAL UPDATER）

現在可用環境變數關閉：
```powershell
$env:ENABLE_GLOBAL_UPDATER = "false"
docker compose -f docker-compose.prod.yml up -d web
```
或在 `docker-compose.prod.yml` 的 `web.environment` 增加：
```yaml
	- ENABLE_GLOBAL_UPDATER=false
```
再次啟動後，日誌不會再出現 `[startup] global updater started`。

#### 為什麼容器內 curl 不存在？

基底映像是 PyTorch runtime，未預裝 curl。可改用：
```powershell
docker compose -f docker-compose.prod.yml exec web python - <<'PY'
import urllib.request;print(urllib.request.urlopen('http://localhost:8000/health',timeout=3).read().decode())
PY
```
如需安裝 curl（除錯用）：
```powershell
docker compose -f docker-compose.prod.yml exec web bash -c "apt-get update && apt-get install -y curl && curl -s http://localhost:8000/health"
```
```

### 可調整的 Build 參數（--build-arg）

- BASE_IMAGE：覆寫基底映像（預設 `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`）
- SKIP_PIP_INSTALL：是否跳過依賴安裝（預設 false；除非你的 BASE_IMAGE 已預先安裝好所有 requirements，否則不要設為 true）
- TORCH_FILTER：是否在安裝時略過 torch/torchvision/torchaudio（預設 true；讓 torch 維持使用基底映像的版本）

範例：

```powershell
# 以 CPU 版 PyTorch 作為基底（適用於沒有 GPU 的機器）
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cpu -t new_project:cpu .

# 強制由 pip 安裝（含 torch）— 通常不建議，僅在你確定需要覆蓋 PyTorch 版本時
docker build --build-arg TORCH_FILTER=false -t new_project:full .
```

### 開發者常用指令速查（PowerShell）

```powershell
# 列出容器與映像
docker ps -a; docker images

# 進入容器（互動 shell）
docker exec -it stock-ai bash

# 清理暫存/中止的容器與懸掛映像
docker container prune -f; docker image prune -f
```

> 提醒：Windows 請先啟動 Docker Desktop（鯨魚圖示為 Running）。

---

## 附註

- 本服務僅使用「個股 CSV」（data/{symbol}_short_term_with_lag3.csv），不再依賴聚合檔。
- 多數資料/統計端點皆需帶 symbol 參數（例如 `/api/diagnostics?symbol=AAPL`）。
- 本專案已移除批次腳本與多餘的工具腳本；如需批次或自動更新，建議改用 API（`/api/build_symbol`、`/api/bulk_build_*`）。
