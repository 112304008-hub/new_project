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
Invoke-WebRequest -Uri "http://localhost:8000/api/build_symbol?symbol=AAPL" | ConvertFrom-Json
```

2. 執行預測（需要 models/ 中已有已訓練模型檔 e.g. rf_pipeline.pkl / rf_threshold.pkl；symbol 必填）

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/api/draw?model=rf&symbol=AAPL" | ConvertFrom-Json
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

1) 只有後端（直接聽 8000 埠，最快測起來）

```powershell
# 在專案根目錄建置映像檔
docker build -t new_project:latest .

# 切換到 compose 目錄，只啟動 web 服務
Set-Location infra/compose
docker compose -f docker-compose.prod.yml up -d web

# 健康檢查（服務直接在 8000）
Invoke-WebRequest -Uri "http://localhost:8000/health"

# 觀察日誌與停止
docker compose -f docker-compose.prod.yml logs -f web
docker compose -f docker-compose.prod.yml down
```

2) 含 Caddy 反向代理與 HTTPS（80/443）

# 在專案根目錄建置映像檔
docker build -t new_project:latest .

# 設定必要環境變數（或改用 .env）
$env:DOMAIN = "your-domain.example"  # 你的網域
$env:ACME_EMAIL = "you@example.com"  # 憑證註冊 email（可選）
# 若要保護 /api/*：
# $env:API_KEY = "your-secret-key"

# 切到 compose 目錄並啟動所有服務（web + caddy）
Set-Location infra/compose
docker compose -f docker-compose.prod.yml up -d

# 用網域檢查健康情況
Invoke-WebRequest -Uri "http://$env:DOMAIN/health"

# 觀察日誌與停止
docker compose -f docker-compose.prod.yml logs -f web
docker compose -f docker-compose.prod.yml logs -f caddy
docker compose -f docker-compose.prod.yml down
```

說明：
- 正式 compose 會使用 named volumes 保存 `/app/data` 與 `/app/models`，重啟不會遺失資料。
- 更新程式：重新 `docker build -t new_project:latest .` 後，再 `docker compose -f docker-compose.prod.yml up -d` 即可滾更。
- 若要使用外部排程取代內建全域更新，可關閉 `ENABLE_GLOBAL_UPDATER` 並定期呼叫 `/api/bulk_build_start`。

## 🚀 CI/CD（GitHub Actions）與雲端依賴層

本專案提供單一工作流（`.github/workflows/docker.yml`）來同時處理「依賴層（deps）」與「應用層（app）」的建置與發佈：

- 依賴層（deps）：依 `requirements.txt` 計算 SHA-12 指紋，建置並推送
  - 產物：`ghcr.io/<owner>/<repo>/py311-deps:<sha12>`
- 應用層（app）：以 deps 當 `BASE_IMAGE`，並設 `SKIP_PIP_INSTALL=true` 跳過安裝，加速建置
  - 觸發：push 到 `main`、建立 tag、或手動觸發
  - 推送標籤：
    - `ghcr.io/<owner>/<repo>/app:<git_sha>`（每次 build 都有）
    - `:latest`（僅 tag 釋出時）
    - `:<tag>`（當你打 tag 時）

如何本機重用雲端依賴層做「薄層 build」：

```powershell
# 計算 requirements 指紋（12 碼）
$reqHash = (Get-FileHash .\requirements.txt -Algorithm SHA256).Hash.Substring(0,12)

# 使用 GHCR 的 deps 當 BASE_IMAGE，並跳過 pip 安裝
docker build -f Dockerfile `
  --build-arg BASE_IMAGE=ghcr.io/112304008-hub/new_project/py311-deps:$reqHash `
  --build-arg SKIP_PIP_INSTALL=true `
  -t new_project:dev .
```

小提醒：
- 若 GHCR 套件是私有，先 `docker login ghcr.io`（需要 PAT，權限含 Packages:read/write）。
- 只要 `requirements.txt` 沒變，`py311-deps:<sha12>` 可長期重用，App 重建只需幾秒。

## 📦 從 GHCR 拉取與啟動（完成 CI 後）

> 前提：若 GHCR 套件是私有，請先 `docker login ghcr.io`；若公開則可直接拉。

```powershell
# 建議使用特定版本（tag 或 git sha）
docker pull ghcr.io/112304008-hub/new_project/app:v0.1.0
# 或
docker pull ghcr.io/112304008-hub/new_project/app:<git_sha>

# 執行（服務在 8000 埠）
docker run --rm -p 8000:8000 ghcr.io/112304008-hub/new_project/app:v0.1.0

# 健康檢查
Invoke-WebRequest -Uri "http://localhost:8000/health"
```

> 註：`:latest` 只有在「打 tag」時才會由 CI 發佈；平常請用 `:<git_sha>` 或 `:<tag>` 鎖定版本。

## 🛠️ 本機建置映像（兩種方式）

1) 極速（重用雲端依賴層，推薦開發時使用）

```powershell
# 方式 A：一鍵腳本（建議）
scripts\build_from_ghcr.ps1 -AppTag dev
# 產出：new_project:dev

# 方式 B：手動（直接使用 GHCR 依賴映像當 BASE_IMAGE）
$reqHash = (Get-FileHash .\requirements.txt -Algorithm SHA256).Hash.Substring(0,12)
docker build --build-arg BASE_IMAGE=ghcr.io/112304008-hub/new_project/py311-deps:$reqHash --build-arg SKIP_PIP_INSTALL=true -t new_project:dev .
```

2) 備用（不依賴雲端，直接完整安裝 requirements）

```powershell
docker build -t new_project:latest .
```

> 小提醒：Windows 請先啟動 Docker Desktop（鯨魚圖示為 Running）。

---

## 附註

- 本服務僅使用「個股 CSV」（data/{symbol}_short_term_with_lag3.csv），不再依賴聚合檔。
- 多數資料/統計端點皆需帶 symbol 參數（例如 /api/diagnostics?symbol=AAPL）。

- 若不使用 Makefile，可直接照上述命令操作；Makefile 只是幫你把常用命令取個別名（見下）。
- 本專案已移除批次腳本與多餘的工具腳本；如需批次或自動更新，建議改用 API（/api/build_symbol）自行外掛排程。

---

## Makefile 是什麼？可以刪嗎？
### ⚡ 加速 Docker 建置：預先烤好的依賴層（強烈推薦）

若每次 `docker build` 都要重新安裝 `requirements.txt` 太慢，您可以先建一個「已安裝好所有套件」的基底映像，之後只要複製程式碼就能秒級完成建置。

步驟（PowerShell）：

```powershell
# 1) 以 requirements.txt 的雜湊值當作標籤，建立依賴映像
$reqHash = (Get-FileHash .\requirements.txt -Algorithm SHA256).Hash.Substring(0,12)
docker build -f Dockerfile.deps --build-arg REQUIREMENTS_SHA=$reqHash -t new_project/py311-deps:$reqHash .

# 2) 使用此依賴映像當作基底，並跳過再次安裝依賴
docker build --build-arg BASE_IMAGE=new_project/py311-deps:$reqHash --build-arg SKIP_PIP_INSTALL=true -t new_project:latest .
```

說明：
- `Dockerfile.deps` 會把 `requirements.txt` 裝進基底映像；只要需求沒變，這層可以長期重用。
- 主 `Dockerfile` 新增 `BASE_IMAGE` 與 `SKIP_PIP_INSTALL` 參數；設為上述依賴映像 + 跳過安裝，即可極速建置。
- 建議把依賴映像推到你的私有/公有 Registry，團隊成員即可直接重用（例如 `ghcr.io/yourorg/new_project/py311-deps:$reqHash`）。

### 🏷️ 使用 GHCR（GitHub Container Registry）映像

本專案的 CI（GitHub Actions）會自動將映像推到 GHCR：

- 依賴映像（已安裝 requirements）：
  - `ghcr.io/<你的帳號>/new_project/py311-deps:<12位requirements雜湊>`
  - 用途：加速後續 App build（作為 BASE_IMAGE）
- App 映像：
  - 永遠會有：`ghcr.io/<你的帳號>/new_project/app:<git_sha>`
  - 只有在「打 tag」時，才會另推：`ghcr.io/<你的帳號>/new_project/app:latest` 與 `app:<tag>`

注意：第一次用 GHCR，請在 GitHub 帳號 Settings > Packages 啟用 GHCR；若要公開下載，記得把 Package 設為 Public。

拉取與運行（PowerShell）：

```powershell
# 若是公開套件可直接拉，若為私有請先： docker login ghcr.io
# 1) 下載標記為 latest（僅 tag 釋出時更新）
docker pull ghcr.io/112304008-hub/new_project/app:latest

# 2) 或下載特定版本（例如標籤 v1.2.3 或特定 git SHA）
docker pull ghcr.io/112304008-hub/new_project/app:v1.2.3
# 或
docker pull ghcr.io/112304008-hub/new_project/app:<git_sha>

# 3) 執行（聽 8000 埠）
docker run --rm -p 8000:8000 ghcr.io/112304008-hub/new_project/app:latest

# 健康檢查
Invoke-WebRequest -Uri "http://localhost:8000/health"
```

在本機重建 App 但重用 GHCR 依賴層（加速 build）：

```powershell
$reqHash = (Get-FileHash .\requirements.txt -Algorithm SHA256).Hash.Substring(0,12)
docker build --build-arg BASE_IMAGE=ghcr.io/112304008-hub/new_project/py311-deps:$reqHash --build-arg SKIP_PIP_INSTALL=true -t new_project:dev .
```

> :latest 只有在打 tag 時才會更新；平時請使用 `app:<git_sha>` 或 `app:<tag>` 來鎖定版本。

也可以使用腳本一鍵拉依賴並建置（PowerShell）：

```powershell
# 在專案根目錄執行
scripts\build_from_ghcr.ps1 -AppTag dev
# 產生的映像為 new_project:dev
```

Makefile 只是把常用命令封裝成短命令（例如 `make dev` 等同 `uvicorn main:app --reload`）。

- 保留的好處：
  - 跨平台快速啟動與測試（在有 `make` 的環境）。
- 可以刪除嗎？
  - 可以。如果你不會用 `make` 或在 Windows 上不裝 `make`，直接照上面命令操作即可。
