# 📈 股價之神 - AI 股票預測系統

基於 FastAPI + 機器學習的股票短期預測服務

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-green.svg)](https://fastapi.tiangolo.com/)

## ✨ 核心功能

- **🎯 股票預測**：支援隨機森林 (RF) 與邏輯回歸 (LR) 模型，預測明日漲跌
- **📊 自動資料建置**：整合 Twelve Data API，自動抓取並生成技術指標特徵
- **⚙️ 全域自動更新**：服務啟動後每 5 分鐘自動更新所有股票資料
- **🔄 批次處理**：支援批次建置多支股票的特徵資料
- **🎨 網頁介面**：提供籤筒風格的預測介面

## 🚀 快速開始

### 1. 安裝依賴

```powershell
# 建立虛擬環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安裝套件
pip install -r requirements.txt
```

### 2. 啟動服務

```powershell
# 開發模式（熱重載）
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 或直接執行
python main.py
```

開啟瀏覽器：http://localhost:8000

### 3. 基本使用

```powershell
# 建置單一股票資料（自動抓取 & 生成特徵）
Invoke-RestMethod -Uri "http://localhost:8000/api/build_symbol?symbol=AAPL"

# 預測（簡潔版）
Invoke-RestMethod -Uri "http://localhost:8000/api/predict?symbol=AAPL&model=rf"

# 預測（完整版，含信心度）
Invoke-RestMethod -Uri "http://localhost:8000/api/draw?symbol=AAPL&model=rf"

# 列出已有資料的股票
Invoke-RestMethod -Uri "http://localhost:8000/api/list_symbols"
```

## 📦 批次建置

使用 `batch_build.py` 可以分批建置多支股票，智能處理 API 限制：

```powershell
# 更新所有現有股票（自動掃描 data/ 目錄）
python batch_build.py

# 指定要建置的股票
python batch_build.py --symbols AAPL,MSFT,NVDA,GOOGL,TSLA

# 續建未完成的股票（跳過已存在的）
python batch_build.py --symbols AAPL,MSFT,NVDA --resume

# 自訂批次大小與等待時間
python batch_build.py --symbols AAPL,MSFT --batch-size 3 --wait-time 90
```

## 🐳 Docker 部署

### ⚙️ 環境設定（重要）

Docker Compose 需要 `.env` 檔案來讀取環境變數。根據 Docker 版本不同：

```powershell
# 新版 Docker Compose v2+：需要在 compose/ 目錄放置 .env
Copy-Item .env compose\.env -Force

# 或者只維護 compose/.env（如果只在新電腦部署）
# 將根目錄 .env 內容複製到 compose/.env
```

詳細說明請參閱：[compose/ENV_SETUP.md](compose/ENV_SETUP.md)

### 方式一：只部署後端（最快）

```powershell
# 建置映像
docker build -t fortune-ticker:latest .

# 執行（使用 8001 埠）
cd infra/compose
docker compose -f docker-compose.prod.yml up -d web

# 健康檢查
Invoke-WebRequest -Uri "http://localhost:8001/health"
```

### 方式二：含 Caddy 反向代理（HTTPS）

```powershell
# 一鍵啟動（從任何目錄）
powershell -File .\infra\compose\run_all.ps1

# 手動設定
$env:DOMAIN = "your-domain.example"
$env:ACME_EMAIL = "you@example.com"
cd infra/compose
docker compose -f docker-compose.prod.yml up -d
```

## 📖 API 文件

### 預測 API

- `GET /api/predict` - 簡潔版預測（只回傳 label + proba）
- `GET /api/draw` - 完整版預測（含 threshold + confidence）

### 資料管理 API

- `GET /api/build_symbol` - 建置單一股票
- `GET /api/build_symbols` - 建置多支股票（逗號分隔）
- `GET /api/list_symbols` - 列出已有資料的股票

### 批次處理 API

- `GET /api/bulk_build_start` - 啟動批次建置
- `GET /api/bulk_build_status` - 查詢批次進度
- `GET /api/bulk_build_stop` - 停止批次任務

### 診斷 API

- `GET /api/diagnostics` - 資料診斷資訊
- `GET /api/stattests` - 統計檢定
- `GET /api/lag_stats` - 滯後特徵分析
- `GET /api/latest_features` - 最新特徵值

### 監控 API

- `GET /health` - 健康檢查
- `GET /version` - 版本資訊
- `GET /api/overview` - 路由總覽

## 🛠️ 技術棧

### 後端
- **FastAPI 0.95.2** - 現代化非同步 Web 框架
- **Uvicorn 0.22.0** - ASGI 服務器
- **Python 3.11+** - 執行環境

### 資料處理 & 機器學習
- **pandas 2.2.2** - 資料處理
- **numpy 1.26.4** - 數值計算
- **scikit-learn 1.3.2** - 機器學習模型
- **scipy 1.11.1** - 科學計算
- **statsmodels 0.14.0** - 統計建模

### 資料源
- **Twelve Data API** - 股票歷史資料

### 部署
- **Docker** - 容器化
- **Docker Compose** - 多容器編排
- **Caddy** - 自動 HTTPS 反向代理

## 📂 專案結構

```
fortune-ticker/
├── main.py                 # FastAPI 主應用程式
├── stock.py                # 資料處理 & 模型推論
├── batch_build.py          # 批次建置工具
├── template2.html          # 前端網頁介面
├── requirements.txt        # Python 依賴
├── Dockerfile              # Docker 映像定義
├── data/                   # 股票資料（CSV）
├── models/                 # 訓練好的模型
├── static/                 # 靜態資源
├── tests/                  # 單元測試
├── scripts/                # 工具腳本
├── docs/                   # 詳細文件
└── infra/                  # 部署配置
    └── compose/            # Docker Compose 設定
```

## ⚙️ 環境變數

```powershell
# API 保護（可選）
$env:API_KEY = "your-secret-key"

# 速率限制（每分鐘請求數）
$env:RATE_LIMIT_PER_MIN = "120"

# 日誌級別
$env:LOG_LEVEL = "INFO"

# 全域自動更新（預設啟用）
$env:ENABLE_GLOBAL_UPDATER = "true"

# 預測時自動建置缺失的 CSV（預設啟用）
$env:ENABLE_AUTO_BUILD_PREDICT = "true"
```

## 🔍 故障排除

### Docker 容器無法啟動

```powershell
# 查看容器日誌
docker compose -f docker-compose.prod.yml logs -f web

# 清理舊容器
docker ps -a --format "{{.Names}}" | Select-String fortune-ticker | ForEach-Object { docker stop $_.Line 2>$null; docker rm $_.Line 2>$null }
```

### 埠口衝突

```powershell
# 使用其他埠口
docker run --rm -p 8001:8000 --name stock-ai fortune-ticker:latest

# 或查看占用的程式
netstat -ano | Select-String ":8000"
```

### API 速率限制

使用 `batch_build.py` 的 `--wait-time` 參數調整批次間等待時間：

```powershell
python batch_build.py --wait-time 90  # 等待 90 秒
```

## 📚 詳細文件

更多資訊請參閱 `docs/` 目錄：

- [架構概覽](docs/01_架構概覽.md) - 系統架構與設計決策
- [資料模型](docs/02_資料模型.md) - 資料結構與欄位定義
- [業務規則](docs/03_業務規則.md) - 業務邏輯與特殊規則
- [開發規範](docs/05_開發規範.md) - 程式碼風格與開發流程

## 🤝 貢獻

歡迎提交 Issue 與 Pull Request！

## 📄 授權

MIT License
