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
- **定時更新**：可設定股票自動更新週期
- **指數追蹤**：自動追蹤指數成分變化
- **失敗重試**：智慧型指數退避策略

### 🔍 監控與診斷
- **健康檢查**：容器健康狀態監控
- **Prometheus 指標**：完整的效能指標
- **診斷工具**：資料統計與模型狀態查詢

---

## 🚀 快速開始

### 方式一：本機開發（推薦新手）

```powershell
# 1. 克隆專案
git clone https://github.com/112304008-hub/new_project.git
cd new-project

# 2. 建立虛擬環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 啟動服務
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

開啟瀏覽器：http://localhost:8000

---

### 方式二：Docker 部署（推薦生產環境）

```powershell
# 1. 建置映像（從專案根目錄）
docker compose -f infra/compose/docker-compose.yml build

# 2. 啟動服務（開發）
docker compose -f infra/compose/docker-compose.yml up -d

# 3. 檢查狀態
docker ps
curl http://localhost:8000/health
```

**生產環境（含自動 HTTPS）**：
```powershell
# 設定環境變數（.env 檔案）
echo "DOMAIN=your-domain.com" > .env
echo "ACME_EMAIL=your@email.com" >> .env

# 啟動（自動取得 Let's Encrypt 憑證）
docker compose -f infra/compose/docker-compose.prod.yml up -d
```

---

## 📂 專案結構

> 更完整腳本/端點與測試覆蓋摘要請見 `docs/SUMMARY.md`。

### 目錄樹
```
new-project/
├── 📄 README.md
├── 📁 docs/
│   ├── 01_架構概覽.md
│   ├── 02_資料模型.md
│   ├── 03_業務規則.md
│   ├── 04_術語詞彙.md
│   ├── 05_開發規範.md
│   └── 06_常見問題.md
├── 📁 scripts/               # 開發、批次與工具腳本
│   ├── Build-And-Run-Prod.ps1
│   ├── Setup-Env.ps1
│   ├── Run-DDNS.ps1
│   ├── batch/
│   │   ├── fetch_sp500_github.py
│   │   ├── fetch_tech_and_start.py
│   │   ├── start_first50.py
│   │   ├── start_next50.py
│   │   ├── start_and_monitor_batch3.py
│   │   └── start_and_monitor_batch4.py
│   ├── dev/
│   │   ├── run_api_smoke.py
│   │   ├── run_bulk_build.py
│   │   ├── run_bulk_task_test.py
│   │   ├── run_predict.py
│   │   └── run_test_bulk.py
│   ├── docs/
│   │   └── convert_to_traditional.py
│   └── tools/
│       └── check_twelve.py
├── 🐍 main.py                # FastAPI 應用入口
├── 🐍 stock.py               # 資料處理與模型邏輯
├── 🐍 test.py                # 輕量工具/範例腳本（legacy）
├── 📁 tests/                 # 測試套件
├── 📁 data/                  # 資料 CSV 與 registry
├── 📁 models/                # 訓練好的模型檔案
├── 📁 static/                # 靜態資源
├── 🐳 Dockerfile             # Docker 映像定義
├── 🐳 docker-compose.yml     # 開發環境配置
├── 🐳 docker-compose.prod.yml  # 生產環境配置（Caddy + HTTPS）
├── 🐳 docker-compose.override.yml  # 本機疊加（可選）
└── 📦 requirements.txt       # Python 依賴
```

### 腳本快速索引
| 類別 | 位置 | 作用 | 典型用法 |
|------|------|------|----------|
| 批次成分股 | scripts/batch/fetch_sp500_github.py | 從 GitHub 抓 S&P500 並建置前 50 | `python -m scripts.batch.fetch_sp500_github` |
| 批次分段 | scripts/batch/start_first50.py | 使用內部方法建置前 50 | `python -m scripts.batch.start_first50` |
| 批次分段 | scripts/batch/start_next50.py | GitHub 清單第 51-100 | `python -m scripts.batch.start_next50` |
| 批次監控 | scripts/batch/start_and_monitor_batch3.py | 101-150 建置 + 監控 | `python -m scripts.batch.start_and_monitor_batch3` |
| 批次監控 | scripts/batch/start_and_monitor_batch4.py | 151-200 建置 + 監控 | `python -m scripts.batch.start_and_monitor_batch4` |
| 開發冒煙 | scripts/dev/run_api_smoke.py | 呼叫函式層快速檢查 | `python -m scripts.dev.run_api_smoke` |
| 預測測試 | scripts/dev/run_predict.py | 測試單一 symbol 推論 | `python -m scripts.dev.run_predict -s AAPL` |
| 多檔建置 | scripts/dev/run_bulk_build.py | 建置多 symbols + 列表 | `python -m scripts.dev.run_bulk_build` |
| Bulk 端點 | scripts/dev/run_test_bulk.py | 用 TestClient 呼叫 bulk API | `python -m scripts.dev.run_test_bulk` |
| Bulk 輪詢 | scripts/dev/run_bulk_task_test.py | 啟動並輪詢任務 | `python -m scripts.dev.run_bulk_task_test` |
| 任務監控 | scripts/dev/monitor_task2.py | 監視已知 task_id 進度 | 修改常數後執行 |
| TwelveData | scripts/tools/check_twelve.py | 額度與取價測試 | 設 TWELVE_API_KEY 後執行 |
| Docs 轉繁 | scripts/docs/convert_to_traditional.py | docs/ 轉繁體 | `python -m scripts.docs.convert_to_traditional` |
| 動態 DNS | scripts/ddns/ddns_updater.py | DuckDNS/CF 更新 A 記錄 | docker compose ddns 服務 |

### 測試覆蓋摘要
| 測試檔 | 核心驗證 |
|--------|----------|
| test_api.py | 健康 / 基礎預測 / 列表 / API key & rate limit 基本 |
| test_api_extras.py | metrics/version / series / latest_features 邊界 |
| test_error_paths.py | 錯誤回應情境 (缺檔、損毀、批次錯誤) |
| test_index_auto.py | 指數與 existing CSV 自動任務 |
| test_rate_and_metrics.py | Rate limit 與背景任務指標 |
| test_stats_endpoints.py | 統計檢定 / lag 特徵/ series |
| test_tasks_and_safety.py | 批次任務（monkeypatch 加速） |
| conftest.py | 臨時模型/資料 fixture 建置 |

> 若新增端點或背景流程，請同步更新本表與 `docs/SUMMARY.md`。

---

## API 快速參考

### 預測相關

| 端點 | 方法 | 說明 | 範例 |
|------|------|------|------|
| `/api/draw` | GET | 執行預測 | `?model=rf&symbol=AAPL` |
| `/api/diagnostics` | GET | 診斷資訊 | - |
| `/api/latest_features` | GET | 最新特徵 | `?symbol=AAPL` |

### 資料建置

| 端點 | 方法 | 說明 | 範例 |
|------|------|------|------|
| `/api/build_symbol` | GET | 建置單一股票 | `?symbol=AAPL` |
| `/api/build_symbols` | GET | 建置多個股票 | `?symbols=AAPL,MSFT,GOOGL` |
| `/api/bulk_build_start` | GET | 批次建置 | `?index=sp500&concurrency=4` |
| `/api/bulk_build_status` | GET | 查詢批次進度 | `?task_id={uuid}` |
| `/api/list_symbols` | GET | 列出已建置股票 | - |

### 自動更新

| 端點 | 方法 | 說明 | 範例 |
|------|------|------|------|
| `/api/auto/start_symbol` | GET | 啟動單股更新 | `?symbol=AAPL&interval=5` |
| `/api/auto/stop_symbol` | GET | 停止單股更新 | `?symbol=AAPL` |
| `/api/auto/start_index` | GET | 啟動指數更新 | `?index=sp500&interval=5` |
| `/api/auto/stop_index` | GET | 停止指數更新 | `?index=sp500` |
| `/api/auto/list_registry` | GET | 查看自動任務 | - |

### 系統監控

| 端點 | 方法 | 說明 |
|------|------|------|
| `/health` | GET | 健康檢查 |
| `/version` | GET | 版本資訊 |
| `/metrics` | GET | Prometheus 指標 |

**完整 API 文件**：啟動服務後訪問 http://localhost:8000/docs

---

## 📖 使用範例

### 1. 預測單一股票

```powershell
# 使用隨機森林模型預測 AAPL
Invoke-WebRequest -Uri "http://localhost:8000/api/draw?model=rf&symbol=AAPL" | ConvertFrom-Json

# 回應範例
{
  "label": "漲",
  "proba": 0.6523,
  "threshold": 0.55,
  "model": "rf",
  "symbol": "AAPL"
}
```

### 2. 批次建置 S&P 500 成分股

```powershell
# 啟動批次任務（並發度 4）
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/bulk_build_start?index=sp500&concurrency=4" | ConvertFrom-Json
$taskId = $response.task_id

# 查詢進度
Invoke-WebRequest -Uri "http://localhost:8000/api/bulk_build_status?task_id=$taskId" | ConvertFrom-Json
```

### 3. 啟動自動更新

```powershell
# 每 5 分鐘自動更新 AAPL 資料
Invoke-WebRequest -Uri "http://localhost:8000/api/auto/start_symbol?symbol=AAPL&interval=5"

# 啟動 S&P 500 指數自動更新（每 10 分鐘，並發度 6）
Invoke-WebRequest -Uri "http://localhost:8000/api/auto/start_index?index=sp500&interval=10&concurrency=6"
```

---

## 🧪 模型訓練

**注意**：生產環境僅提供推論，訓練需在開發環境執行。

```powershell
# 訓練隨機森林模型
python stock.py --train --model rf

# 同時訓練兩種模型
python stock.py --train --model all
```

訓練完成後會產生：
- `models/rf_pipeline.pkl` - 隨機森林模型管道
- `models/rf_threshold.pkl` - 最佳分類閾值
- `models/lr_pipeline.pkl` - 邏輯回歸模型管道
- `models/lr_threshold.pkl` - 最佳分類閾值

---

## ⚙️ 環境變數配置

| 變數名稱 | 說明 | 預設值 | 範例 |
|---------|------|--------|------|
| `API_KEY` | API 金鑰（可選） | 無 | `your-secret-key` |
| `RATE_LIMIT_PER_MIN` | 每分鐘請求限制 | 120 | 200 |
| `LOG_LEVEL` | 日誌層級 | INFO | DEBUG |
| `DATA_DIR_WRITE` | 資料寫入目錄 | `./data_work` | `/mnt/data` |
| `DOMAIN` | 網域名稱（生產） | - | `api.example.com` |
| `ACME_EMAIL` | Let's Encrypt 郵箱 | - | `admin@example.com` |

**設定方式**：建立 `.env` 檔案
```bash
API_KEY=your-secret-key-here
RATE_LIMIT_PER_MIN=200
LOG_LEVEL=INFO
```

---

## 🛠️ 疑難排解

### 常見問題

**Q: Docker 建置卡在安裝 numpy/scipy？**
```powershell
# 使用國內映像加速
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
```

**Q: 容器健康檢查一直 starting？**
```powershell
# 查看容器日誌
docker logs <container_id>

# 手動測試健康端點
docker exec <container_id> curl http://localhost:8000/health
```

**Q: 模型預測返回 404？**
```
確認 models/ 目錄包含：
- rf_pipeline.pkl
- rf_threshold.pkl
```

**更多問題**：查看 [常見問題文檔](docs/06_常見問題.md)

---

## 📚 完整文檔

本 README 提供快速上手指南。完整技術文檔請參考：

- **[文檔導航](docs/README.md)** - 文檔索引與閱讀路徑
- **[架構概覽](docs/01_架構概覽.md)** - 系統設計與技術選型
- **[資料模型](docs/02_資料模型.md)** - 資料結構與欄位定義
- **[業務規則](docs/03_業務規則.md)** - 業務邏輯與特殊規則
- **[術語詞彙](docs/04_術語詞彙.md)** - 統一術語與編碼規範
- **[開發規範](docs/05_開發規範.md)** - 代碼風格與測試要求
- **[常見問題](docs/06_常見問題.md)** - FAQ 與故障排查

---

## 🧪 測試

```powershell
# 安裝測試依賴
pip install pytest pytest-cov httpx

# 執行所有測試
pytest tests/

# 產生覆蓋率報告
pytest --cov=. --cov-report=html tests/
```

---

## 🤝 貢獻指南

1. Fork 本專案
2. 建立功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

**請遵循**：[開發規範](docs/05_開發規範.md)

---

## 📝 變更日誌

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本歷史。

---

## 📄 授權

本專案採用 MIT 授權 - 查看 [LICENSE](LICENSE) 檔案了解詳情。

---

## 👥 作者

- **開發團隊** - [112304008-hub](https://github.com/112304008-hub)

---

## 🙏 致謝

- [FastAPI](https://fastapi.tiangolo.com/) - 現代化 Python Web 框架
- [scikit-learn](https://scikit-learn.org/) - 機器學習庫
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance 資料源
- [Caddy](https://caddyserver.com/) - 自動 HTTPS 反向代理

---

## 📧 聯絡方式

- **問題回報**：[GitHub Issues](https://github.com/112304008-hub/new_project/issues)
- **功能建議**：[GitHub Discussions](https://github.com/112304008-hub/new_project/discussions)

---

<div align="center">

**⭐ 如果這個專案對你有幫助，請給個星星！**

Made with ❤️ by the development team

</div>

</div>
| `/health` 失敗 | docker logs / 檢查依賴 | 重建或確認套件版本 |
| 模型未準備 | `/health` models_ready=0 | 確認掛載 models/ 與檔名格式 *_pipeline.pkl |
| 預測慢 | 查看 latency histogram | 減少同時 bulk build / 增 cache |
| 指數 loop 休眠時間異常變長 | backoff 生效 (連續失敗) | 查日誌找出失敗根因 |
| 記憶體攀升 | process_resident_memory_bytes | 降低自動更新數量 / 增容器限制 |
| 429 Too Many Requests | 日誌計數 | 調整 RATE_LIMIT_PER_MIN 或導入 API Key 分流 |

---

## 17. Roadmap / 未來方向
- 模型版本管理 (MLflow or manifest)
- 多語系 / 英文 README 分版
- Retrain API（需角色 / scope）
- WebSocket 進度推播
- OpenTelemetry Tracing
- Redis 共享 rate limit / 任務隊列

---

## 18. 授權與支援 (License & Support)
授權：MIT（若新增 `LICENSE` 請同步更新本段）。

支援步驟：
1. `docker logs <container>` 收集錯誤
2. `curl /health` / `curl /version`
3. `curl /metrics | grep app_http_requests_total`
4. 確认 models/ 與 data/ 是否掛載
5. 檔案/環境變數差異清單

回報問題時建議附：FastAPI 版本、git SHA、執行環境（Docker / OS）、錯誤片段。

---

> 本文件已提供開發、部署、維運、監控與安全所需之完整參考。若需英文版或進一步自動化 (CI/CD / Kubernetes) 可再提出需求。

---

## 🌐 生產環境自動 HTTPS（Caddy）
我們在 `docker-compose.prod.yml` 加入了 Caddy 反向代理，會自動透過 Let’s Encrypt 申請與續約 TLS 憑證。

步驟：
1) 建置映像（烤入資料 / 模型）
  - PowerShell
    - `$sha = git rev-parse --short HEAD; $ts = (Get-Date -Format o)`
    - `docker build --build-arg APP_GIT_SHA=$sha --build-arg APP_BUILD_TIME=$ts -t new_project:$sha -t new_project:latest .`
2) DNS：將你的 `DOMAIN`（例如 `app.example.com`）的 A 記錄指向伺服器 Public IP。
3) 建立 `.env`（參考 `.env.example`）
  - `DOMAIN=app.example.com`
  - `ACME_EMAIL=you@example.com`
  - 可選擇加入 `API_KEY` 與其他變數。
4) 啟動：
  - `docker compose -f infra/compose/docker-compose.prod.yml up -d`
  - Caddy 設定檔：`infra/caddy/conf/Caddyfile`
  - Caddy 憑證/設定資料：`infra/caddy/data`, `infra/caddy/config`
5) 驗證：
  - 瀏覽 `https://app.example.com/health`

說明：
- `web` 服務只在內部網路上 `expose: 8000`，公開的 80/443 由 `caddy` 服務對外提供。
- `Caddyfile` 支援用 `{$DOMAIN}` 讀取環境變數，並自動配置 TLS。
- 若你無 Public DNS（純內網），可改用自簽或在 Caddy 中加 `tls internal`（僅供測試）。

---

## 🔄 免費 DNS / 動態 DNS（DDNS）選項
若你沒有自己的網域，或伺服器是動態 IP，建議使用下列其中一種：

- DuckDNS（完全免費）：https://www.duckdns.org/
  - 註冊後建立一個子網域（如 `yourname.duckdns.org`）並取得 Token。
  - 在 `.env` 設定：
    - `DDNS_PROVIDER=duckdns`
    - `DUCKDNS_DOMAIN=yourname`
    - `DUCKDNS_TOKEN=<your-token>`
  - 啟動生產 compose 後，`ddns` 服務會每 5 分鐘自動把 A 記錄更新成目前伺服器的 Public IP。

- Cloudflare（需要你擁有網域）：https://dash.cloudflare.com/
  - 把你的網域託管到 Cloudflare，建立 API Token（權限：Zone.DNS Edit）。
  - 在 `.env` 設定：
    - `DDNS_PROVIDER=cloudflare`
    - `CLOUDFLARE_API_TOKEN=<token-with-dns-edit>`
    - `CF_ZONE_NAME=example.com`
    - `CF_RECORD_NAME=app.example.com`
  - `ddns` 服務會自動建立 / 更新 A 記錄，TTL=60 秒，不會開啟橘色雲（proxied=false）。

啟用步驟：
1) 填好 `.env` 的 DDNS 相關變數（見 `.env.example`）。
2) 啟動：
  - `docker compose -f infra/compose/docker-compose.prod.yml up -d`
3) 驗證：
   - `nslookup <你的網域>` 應回到你的伺服器 Public IP。
   - 等待 DNS 解析生效後，Caddy 就會自動簽發 HTTPS 憑證。

備註：若使用 Cloudflare，請將 A 記錄暫時關閉 Proxy（灰色雲），以便 Let’s Encrypt HTTP-01 驗證。待簽發完成再視需要開啟。

---

## 🧰 Windows 快速開發環境

專案提供 PowerShell 腳本協助建立/更新虛擬環境與依賴：

```powershell
# 在專案根目錄執行（建立/更新 .venv 並安裝 requirements）
pwsh -File .\scripts\Setup-Env.ps1

# 重新建立虛擬環境（可選）
pwsh -File .\scripts\Setup-Env.ps1 -Reinstall

# 啟用虛擬環境（目前 Shell）
. .\.venv\Scripts\Activate.ps1

# 啟動開發伺服器
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

如果遇到腳本執行權限限制，可執行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## 🧪 常用工具腳本（模組方式）

專案已將零散腳本整併到 `scripts/` 目錄，建議用 `python -m` 從專案根目錄執行：

```powershell
# 單股預測（以 AAPL 為例；支援 --model rf|lr）
python -m scripts.dev.run_predict --symbol AAPL --model rf

# API 冒煙測試
python -m scripts.dev.run_api_smoke

# 啟動 S&P 500 批次建置（第一批）
python -m scripts.batch.start_first50

# 抓取 GitHub 上的 S&P 500 清單並啟動批次
python -m scripts.batch.fetch_sp500_github
```

也可直接以檔案路徑執行，例如：`python .\scripts\dev\run_predict.py`。

---

## 🈶️ 簡體轉繁體工具（docs/）

使用 `scripts/docs/convert_to_traditional.py` 遞迴將 `docs/` 下的所有可讀文字檔與其名稱轉為繁體（預設使用 OpenCC）。建議使用模組方式執行，確保相對路徑正確：

```powershell
# 從專案根目錄執行
python -m scripts.docs.convert_to_traditional
```

轉換結果會列在終端輸出，若發現名稱衝突，腳本會在新名稱後加上 `_trad` 以避免覆蓋。

若你偏好直接執行檔案，也可：
```powershell
python .\scripts\docs\convert_to_traditional.py
```

兩種方式等效。

---

## 🌐 一次性更新 DDNS（固定 IP 模式）
## 🛠️ Makefile 與 Windows 快捷腳本

已新增 `Makefile` 與 `scripts/win/dev_shortcuts.ps1`：

```powershell
# Unix / WSL
make install
make dev
make train-all

# Windows PowerShell (dot-source 以載入函式)
. .\scripts\win\dev_shortcuts.ps1
Start-Dev
Train-All
Bulk-SP500
```

若缺少 make，可直接閱讀 Makefile 對應命令複製執行。

## 🧪 DDNS 本地測試建議

DuckDNS 測試（oneshot）：
```powershell
$env:DDNS_PROVIDER='duckdns'
$env:DUCKDNS_DOMAIN='yourdomain'
$env:DUCKDNS_TOKEN='token'
$env:DDNS_ONESHOT='true'
python -m scripts.ddns.ddns_updater
```

Cloudflare 測試：
```powershell
$env:DDNS_PROVIDER='cloudflare'
$env:CLOUDFLARE_API_TOKEN='cf_api_token'
$env:CF_ZONE_NAME='example.com'
$env:CF_RECORD_NAME='ddns.example.com'
$env:DDNS_ONESHOT='true'
python -m scripts.ddns.ddns_updater
```

看到 `ddns: updated <provider> record to <IP>` 即代表成功。

使用 PowerShell 腳本載入 `.env` 後執行 DDNS 更新（支援 DDNS_STATIC_IP + DDNS_ONESHOT）：

```powershell
pwsh -File .\scripts\Run-DDNS.ps1
```

請先在 `.env` 設定：

```
DDNS_PROVIDER=duckdns
DUCKDNS_DOMAIN=<你的子網域>
DUCKDNS_TOKEN=<你的 Token>
DDNS_STATIC_IP=<你的固定公網IP>
DDNS_ONESHOT=true
```
