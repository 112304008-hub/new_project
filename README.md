<div align="center">

# 📈 股價短期預測 / 批次建置服務 (new_project)

以 FastAPI 建構的股票短期特徵建置與預測服務：提供互動式網頁、REST API、批次背景任務、指數（S&P500 / Nasdaq-100 / 台股部分）成分自動擷取，以及模型推論。可本地快速開發，也可用 Docker 部署。

</div>

---

## ✨ 核心功能
| 類型 | 說明 |
|------|------|
| Web 前端 | `template2.html` 提供簡單抽籤 / 預測互動頁面 |
| 預測 API | `/api/draw` 回傳模型推論（機率 + 標籤） |
| 單 / 多股票資料建置 | `/api/build_symbol`, `/api/build_symbols` 建立指定 CSV |
| 指數批次建置 | `/api/bulk_build_start?index=sp500` 等啟動背景任務 |
| 背景任務狀態 | `/api/bulk_build_status?task_id=...` 查詢進度 |
| 自動循環更新 | `/api/auto/start_symbol` 啟動每 X 分鐘更新某股票資料 (支援 backoff) |
| 指數自動更新 | `/api/auto/start_index` 以單一 loop 更新整個指數成分 (含動態合併現有 CSV) |
| 批量啟動既有 CSV | `/api/auto/start_existing_csvs` 為 data/ 下所有現有 CSV 建立 symbol loop |
| 模型與閾值 | `models/` 內存放 `*_pipeline.pkl` 與對應 threshold |
| 診斷資訊 | `/api/diagnostics` 與 `/api/latest_features` 等端點 |
| 健康檢查 | `/health` 提供容器與依賴狀態回報 |

---

## 📂 目錄重點
```
main.py            # FastAPI 入口與所有 API 定義
stock.py           # 資料處理 / 建置 / 預測邏輯
template2.html     # 前端頁面
data/              # 產生的特徵 CSV、registry、*_last_update
models/            # 已訓練模型與 threshold artifacts
Dockerfile         # 精簡化 Python 3.11-slim 基底映像
docker-compose.yml # 啟動服務 (web)；已移除過時 version 欄位
requirements.txt   # 依賴版本（已修正 numpy pin）
```

---

## 🚀 快速開始（本機開發，不使用 Docker）
```powershell
git clone https://github.com/112304008-hub/new_project.git
cd new-project
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
開啟瀏覽器：http://localhost:8000

---

## 🐳 使用 Docker / Compose
先安裝 Docker Desktop (Windows) 或 Docker Engine (Linux)。

建置映像：
```powershell
docker compose build
```
啟動服務：
```powershell
docker compose up -d
```
查看容器：
```powershell
docker ps
```
查看健康檢查 JSON：
```powershell
Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing | Select-Object -ExpandProperty Content
```
（若 STATUS 長時間停留在 `health: starting`，可檢查 Dockerfile HEALTHCHECK 或容器內部日誌。）

停止與移除：
```powershell
docker compose down
```

### Docker 健康檢查說明
目前 HEALTHCHECK 每 30 秒呼叫 `/health`，成功條件為 HTTP 2xx 並且 JSON `status == "ok"`。

---

## 🔍 主要 API 快速參考
| Endpoint | 方法 | 說明 | 主要參數 |
|----------|------|------|----------|
| `/health` | GET | 簡易健康狀態 | - |
| `/api/draw` | GET | 進行單次預測 | `model=rf|lr`, `symbol` (可選) |
| `/api/build_symbol` | GET | 建構單一股票特徵 CSV | `symbol=` |
| `/api/build_symbols` | GET | 多股票批次建構 | `symbols=2330,2317,AAPL` |
| `/api/bulk_build_start` | GET | 啟動指數或自訂列表背景批次 | `index=sp500` 或 `symbols=`、`concurrency=` |
| `/api/bulk_build_status` | GET | 查詢背景任務進度 | `task_id=` |
| `/api/auto/start_symbol` | GET | 啟動某 symbol 週期更新 | `symbol=`, `interval=`(分鐘) |
| `/api/auto/stop_symbol` | GET | 停止週期更新 | `symbol=` |
| `/api/diagnostics` | GET | 回傳最新資料統計、模型清單 | `n_bins` (可選) |
| `/api/latest_features` | GET | 最新一列特徵過濾 | `features` / `pattern` / `symbol` |

> 詳細行為與例外請參閱 `main.py`。

---

## 🧪 預測範例
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/api/draw?model=rf" -UseBasicParsing | Select -Expand Content
```
或指定 symbol：
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/api/draw?model=rf&symbol=AAPL" -UseBasicParsing | Select -Expand Content
```

---

## 🛠️ 常見問題 (FAQ)
| 問題 | 可能原因 | 解法 |
|------|----------|------|
| Docker build 卡在 numpy / scipy | 版本不存在或無法抓取 wheel | 已 pin numpy=1.26.4；確認網路或換用官方 registry |
| 容器 health 一直 starting | HEALTHCHECK Python -c 執行失敗或被殺 | 進容器 `docker logs <container>`；簡化 HEALTHCHECK 腳本 |
| `/api/draw` 回傳模型未準備 | `models/` 缺少 `*_pipeline.pkl` 或 threshold | 確認訓練流程已產出並掛載 `models/` 目錄 |
| `/api/build_symbol` 失敗 | 無法連到 Yahoo Finance 或 symbol 無效 | 測試連線；改用其他 symbol；稍後再試 |
| 批次任務 progress 不動 | 網路取資料慢或遭 rate-limit | 降低 concurrency；分批執行 |

---

## 📊 健康與可觀察性
最輕量的存活檢查：`/health`
更深入：`/api/diagnostics`（含最新資料行、模型清單、特徵統計）。
背景任務監控：輪詢 `/api/bulk_build_status?task_id=...`。

---

## 🧱 部署建議（非容器）
### Linux (systemd)
建立服務單元：
```
[Unit]
Description=NewProject FastAPI
After=network.target

[Service]
User=youruser
WorkingDirectory=/path/to/new-project
Environment="PATH=/path/to/new-project/.venv/bin"
ExecStart=/path/to/new-project/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
<div align="center">

# 📈 股價短期預測 / 批次建置服務 (new_project)

FastAPI 驅動的短期股價特徵建置與推論平台：整合資料抓取、批次建置、自動更新、模型推論、監控與基本安全控制，可快速於本地或雲端部署並長期維運。

</div>

---

## 目錄 (Table of Contents)
1. 概述 (Overview)
2. 核心能力 (Key Features)
3. 系統架構 (Architecture)
4. 目錄結構 (Repository Layout)
5. 快速開始 (Quick Start)
6. 使用 Docker 與部署模式 (Deployment Modes)
7. 組態與環境變數 (Configuration / Env Vars)
8. API 一覽 (API Matrix)
9. 資料與模型生命週期 (Data & Model Lifecycle)
10. 背景任務與排程 (Background Tasks)
11. 觀測性 / 指標 / 日誌 (Observability)
12. 安全與存取控制 (Security)
13. 效能與調校 (Performance Tuning)
14. 部署指引（Systemd / Nginx / 反向代理）
15. 災難復原與備份 (DR & Backup)
16. 疑難排解 (Troubleshooting)
17. Roadmap / 未來方向
18. 授權與支援 (License & Support)

---

## 1. 概述 (Overview)
本專案提供：
- 以歷史資料 + 衍生特徵（包含滯後欄位）進行短期方向機率推論。
- 自動化批次抓取 / 建置（指數成分、指定多檔），並可背景執行與輪詢進度。
- 模型輸出與分類閾值（threshold）載入後僅供推論，避免在佈署容器內做重量級訓練。
- 健康監控、Prometheus 指標、速率限制、可選 API Key、安全最佳實務基礎。

適用場景：量化研究 PoC、內部工具、輕量服務對外試營運。

---

## 2. 核心能力 (Key Features)
| 類別 | 說明 | 成熟度 |
|------|------|--------|
| 預測 API | `/api/draw` 提供機率與標籤 | 穩定 |
| 特徵資料建置 | 單檔 / 多檔 / 指數批次 | 穩定 |
| 背景任務 | Bulk build / 自動更新 Symbol | 穩定 |
| 自動再啟動註冊 | 透過 registry 檔案重啟後恢復 | 基礎 |
| 健康檢查 | `/health` + Docker HEALTHCHECK | 穩定 |
| 觀測性 | `/metrics` Prometheus 指標 + `/version` | 穩定 |
| 安全 | 速率限制 + API Key (可選) | 基礎 |
| 日誌 | key=value 統一格式 | 基礎 |
| 模型管理 | 基於檔案 (pipeline + threshold) | 基礎 |

---

## 3. 系統架構 (Architecture)
邏輯分層：
```
┌────────────────────────────────────┐
│          Client / Browser          │ -> 使用 template2.html 或 API 呼叫
└────────────────────────────────────┘
                │ HTTP
┌────────────────────────────────────┐
│            FastAPI (main.py)       │
│  - 路由與輸入驗證                  │
│  - 中介層：日誌 / 速率限制 / API Key │
│  - 背景任務排程 (async tasks)       │
└────────────────────────────────────┘
                │
┌────────────────────────────────────┐
│        特徵 / 模型工具 (stock.py)  │
│  - 資料抓取 (Yahoo Finance)         │
│  - 特徵建置                         │
│  - 模型載入 / 預測                  │
└────────────────────────────────────┘
                │
┌────────────────────────────────────┐
│  持久化層 (data/, models/, registry) │
│  - CSV / last_update / auto registry │
│  - pipeline.pkl / threshold.pkl      │
└────────────────────────────────────┘
```

---

## 4. 目錄結構 (Repository Layout)
```
main.py               # FastAPI 入口、API、middleware、/metrics /version /health
stock.py              # 資料 / 模型工具與預測函式
template2.html        # 前端頁面 (簡易互動 UI)
data/                 # 資料輸出與狀態檔 (符號 CSV, *_last_update, registry)
models/               # 模型與 threshold artifacts
Dockerfile            # 部署映像：精簡 + 健康檢查
docker-compose.yml    # 開發 / 單機部署服務定義
requirements.txt      # 依賴 (含 prometheus-client)
README.md             # 說明文件（本檔）
```

---

## 5. 快速開始 (Quick Start - Dev w/out Docker)
```powershell
git clone https://github.com/112304008-hub/new_project.git
cd new-project
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
瀏覽器開啟：http://localhost:8000

---

## 6. 部署模式 (Deployment Modes)
| 模式 | 說明 | 適合 | 備註 |
|------|------|------|------|
| 本地直接執行 | venv + uvicorn | 開發 | 變更快速, 無隔離 |
| Docker 開發 (掛載) | `docker-compose.yml` | 開發 / 測試 | 掛載原始碼與 data/models 熱更新 |
| Docker 生產（烤入資料） | `docker-compose.prod.yml` | 交付 / 發佈 | 不掛載；重建 image 才更新資料模型 |
| 多主機（手動） | 手動分發 image | 內部測試 | 可 docker save / load |
| Registry 發佈 | push 到 Docker Hub / ECR | 穩定長期 | 建議加 CI/CD |
| K8s (未內建範例) | Deployment + Service + Ingress | 水平擴展 | 需加共享儲存 & 分布式限流 |

---

## 7. 組態與環境變數 (Configuration)
| 變數 | 功能 | 預設 | 備註 |
|------|------|------|------|
| API_KEY | 啟用 API Key 驗證 `/api/*` | 無 (停用) | Header: x-api-key |
| RATE_LIMIT_PER_MIN | 每 IP 每分鐘請求上限 | 120 | 不含 /health /metrics /version /static / |
| LOG_LEVEL | 日誌層級 | INFO | DEBUG / WARNING / ERROR |
| APP_GIT_SHA | Build 時注入 commit | UNKNOWN | Docker ARG 傳入 |
| APP_BUILD_TIME | Build UTC 時間 | UNKNOWN | Docker ARG 傳入 |

版本資訊建置（PowerShell）：
```powershell
$sha = (git rev-parse --short HEAD)
$ts = (Get-Date -Format o)
docker build --build-arg APP_GIT_SHA=$sha --build-arg APP_BUILD_TIME=$ts -t new-project-web:$sha .
```

---

## 8. API 一覽 (API Matrix)
| Endpoint | Method | 描述 | 重要參數 | 保護 (需 API Key?) |
|----------|--------|------|----------|-------------------|
| `/` | GET | 前端頁面 | - | 否 |
| `/health` | GET | 健康狀態 | - | 否 |
| `/version` | GET | Build / 版本資訊 | - | 否 |
| `/metrics` | GET | Prometheus 指標 | - | 否 |
| `/api/draw` | GET | 單次推論 | model, symbol? | 是 (若啟用) |
| `/api/build_symbol` | GET | 建構單檔 | symbol | 是 |
| `/api/build_symbols` | GET | 建構多檔 | symbols CSV | 是 |
| `/api/bulk_build_start` | GET | 啟動批次 | index / symbols / concurrency | 是 |
| `/api/bulk_build_status` | GET | 批次進度 | task_id | 是 |
| `/api/auto/start_symbol` | GET | 自動更新 symbol | symbol, interval, backoff_factor?, max_backoff? | 是 |
| `/api/auto/start_index` | GET | 指數集中式自動更新 | index, interval, concurrency, backoff_factor?, max_backoff? | 是 |
| `/api/auto/list_index` | GET | 顯示指數 loop registry 與執行中 | - | 是 |
| `/api/auto/start_existing_csvs` | GET | 為現有 CSV 啟動 loops | interval, backoff_factor?, max_backoff? | 是 |
| `/api/auto/stop_symbol` | GET | 停止自動 | symbol | 是 |
| `/api/diagnostics` | GET | 診斷統計 | n_bins | 是 |
| `/api/latest_features` | GET | 最新特徵 | features / pattern / symbol | 是 |

---

## 9. 資料與模型生命週期 (Data & Model Lifecycle)
| 階段 | 動作 | 來源 / 產出 | 說明 |
|------|------|-------------|------|
| 抓取 | Yahoo Finance / Wikipedia | 原始價量 / 指數成分 | 視網路與頻率限制 |
| 特徵建置 | stock.py `_build_from_yfinance` | `*_short_term_with_lag3.csv` | 產生滯後與統計欄位 |
| 模型訓練 (外部) | (不在容器內) | `*_pipeline.pkl` + `*_threshold.pkl` | 推薦離線訓練後掛載 |
| 推論 | `/api/draw` | JSON 結果 | 輕量、無 state |
| 保留 / 清理 | data/ models/ | 老舊 CSV/模型 | 排程清理避免膨脹 |

---

## 10. 背景任務與排程 (Background Tasks)
### 10.1 類型總覽
| 類型 | 說明 | 適用情境 | 優點 | 風險 |
|------|------|----------|------|------|
| Symbol Loop | 為單一股票建立固定週期更新 | 少量重要股票 | 精準控制個別 interval | 大量時產生許多協程開銷 |
| Index Loop | 單一協程批量更新指數成分 + 動態合併現有 CSV | 上百檔 / 全市場 | 降低協程數量；集中節流 | 單點延遲影響整批完成時間 |
| Existing CSV Bootstrap | 掃描 data/ 啟動所有現有 symbol loop | 已先行批次建立大量 CSV | 快速接管既有檔案 | 可能一次啟動過多任務 |

### 10.2 Backoff / Retry 策略
自動更新 loop (symbol / index) 支援指數式後退：
```
sleep = min(max_backoff, interval * backoff_factor ** consecutive_failures)
```
一旦成功執行即重置 `consecutive_failures`。

Default：`backoff_factor=2.0`, `max_backoff=30 (分鐘)`。

啟動範例：
```
/api/auto/start_symbol?symbol=AAPL&interval=5&backoff_factor=2&max_backoff=20
/api/auto/start_index?index=sp500&interval=5&concurrency=6&backoff_factor=1.8&max_backoff=25
```

### 10.3 Registry 格式變更
`auto_registry.json` / `index_auto_registry.json` 內部條目已從：
```
"AAPL": 5
```
演進為：
```
"AAPL": { "interval": 5, "backoff_factor": 2.0, "max_backoff": 30 }
```
啟動時向下相容（偵測為 int 則套用預設 backoff 參數）。

### 10.4 何時選擇哪一種？
| 目標 | 推薦方式 | 備註 |
|------|----------|------|
| 少於 10 檔關鍵股票 | 多個 Symbol Loop | 最簡單直觀 |
| 50~500 檔大量更新 | Index Loop | 降低協程；統一節奏 |
| 已先批次拉資料後希望同步 | Existing CSV + Index Loop | 先 bootstrap，再切換集中管理 |
| 想依個別股票調整頻率 | Symbol Loop | 支援不同 interval |

### 10.5 指數 Loop 動態合併
每輪執行前會掃描 `data/` 內 `*_short_term_with_lag3.csv`，若出現新檔案（例如你離線產生或透過 bulk 建置），自動納入下一輪更新。

### 10.6 常見調校建議
| 症狀 | 調整建議 |
|------|----------|
| Yahoo Finance 頻繁失敗 | 降低 `concurrency` 或提高 `interval`；調整 `backoff_factor` > 2.0 |
| 更新延遲放大 | 檢查是否發生多輪失敗導致 backoff；觀察日誌 | 
| CPU 過高 | 降低 Index Loop `concurrency`；避免同時 Symbol + Index 重疊 |
| 記憶體成長 | 減少同時執行的 Symbol Loop 數量 |

### 10.7 停止與檢視
```
/api/auto/stop_symbol?symbol=AAPL
/api/auto/stop_index?index=sp500
/api/auto/list_index  # 檢視正在運作與 registry 設定
/api/auto/list_registry  # 檢視 symbol registry
```

### 10.8 移轉策略建議 (大量股票)
1. 先執行一次 bulk 建置：`/api/bulk_build_start?index=sp500&concurrency=8`
2. 等待完成 → 啟動 index loop：`/api/auto/start_index?index=sp500&interval=5&concurrency=6`
3. 可選：對極少數高頻 symbol 額外啟動獨立 loop（不同 interval）。
4. 避免：為全部成分同時啟動數百個 symbol loop（管理成本與資源佔用高）。

---

## 11. 觀測性 (Observability)
| 類別 | 工具 / 端點 | 指標 |
|------|-------------|------|
| 健康 | `/health` | status / models_ready / data_ready |
| 版本 | `/version` | git_sha / build_time / package versions |
| 指標 | `/metrics` | app_http_requests_total / _duration_seconds / app_models_ready / app_data_ready / app_background_tasks |
| 日誌 | STDOUT | key=value：req_id / method / path / status / ms |

Prometheus 抓取設定：
```
scrape_configs:
  - job_name: new_project
    static_configs:
      - targets: ['host:8000']
```

告警參考：
| 指標 | 規則 | 意義 |
|------|------|------|
| app_http_request_duration_seconds_bucket | p95 > 1s for 5m | 延遲異常 |
| app_http_requests_total{status=~"5.."} / sum(all) | >2% | 失敗率上升 |
| app_models_ready == 0 | 任何時間 | 模型遺失 |
| app_background_tasks 高於基線 | 持續 10m | 堆積 |
| 指數更新日志多次連續失敗 | 觀察 backoff 日誌訊息 | 網路或來源阻擋 |

---

## 12. 安全與存取控制 (Security)
現有：API Key（header: x-api-key）、速率限制 (in-memory)、基本 logging。
建議進階：
| 面向 | 建議 |
|------|------|
| 傳輸 | 加 TLS (Nginx / Caddy / Cloudflare) |
| 驗證 | 分層 scope / JWT（視未來需求） |
| 授權 | 僅對 retrain / destructive API 加更嚴保護 |
| 限流 | 多副本時改 Redis / external rate limiter |
| 供應鏈 | pip hash / 週期性漏洞掃描 (trivy) |

---

## 13. 效能與調校 (Performance)
| 項目 | 策略 |
|------|------|
| 冷啟動 | 精簡 base image + 預載模型於首次呼叫 |
| 預測延遲 | 模型載入快（joblib pickle），可加簡單 LRU cache |
| I/O | 大量 symbol 時避免同時 burst；控制 concurrency |
| 記憶體 | 監控 RSS 指標（/metrics process_*） |
| 映像大小 | 可改 multi-stage 或使用 `python:3.11-slim` + 移除多餘檔案 |

---

## 14. 部署指引摘要
### Docker Compose (開發模式)
掛載本機目錄（原本 `docker-compose.yml`）：
```bash
docker compose build \
  --build-arg APP_GIT_SHA=$(git rev-parse --short HEAD) \
  --build-arg APP_BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
API_KEY=YourKey RATE_LIMIT_PER_MIN=200 docker compose up -d
```

### 生產模式：烤入資料與模型（無 bind mount）
1. 確保要隨映像提供的 `models/` 與最小 `data/` CSV 已放入專案根目錄。
2. 建置映像：
```bash
docker build \
  --build-arg APP_GIT_SHA=$(git rev-parse --short HEAD) \
  --build-arg APP_BUILD_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t new_project:$(git rev-parse --short HEAD) -t new_project:latest .
```
3. 啟動（不掛載資料）：
```bash
docker compose -f docker-compose.prod.yml up -d
```
4. 驗證：
```bash
curl http://localhost:8000/health
```
5. 更新模型或資料：替換本機檔案 → 重新 build → 重新 up：
```bash
docker compose -f docker-compose.prod.yml down
docker build -t new_project:latest .
docker compose -f docker-compose.prod.yml up -d
```
6. 導出映像（離線交付）：
```bash
docker save new_project:latest -o new_project_latest.tar
```

### Systemd (Linux)
見先前範本；確保 `Environment="API_KEY=YourKey"` 加入。

### Nginx 反向代理
```
server {
  listen 443 ssl;
  server_name example.com;
  location / { proxy_pass http://127.0.0.1:8000; }
}
```

---

## 15. 災難復原與備份 (DR & Backup)
| 資產 | 重要性 | 建議備份頻率 | 備註 |
|------|--------|--------------|------|
| models/ | 高 | 每模型更新 | 可加版本號副檔名；若生產烤入需 rebuild |
| data/*.csv | 中 | 每日 / 產生後 | 可壓縮存放 object storage；若生產烤入需 rebuild |
| auto_registry.json | 中 | 每次變動 | 復原自動任務狀態 |
| baked Docker image | 中 | 每次 build | Tag 含 SHA；可 `docker save` 備份 |

恢復流程：拉回 image → 還原 models/ → 還原 data/ → 啟動 → 驗證 `/health`。

---

## 16. 疑難排解 (Troubleshooting)
| 症狀 | 排查步驟 | 修復 |
|------|----------|------|
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
4. 確認 models/ 與 data/ 是否掛載
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
  - `docker compose -f docker-compose.prod.yml up -d`
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
   - `docker compose -f docker-compose.prod.yml up -d`
3) 驗證：
   - `nslookup <你的網域>` 應回到你的伺服器 Public IP。
   - 等待 DNS 解析生效後，Caddy 就會自動簽發 HTTPS 憑證。

備註：若使用 Cloudflare，請將 A 記錄暫時關閉 Proxy（灰色雲），以便 Let’s Encrypt HTTP-01 驗證。待簽發完成再視需要開啟。
