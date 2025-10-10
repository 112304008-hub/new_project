# 開發人員指南（Developers Only）

本文件給團隊開發者使用，說明系統架構、資料與模型契約、背景任務、監控與在地開發流程。使用者文件請看 `README.md`（精簡版）。

## 系統架構概覽

- Framework: FastAPI（`main.py`）
- Domain Logic: `stock.py`
  - 預測：讀取已訓練模型（`models/*_pipeline.pkl`, `*_threshold.pkl`）
  - 資料建置：必要時透過 yfinance 下載歷史價量並產出特徵 CSV
- 資料目錄：
  - 讀取：`data/`
  - 寫入：環境變數 `DATA_DIR_WRITE` 指向的資料夾，未設定時回退 `data_work/`
  - 自動任務註冊檔：`auto_registry.json`、`index_auto_registry.json` 儲存在可寫目錄
- 指標/健康：
  - `/metrics`（Prometheus 格式）
  - `/health`（輕量健康檢查）
  - `/version`（版本資訊）
- 前端頁面：`template2.html`（首頁 `/` 若不存在則 404 JSON）

## 目錄速覽

- `main.py`：FastAPI 服務、端點、背景任務、監控
- `stock.py`：特徵建置與推論核心
- `data/`：預設資料（唯讀情境下不寫）
- `data_work/`：預設可寫資料夾（若未設 `DATA_DIR_WRITE`）
- `models/`：已訓練模型（必須自行放置）
- `static/`：靜態檔案（由 `/static` 提供）
- `tests/`：測試（若有）

## 在地開發

- 需求（`requirements.txt` 安裝）：
  - FastAPI, Uvicorn, pandas, numpy, scipy, statsmodels, prometheus-client, joblib, requests 等

- 啟動（熱重載）：
  ```powershell
  python -m uvicorn main:app --reload --port 8000
  # 或直接
  python .\main.py
  ```

- 重要環境變數：
  - `API_KEY`（可選）：若設定，所有 `/api/*` 端點需於 header 帶 `x-api-key`
  - `RATE_LIMIT_PER_MIN`（預設 120）：每 IP 每分鐘速率限制（首頁、/health、/metrics、/version、/static 免限）
  - `DATA_DIR_WRITE`：指定可寫資料夾；未設時回退 `data_work/`
  - `LOG_LEVEL`：INFO/DEBUG...
  - `APP_GIT_SHA`, `APP_BUILD_TIME`：CI/CD 注入版本資訊

- 快速導覽 API 與背景任務：
  - 瀏覽 `http://localhost:8000/api/overview` 取得所有路由摘要與任務統計

## API 重點（對開發者）

- 抽籤/預測：`GET /api/draw?model=rf&symbol=AAPL`
  - 僅提供推論，必須先放置 `models/{model}_pipeline.pkl` 與 `{model}_threshold.pkl`
  - 若未傳 `symbol`，使用主 CSV：`data/short_term_with_lag3.csv`
- 建置特徵：
  - 單檔：`GET /api/build_symbol?symbol=AAPL`
  - 多檔：`GET /api/build_symbols?symbols=AAPL,MSFT,2330`
  - 列表：`GET /api/list_symbols`
- 批次建置：
  - 啟動：`GET /api/bulk_build_start?index=sp500` 或 `?symbols=...&concurrency=4`
  - 查詢：`GET /api/bulk_build_status?task_id=...`
  - 停止：`GET /api/bulk_build_stop?task_id=...`
- 自動任務（個股）：
  - 啟動：`GET /api/auto/start_symbol?symbol=AAPL&interval=5`
  - 停止：`GET /api/auto/stop_symbol?symbol=AAPL`
  - 批次啟動已存在 CSV：`GET /api/auto/start_existing_csvs`
  - 以列表啟動：`GET /api/auto/start_many?symbols=...`
  - 多檔停止：`GET /api/auto/stop_many?symbols=...`
  - 查看註冊檔：`GET /api/auto/list_registry`
- 自動任務（指數）：
  - 啟動：`GET /api/auto/start_index?index=sp500&interval=5&concurrency=4`
  - 停止：`GET /api/auto/stop_index?index=sp500`
  - 概況：`GET /api/auto/list_index`
- 診斷與統計：
  - `/api/diagnostics`, `/api/stattests`, `/api/lag_stats`, `/api/series`, `/api/latest_features`

## 背景任務與持久化

- 個股自動任務：`SYMBOL_TASKS: Dict[str, asyncio.Task]`
- 指數自動任務：`INDEX_TASKS: Dict[str, asyncio.Task]`
- 批次建置任務：`BULK_TASKS: Dict[str, Dict]`
- 啟動還原（rehydrate）：
  - `@app.on_event('startup')` 會讀取 `auto_registry.json` 與 `index_auto_registry.json` 重啟任務
- Backoff 策略：
  - 個股與指數迴圈遇連續失敗會以 `interval * backoff_factor^n` 延後，並上限到 `max_backoff`

## 資料與模型契約

- Symbol CSV 路徑慣例：`{symbol}_short_term_with_lag3.csv`
  - 讀取優先順序：`DATA_DIR_WRITE` > `data/`
  - 寫入一律於 `DATA_DIR_WRITE`（不存在則建立）
  - 伴隨會寫入 `{symbol}_last_update.txt`（ISO 時戳）
- 主 CSV：`data/short_term_with_lag3.csv`
- 模型檔：
  - `models/{model}_pipeline.pkl`
  - `models/{model}_threshold.pkl`
- 推論特徵對齊：
  - 若 pipeline/step 有 `feature_names_in_` 會依據此順序補齊缺失欄位（以 0.0 或中位數填補）

## 監控與日誌

- Prometheus 指標：`/metrics`
  - HTTP 計數：`app_http_requests_total`
  - 延遲分佈：`app_http_request_duration_seconds`
  - 背景任務：`app_background_tasks`
  - 模型/資料就緒：`app_models_ready`, `app_data_ready`
- 健康檢查：`/health`
- 版本：`/version`
- 日誌格式：`ts=<time> level=<level> msg=<msg> module=<module>`（可用 `LOG_LEVEL` 控制）

## 安全與限流

- 若設 `API_KEY`：所有 `/api/*` 端點需要 header `x-api-key: <token>`
- 記憶體級簡單限流：每 IP 每分鐘 `RATE_LIMIT_PER_MIN`，首頁/健康/指標/版本/靜態免限

## 測試與除錯

- 單元測試（若有）：
  ```powershell
  pytest -q
  ```
- 快速驗證預測（開發腳本，若存在）：
  ```powershell
  python .\scripts\dev\run_predict.py --model rf --symbol AAPL
  ```
- 常見錯誤：
  - 404 模型未準備：尚未放置 `*_pipeline.pkl`/`*_threshold.pkl`
  - 404 主 CSV 不存在：請先建立資料（或以 `build_symbol` 生成 symbol CSV 再於 `/api/draw?symbol=` 使用）
  - 429 限流：調高 `RATE_LIMIT_PER_MIN` 或改用 `/api/overview` 檢視路由後再測試

## 開發規範

- 說明統一中文（docstring 第一行應為簡短摘要）
- 新增端點：
  - 撰寫 docstring、必要的參數驗證、清楚的錯誤訊息
  - 如會觸發長作業，採背景任務並提供狀態查詢端點
  - 視需要更新 Prometheus 指標
- 調整資料 schema 或模型：
  - 同步更新 `_update_gauges` 與 `/health` 的就緒判斷
  - 變更 CSV 欄位時，確認 `stock.py` 與推論對齊邏輯

## 部署備註

- 唯讀根檔系統：將 `DATA_DIR_WRITE` 指向可寫 volume；自動註冊檔亦會寫入該處
- 指數迴圈 vs 個股迴圈：避免大量重疊（負載飆升）
- 監控：抓 `/metrics`；必要時新增自定義指標

---
本文件只面向開發者；使用者操作請參考 `README.md`。