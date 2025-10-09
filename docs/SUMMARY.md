# 文檔索引 (SUMMARY)

> 本檔案自動補充：列出專案主要腳本、端點與測試覆蓋摘要，便於快速導航。

## 1. 架構 / 模組導覽

| 區塊 | 檔案/目錄 | 說明 |
|------|-----------|------|
| Web API | `main.py` | FastAPI 主應用：預測、資料建置、自動任務、診斷、監控 |
| 模型邏輯 | `stock.py` | 模型訓練 / 推論（rf, lr）與特徵對齊、threshold 儲存 |
| 資料建構 | `test.py` | Symbol 特徵建構 + 外生因子 + 自動更新（legacy builder） |
| Scripts - batch | `scripts/batch/*.py` | 指數/批次抓取與分段建置工具 |
| Scripts - dev | `scripts/dev/*.py` | 開發/冒煙/批次測試輔助腳本 |
| Scripts - tools | `scripts/tools/check_twelve.py` | Twelve Data 額度檢查 |
| Scripts - ddns | `scripts/ddns/ddns_updater.py` | DuckDNS / Cloudflare 動態 DNS 更新 |
| Docs 工具 | `scripts/docs/convert_to_traditional.py` | 將 docs/ 轉繁體 (OpenCC) |
| 模型輸出 | `models/` | *_pipeline.pkl / *_threshold.pkl |
| 資料 | `data/` | *_short_term_with_lag3.csv + registry JSON |

## 2. 主要 HTTP 端點分類

### 預測 / 資料
- `GET /api/draw`：載入模型 + 最新特徵，回傳籤筒格式預測
- `GET /api/build_symbol` / `build_symbols`：即時建構特徵 CSV
- `GET /api/list_symbols`：列出現有 symbol CSV

### 批次 / 背景
- `GET /api/bulk_build_start` / `bulk_build_status` / `bulk_build_stop`
- 背景狀態儲存在記憶體結構 `BULK_TASKS`

### 自動更新 (Individual / Index)
- `GET /api/auto/start_symbol` / `stop_symbol` / `start_many` / `stop_many` / `start_existing_csvs`
- `GET /api/auto/start_index` / `stop_index` / `list_index`

### 統計 / 特徵
- `GET /api/diagnostics`：最新列、統計摘要、histograms、模型列表
- `GET /api/stattests`：單一數值欄位 Shapiro / t-test / ADF
- `GET /api/lag_stats`：所有 lag* 特徵的檢定集合
- `GET /api/series`：取某欄位最近 N 筆
- `GET /api/latest_features`：最新列 + 欄位選擇或 regex 篩選

### 監控 / 健康
- `GET /health`：輕量健康檢查（不重建/不重 IO）
- `GET /metrics`：Prometheus 指標 (含 HTTP latency, 任務計數)
- `GET /version`：編譯 / Git SHA / 依賴版本

## 3. Prometheus 自訂指標
| 指標 | 標籤 | 說明 |
|------|------|------|
| `app_http_requests_total` | method, path, status | HTTP 請求計數 |
| `app_http_request_duration_seconds` | method, path | HTTP 延遲分佈 |
| `app_background_tasks` | (none) | 背景任務併發數 |
| `app_models_ready` | (none) | 是否存在任何模型檔 |
| `app_data_ready` | (none) | 主要 CSV 可用性 |

## 4. 背景任務資料結構
```python
BULK_TASKS[task_id] = {
  'status': 'running|completed|cancelled',
  'total': int,
  'done': int,
  'errors': {symbol: message},
  'started_at': epoch_ts,
  'finished_at': epoch_ts?,
  'task': asyncio.Task
}
SYMBOL_TASKS[symbol] = asyncio.Task
INDEX_TASKS[index] = asyncio.Task
```

## 5. 測試覆蓋摘要
| 測試檔 | 覆蓋範圍 |
|--------|----------|
| `test_api.py` | 核心健康 / 預測 / 列表 / API Key & rate limit 基礎 |
| `test_api_extras.py` | metrics/version, series, latest_features, build_symbol 邊界 |
| `test_error_paths.py` | 各種錯誤與缺失資源處理 |
| `test_index_auto.py` | 指數自動任務啟停與 existing CSV loop |
| `test_rate_and_metrics.py` | Rate limit 行為 & 背景指標出現 |
| `test_stats_endpoints.py` | stattests / lag_stats / series 功能 |
| `test_tasks_and_safety.py` | bulk 建置流程（monkeypatch 加速） |
| `conftest.py` | 臨時環境 / 模型準備 fixtures |

## 6. 常見維運排查對照
| 問題 | 指標 / 行為 | 行動 |
|------|-------------|------|
| 模型不可用 | `app_models_ready=0` | 確認 models/ 掛載與檔名 *_pipeline.pkl |
| 資料空或缺 | `app_data_ready=0` | 觸發 /api/build_symbol 或檢查自動任務 |
| 預測錯誤增多 | HTTP 500 計數偏高 | 查看 log + 確認來源 CSV schema |
| 任務爆量 | `app_background_tasks` 高 | 降低 concurrency / 合併使用 index loop |
| 延遲升高 | latency histogram 漏斗右偏 | 檢查 I/O / 降低批次同時觸發 |

## 7. 文件維護指南
- 新增端點：同步更新 README 快速參考 + 此 SUMMARY 端點列表。
- 新增指標：更新 Prometheus 指標表以利 SRE 設定警報。
- 新增背景流程：補充『背景任務資料結構』段落確保欄位一致。

## 8. 待辦 / Roadmap 對照 (摘自 README)
- 模型版本管理 (MLflow / manifest)
- Retrain API with RBAC
- WebSocket 任務進度推播
- OpenTelemetry Tracing
- Redis 分散式 rate limit / 任務佇列

---
本檔為生成基礎索引，可視需要新增章節（例如：資料欄位字典、錯誤碼對照）。
