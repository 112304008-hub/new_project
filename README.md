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
| 自動循環更新 | `/api/auto/start_symbol` 啟動每 X 分鐘更新某股票資料 |
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

```bash
sudo systemctl daemon-reload
sudo systemctl enable newproject
sudo systemctl start newproject
```

### Windows (NSSM) 重點欄位
Path: `C:\path\to\new-project\.venv\Scripts\python.exe`  
Arguments: `-m uvicorn main:app --host 0.0.0.0 --port 8000`  
Start directory: `C:\path\to\new-project`

---

## 🔐 安全 / 上線注意事項
- 若公開：加上反向代理（Nginx / Caddy）+ TLS。
- 加入基本認證或 API key（可在 FastAPI 中加一個 dependency）。
- 限制速率（可用中介層或外部 API Gateway）。
- 排程清理過舊 CSV / log。

---

## 🧩 後續可能增強
- Multi-stage Docker build（壓縮映像體積）
- /metrics (Prometheus) 暴露
- 模型版本管理（e.g. MLflow 或自訂 manifest）
- 前端 UI 加入批次進度輪詢與圖表

---

## 📄 授權
MIT（若新增 LICENSE 檔請同步更新此段）。

---

## 🙋 支援
遇到問題可：
1. 檢查日誌：`docker logs <container>`
2. 驗證健康：`/health`
3. 確認資料：`data/` 內是否有對應 CSV
4. 確認模型：`models/` 內是否有 `*_pipeline.pkl`

---

> 本 README 已整合本地、Docker、健康檢查與常見問題，方便快速上線與維運。
