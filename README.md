# 股價短期預測服務

簡介  
本專案是一個以 FastAPI 建置的輕量化股價短期預測系統。系統以歷史交易資料為基礎，提供模型推論服務與互動式前端，可對單一股票產生下一交易日上漲機率與二元標籤（漲 / 跌）。設計目標為可重現、容易整合與便於本地開發測試。

重點功能
- Web 前端：互動式頁面供使用者選擇股票、觸發預測並檢視推論與統計檢定結果。  
- REST API：簡潔的 API（如 /api/draw）供自動化流程或外部系統呼叫。  
- 多模型支援：內建隨機森林（rf）與邏輯迴歸（lr）兩種模型。  
- 門檻最佳化：於驗證集上搜尋最佳分類閾值（以 F1 為準則）。  
- 資料處理：載入 CSV、計算滯後與衍生特徵並執行預測流程。

目錄結構（關鍵）
- main.py — FastAPI 應用程式入口  
- stock.py — 資料前處理、模型訓練與 predict 函式  
- template2.html — 前端頁面（互動 UI 與顯示）  
- check_twelve.py — 若需，透過 Twelve Data 取得或更新歷史資料  
- data/short_term_with_lag3.csv — 預期之歷史資料檔案

快速上手（Windows 範例）
1. 取得原始碼
```bash
git clone <repository-url>
cd new-project
```
# 股價短期預測服務 (new_project)

本專案提供一個以 FastAPI 為基礎的股價短期預測系統，包含資料處理、模型訓練、批次建置與 REST API。設計重點為可在本地快速部署、方便測試與整合自動化批次處理。

主要功能
- Web 前端：`template2.html` 提供簡單互動 UI，可觸發預測並檢視結果。  
- REST API：以 `main.py` 為入口，提供單一/多檔 symbol 建置、啟動背景批次、查詢批次狀態等端點。  
- 批次與自動化：支援以指數（如 S&P500）或逗號分隔的 symbols 字串啟動背景工作（參考 `fetch_sp500_github.py`, `start_first50.py`）。  
- 模型支援：專案中包含訓練/載入與預測流程（目前以隨機森林與簡單分類器為主，實作位於 `stock.py`）。

檔案重點說明
- `main.py` — FastAPI 應用與批次管理（重要符號：`_fetch_index_tickers`, `bulk_build_start`, `bulk_build_status`, `DATA_DIR`, `BULK_TASKS`）。  
- `stock.py` — 資料前處理、模型訓練與 `predict` 相關函式。  
- `template2.html` — 前端頁面範本。  
- `fetch_sp500_github.py`, `start_first50.py`, `start_next50.py` — 取得 S&P500 並啟動批次建置的範例腳本。  
- `run_bulk_task_test.py`, `run_bulk_build.py` — 批次任務啟動與測試用腳本。  
- `data/`、`models/` — CSV 與模型輸出目錄。

快速上手（在開發機）
1) 取得原始碼

```powershell
git clone https://github.com/112304008-hub/new_project.git
cd new-project
```

2) 建議建立虛擬環境並安裝套件（Windows 範例）

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

若沒有 `requirements.txt` 或需手動安裝最小需求：

```powershell
pip install fastapi uvicorn scikit-learn pandas requests
```

3) 啟動 API（開發模式）

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

打開瀏覽器前往 http://localhost:8000

常用 API 範例（請以 `main.py` 的實作為準）
- 啟動批次（伺服器端）：`/api/bulk_build_start`（可傳 index=sp500 或 symbols=逗號分隔字串）。
- 查詢批次狀態：`/api/bulk_build_status?task_id=<id>`。
- 其他：`/api/build_symbol`、`/api/build_symbols`、`/api/draw`（詳細請參見 `main.py`）。

執行範例腳本
- 啟動前 50 檔的批次（本機測試）：

```powershell
python start_first50.py
```

- 從 GitHub raw 取得 S&P500 並啟動下一組：

```powershell
python start_next50.py
python fetch_sp500_github.py
```

部署到另一台電腦 — 環境整理與步驟
以下提供 Windows 與 Linux（Ubuntu）常見部署作法與檢查清單，含 systemd（Linux）與 NSSM（Windows）範例，方便你把服務常駐化。

共通前置
- Python：建議使用 Python 3.10 或更新版本（與 `requirements.txt` 相容）。
- 網路：確認部署主機可以存取第三方資料來源（如 GitHub raw、Yahoo Finance 等）。
- 權限：確保 `data/` 與 `models/` 目錄有適當的讀寫權限。

Linux (Ubuntu) 建議步驟
1. 安裝系統套件並建立虛擬環境

```bash
sudo apt update; sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 測試啟動

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. 建立 systemd 服務單元（範例：`/etc/systemd/system/newproject.service`）

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

啟用並啟動：

```bash
sudo systemctl daemon-reload
sudo systemctl enable newproject
sudo systemctl start newproject
sudo systemctl status newproject
```

4. （可選）設定 Nginx 反向代理與 TLS，或使用 Cloud 反向代理

Windows 建議步驟
1. 建立虛擬環境並安裝套件（PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 測試啟動

```powershell
.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. 將 uvicorn 註冊為 Windows Service（選項：NSSM）
- 下載 NSSM（https://nssm.cc/）並將 `uvicorn` 指向 Python venv 的可執行檔。
- 範例：
	- Path: C:\path\to\new-project\.venv\Scripts\python.exe
	- Arguments: -m uvicorn main:app --host 0.0.0.0 --port 8000
	- Start directory: C:\path\to\new-project

4. 也可用 Task Scheduler 或建立 PowerShell 腳本搭配 NSSM / 服務註冊。

資料與權限
- 確認 `data/`、`models/` 可寫。應用程式會將輸出寫入 `data/` 與 `models/`（請參考 `main.py` 中 `DATA_DIR` 常數）。

測試與驗收清單
- Python 與套件安裝完成（pip list 或 pip freeze）。
- Uvicorn 成功啟動並回應根目錄（curl http://localhost:8000/ 或在 Windows 用 Invoke-WebRequest）。
- 執行範例批次：`python start_first50.py` 並觀察是否回傳 task id 與狀態。若有 `run_bulk_task_test.py`，可用它做完整輪詢測試。

進階建議（可選）
- 容器化：若需要更穩定的部署，建議用 Docker 並用 docker-compose 或 Kubernetes 管理。  
- 日誌：串接 systemd 日誌（Linux）或 Windows 事件查看器，並定期清理 `data/` 與 `models/` 的舊檔。  
- 安全：若暴露 API 到公網，務必加入基礎認證或反向代理層的限制與 TLS。

我已經更新 README。接下來我會：
1. 把 todo id=2 標示為已完成，並把 id=3 標示為 in-progress，檢查檔案寫入成功後回報簡短摘要與下一步建議。

授權
本專案採 MIT 授權。請參閱 `LICENSE`（若有）以取得詳細條款。
