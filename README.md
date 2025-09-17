# 新專案

這是一個使用 **FastAPI** 建立的應用程式，用於進行股價預測。它提供一個網頁介面，讓使用者與預測模型互動，並根據歷史數據獲取預測結果。

## 功能特色

- **網頁介面**：簡單的網頁入口，作為使用者的主要互動平台。
- **API 端點**：提供 API 介面，讓使用者根據所選模型獲取預測。
- **資料處理**：整合 CSV 檔案，提供即時的預測能力。
- **模型支援**：支援兩種機器學習模型：隨機森林（`rf`）與邏輯迴歸（`lr`）。
- **自訂門檻值**：利用驗證資料動態決定最佳預測門檻值。

## 安裝與設定步驟

1. **下載專案**：
   ```bash
   git clone <repository-url>
   cd new-project
   ```

2. **安裝依賴套件**：
   確保已安裝 Python，然後執行以下指令安裝所需套件：
   ```bash
   pip install fastapi uvicorn scikit-learn pandas
   ```

3. **準備資料**：
   - 確保存在 `data/short_term_with_lag3.csv` 檔案，該檔案包含歷史股價數據，作為訓練與預測的依據。
   - 若缺少該檔案，可以執行 `check_twelve.py` 腳本，透過 Twelve Data API 抓取資料。

4. **啟動應用程式**：
   使用以下指令啟動 FastAPI：
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **存取網頁介面**：
   開啟瀏覽器並前往 `http://localhost:8000` 以進入應用程式。

## 使用方式

### 網頁介面
- 首頁（`/`）提供簡單的使用者互動介面。
- 若 `template2.html` 檔案缺失，系統將回傳 JSON 格式的錯誤訊息。

### API 端點
- **`/api/draw`**：根據所選模型生成預測結果。
  - 查詢參數：
    - `model`: 選擇 `"rf"`（隨機森林）或 `"lr"`（邏輯迴歸），預設為 `"rf"`。
  - 範例：
    ```bash
    curl "http://localhost:8000/api/draw?model=rf"
    ```

### 命令列介面
- 可直接透過命令列執行 `stock.py` 進行預測：
  ```bash
  python stock.py --model rf
  ```
  - 可選參數：
    - `--csv`: 指定 CSV 檔案路徑（預設為 `data/short_term_with_lag3.csv`）。
    - `--model`: 選擇 `"rf"` 或 `"lr"`。

### 預測流程

1. **資料前處理**：
   - `stock.py` 中的 `predict` 函數會讀取 CSV 並進行前處理。
   - 缺失值會用訓練數據的中位數補齊。

2. **模型訓練**：
   - 訓練兩個模型：隨機森林與邏輯迴歸。
   - 資料會拆分成訓練核心集與驗證集，用來決定最佳門檻值。

3. **門檻值最佳化**：
   - `_best_threshold` 函數會根據 F1 分數計算最佳預測門檻。

4. **預測**：
   - 使用 CSV 檔案的最後一列資料作為輸入，預測下一個交易日股價走勢。
   - 預測結果包含：上漲機率、門檻值與最終標籤（"漲" / "跌"）。

## 專案結構

- `main.py`: FastAPI 應用程式的入口。
- `stock.py`: 包含資料前處理、模型訓練與預測邏輯。
- `check_twelve.py`: 與 Twelve Data API 互動的工具腳本。
- `data/short_term_with_lag3.csv`: 訓練與預測用的主要數據檔案。
- `README.md`: 專案文件說明。

## 授權

本專案採用 **MIT 授權條款**。