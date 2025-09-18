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

2. 建議建立虛擬環境並安裝套件
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
```
備註：若沒有 requirements.txt，可安裝基本套件：
```powershell
pip install fastapi uvicorn scikit-learn pandas
```

3. 準備資料  
- 將歷史資料放在 data/short_term_with_lag3.csv（欄位格式請參照專案中 stock.py 所需欄位）。  
- 若無現成資料，可使用 check_twelve.py 並提供 Twelve Data API key 以抓取。

4. 啟動服務
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
打開瀏覽器前往 http://localhost:8000

API 使用範例
- 產生一次預測（GET）
```bash
curl "http://localhost:8000/api/draw?model=rf"
```
參數：
- model: "rf"（隨機森林）或 "lr"（邏輯迴歸），預設 "rf"

CLI 使用（快速測試）
```bash
python stock.py --model rf --csv data/short_term_with_lag3.csv
```
常用選項：--model, --csv

系統流程概述
1. 資料前處理：讀取 CSV、填補缺值、產生滯後與衍生特徵。  
2. 模型訓練/載入：訓練或載入預先儲存之模型（視程式實作）。  
3. 門檻搜尋：在驗證資料上搜尋最佳分類閾值以最大化 F1。  
4. 預測回傳：輸出上漲機率、使用門檻與二元標籤；前端同時顯示詳細欄位檢定結果。

開發與測試建議
- 本地開發請使用 uvicorn --reload 以便即時看到變更。  
- 將資料處理與模型邏輯撰寫為可測試的函式，使用 pytest 撰寫單元測試（資料邏輯、閾值搜尋、輸出格式等）。

部署與注意事項
- 建議在生產環境使用容器化（Docker）與反向代理（如 nginx）。  
- 對外服務時請做好 API 金鑰（若使用第三方資料）的保護與日誌控管。  
- 模型結果僅供參考，不構成投資建議；使用者應自行承擔交易風險。

授權
本專案採 MIT 授權。請參閱 LICENSE 檔案以取得授權條款細節。
