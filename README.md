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
<div align="center">

# � 股價之神 - 最小化使用指南

只保留核心：FastAPI 服務 (`main.py`) 與業務邏輯 (`stock.py`)。

</div>

---

## 1) 啟動服務（開發模式）

```powershell
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

2. 執行預測（需要 models/ 中已有已訓練模型檔 e.g. rf_pipeline.pkl / rf_threshold.pkl）

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

## 附註

- 若不使用 Makefile，可直接照上述命令操作；Makefile 只是幫你把常用命令取個別名（見下）。
- 本專案已移除批次腳本與多餘的工具腳本；如需批次或自動更新，建議改用 API（/api/build_symbol）自行外掛排程。

---

## Makefile 是什麼？可以刪嗎？

Makefile 只是把常用命令封裝成短命令（例如 `make dev` 等同 `uvicorn main:app --reload`）。

- 保留的好處：
  - 跨平台快速啟動與測試（在有 `make` 的環境）。
- 可以刪除嗎？
  - 可以。如果你不會用 `make` 或在 Windows 上不裝 `make`，直接照上面命令操作即可。
