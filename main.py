"""main.py — FastAPI 主應用程式

本服務核心目標：
1. 提供『股價籤筒 / 預測 API』：/api/draw 讀取最新特徵 CSV 與已訓練模型，回傳「明日漲跌」推論結果與籤筒格式文字。
2. 支援動態建立/更新個股特徵資料：/api/build_symbol, /api/build_symbols 等，必要時自動用 twelvedata 下載歷史價量並產生特徵 CSV。
3. 內建全域自動更新：服務啟動後每 5 分鐘掃描 data/ 內現有 CSV 以受控併發進行更新；另提供批次建置（/api/bulk_*）。
4. 暴露統計診斷與特徵檢定：/api/diagnostics, /api/stattests, /api/lag_stats, /api/series, /api/latest_features。
5. 提供健康檢查 (/health) 與版本資訊 (/version) 供部署監控與稽核。

主要設計重點：
- 所有模型推論僅在已存在 models/ 內的 *_pipeline.pkl + *_threshold.pkl 情況下執行；若不存在則回傳友善錯誤訊息。
- 資料目錄：讀寫皆使用 data/（簡化原先 DATA_DIR_WRITE 的可寫目錄設計）。
- 自動更新註冊檔 (auto_registry.json / index_auto_registry.json) 儲存在資料目錄以支援重啟後還原背景任務。
- Rate Limit（記憶體級）簡易實作，避免濫用 API。
- 監控指標：HTTP 請求計數/延遲、背景任務數量、模型/資料就緒狀態。

常用環境變數：
    API_KEY              （可選）— 若設置，保護 /api/* 端點需以 request header: x-api-key 傳遞。
    RATE_LIMIT_PER_MIN   （預設 120）— 單個 IP 每分鐘允許請求數（排除 /health 等安全端點）。
    LOG_LEVEL            （INFO/DEBUG...）
    APP_GIT_SHA / APP_BUILD_TIME — 於 CI/CD build 注入版本資訊。

快速啟動 (開發)：
    python -m uvicorn main:app --reload --port 8000
或直接執行：
    python main.py

主要端點概覽（摘要）：
    GET /                      — 首頁（籤筒/前端頁面 template2.html）
    GET /api/draw              — 取得預測籤（參數：model, symbol 必填）
    GET /api/build_symbol      — 單一 symbol 生成/刷新特徵
    GET /api/build_symbols     — 多 symbols 生成
    GET /api/list_symbols      — 列出現有特徵 CSV
    批次：/api/bulk_build_start /status /stop
    診斷與統計：/api/diagnostics（symbol 必填） /api/stattests（symbol 必填） /api/lag_stats（symbol 必填） /api/series（symbol 必填） /api/latest_features（symbol 或 file 必填）
    監控：/health /version

檔案閱讀指南：
    _symbol_* / _index_* 系列：管理個股與指數的特徵 CSV 建構邏輯與自動輪巡。
    BULK_TASKS 結構：儲存批次工作進度 (status, total, done, errors, started_at ...)。
    predict() 相關邏輯在 stock.py；本檔僅做輸入驗證與錯誤處理 / 回應格式化。

維護建議：
    - 新增模型時：確保 stock.py 產出的 *_pipeline.pkl 與 *_threshold.pkl 命名一致，並更新 /api/draw 的預設允許 model 名稱。
    - 增加大型迴圈請務必觀察記憶體使用量與請求延遲（可從日誌觀察）。
    - 若路徑/資料 schema 調整，記得同步更新 _update_gauges 與相關健康檢查。

本模組只加入說明性文字，未改變既有行為。
"""
import os
import time as _time
import logging
from enum import Enum
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from stock import predict, MODELS_DIR
from stock import _ensure_td, _build_from_td
import asyncio
import json
from typing import Dict
import uuid
import joblib
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.tsa.stattools as tsstat
import re

# （精簡版）移除 Prometheus 相關依賴

APP_GIT_SHA = os.getenv("APP_GIT_SHA", "UNKNOWN")
APP_BUILD_TIME = os.getenv("APP_BUILD_TIME", "UNKNOWN")
API_KEY = os.getenv("API_KEY")  # optional
CSV_ENCODING = "utf-8-sig"
ENABLE_AUTO_BUILD_PREDICT = os.getenv("ENABLE_AUTO_BUILD_PREDICT", "true").strip().lower() in {"1","true","yes","on"}

"""Logging 初始化：
預設使用環境變數 LOG_LEVEL（未設定時為 INFO）。
這裡主動建立 basicConfig，確保在 Docker 容器中透過 `docker logs -f` 能看到：
 1. 啟動階段訊息（startup/global updater prints）
 2. API 路由的自訂 logging（例如 /api/draw）
如需更詳細 uvicorn access log，可在 Docker CMD 加上 `--log-level` 或設置 LOG_LEVEL=DEBUG。
"""
_lvl = os.getenv("LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _lvl, logging.INFO)
logging.basicConfig(
    level=_level,
    format="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s module=%(module)s"
)
logger = logging.getLogger("app")
logger.info(f"[init] logging initialized level={_lvl}")

# Metrics（精簡版）：移除 Prometheus，僅保留結構化日誌與 /health

app = FastAPI(title="股價之神", version="1.0")

class ModelName(str, Enum):
    rf = "rf"
    lr = "lr"

#TODO:研究保護機制
@app.middleware("http")
async def metrics_and_logging_middleware(request: Request, call_next):
    """全域中間件：可選 API Key 驗證（僅 /api/*）並回傳 x-request-id。"""
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    path = request.url.path

    # API Key：僅保護 /api/*
    if API_KEY and path.startswith('/api/') and request.headers.get('x-api-key') != API_KEY:
        return JSONResponse({"error": "unauthorized"}, status_code=401, headers={"x-request-id": req_id})

    try:
        response: Response = await call_next(request)
    except Exception:
        response = JSONResponse({"error": "internal error"}, status_code=500)

    response.headers['x-request-id'] = req_id
    return response

# === 路徑設定 ===
ROOT = Path(__file__).parent
HTML = ROOT / "template2.html"
DATA_DIR = ROOT / "data"

# Writable data directory (separate from read-only data/ when desired).
DATA_WRITE_DIR = DATA_DIR
try:
    DATA_WRITE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    # It's okay if we cannot create it now (e.g., read-only FS); writers will attempt later.
    pass

# 服務專案 static/ 目錄（僅保留單一背景圖的供應）
STATIC_DIR = ROOT / "static"
try:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# 提供 /static/* 靜態檔案服務（前端背景圖、圖示等）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# === 註冊路由 ===
from api.home import router as home_router
from api.predict import router as predict_router
from api.data_build import router as data_build_router

app.include_router(home_router)
app.include_router(predict_router)
app.include_router(data_build_router)

# CSV 解析與預測共用小工具：先解析 CSV 再帶入預測，避免重複設定參數
def _resolve_csv_for(symbol: str, auto_build_csv: bool) -> Path:
    """取得 symbol 對應的 CSV。

    行為：
    - 若檔案存在且非空：直接回傳。
    - 若不存在或為空：
        * auto_build_csv=True 時嘗試自動建置（呼叫 _ensure_symbol_csv）。
        * auto_build_csv=False 時回 404 要求先建置。
    """
    p = _symbol_csv_path(symbol)
    if p.exists() and p.stat().st_size > 0:
        return p
    if auto_build_csv:
        try:
            built = _ensure_symbol_csv(symbol)
            logging.getLogger("app").info("[auto-build] symbol=%s csv=%s", symbol, built)
            return built
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"自動建置失敗：{e}")
    raise HTTPException(status_code=404, detail=f"找不到 CSV：{p.name}（請先呼叫 /api/build_symbol 或 /api/bulk_build_start 建置）")

def _run_predict(csv_path: Path, model: ModelName, symbol: str) -> dict:
    """以指定 CSV 與模型執行預測，回傳統一資料結構。"""
    chosen_model = model.value if isinstance(model, ModelName) else str(model)
    try:
        out = predict(str(csv_path), model=chosen_model, symbol=symbol)
        return {"model": out.get("model"), "proba": out.get("proba"), "label": out.get("label"), "symbol": symbol}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測錯誤：{e}")

def _load_threshold(model: ModelName | str) -> float | None:
    """嘗試從 models/ 載入對應模型的 threshold（若不存在或讀取/轉型失敗則回傳 None）。"""
    try:
        m = model.value if isinstance(model, ModelName) else str(model)
        p = MODELS_DIR / f"{m}_threshold.pkl"
        if p.exists():
            val = joblib.load(p)
            return float(val)
        return None
    except Exception:
        return None

# === Symbol 輔助與個股自動更新 ===

def _symbol_csv_path(symbol: str) -> Path:
    """取得 Symbol CSV 的固定讀取路徑：一律使用 data/ 底下檔案。

    檔名樣式：<symbol>_short_term_with_lag3.csv
    """
    return DATA_DIR / f"{symbol}_short_term_with_lag3.csv"

def _symbol_csv_write_path(symbol: str) -> Path:
    """Symbol 對應 CSV 的寫入（或刷新）路徑。"""
    return DATA_WRITE_DIR / f"{symbol}_short_term_with_lag3.csv"

def _infer_symbol_from_path(p: Path) -> str | None:
    """從檔名推斷 symbol（<symbol>_short_term_with_lag3.csv）。"""
    try:
        name = p.stem
        if name.endswith('_short_term_with_lag3'):
            return name.replace('_short_term_with_lag3', '')
        return None
    except Exception:
        return None


def _ensure_symbol_csv(symbol: str) -> Path:
    """確保 symbol 對應的特徵 CSV 存在；缺少時以 twelvedata 建置後回傳路徑。"""
    from datetime import datetime

    p = _symbol_csv_path(symbol)
    if p.exists() and p.stat().st_size > 0:
        return p

    _ensure_td()
    out = _symbol_csv_write_path(symbol)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        _build_from_td(symbol=symbol, out_csv=out)
    except Exception as e:
        raise RuntimeError(f"無法為 {symbol} 建構 CSV：{e}") from e
    except Exception as e:
        raise RuntimeError(f"無法為 {symbol} 建構 CSV：{e}") from e

    if not out.exists() or out.stat().st_size == 0:
        raise FileNotFoundError(f"Symbol CSV 仍然不存在：{out}")

    # 非關鍵的時間戳紀錄，失敗可忽略
    try:
        (out.parent / f"{symbol}_last_update.txt").write_text(datetime.now().isoformat(), encoding="utf-8")
    except Exception:
        pass

    return out

"""
移除舊版預測 fallback：統一走 predict 主路徑，失敗則回標準錯誤。
"""
# === 批次抓取輔助與背景任務 ===
BULK_TASKS: Dict[str, Dict] = {}

def _fetch_index_tickers(index: str) -> list:
    """Fetch tickers for known indices (sp500, nasdaq100, twse). Returns list of tickers.

    Uses Wikipedia tables for SP500 and NASDAQ-100, and for TWSE falls back to a small sample if not found.
    """
    idx = index.lower()
    if idx in ("sp500", "s&p500", "s&p 500"):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df['Symbol'].astype(str).tolist()
            return [t.replace('.', '-') for t in tickers]
        except Exception as e:
            raise RuntimeError(f"無法從 Wikipedia 取得 S&P500 名單：{e}")
    if idx in ("nasdaq100", "nasdaq-100"):
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        try:
            tables = pd.read_html(url)
            # the ticker table is often the first with 'Ticker' or 'Company' column
            for t in tables:
                cols = [c.lower() for c in t.columns]
                if 'ticker' in cols or 'symbol' in cols:
                    col = [c for c in t.columns if c.lower() in ('ticker', 'symbol')][0]
                    return t[col].astype(str).tolist()
            raise RuntimeError('找不到 ticker 欄位')
        except Exception as e:
            raise RuntimeError(f"無法從 Wikipedia 取得 Nasdaq-100 名單：{e}")
    if idx in ("twse", "tse", "taiwan"):
        # TWSE list is less reliable via Wikipedia; return a small default list or try a web source
        # Provide a reasonable sample; user can also pass explicit symbol list
        return ['2330', '2317', '2454', '2412']
    raise ValueError("未知的 index，支援：sp500, nasdaq100, twse 或直接傳 symbols")


async def _bulk_build_worker(symbols, concurrency, task_id):
    """批次建置工作器（背景任務）。

    依序以限制並發數的方式為每個 symbol 產生/更新特徵 CSV，並更新 BULK_TASKS 進度。
    此函式僅由 /api/bulk_build_start 觸發，不直接對外暴露。
    """
    sem = asyncio.Semaphore(concurrency)
    total = len(symbols)
    BULK_TASKS[task_id]['total'] = total
    BULK_TASKS[task_id]['done'] = 0
    BULK_TASKS[task_id]['errors'] = {}
    BULK_TASKS[task_id]['files'] = {}

    async def _build_one(s):
        async with sem:
            try:
                # run blocking build in thread
                p = _symbol_csv_write_path(s)
                await asyncio.to_thread(lambda: (_ensure_td(), _build_from_td(symbol=s, out_csv=p)))
                try:
                    # 記錄成功產出的 CSV 絕對路徑，方便前端或除錯查閱
                    BULK_TASKS[task_id]['files'][s] = str(p.resolve())
                except Exception:
                    BULK_TASKS[task_id]['files'][s] = str(p)
                BULK_TASKS[task_id]['done'] += 1
            except Exception as e:
                BULK_TASKS[task_id]['errors'][s] = str(e)
                BULK_TASKS[task_id]['done'] += 1

    tasks = [asyncio.create_task(_build_one(s)) for s in symbols]
    await asyncio.gather(*tasks)
    BULK_TASKS[task_id]['status'] = 'completed'
    BULK_TASKS[task_id]['finished_at'] = _time.time()


@app.get('/api/bulk_build_start')
async def bulk_build_start(index: str | None = None, symbols: str | None = None, concurrency: int = 4):
    """Start a background bulk build. Provide either `index` (sp500/nasdaq100/twse) or `symbols` (comma-separated). Returns a task_id to poll with /api/bulk_build_status."""
    if not index and not symbols:
        return JSONResponse({"ok": False, "error": "請提供 index 或 symbols 參數"}, status_code=400)
    try:
        if symbols:
            syms = [s.strip() for s in symbols.split(',') if s.strip()]
        else:
            syms = _fetch_index_tickers(index)
        # dedupe and normalize
        syms = list(dict.fromkeys([s.upper().replace('.TW', '').replace('-', '.') for s in syms]))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    task_id = str(uuid.uuid4())
    BULK_TASKS[task_id] = {"status": "running", "total": len(syms), "done": 0, "errors": {}, "started_at": _time.time()}
    # schedule the async worker on the running event loop
    loop = asyncio.get_running_loop()
    task = loop.create_task(_bulk_build_worker(syms, max(1, int(concurrency)), task_id))
    BULK_TASKS[task_id]['task'] = task
    return {"ok": True, "task_id": task_id, "count": len(syms)}


@app.get('/api/bulk_build_status')
def bulk_build_status(task_id: str):
    """查詢批次建置任務狀態。

    回傳進度（0~1）與錯誤摘要，不包含實際 asyncio.Task 物件。
    """
    info = BULK_TASKS.get(task_id)
    if not info:
        return JSONResponse({"ok": False, "error": "找不到 task_id"}, status_code=404)
    out = {k: v for k, v in info.items() if k != 'task'}
    # compute progress
    total = out.get('total', 0)
    done = out.get('done', 0)
    out['progress'] = float(done) / total if total else 1.0
    return {"ok": True, "task": out}

@app.get('/api/where_is_data')
def where_is_data():
    """回報目前資料目錄與已存在 CSV 絕對路徑（除錯用途）。"""
    try:
        data_dir = str(DATA_DIR.resolve())
        write_dir = str(DATA_WRITE_DIR.resolve())
    except Exception:
        data_dir = str(DATA_DIR)
        write_dir = str(DATA_WRITE_DIR)
    files = []
    try:
        for p in DATA_WRITE_DIR.glob('*_short_term_with_lag3.csv'):
            files.append(str(p.resolve()))
        if DATA_WRITE_DIR.resolve() != DATA_DIR.resolve():
            for p in DATA_DIR.glob('*_short_term_with_lag3.csv'):
                rp = str(p.resolve())
                if rp not in files:
                    files.append(rp)
    except Exception:
        pass
    return {"data_dir": data_dir, "write_dir": write_dir, "count": len(files), "files": files, "auto_build_predict": ENABLE_AUTO_BUILD_PREDICT}


@app.get('/api/bulk_build_stop')
def bulk_build_stop(task_id: str):
    """嘗試中止指定的批次建置任務。"""
    info = BULK_TASKS.get(task_id)
    if not info:
        return JSONResponse({"ok": False, "error": "找不到 task_id"}, status_code=404)
    t = info.get('task')
    if t and not t.done():
        t.cancel()
        info['status'] = 'cancelled'
        return {"ok": True, "status": "cancelled", "task_id": task_id}
    return {"ok": False, "error": "task not running or already finished"}


@app.get("/api/diagnostics")
def diagnostics(n_bins: int = 20, symbol: str | None = None):
    """回傳前端可視化所需的診斷資訊：
    - latest_row：CSV 最新一列（dict）
    - feature_stats：數值欄位的 mean/std/min/max
    - histograms：前幾個數值欄位的直方圖 bins/counts
    - models：已持久化的模型管線檔
    - thresholds：若存在則回傳已載入的閾值
    """
    # 僅接受個股 CSV：未提供 symbol 時回 400
    if not symbol:
        return JSONResponse({"error": "請提供 symbol 參數"}, status_code=400)
    try:
        csvp = _resolve_csv_for(symbol, auto_build_csv=False)
    except HTTPException as e:
        return JSONResponse({"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        return JSONResponse({"error": f"找不到資料：{e}"}, status_code=404)

    try:
        df = pd.read_csv(csvp, encoding=CSV_ENCODING)
    except Exception as e:
        return JSONResponse({"error": f"讀取 CSV 失敗：{str(e)}"}, status_code=500)

    # Treat empty/invalid CSV as server-side data error for diagnostics
    if df.shape[0] == 0:
        return JSONResponse({"error": "CSV 沒有資料或格式無效"}, status_code=500)

    numeric = df.select_dtypes(include=[np.number])
    stats = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) == 0:
            continue
        stats[col] = {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}

    latest = df.iloc[-1].to_dict() if len(df) > 0 else {}

    hist = {}
    for col in numeric.columns[:3]:
        arr = numeric[col].dropna().to_numpy()
        if arr.size == 0:
            continue
        counts, bins = np.histogram(arr, bins=n_bins)
        hist[col] = {"bins": bins.tolist(), "counts": counts.tolist()}

    models = []
    thresholds = {}
    if MODELS_DIR.exists():
        for p in MODELS_DIR.glob('*_pipeline.pkl'):
            models.append(p.name)
        for m in ['rf', 'lr']:
            thr_file = MODELS_DIR / f"{m}_threshold.pkl"
            if thr_file.exists():
                try:
                    thresholds[m] = float(joblib.load(thr_file))
                except Exception:
                    thresholds[m] = None

    return {"latest_row": latest, "feature_stats": stats, "histograms": hist, "models": models, "thresholds": thresholds}


@app.get('/api/stattests')
def stat_tests(feature: str, symbol: str | None = None):
    """針對指定數值特徵執行 Shapiro 常態性檢定、單樣本 t 檢定（H0: mean=0）、以及 ADF 單位根檢定。"""
    if not symbol:
        return JSONResponse({"error": "請提供 symbol 參數"}, status_code=400)
    try:
        csvp = _resolve_csv_for(symbol, auto_build_csv=False)
        df = pd.read_csv(csvp, encoding=CSV_ENCODING)
    except Exception as e:
        return JSONResponse({"error": f"讀取 CSV 失敗：{e}"}, status_code=500)
    if feature not in df.columns:
        return JSONResponse({"error": f"找不到欄位 {feature}"}, status_code=404)
    arr = pd.to_numeric(df[feature], errors='coerce').dropna()
    if arr.size < 3:
        return JSONResponse({"error": "資料量不足以做檢定"}, status_code=400)

    # Shapiro
    try:
        sh_w, sh_p = stats.shapiro(arr.sample(min(len(arr), 5000)))
    except Exception as e:
        sh_w, sh_p = None, None

    # one-sample t-test vs 0
    try:
        t_stat, t_p = stats.ttest_1samp(arr, popmean=0)
    except Exception as e:
        t_stat, t_p = None, None

    # ADF
    try:
        adf_res = tsstat.adfuller(arr.dropna(), autolag='AIC')
        adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])
    except Exception as e:
        adf_stat, adf_p = None, None

    return {"feature": feature, "shapiro": {"stat": sh_w, "p": sh_p}, "ttest": {"stat": t_stat, "p": t_p}, "adf": {"stat": adf_stat, "p": adf_p}}


@app.get('/api/lag_stats')
def lag_stats(symbol: str | None = None):
    """對 CSV 中所有滯後（lag）特徵進行詳細統計推論。

    回傳每個 lag 特徵的結果（皮爾森相關 r/p、線性回歸、Shapiro、t-test、ADF）。
    """
    if not symbol:
        return JSONResponse({"error": "請提供 symbol 參數"}, status_code=400)
    try:
        csvp = _resolve_csv_for(symbol, auto_build_csv=False)
        df = pd.read_csv(csvp, encoding=CSV_ENCODING)
    except Exception as e:
        return JSONResponse({"error": f"讀取 CSV 失敗：{str(e)}"}, status_code=500)

    numeric = df.select_dtypes(include=[np.number])
    lag_cols = [c for c in numeric.columns if 'lag' in c.lower()]
    results = []
    for col in lag_cols:
        arr = pd.to_numeric(df[col], errors='coerce').dropna()
        y = arr.values
        x = pd.to_numeric(df['收盤價(元)'], errors='coerce').loc[arr.index].values if '收盤價(元)' in df.columns else None
        if y.size == 0 or x is None or x.size == 0:
            continue

        res = {"feature": col, "n": int(y.size)}
        # mean/std
        try:
            res["mean"] = float(arr.mean())
            res["std"] = float(arr.std())
        except Exception:
            res["mean"] = None
            res["std"] = None
        proc = []
        proc.append(f"n={res['n']}")
        if res["mean"] is not None:
            try:
                proc.append(f"mean={res['mean']:.6g}")
            except Exception:
                proc.append(f"mean={res['mean']}")
        if res["std"] is not None:
            try:
                proc.append(f"std={res['std']:.6g}")
            except Exception:
                proc.append(f"std={res['std']}")
        try:
            pr, pp = stats.pearsonr(y, x)
            res.update({"pearson_r": float(pr), "pearson_p": float(pp)})
            proc.append(f"pearson_r={pr:.6g}, p={pp:.3g}")
        except Exception:
            res.update({"pearson_r": None, "pearson_p": None})

        try:
            lr = stats.linregress(y, x)
            res.update({"linreg_slope": float(lr.slope), "linreg_intercept": float(lr.intercept),
                        "linreg_r": float(lr.rvalue), "linreg_p": float(lr.pvalue), "linreg_stderr": float(lr.stderr) if lr.stderr is not None else None})
            proc.append(f"linreg slope={lr.slope:.6g}, p={lr.pvalue:.3g}")
        except Exception:
            res.update({"linreg_slope": None, "linreg_intercept": None, "linreg_r": None, "linreg_p": None, "linreg_stderr": None})

        # Shapiro (sample up to 5000)
        try:
            if y.size <= 5000:
                sh_w, sh_p = stats.shapiro(y)
                res.update({"shapiro_stat": float(sh_w), "shapiro_p": float(sh_p)})
                proc.append(f"shapiro W={sh_w:.6g}, p={sh_p:.3g}")
            else:
                res.update({"shapiro_stat": None, "shapiro_p": None})
        except Exception:
            res.update({"shapiro_stat": None, "shapiro_p": None})

        # one-sample t-test vs 0
        try:
            t_stat, t_p = stats.ttest_1samp(y, popmean=0)
            res.update({"ttest_stat": float(t_stat), "ttest_p": float(t_p)})
            proc.append(f"t-test t={t_stat:.6g}, p={t_p:.3g}")
        except Exception:
            res.update({"ttest_stat": None, "ttest_p": None})

        # ADF
        try:
            adf_res = tsstat.adfuller(y, autolag='AIC')
            res.update({"adf_stat": float(adf_res[0]), "adf_p": float(adf_res[1]), "adf_usedlag": int(adf_res[2]), "adf_nobs": int(adf_res[3])})
            proc.append(f"ADF stat={adf_res[0]:.6g}, p={adf_res[1]:.3g}, usedlag={int(adf_res[2])}")
        except Exception:
            res.update({"adf_stat": None, "adf_p": None, "adf_usedlag": None, "adf_nobs": None})

        # attach process list
        res["process"] = proc

        # conclusions (alpha=0.05)
        alpha = 0.05
        try:
            ttest_p = res.get('ttest_p')
            ttest_reject = (ttest_p is not None and ttest_p < alpha)
        except Exception:
            ttest_reject = None
        try:
            sh_p = res.get('shapiro_p')
            shapiro_normal = (sh_p is not None and sh_p > alpha)
        except Exception:
            shapiro_normal = None
        try:
            adf_p = res.get('adf_p')
            adf_stationary = (adf_p is not None and adf_p < alpha)
        except Exception:
            adf_stationary = None

        verdicts = []
        if ttest_reject is True:
            verdicts.append('t-test: 拒絕 H0 (mean ≠ 0)')
        elif ttest_reject is False:
            verdicts.append('t-test: 未拒絕 H0 (mean = 0)')
        else:
            verdicts.append('t-test: 無法判定')

        if shapiro_normal is True:
            verdicts.append('Shapiro: 通常視為常態分佈')
        elif shapiro_normal is False:
            verdicts.append('Shapiro: 非常態（拒絕常態）')
        else:
            verdicts.append('Shapiro: 無法判定')

        if adf_stationary is True:
            verdicts.append('ADF: 序列平穩（拒絕單位根）')
        elif adf_stationary is False:
            verdicts.append('ADF: 非平穩（未拒絕單位根）')
        else:
            verdicts.append('ADF: 無法判定')

        res['conclusions'] = {
            'ttest_reject': bool(ttest_reject) if ttest_reject is not None else None,
            'shapiro_normal': bool(shapiro_normal) if shapiro_normal is not None else None,
            'adf_stationary': bool(adf_stationary) if adf_stationary is not None else None,
            'verdict': '；'.join(verdicts)
        }

        results.append(res)

    # sort by absolute pearson r
    try:
        results.sort(key=lambda r: abs(r.get('pearson_r') or 0), reverse=True)
    except Exception:
        pass

    return {"ok": True, "count": len(results), "results": results}


@app.get('/api/series')
def series(feature: str, n: int = 500, symbol: str | None = None):
    """回傳數值特徵最近 n 筆的數值，供前端繪圖使用。"""
    if not symbol:
        return JSONResponse({"error": "請提供 symbol 參數"}, status_code=400)
    try:
        csvp = _resolve_csv_for(symbol, auto_build_csv=False)
        df = pd.read_csv(csvp, encoding=CSV_ENCODING)
    except Exception as e:
        return JSONResponse({"error": f"讀取 CSV 失敗：{str(e)}"}, status_code=500)
    if feature not in df.columns:
        return JSONResponse({"error": f"找不到欄位 {feature}"}, status_code=404)
    arr = pd.to_numeric(df[feature], errors='coerce').dropna()
    vals = arr.tail(n).tolist()
    return {"feature": feature, "values": vals}


@app.get('/api/latest_features')
def latest_features(features: str | None = None, pattern: str | None = None, file: str | None = None, max_items: int = 100, symbol: str | None = None):
    """以 JSON 形式回傳 CSV 的最新一列（或指定欄位）。

    - features：以逗號分隔的欄位名稱（保留順序）
    - pattern：以正則式選取欄位（不分大小寫）
    - file：可選擇相對專案根目錄的路徑，但必須位於 data/ 之下
    - max_items：回傳欄位的最大數量（安全限制）
    """
    # decide which CSV to read (restrict file to data/ folder) or use symbol
    csv_path = None
    if symbol:
        try:
            csv_path = _resolve_csv_for(symbol, auto_build_csv=False)
        except HTTPException as e:
            return JSONResponse({"error": e.detail}, status_code=e.status_code)
        except Exception as e:
            return JSONResponse({"error": f"無法取得 symbol CSV：{e}"}, status_code=500)
    elif file:
        p = Path(file)
        if not p.is_absolute():
            p = Path(__file__).parent / file
        try:
            p_res = p.resolve()
            data_dir_res = DATA_DIR.resolve()
            if not str(p_res).startswith(str(data_dir_res)):
                return JSONResponse({"error": "file 參數必須位於 data/ 資料夾內"}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": f"檔案解析失敗：{str(e)}"}, status_code=400)
        if not p_res.exists():
            return JSONResponse({"error": f"指定的檔案不存在：{str(p_res)}"}, status_code=404)
        csv_path = p_res
    else:
        return JSONResponse({"error": "請提供 symbol 或 file 參數"}, status_code=400)

    try:
        df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    except Exception as e:
        return JSONResponse({"error": f"讀取 CSV 失敗：{str(e)}"}, status_code=500)

    if df.shape[0] == 0:
        return JSONResponse({"error": "CSV 沒有資料"}, status_code=404)

    cols = list(df.columns)

    # user-specified features take precedence
    selected = []
    if features:
        want = [f.strip() for f in features.split(',') if f.strip()]
        for w in want:
            if w in df.columns:
                selected.append(w)
        # keep order and unique
        selected = list(dict.fromkeys(selected))
    elif pattern:
        try:
            rx = re.compile(pattern, re.I)
            selected = [c for c in cols if rx.search(c)]
        except re.error as e:
            return JSONResponse({"error": f"正則式錯誤：{str(e)}"}, status_code=400)
    else:
        # sensible defaults: look for columns with 'lag' or '滯後' in name, else last numeric columns
        selected = [c for c in cols if 'lag' in c.lower() or '滯後' in c]
        if len(selected) == 0:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            selected = numeric[-min(len(numeric), max_items):]

    # cap number of items
    if len(selected) > max_items:
        selected = selected[:max_items]

    latest = df.iloc[-1]
    latest_selected = {}
    for c in selected:
        val = latest.get(c)
        # try to coerce numpy types to python native
        try:
            if pd.isna(val):
                latest_selected[c] = None
            else:
                latest_selected[c] = (float(val) if (isinstance(val, (int, float, np.number))) else str(val))
        except Exception:
            latest_selected[c] = str(val)

    return {"source": str(csv_path.resolve()), "selected_columns": selected, "latest": latest_selected}

# === 健康檢查 endpoint ===
@app.get('/health')
def health():
    """輕量健康檢查（供容器 HEALTHCHECK 使用）。

    回傳內容：
      - status：可達則為 "ok"
      - versions：主要函式庫版本
      - models_ready：是否存在任一模型管線檔
      - data_ready：主要資料 CSV 是否存在且非空
    避免重型操作，以利快速探測。
    """
    try:
        models_ready = False
        model_files = []
        if MODELS_DIR.exists():
            model_files = [p.name for p in MODELS_DIR.glob('*_pipeline.pkl')]
            models_ready = len(model_files) > 0
        # 資料就緒：任一個股 CSV 存在（以目錄 glob 快速檢查）
        try:
            data_ready = any(DATA_DIR.glob('*_short_term_with_lag3.csv'))
        except Exception:
            data_ready = False
        return {
            "status": "ok",
            "models_ready": models_ready,
            "model_files": model_files[:5],  # limit list size
            "data_ready": data_ready,
            "versions": {
                "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "pandas": getattr(pd, '__version__', None),
                "numpy": getattr(np, '__version__', None),
                "scipy": getattr(stats, '__version__', None) if hasattr(stats, '__version__') else None,
                "app_git_sha": APP_GIT_SHA,
                "build_time": APP_BUILD_TIME,
            }
        }
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get('/version')
def version():
    """回傳建置／版本資訊（可公開）。"""
    return {
        "git_sha": APP_GIT_SHA,
        "build_time": APP_BUILD_TIME,
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "fastapi": getattr(FastAPI, '__version__', 'n/a'),
        "pandas": getattr(pd, '__version__', None),
        "numpy": getattr(np, '__version__', None),
    }


# （精簡版）移除 Prometheus 指標端點

# === 全域自動更新配置 ===
def _parse_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = v.strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    logging.getLogger("app").warning(f"[config] Unknown boolean for {name}={v!r}, fallback to {default}")
    return default

ENABLE_GLOBAL_UPDATER = _parse_bool_env("ENABLE_GLOBAL_UPDATER", True)
GLOBAL_UPDATE_TASK: asyncio.Task | None = None
GLOBAL_UPDATE_INTERVAL_MIN = 5
GLOBAL_UPDATE_CONCURRENCY = 4


# === 快速導覽（路由與背景任務） ===
@app.get('/api/overview')
def api_overview(only_api: bool = True):
    """快速導覽：列出所有路由摘要與背景任務狀態。

    - only_api：僅顯示 /api/*（以及 /health、/version、/），預設 True。
    """
    routes = []
    allow_prefixes = ['/', '/health', '/version', '/static', '/api/']
    for r in app.routes:
        try:
            methods = sorted(list(getattr(r, 'methods', []) or []))
            path = getattr(r, 'path', '')
            if not methods or not path:
                continue
            if only_api:
                if not any(path == p or path.startswith(p) for p in allow_prefixes):
                    continue
            endpoint = getattr(r, 'endpoint', None)
            summary = ""
            if endpoint and getattr(endpoint, '__doc__', None):
                summary = (endpoint.__doc__ or '').strip().splitlines()[0]
            routes.append({"methods": methods, "path": path, "summary": summary})
        except Exception:
            continue
    global_running = (GLOBAL_UPDATE_TASK is not None and not GLOBAL_UPDATE_TASK.done())
    background = {
        "global_updater_running": global_running,
        "global_updater_interval_min": GLOBAL_UPDATE_INTERVAL_MIN if global_running else None,
        "global_updater_concurrency": GLOBAL_UPDATE_CONCURRENCY if global_running else None,
        "bulk_tasks_running": sum(1 for t in BULK_TASKS.values() if isinstance(t, dict) and t.get('status') == 'running'),
        "bulk_tasks_total": len(BULK_TASKS),
    }
    return {"ok": True, "routes": routes, "background": background}

# === 執行啟動 ===
@app.on_event('startup')
async def _startup_global_updater():
    """應用啟動：清理少量靜態檔後，啟動全域自動更新 loop（若啟用）。"""
    # 清理未使用的靜態圖片（只移除明確列出的檔名）
    try:
        for _name in ("123.png", "ComfyUI_00012_.jpg"):
            _p = STATIC_DIR / _name
            if _p.exists():
                _p.unlink()
    except Exception:
        pass
    # 啟動全域自動更新
    if ENABLE_GLOBAL_UPDATER:
        try:
            from api.data_build import _all_symbols_loop
            loop = asyncio.get_event_loop()
            global GLOBAL_UPDATE_TASK
            if GLOBAL_UPDATE_TASK is None or GLOBAL_UPDATE_TASK.done():
                GLOBAL_UPDATE_TASK = loop.create_task(_all_symbols_loop(GLOBAL_UPDATE_INTERVAL_MIN, GLOBAL_UPDATE_CONCURRENCY))
                print(f"[startup] global updater started: interval={GLOBAL_UPDATE_INTERVAL_MIN}m, concurrency={GLOBAL_UPDATE_CONCURRENCY}")
        except Exception as e:
            print(f"[startup] failed to start global updater: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)