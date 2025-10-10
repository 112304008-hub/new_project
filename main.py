"""main.py — FastAPI 主應用程式

本服務核心目標：
1. 提供『股價籤筒 / 預測 API』：/api/draw 讀取最新特徵 CSV 與已訓練模型，回傳「明日漲跌」推論結果與籤筒格式文字。
2. 支援動態建立/更新個股特徵資料：/api/build_symbol, /api/build_symbols 等，必要時自動用 yfinance 下載歷史價量並產生特徵 CSV。
3. 提供批次背景建置與自動更新機制：
     - 單一或多個 symbol 的自動輪巡 (_symbol_loop)
     - 指數成分（sp500, nasdaq100, twse）整體迴圈 (_index_loop)
4. 暴露統計診斷與特徵檢定：/api/diagnostics, /api/stattests, /api/lag_stats, /api/series, /api/latest_features。
5. 提供 Prometheus 指標 (/metrics) 與健康檢查 (/health) 供部署監控；版本資訊 (/version) 供快速稽核。

主要設計重點：
- 所有模型推論僅在已存在 models/ 內的 *_pipeline.pkl + *_threshold.pkl 情況下執行；若不存在則回傳友善錯誤訊息。
- 資料目錄分離：讀取 data/，可另設 DATA_DIR_WRITE（環境變數）指向可寫入的 data_work/ 以利唯讀環境（Docker read-only layer / Kubernetes volume）。
- 自動更新註冊檔 (auto_registry.json / index_auto_registry.json) 儲存在可寫目錄以支援重啟後還原背景任務。
- Rate Limit（記憶體級）簡易實作，避免濫用 API。
- 監控指標：HTTP 請求計數/延遲、背景任務數量、模型/資料就緒狀態。

常用環境變數：
    API_KEY              （可選）— 若設置，保護 /api/* 端點需以 request header: x-api-key 傳遞。
    RATE_LIMIT_PER_MIN   （預設 120）— 單個 IP 每分鐘允許請求數（排除 /health, /metrics 等安全端點）。
    DATA_DIR_WRITE        指定可寫入資料夾；未設置時 fallback 到專案內 data_work/。
    LOG_LEVEL            （INFO/DEBUG...）
    APP_GIT_SHA / APP_BUILD_TIME — 於 CI/CD build 注入版本資訊。

快速啟動 (開發)：
    python -m uvicorn main:app --reload --port 8000
或直接執行：
    python main.py

主要端點概覽（摘要）：
    GET /                      — 首頁（籤筒/前端頁面 template2.html）
    GET /api/draw              — 取得預測籤（參數：model, symbol 可選）
    GET /api/build_symbol      — 單一 symbol 生成/刷新特徵
    GET /api/build_symbols     — 多 symbols 生成
    GET /api/list_symbols      — 列出現有特徵 CSV
    批次：/api/bulk_build_start /status /stop
    自動任務（個股）：/api/auto/start_symbol /stop_symbol /start_many /stop_many /start_existing_csvs /list_registry
    自動任務（指數）：/api/auto/start_index /stop_index /list_index
    診斷與統計：/api/diagnostics /api/stattests /api/lag_stats /api/series /api/latest_features
    監控：/health /metrics /version

檔案閱讀指南：
    _symbol_* / _index_* 系列：管理個股與指數的特徵 CSV 建構邏輯與自動輪巡。
    BULK_TASKS 結構：儲存批次工作進度 (status, total, done, errors, started_at ...)。
    predict() 相關邏輯在 stock.py；本檔僅做輸入驗證與錯誤處理 / 回應格式化。

維護建議：
    - 新增模型時：確保 stock.py 產出的 *_pipeline.pkl 與 *_threshold.pkl 命名一致，並更新 /api/draw 的預設允許 model 名稱。
    - 增加大型迴圈請務必觀察 Prometheus 指標與記憶體使用量。
    - 若路徑/資料 schema 調整，記得同步更新 _update_gauges 與相關健康檢查。

本模組只加入說明性文字，未改變既有行為。
"""
import os
import time as _time
import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from stock import predict, MODELS_DIR
from stock import _ensure_yf, _build_from_yfinance
import asyncio
import json
from typing import Dict
import uuid
import requests
import time
import joblib
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.tsa.stattools as tsstat
import re

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

APP_GIT_SHA = os.getenv("APP_GIT_SHA", "UNKNOWN")
APP_BUILD_TIME = os.getenv("APP_BUILD_TIME", "UNKNOWN")
API_KEY = os.getenv("API_KEY")  # optional

# Basic key=value logging format for easy ingestion
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="ts=%(asctime)s level=%(levelname)s msg=%(message)s module=%(module)s",
)
logger = logging.getLogger("app")

# Prometheus metrics
HTTP_REQ_COUNT = Counter(
    "app_http_requests_total", "HTTP request count", ["method", "path", "status"]
)
HTTP_REQ_LATENCY = Histogram(
    "app_http_request_duration_seconds", "HTTP request latency", ["method", "path"]
)
BACKGROUND_TASKS_GAUGE = Gauge(
    "app_background_tasks", "Number of running background tasks"
)
MODELS_READY_GAUGE = Gauge(
    "app_models_ready", "1 if at least one model pipeline file present else 0"
)
DATA_READY_GAUGE = Gauge(
    "app_data_ready", "1 if main data CSV present and non-empty else 0"
)

app = FastAPI(title="股價之神", version="1.0")


def _update_gauges():
    """更新 Prometheus 指標狀態。

    - models_ready: 是否存在任一已訓練模型檔 (*_pipeline.pkl)
    - data_ready: 主要資料檔是否存在且非空 (data/short_term_with_lag3.csv)
    - app_background_tasks: 目前執行中的背景任務數（批次 + 自動任務）
    """
    try:
        models_ready = MODELS_DIR.exists() and any(MODELS_DIR.glob('*_pipeline.pkl'))
        MODELS_READY_GAUGE.set(1 if models_ready else 0)
        data_ready = DATA.exists() and DATA.stat().st_size > 0
        DATA_READY_GAUGE.set(1 if data_ready else 0)
        # background bulk + symbol tasks
        running_tasks = sum(1 for t in BULK_TASKS.values() if isinstance(t, dict) and t.get('status') == 'running')
        running_tasks += sum(1 for t in SYMBOL_TASKS.values() if not t.done())
        BACKGROUND_TASKS_GAUGE.set(running_tasks)
    except Exception as e:
        logger.debug(f"gauge update failed: {e}")


@app.middleware("http")
async def metrics_and_logging_middleware(request: Request, call_next):
    """全域中間件：記錄請求、收集延遲指標、簡易速率限制與 API Key 驗證。

    注意：/health, /metrics, /version, /static 與首頁不適用速率限制。
    若設定 API_KEY，僅保護 /api/* 端點。
    """
    start = _time.perf_counter()
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    # simple rate limit (memory) excluding safe paths
    path = request.url.path
    rate_exempt = path.startswith('/health') or path.startswith('/metrics') or path.startswith('/version') or path.startswith('/static') or path == '/'
    client_ip = request.client.host if request.client else "-"
    if not hasattr(app.state, 'rate_bucket'):
        app.state.rate_bucket = {}
    bucket = app.state.rate_bucket
    now = _time.time()
    window = 60
    limit = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))
    key = f"{client_ip}" if not rate_exempt else None
    if key:
        rec = bucket.get(key)
        if not rec or now - rec['ts'] > window:
            rec = {'ts': now, 'count': 0}
        rec['count'] += 1
        bucket[key] = rec
        if rec['count'] > limit:
            return JSONResponse({"error": "rate limit exceeded"}, status_code=429, headers={"x-request-id": req_id})

    # API key check (only protect /api/* endpoints unless exempt list)
    if API_KEY and path.startswith('/api/'):
        provided = request.headers.get('x-api-key')
        if provided != API_KEY:
            return JSONResponse({"error": "unauthorized"}, status_code=401, headers={"x-request-id": req_id})

    try:
        response: Response = await call_next(request)
    except Exception as e:
        logger.exception(f"request failed req_id={req_id} path={path} error={e}")
        response = JSONResponse({"error": "internal error"}, status_code=500)
    duration = _time.perf_counter() - start
    status = response.status_code
    # record metrics (path collapsed for very dynamic routes if needed)
    label_path = path if len(path) < 64 else path[:60] + '...'
    if not path.startswith('/static'):
        HTTP_REQ_COUNT.labels(request.method, label_path, str(status)).inc()
        HTTP_REQ_LATENCY.labels(request.method, label_path).observe(duration)
    response.headers['x-request-id'] = req_id
    logger.info(
        f"req_id={req_id} method={request.method} path={path} status={status} ms={duration*1000:.2f} ip={client_ip} ua=\"{request.headers.get('user-agent','')}\""
    )
    return response

# === 路徑設定 ===
ROOT = Path(__file__).parent
HTML = ROOT / "template2.html"
DATA = ROOT / "data" / "short_term_with_lag3.csv"

# Writable data directory (separate from read-only data/ when desired).
def _get_write_dir() -> Path:
    """取得可寫入的資料目錄。

    優先使用環境變數 DATA_DIR_WRITE；未設置時回退至專案內的 data_work/。
    """
    env = os.getenv("DATA_DIR_WRITE")
    if env:
        try:
            p = Path(env)
            return p
        except Exception:
            pass
    return ROOT / "data_work"

DATA_DIR = DATA.parent
DATA_WRITE_DIR = _get_write_dir()
try:
    DATA_WRITE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    # It's okay if we cannot create it now (e.g., read-only FS); writers will attempt later.
    pass

# 服務專案 static/ 目錄下的靜態資源（例如 static/temple.jpg）
STATIC_DIR = ROOT / "static"
if not STATIC_DIR.exists():
    try:
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# === 首頁：籤筒網頁 ===
@app.get("/")
def home():
    """首頁：若存在 `template2.html`，回傳檔案；否則回應 404 JSON。"""
    return FileResponse(HTML) if HTML.exists() else JSONResponse({"error": "template2.html 不存在"}, status_code=404)

# === API: 抽籤（預測）=== 
@app.get("/api/draw")
def draw(model: str = "rf", symbol: str | None = None):
    """執行單次預測。

    - 若未提供 symbol，使用主要資料檔 DATA；須先存在訓練好的模型檔。
    - 回傳包含 label/proba/threshold 與可讀的 fortune 文案。
    """
    # 若未指定 symbol，需確保預設 CSV 存在
    if not symbol:
        if not DATA.exists() or DATA.stat().st_size == 0:
            return JSONResponse({
                "error": "找不到有效的 short_term_with_lag3.csv，請先執行 test.py 建立資料",
                "fortune": {
                    "title": "找不到資料",
                    "text": ["請先執行 test.py 建立最新特徵資料。"],
                    "advice": "請確認 test.py 有啟動成功並產生 CSV。"
                }
            }, status_code=404)

    # 僅允許推論：需要已訓練的 pipeline 與 threshold 檔案存在
    model_file = MODELS_DIR / f"{model}_pipeline.pkl"
    thr_file = MODELS_DIR / f"{model}_threshold.pkl"
    if not model_file.exists() or not thr_file.exists():
        return JSONResponse({
            "error": "未發現已訓練的模型，API 僅提供推論。請先在主機上訓練並將模型儲存在 models/ 資料夾。",
            "fortune": {
                "title": "模型未準備",
                "text": [
                    f"請在伺服器上執行訓練以產生 {model_file.name} 與 {thr_file.name}。",
                    "訓練完成後再呼叫此 API。"
                ],
                "advice": "可建立 CI 或排程作業定期訓練／更新模型"
            }
        }, status_code=404)

    try:
    # 既有模型檔存在 -> 呼叫 predict 進行推論
        if symbol:
            # 解析或建立 symbol 對應 CSV，並傳入明確路徑
            csvp = _ensure_symbol_csv(symbol)
            out = predict(str(csvp), model=model, symbol=symbol)
        else:
            out = predict(str(DATA), model=model)
        fortune = {
            "title": "預測結果",
            "text": [
                f"模型：{out['model']}",
                f"預測：{out['label']}",
                f"預測機率：{out['proba']*100:.2f}%",
            ],
            "label": out["label"],
            "prob_up": out["proba"],
            "threshold": out["threshold"],
            "confidence": 0.85,
            "advice": "此籤僅供參考，請謹慎投資"
        }
        resp = {
            "ok": True,
            "model": out["model"],
            "threshold": out["threshold"],
            "proba": out["proba"],
            "label": out["label"],
            "fortune": fortune
        }
    # 若 predict 回傳包含 symbol/csv，則一併回傳
        if isinstance(out, dict) and out.get('symbol'):
            resp['symbol'] = out.get('symbol')
        if isinstance(out, dict) and out.get('csv'):
            resp['csv'] = out.get('csv')
        return resp
    except Exception as e:
    # 退而求其次：若有提供 symbol，改走直接載入 pipeline 推論以避免路徑問題
        if symbol:
            try:
                csvp = _symbol_csv_path(symbol)
                df_raw = pd.read_csv(csvp, encoding="utf-8-sig")
                df = df_raw.copy()
                # reconstruct X_all like stock.predict
                df["y_final"] = pd.Series(dtype=float)
                drop_cols = ["年月日", "y_明天漲跌", "明天收盤價", "y_final"]
                X_all = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
                for c in X_all.columns:
                    X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
                # load persisted pipeline + threshold
                import joblib as _joblib
                from sklearn.pipeline import Pipeline as _Pipeline
                model_file = MODELS_DIR / f"{model}_pipeline.pkl"
                thr_file = MODELS_DIR / f"{model}_threshold.pkl"
                pipeline: _Pipeline = _joblib.load(model_file)
                thr = _joblib.load(thr_file)
                # 推斷模型所需特徵欄位
                expected = None
                for step_name in ("scaler", "rf", "lr", "mdl"):
                    if step_name in getattr(pipeline, 'named_steps', {}):
                        step = pipeline.named_steps[step_name]
                        if hasattr(step, 'feature_names_in_'):
                            expected = list(step.feature_names_in_)
                            break
                if expected is None and hasattr(pipeline, 'feature_names_in_'):
                    expected = list(pipeline.feature_names_in_)
                if expected is None and hasattr(pipeline, 'n_features_in_'):
                    n = int(pipeline.n_features_in_)
                    numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
                    expected = numeric_cols[-n:] if n and len(numeric_cols) >= n else numeric_cols
                # 確保必要欄位存在
                if expected:
                    missing = [c for c in expected if c not in X_all.columns]
                    for c in missing:
                        X_all[c] = 0.0
                    x_latest = X_all[expected].iloc[[-1]].fillna(X_all.median(numeric_only=True))
                else:
                    x_latest = X_all.tail(1).fillna(X_all.median(numeric_only=True))
                p1 = float(pipeline.predict_proba(x_latest)[:, 1][0])
                yhat = 1 if p1 >= float(thr) else 0
                fortune = {
                    "title": "預測結果",
                    "text": [
                        f"模型：{model}",
                        f"預測：{'漲' if yhat==1 else '跌'}",
                        f"預測機率：{p1*100:.2f}%",
                    ],
                    "label": '漲' if yhat==1 else '跌',
                    "prob_up": p1,
                    "threshold": float(thr),
                    "confidence": 0.85,
                    "advice": "此籤僅供參考，請謹慎投資"
                }
                return {"ok": True, "model": model, "threshold": float(thr), "proba": p1, "label": ('漲' if yhat==1 else '跌'), "fortune": fortune, "symbol": symbol, "csv": str(csvp.resolve())}
            except Exception as e2:
                import traceback
                err_msg = traceback.format_exc()
                return JSONResponse({
                    "error": f"預測錯誤：{str(e)} | fallback 失敗：{str(e2)}",
                    "fortune": {
                        "title": "預測錯誤",
                        "text": ["系統忙碌或資料錯誤，請稍後再試。"],
                        "advice": "請重新整理或查看 log"
                    }
                }, status_code=500)
    # 未提供 symbol 或退路亦失敗
        import traceback
        err_msg = traceback.format_exc()
        return JSONResponse({
            "error": f"預測錯誤：{str(e)}",
            "fortune": {
                "title": "預測錯誤",
                "text": ["系統忙碌或資料錯誤，請稍後再試。"],
                "advice": "請重新整理或查看 log"
            }
        }, status_code=500)


# === Symbol 輔助與個股自動更新 ===

def _symbol_csv_path(symbol: str) -> Path:
    """取得 Symbol CSV 的優先讀取路徑：先找可寫目錄中的副本，否則回退到唯讀 data 目錄。"""
    p_w = DATA_WRITE_DIR / f"{symbol}_short_term_with_lag3.csv"
    if p_w.exists():
        return p_w
    return DATA_DIR / f"{symbol}_short_term_with_lag3.csv"

def _symbol_csv_write_path(symbol: str) -> Path:
    """Symbol 對應 CSV 的寫入（或刷新）路徑。"""
    return DATA_WRITE_DIR / f"{symbol}_short_term_with_lag3.csv"

def _ensure_symbol_csv(symbol: str) -> Path:
    """確保取得 Symbol 的 CSV 路徑；若不存在則嘗試透過 yfinance 建構。"""
    # Prefer existing in write dir; else in data dir; else build into write dir
    p_read = _symbol_csv_path(symbol)
    if p_read.exists() and p_read.stat().st_size > 0:
        return p_read
    # try to build using stock helper
    try:
        _ensure_yf()
        p_write = _symbol_csv_write_path(symbol)
        p_write.parent.mkdir(parents=True, exist_ok=True)
        _build_from_yfinance(symbol=symbol, out_csv=p_write)
        if p_write.exists():
            # also write/update last_update timestamp alongside
            try:
                from datetime import datetime
                ts_path = p_write.parent / f"{symbol}_last_update.txt"
                with ts_path.open('w', encoding='utf-8') as f:
                    f.write(datetime.now().isoformat())
            except Exception:
                pass
            return p_write
    except Exception as e:
        raise RuntimeError(f"無法為 {symbol} 建構 CSV：{e}")
    # Fallback message if file still missing
    raise FileNotFoundError(f"Symbol CSV 仍然不存在：{_symbol_csv_write_path(symbol)}")

# 管理個股自動任務
SYMBOL_TASKS: Dict[str, asyncio.Task] = {}
# 管理指數級自動任務（例如 sp500、nasdaq100），避免一次產生過多協程
INDEX_TASKS: Dict[str, asyncio.Task] = {}

# 自動任務的持久化註冊檔（symbol -> interval_min 等設定）
_REG_FALLBACK_DIR = DATA.parent
_REG_DIR = DATA_WRITE_DIR if DATA_WRITE_DIR else DATA.parent
AUTO_REG_FILE = _REG_DIR / "auto_registry.json"
INDEX_AUTO_REG_FILE = _REG_DIR / "index_auto_registry.json"

def _load_auto_registry() -> dict:
    try:
        if AUTO_REG_FILE.exists():
            with AUTO_REG_FILE.open('r', encoding='utf-8') as f:
                return json.load(f)
        # fallback to old location (read-only data dir)
        fb = _REG_FALLBACK_DIR / "auto_registry.json"
        if fb.exists():
            with fb.open('r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_auto_registry(reg: dict):
    try:
        with AUTO_REG_FILE.open('w', encoding='utf-8') as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[warning] 無法寫入 auto registry: {e}")

def _load_index_auto_registry() -> dict:
    try:
        if INDEX_AUTO_REG_FILE.exists():
            with INDEX_AUTO_REG_FILE.open('r', encoding='utf-8') as f:
                return json.load(f)
        fb = _REG_FALLBACK_DIR / "index_auto_registry.json"
        if fb.exists():
            with fb.open('r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_index_auto_registry(reg: dict):
    try:
        with INDEX_AUTO_REG_FILE.open('w', encoding='utf-8') as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[warning] 無法寫入 index auto registry: {e}")

async def _symbol_loop(symbol: str, interval_min: int, backoff_factor: float = 2.0, max_backoff_min: int = 30):
    """Per-symbol auto update loop with exponential backoff on consecutive failures.

    Success path: sleep 固定 base interval。
    Failure path: sleep = min(interval * backoff_factor**consecutive_failures, max_backoff_min).
    任何一次成功即重置失敗計數。
    """
    print(f"[symbol auto] 啟動自動更新：{symbol} every {interval_min} min (backoff_factor={backoff_factor}, max_backoff={max_backoff_min}m)")
    consecutive_failures = 0
    base = max(1, int(interval_min))
    while True:
        success = False
        try:
            p = _symbol_csv_write_path(symbol)
            await asyncio.to_thread(lambda: (_ensure_yf(), _build_from_yfinance(symbol=symbol, out_csv=p)))
            ts_path = p.parent / f"{symbol}_last_update.txt"
            try:
                from datetime import datetime
                with ts_path.open('w', encoding='utf-8') as f:
                    f.write(datetime.now().isoformat())
            except Exception:
                pass
            success = True
        except Exception as e:
            consecutive_failures += 1
            print(f"[symbol auto] 更新 {symbol} 失敗 (#{consecutive_failures}): {e}")
        if success:
            if consecutive_failures:
                print(f"[symbol auto] {symbol} 恢復成功，重置 backoff")
            consecutive_failures = 0
            sleep_for = base
        else:
            sleep_for = min(int(max_backoff_min), int(base * (backoff_factor ** consecutive_failures)))
        await asyncio.sleep(sleep_for * 60)

async def _index_loop(index: str, interval_min: int, concurrency: int = 4, backoff_factor: float = 2.0, max_backoff_min: int = 30):
    """指數級自動更新迴圈：每隔 interval 分鐘（重新）建置該指數所有成分的特徵 CSV。

    策略：啟動時取得成分列表（若失敗則在迴圈內重試），每一輪：
      - 以限制並發數的方式分批處理 symbols
      - 若 CSV 缺失或過舊即重建（目前採無條件重建以保持最新）
    這種作法避免每個 symbol 都常駐一個協程，將排程集中管理。
    """
    print(f"[index auto] 啟動指數自動更新：{index} every {interval_min} min (concurrency={concurrency}, backoff_factor={backoff_factor}, max_backoff={max_backoff_min}m)")
    syms: list[str] = []  # cached index symbols
    consecutive_failures = 0
    base = max(1, int(interval_min))
    while True:
        cycle_failed = False
        try:
            if not syms:
                try:
                    # reuse _fetch_index_tickers for supported indices
                    syms = _fetch_index_tickers(index)
                    syms = list(dict.fromkeys([s.upper() for s in syms]))
                except Exception as e:
                    print(f"[index auto] 取得指數 {index} 成分失敗: {e}; 5 分鐘後重試")
                    cycle_failed = True
                    await asyncio.sleep(300)
                    continue
            # --- Feature C: merge any new existing CSV symbols in data/ directory ---
            try:
                existing = []
                for p in DATA_DIR.glob('*_short_term_with_lag3.csv'):
                    sym = p.stem.replace('_short_term_with_lag3', '')
                    existing.append(sym.upper())
                merged = list(dict.fromkeys(syms + existing))
                if len(merged) != len(syms):
                    added = set(merged) - set(syms)
                    if added:
                        print(f"[index auto] 偵測到新 CSV symbols 將納入自動更新: {sorted(list(added))[:8]}{'...' if len(added)>8 else ''}")
                    syms = merged
            except Exception as e:
                print(f"[index auto] merge existing csv symbols 失敗: {e}")
                cycle_failed = True
            sem = asyncio.Semaphore(max(1, int(concurrency)))
            async def _build_one(sym: str):
                async with sem:
                    p = _symbol_csv_write_path(sym)
                    try:
                        await asyncio.to_thread(lambda: (_ensure_yf(), _build_from_yfinance(symbol=sym, out_csv=p)))
                        ts_path = p.parent / f"{sym}_last_update.txt"
                        from datetime import datetime
                        try:
                            with ts_path.open('w', encoding='utf-8') as f:
                                f.write(datetime.now().isoformat())
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[index auto] 更新 {sym} 失敗: {e}")
                        nonlocal_cycle_failed.append(True)
            tasks = [asyncio.create_task(_build_one(s)) for s in syms]
            # track per-symbol failures by capturing in list (Python scoping workaround)
            nonlocal_cycle_failed = []
            await asyncio.gather(*tasks)
            if nonlocal_cycle_failed:
                cycle_failed = True
        except asyncio.CancelledError:
            print(f"[index auto] 指數 {index} loop 已取消")
            raise
        except Exception as e:
            print(f"[index auto] 迴圈錯誤：{e}")
            cycle_failed = True
        # backoff sleep logic
        if cycle_failed:
            consecutive_failures += 1
            sleep_for = min(int(max_backoff_min), int(base * (backoff_factor ** consecutive_failures)))
            print(f"[index auto] {index} 本輪有失敗，使用 backoff 休息 {sleep_for} 分鐘 (連續失敗 {consecutive_failures})")
        else:
            if consecutive_failures:
                print(f"[index auto] {index} 復原成功，重置 backoff")
            consecutive_failures = 0
            sleep_for = base
        await asyncio.sleep(sleep_for * 60)


@app.get('/api/build_symbol')
def build_symbol(symbol: str):
    """按需建置單一 symbol 的 CSV，成功則回傳路徑，否則回傳錯誤。"""
    if not symbol:
        return JSONResponse({"ok": False, "error": "請提供 symbol 參數"}, status_code=400)
    try:
        p = _ensure_symbol_csv(symbol)
        return {"ok": True, "symbol": symbol, "csv": str(p.resolve())}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get('/api/build_symbols')
def build_symbols(symbols: str):
    """一次建置多個 symbol 的 CSV。參數 `symbols` 以逗號分隔，例如 '2330,2317,AAPL'。"""
    if not symbols:
        return JSONResponse({"ok": False, "error": "請提供 symbols 參數"}, status_code=400)
    syms = [s.strip() for s in symbols.split(',') if s.strip()]
    results = {}
    for s in syms:
        try:
            p = _ensure_symbol_csv(s)
            results[s] = {"ok": True, "csv": str(p.resolve())}
        except Exception as e:
            results[s] = {"ok": False, "error": str(e)}
    return {"ok": True, "results": results}


@app.get('/api/list_symbols')
def list_symbols():
    """列出存在於 data/（或可寫目錄）中的 symbol CSV（樣式：<symbol>_short_term_with_lag3.csv）。"""
    out = []
    seen = set()
    try:
        # Prefer files from write dir, then from read-only data dir
        for dir_ in (DATA_WRITE_DIR, DATA.parent):
            if not dir_.exists():
                continue
            for p in dir_.glob('*_short_term_with_lag3.csv'):
                sym = p.stem.replace('_short_term_with_lag3', '').upper()
                if sym in seen:
                    continue
                seen.add(sym)
                out.append({"symbol": sym, "csv": str(p.resolve()), "size": p.stat().st_size})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "count": len(out), "symbols": out}


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

    async def _build_one(s):
        async with sem:
            try:
                # run blocking build in thread
                p = _symbol_csv_write_path(s)
                await asyncio.to_thread(lambda: (_ensure_yf(), _build_from_yfinance(symbol=s, out_csv=p)))
                BULK_TASKS[task_id]['done'] += 1
            except Exception as e:
                BULK_TASKS[task_id]['errors'][s] = str(e)
                BULK_TASKS[task_id]['done'] += 1

    tasks = [asyncio.create_task(_build_one(s)) for s in symbols]
    await asyncio.gather(*tasks)
    BULK_TASKS[task_id]['status'] = 'completed'
    BULK_TASKS[task_id]['finished_at'] = time.time()


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
    BULK_TASKS[task_id] = {"status": "running", "total": len(syms), "done": 0, "errors": {}, "started_at": time.time()}
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


@app.get('/api/auto/start_symbol')
async def auto_start_symbol(symbol: str, interval: int = 5, backoff_factor: float = 2.0, max_backoff: int = 30):
    """啟動單一 symbol 的自動更新迴圈（每隔 `interval` 分鐘執行）。"""
    if not symbol:
        return JSONResponse({"ok": False, "error": "請提供 symbol 參數"}, status_code=400)
    if symbol in SYMBOL_TASKS and not SYMBOL_TASKS[symbol].done():
        return {"ok": True, "status": "already running", "symbol": symbol}
    loop = asyncio.get_event_loop()
    task = loop.create_task(_symbol_loop(symbol, interval, backoff_factor=backoff_factor, max_backoff_min=max_backoff))
    SYMBOL_TASKS[symbol] = task
    # persist
    reg = _load_auto_registry()
    reg[symbol] = {"interval": int(interval), "backoff_factor": float(backoff_factor), "max_backoff": int(max_backoff)}
    _save_auto_registry(reg)
    return {"ok": True, "status": "started", "symbol": symbol, "interval_min": interval, "backoff_factor": backoff_factor, "max_backoff": max_backoff}


@app.get('/api/auto/stop_symbol')
async def auto_stop_symbol(symbol: str):
    """停止單一 symbol 的自動更新 loop，並自 registry 移除。"""
    if not symbol:
        return JSONResponse({"ok": False, "error": "請提供 symbol 參數"}, status_code=400)
    t = SYMBOL_TASKS.get(symbol)
    if t and not t.done():
        t.cancel()
        SYMBOL_TASKS.pop(symbol, None)
        # persist
        reg = _load_auto_registry()
        if symbol in reg:
            reg.pop(symbol, None)
            _save_auto_registry(reg)
        return {"ok": True, "status": "stopped", "symbol": symbol}
    return {"ok": False, "error": "not running", "symbol": symbol}


@app.get('/api/auto/list_registry')
def auto_list_registry():
    """列出目前持久化的自動任務設定（symbol -> interval/backoff 等）。"""
    reg = _load_auto_registry()
    return {"ok": True, "registry": reg}

@app.get('/api/auto/start_index')
async def auto_start_index(index: str, interval: int = 5, concurrency: int = 4, backoff_factor: float = 2.0, max_backoff: int = 30):
    """啟動整個指數（sp500 / nasdaq100 / twse）的自動更新迴圈。
    以單一迴圈管理全部成分，避免啟動數百個個股協程。
    注意：若同時對重疊的 symbols 啟動個股與指數迴圈，可能造成額外負載。
    """
    if not index:
        return JSONResponse({"ok": False, "error": "請提供 index 參數"}, status_code=400)
    ix = index.lower()
    if ix in INDEX_TASKS and not INDEX_TASKS[ix].done():
        return {"ok": True, "status": "already running", "index": ix}
    loop = asyncio.get_running_loop()
    task = loop.create_task(_index_loop(ix, interval, concurrency, backoff_factor=backoff_factor, max_backoff_min=max_backoff))
    INDEX_TASKS[ix] = task
    reg = _load_index_auto_registry()
    reg[ix] = {"interval": int(interval), "concurrency": int(concurrency), "backoff_factor": float(backoff_factor), "max_backoff": int(max_backoff)}
    _save_index_auto_registry(reg)
    return {"ok": True, "status": "started", "index": ix, "interval_min": interval, "concurrency": concurrency, "backoff_factor": backoff_factor, "max_backoff": max_backoff}

@app.get('/api/auto/stop_index')
def auto_stop_index(index: str):
    """停止指數級別自動更新 loop，並自 index registry 移除。"""
    if not index:
        return JSONResponse({"ok": False, "error": "請提供 index 參數"}, status_code=400)
    ix = index.lower()
    t = INDEX_TASKS.get(ix)
    if t and not t.done():
        t.cancel()
        INDEX_TASKS.pop(ix, None)
        reg = _load_index_auto_registry()
        if ix in reg:
            reg.pop(ix, None)
            _save_index_auto_registry(reg)
        return {"ok": True, "status": "stopped", "index": ix}
    return {"ok": False, "error": "not running", "index": ix}

@app.get('/api/auto/list_index')
def auto_list_index():
    """列出記憶體中正在運行的指數迴圈與持久化設定（registry）。"""
    reg = _load_index_auto_registry()
    running = [k for k, v in INDEX_TASKS.items() if not v.done()]
    return {"ok": True, "registry": reg, "running": running}

@app.get('/api/auto/start_existing_csvs')
async def auto_start_existing_csvs(interval: int = 5, backoff_factor: float = 2.0, max_backoff: int = 30):
    """（Feature A）對 data/ 目錄下所有已存在的 *_short_term_with_lag3.csv 啟動個股自動迴圈。

    會枚舉現有 CSV 檔並為每個 symbol 建立迴圈（類似批次呼叫 start_many）。
    警告：若 symbols 過多，將產生大量併發協程；對大型集合建議使用指數迴圈。
    """
    syms = []
    for p in DATA_DIR.glob('*_short_term_with_lag3.csv'):
        sym = p.stem.replace('_short_term_with_lag3', '')
        syms.append(sym)
    syms = list(dict.fromkeys([s for s in syms if s]))
    if not syms:
        return JSONResponse({"ok": False, "error": "找不到任何已存在的 symbol CSV"}, status_code=404)
    reg = _load_auto_registry()
    loop = asyncio.get_running_loop()
    started = {}
    for s in syms:
        if s in SYMBOL_TASKS and not SYMBOL_TASKS[s].done():
            started[s] = "already running"
            reg[s] = {"interval": int(interval), "backoff_factor": float(backoff_factor), "max_backoff": int(max_backoff)}
            continue
        try:
            task = loop.create_task(_symbol_loop(s, interval, backoff_factor=backoff_factor, max_backoff_min=max_backoff))
            SYMBOL_TASKS[s] = task
            reg[s] = {"interval": int(interval), "backoff_factor": float(backoff_factor), "max_backoff": int(max_backoff)}
            started[s] = "started"
        except Exception as e:
            started[s] = f"error: {e}"
    _save_auto_registry(reg)
    return {"ok": True, "count": len(started), "interval_min": interval, "backoff_factor": backoff_factor, "max_backoff": max_backoff, "results": started}


@app.get('/api/auto/start_many')
async def auto_start_many(symbols: str, interval: int = 5):
    """為以逗號分隔的 symbols 啟動個股迴圈並持久化設定。

    本端點為 async，可在 HTTP worker 執行緒觸發時將任務排入事件迴圈。
    """
    syms = [s.strip() for s in symbols.split(',') if s.strip()]
    out = {}
    reg = _load_auto_registry()
    loop = asyncio.get_running_loop()
    for s in syms:
        try:
            if s in SYMBOL_TASKS and not SYMBOL_TASKS[s].done():
                out[s] = {"ok": True, "status": "already running"}
                # preserve any existing advanced config
                prev = reg.get(s)
                if isinstance(prev, dict):
                    prev.update({"interval": int(interval)})
                    reg[s] = prev
                else:
                    reg[s] = {"interval": int(interval), "backoff_factor": 2.0, "max_backoff": 30}
                continue
            task = loop.create_task(_symbol_loop(s, interval))  # default backoff values (legacy)
            SYMBOL_TASKS[s] = task
            reg[s] = {"interval": int(interval), "backoff_factor": 2.0, "max_backoff": 30}
            out[s] = {"ok": True, "status": "started"}
        except Exception as e:
            out[s] = {"ok": False, "error": str(e)}
    _save_auto_registry(reg)
    return {"ok": True, "results": out}


@app.get('/api/auto/stop_many')
def auto_stop_many(symbols: str):
    """一次停止多個 symbol 的自動 loop，並從 registry 中移除相應項目。"""
    syms = [s.strip() for s in symbols.split(',') if s.strip()]
    out = {}
    reg = _load_auto_registry()
    for s in syms:
        t = SYMBOL_TASKS.get(s)
        if t and not t.done():
            try:
                t.cancel()
                SYMBOL_TASKS.pop(s, None)
                if s in reg:
                    reg.pop(s, None)
                out[s] = {"ok": True, "status": "stopped"}
            except Exception as e:
                out[s] = {"ok": False, "error": str(e)}
        else:
            out[s] = {"ok": False, "error": "not running"}
    _save_auto_registry(reg)
    return {"ok": True, "results": out}


@app.get("/api/diagnostics")
def diagnostics(n_bins: int = 20):
    """回傳前端可視化所需的診斷資訊：
    - latest_row：CSV 最新一列（dict）
    - feature_stats：數值欄位的 mean/std/min/max
    - histograms：前幾個數值欄位的直方圖 bins/counts
    - models：已持久化的模型管線檔
    - thresholds：若存在則回傳已載入的閾值
    """
    if not DATA.exists() or DATA.stat().st_size == 0:
        return JSONResponse({"error": "找不到資料 short_term_with_lag3.csv"}, status_code=404)

    try:
        df = pd.read_csv(DATA, encoding="utf-8-sig")
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
    try:
        if symbol:
            csvp = _ensure_symbol_csv(symbol)
            df = pd.read_csv(csvp, encoding='utf-8-sig')
        else:
            if not DATA.exists():
                return JSONResponse({"error": "資料不存在"}, status_code=404)
            df = pd.read_csv(DATA, encoding='utf-8-sig')
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
    try:
        if symbol:
            csvp = _ensure_symbol_csv(symbol)
            df = pd.read_csv(csvp, encoding='utf-8-sig')
        else:
            if not DATA.exists():
                return JSONResponse({"error": "資料不存在"}, status_code=404)
            df = pd.read_csv(DATA, encoding='utf-8-sig')
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
    try:
        if symbol:
            csvp = _ensure_symbol_csv(symbol)
            df = pd.read_csv(csvp, encoding='utf-8-sig')
        else:
            if not DATA.exists():
                return JSONResponse({"error": "資料不存在"}, status_code=404)
            df = pd.read_csv(DATA, encoding='utf-8-sig')
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
    csv_path = DATA
    if symbol:
        try:
            csv_path = _ensure_symbol_csv(symbol)
        except Exception as e:
            return JSONResponse({"error": f"無法取得 symbol CSV：{e}"}, status_code=500)
    elif file:
        p = Path(file)
        if not p.is_absolute():
            p = Path(__file__).parent / file
        try:
            p_res = p.resolve()
            data_dir_res = DATA.parent.resolve()
            if not str(p_res).startswith(str(data_dir_res)):
                return JSONResponse({"error": "file 參數必須位於 data/ 資料夾內"}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": f"檔案解析失敗：{str(e)}"}, status_code=400)
        if not p_res.exists():
            return JSONResponse({"error": f"指定的檔案不存在：{str(p_res)}"}, status_code=404)
        csv_path = p_res

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
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
        data_ready = DATA.exists() and DATA.stat().st_size > 0
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


@app.get('/metrics')
def metrics():
    """Prometheus 指標端點。"""
    _update_gauges()
    data = generate_latest()  # bytes
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# === 快速導覽（路由與背景任務） ===
@app.get('/api/overview')
def api_overview(only_api: bool = True):
    """快速導覽：列出所有路由摘要與背景任務狀態。

    - only_api：僅顯示 /api/*（以及 /health、/metrics、/version、/），預設 True。
    """
    routes = []
    allow_prefixes = ['/', '/health', '/metrics', '/version', '/static', '/api/']
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
    background = {
        "symbol_loops": len([t for t in SYMBOL_TASKS.values() if not t.done()]),
        "index_loops": len([t for t in INDEX_TASKS.values() if not t.done()]),
        "bulk_tasks_running": sum(1 for t in BULK_TASKS.values() if isinstance(t, dict) and t.get('status') == 'running'),
        "bulk_tasks_total": len(BULK_TASKS),
    }
    return {"ok": True, "routes": routes, "background": background}

# === 執行啟動 ===
@app.on_event('startup')
async def _rehydrate_auto_registry():
    """應用啟動時自動恢復（rehydrate）自動任務。

    - 從 auto_registry.json 重啟單股 loop
    - 從 index_auto_registry.json 重啟指數 loop
    """
    # on server startup, read auto_registry.json and start symbol loops
    try:
        reg = _load_auto_registry()
        if reg:
            print(f"[startup] rehydrating auto symbol registry: {reg}")
            loop = asyncio.get_event_loop()
            for sym, cfg in reg.items():
                try:
                    if sym in SYMBOL_TASKS and not SYMBOL_TASKS[sym].done():
                        continue
                    # backward compatibility: int -> interval only
                    if isinstance(cfg, int):
                        interval = int(cfg)
                        bf, mb = 2.0, 30
                    elif isinstance(cfg, dict):
                        interval = int(cfg.get('interval', 5))
                        bf = float(cfg.get('backoff_factor', 2.0))
                        mb = int(cfg.get('max_backoff', 30))
                    else:
                        interval, bf, mb = 5, 2.0, 30
                    task = loop.create_task(_symbol_loop(sym, interval, backoff_factor=bf, max_backoff_min=mb))
                    SYMBOL_TASKS[sym] = task
                except Exception as e:
                    print(f"[startup] failed to start auto for {sym}: {e}")
    except Exception as e:
        print(f"[startup] failed to load auto registry: {e}")
    # Rehydrate index loops
    try:
        ireg = _load_index_auto_registry()
        if ireg:
            print(f"[startup] rehydrating index auto registry: {ireg}")
            loop = asyncio.get_event_loop()
            for ix, cfg in ireg.items():
                if ix in INDEX_TASKS and not INDEX_TASKS[ix].done():
                    continue
                try:
                    interval = int(cfg.get('interval', 5))
                    concurrency = int(cfg.get('concurrency', 4))
                    bf = float(cfg.get('backoff_factor', 2.0))
                    mb = int(cfg.get('max_backoff', 30))
                    task = loop.create_task(_index_loop(ix, interval, concurrency, backoff_factor=bf, max_backoff_min=mb))
                    INDEX_TASKS[ix] = task
                except Exception as e:
                    print(f"[startup] failed to start index auto for {ix}: {e}")
    except Exception as e:
        print(f"[startup] failed to load index auto registry: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)