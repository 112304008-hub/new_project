# test.py — 最終版：固定檔名、UTF-8-BOM、啟動自動建檔、五分鐘自動更新、yfinance 備援
import os, asyncio
from pathlib import Path
from typing import Optional, Iterable, Literal

import requests, pandas as pd, numpy as np
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from zoneinfo import ZoneInfo

# ===== 懶載入 yfinance =====
yf = None
def _ensure_yf():
    global yf
    if yf is None:
        import importlib
        yf = importlib.import_module("yfinance")

# ===== 基本設定 =====
app = FastAPI(title="TSMC CSV Builder")
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTCSV = DATA_DIR / "short_term_with_lag3.csv"
TZ_TW, TZ_US = "Asia/Taipei", "America/New_York"

# ===== 小工具 =====
def _key(override: Optional[str]) -> str:
    # Return key if provided or env var; return None if not set to allow yfinance-only mode
    k = (override or os.getenv("TWELVE_DATA_KEY") or "").strip()
    return k or None

def _td_series(symbol: str, key: str, start="2020-01-01", end=None, tz="America/New_York") -> pd.DataFrame:
    """回傳 [date, close] 升冪"""
    p = {
        "symbol": symbol, "interval": "1day", "order": "asc", "outputsize": 5000,
        "timezone": tz, "apikey": key, "start_date": start
    }
    if end: p["end_date"] = end
    r = requests.get("https://api.twelvedata.com/time_series", params=p, timeout=20)
    r.raise_for_status()
    j = r.json()
    if j.get("status") == "error": raise RuntimeError(j.get("message", "TD 錯誤"))
    vals = j.get("values") or []
    if not vals: raise RuntimeError("TD 空資料")
    df = pd.DataFrame(vals)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)

def _winsor(df: pd.DataFrame, cols: Iterable[str], lo=0.01, hi=0.99) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            ql, qh = df[c].quantile([lo, hi])
            df[c] = df[c].clip(ql, qh)
    return df

def _lags(df: pd.DataFrame, cols: Iterable[str], lags=range(1, 4)) -> pd.DataFrame:
    print("執行 _lags 函數，生成滯後特徵...")
    df = df.copy()
    for c in cols:
        if c in df.columns:
            for L in lags:
                df[f"{c}_lag{L}"] = df[c].shift(L)
    print("滯後特徵生成完成。")
    return df

def _stat_tests(df: pd.DataFrame, target: str, lags: Iterable[int]):
    """對每個滯後欄位進行詳細的統計推論，回傳 list[dict]（JSON-friendly）。

    每個 feature 會包含：n, mean, std, pearson_r, pearson_p, linregress(slope/intercept/r/p/stderr),
    shapiro_stat/p（若樣本數 <= 5000）、ADF 統計量與 p-value（若可算）。
    """
    print("執行 _stat_tests（詳細）...")
    results = []
    import math
    try:
        from scipy import stats
    except Exception:
        stats = None
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:
        adfuller = None

    def safe_float(x):
        try:
            if x is None:
                return None
            if isinstance(x, (float, int)):
                if isinstance(x, float) and math.isnan(x):
                    return None
                return float(x)
            return float(x)
        except Exception:
            return None

    for col in df.columns:
        if any(f"_lag{L}" in col for L in lags):
            pair = df[[target, col]].dropna()
            if pair.empty:
                continue
            x = pair[target].values
            y = pair[col].values
            n = int(len(pair))
            mean_y = safe_float(np.mean(y))
            std_y = safe_float(np.std(y, ddof=1))

            pearson_r = pearson_p = None
            if stats is not None:
                try:
                    pearson_r, pearson_p = stats.pearsonr(y, x)
                    pearson_r = safe_float(pearson_r)
                    pearson_p = safe_float(pearson_p)
                except Exception:
                    pearson_r = pearson_p = None

            lr_slope = lr_intercept = lr_r = lr_p = lr_stderr = None
            if stats is not None:
                try:
                    lr = stats.linregress(y, x)
                    lr_slope = safe_float(lr.slope)
                    lr_intercept = safe_float(lr.intercept)
                    lr_r = safe_float(lr.rvalue)
                    lr_p = safe_float(lr.pvalue)
                    lr_stderr = safe_float(lr.stderr)
                except Exception:
                    lr_slope = lr_intercept = lr_r = lr_p = lr_stderr = None

            shapiro_stat = shapiro_p = None
            if stats is not None:
                try:
                    # Shapiro-Wilk has sample size limitation; only run for reasonable sizes
                    if n <= 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(y)
                        shapiro_stat = safe_float(shapiro_stat)
                        shapiro_p = safe_float(shapiro_p)
                except Exception:
                    shapiro_stat = shapiro_p = None

            adf_stat = adf_p = adf_usedlag = adf_nobs = None
            if adfuller is not None:
                try:
                    adf_res = adfuller(y, autolag='AIC')
                    adf_stat = safe_float(adf_res[0])
                    adf_p = safe_float(adf_res[1])
                    adf_usedlag = int(adf_res[2])
                    adf_nobs = int(adf_res[3])
                except Exception:
                    adf_stat = adf_p = adf_usedlag = adf_nobs = None

            proc = []
            proc.append(f"n={n}")
            proc.append(f"mean={mean_y:.6g}" if mean_y is not None else "mean=None")
            proc.append(f"std={std_y:.6g}" if std_y is not None else "std=None")
            if pearson_r is not None:
                proc.append(f"pearson_r={pearson_r:.6g}, p={pearson_p:.3g}")
            if lr_slope is not None:
                proc.append(f"linreg slope={lr_slope:.6g}, p={lr_p:.3g}")
            if shapiro_stat is not None:
                proc.append(f"shapiro W={shapiro_stat:.6g}, p={shapiro_p:.3g}")
            if adf_stat is not None:
                proc.append(f"ADF stat={adf_stat:.6g}, p={adf_p:.3g}, usedlag={adf_usedlag}")

            results.append({
                "feature": col,
                "n": n,
                "mean": mean_y,
                "std": std_y,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "linreg_slope": lr_slope,
                "linreg_intercept": lr_intercept,
                "linreg_r": lr_r,
                "linreg_p": lr_p,
                "linreg_stderr": lr_stderr,
                "shapiro_stat": shapiro_stat,
                "shapiro_p": shapiro_p,
                "adf_stat": adf_stat,
                "adf_p": adf_p,
                "adf_usedlag": adf_usedlag,
                "adf_nobs": adf_nobs,
                "process": proc
            })

    print("統計檢定（詳細）完成，共檢測 %d 個特徵。" % len(results))
    # Sort by absolute Pearson r (desc) when available, else by feature name
    try:
        results.sort(key=lambda r: abs(r.get("pearson_r") or 0), reverse=True)
    except Exception:
        pass
    return results

# ===== 資料來源（含備援） =====
def get_ohlcv(symbol: str, key: str, start="2020-01-01", end=None) -> pd.DataFrame:
    """Return ohlcv DataFrame with columns date, open, high, low, close, volume.
    symbol: e.g. '2330' or '2330:TWSE' or 'MSFT' (TwelveData friendly)."""
    td_symbol = symbol
    if symbol.isdigit():
        td_symbol = f"{symbol}:TWSE"
    # If a TwelveData key is provided, try TD first; otherwise use yfinance directly
    if key:
        try:
            p = {
                "symbol": td_symbol,
                "interval": "1day",
                "order": "asc",
                "outputsize": 5000,
                "timezone": TZ_TW if symbol.isdigit() else TZ_US,
                "apikey": key,
                "start_date": start,
            }
            if end:
                p["end_date"] = end
            r = requests.get("https://api.twelvedata.com/time_series", params=p, timeout=20)
            r.raise_for_status()
            j = r.json()
            if j.get("status") == "error":
                raise RuntimeError(j.get("message", "TD 錯誤"))
            vals = j.get("values") or []
            if not vals:
                raise RuntimeError("TD 空資料")
            df = pd.DataFrame(vals)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
            return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception:
            # fall through to yfinance
            pass
    # yfinance fallback or no key provided
    _ensure_yf()
    yf_sym = symbol if not symbol.isdigit() else f"{symbol}.TW"
    hist = yf.Ticker(yf_sym).history(start=start, end=end, interval="1d", auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance 取回 {yf_sym} 為空")
    hist = hist.reset_index()
    hist["date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
    return hist.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    })[["date","open","high","low","close","volume"]]

def get_usdtwd(key: str, start="2020-01-01", end=None) -> pd.DataFrame:
    try:
        fx = _td_series("USD/TWD", key, start=start, end=end, tz=TZ_TW).rename(columns={"close": "USDTWD"})
        return fx
    except Exception:
        # fallback: yfinance
        _ensure_yf()
        hist = yf.download("TWD=X", start=start, end=end, interval="1d", progress=False)
        if hist is None or hist.empty:
            raise RuntimeError("yfinance TWD=X 為空")
        hist = hist.reset_index()
        hist["date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
        hist["USDTWD"] = pd.to_numeric(hist["Close"], errors="coerce")
        return hist[["date", "USDTWD"]]

# ===== 建通用 symbol 特徵 =====
def build_symbol(symbol: str, key: str, start="2020-01-01", end=None,
                 vol_mode: Literal["log", "rel", "raw", "both"] = "log") -> pd.DataFrame:
    ohlcv = get_ohlcv(symbol, key, start, end).copy()
    df = ohlcv.rename(columns={
        "date": "年月日","open":"開盤價(元)","high":"最高價(元)","low":"最低價(元)",
        "close":"收盤價(元)","volume":"成交量(千股)"
    })
    # normalize volume to 千股 if volume present
    if "成交量(千股)" in df.columns:
        df["成交量(千股)"] = df["成交量(千股)"]/1000.0
    df["年月日"] = pd.to_datetime(df["年月日"])

    close = df["收盤價(元)"]
    prev = close.shift(1)
    df["報酬率％"] = close.pct_change()*100
    df["Gap％"]    = (df["開盤價(元)"]/prev-1)*100
    df["振幅％"]   = ((df["最高價(元)"]-df["最低價(元)"])/prev)*100
    for n in (5,10,20): df[f"MA{n}"] = close.rolling(n).mean()
    df["MA5差％"]  = (close/df["MA5"]-1)*100
    df["MA20差％"] = (close/df["MA20"]-1)*100
    df["週幾"]     = df["年月日"].dt.weekday
    if vol_mode in ("log","both") and "成交量(千股)" in df:
        df["成交量(千股)_log"] = np.log1p(df["成交量(千股)"])
    if vol_mode in ("rel","both") and "成交量(千股)" in df:
        med20 = df["成交量(千股)"].rolling(20).median()
        df["成交量(千股)_rel20％"] = (df["成交量(千股) "]/med20-1)*100

    # 去除極端值
    df = _winsor(df, ["報酬率％","Gap％","振幅％","MA5差％","MA20差％",
                      "成交量(千股)_rel20％"] if "成交量(千股)_rel20％" in df else
                      ["報酬率％","Gap％","振幅％","MA5差％","MA20差％"])
    # 滯後特徵
    lag_cols = ["收盤價(元)","成交量(千股)","報酬率％","Gap％","振幅％","MA5差％","MA20差％"]
    if "成交量(千股)_log" in df: lag_cols.append("成交量(千股)_log")
    if "成交量(千股)_rel20％" in df: lag_cols.append("成交量(千股)_rel20％")
    df = _lags(df, lag_cols, range(1, 6))  # 增加滯後範圍到 5

    # 統計檢定
    stats = _stat_tests(df, target="收盤價(元)", lags=range(1, 6))
    print("統計檢定結果：")
    print(stats)

    df = df.dropna().reset_index(drop=True)
    df["年月日"] = df["年月日"].dt.strftime("%Y-%m-%d")
    return df


# backward compatibility wrapper
def build_2330(key: str, start="2020-01-01", end=None, vol_mode: Literal["log", "rel", "raw", "both"] = "log") -> pd.DataFrame:
    return build_symbol("2330", key, start=start, end=end, vol_mode=vol_mode)

# ===== 併外生因子 =====
def enrich(df: pd.DataFrame, key: str) -> pd.DataFrame:
    out = df.copy()
    # TSM ADR
    tsm = _td_series("TSM", key, tz=TZ_US).rename(columns={"close":"TSM_ADR_close"})
    tsm["TSM_ADR_ret1d"] = tsm["TSM_ADR_close"].pct_change()
    out = out.merge(tsm, left_on="年月日", right_on="date", how="left").drop(columns=["date"])
    out[["TSM_ADR_close","TSM_ADR_ret1d"]] = out[["TSM_ADR_close","TSM_ADR_ret1d"]].ffill()
    # USD/TWD
    fx = get_usdtwd(key).copy()
    fx["USDTWD_ret1d"] = fx["USDTWD"].pct_change()
    out = out.merge(fx, left_on="年月日", right_on="date", how="left").drop(columns=["date"])
    out[["USDTWD","USDTWD_ret1d"]] = out[["USDTWD","USDTWD_ret1d"]].ffill()
    # ADR gap％
    out["TSM_ADR_gap％"] = ((1+out["TSM_ADR_ret1d"])*(1+out["USDTWD_ret1d"])-1)*100
    return out

# ===== API =====

def _write_timestamp_file(path: Path):
    try:
        ts_path = path.parent / "last_update.txt"
        with ts_path.open("w", encoding="utf-8") as f:
            from datetime import datetime
            f.write(datetime.now(TAIPEI).isoformat())
        print(f"[更新時間檔] 已寫入 {ts_path}")
    except Exception as e:
        print(f"[警告] 無法寫入時間檔：{e}")


def perform_update(key: str, symbol: Optional[str] = None) -> dict:
    """執行一次完整的資料建置、併入外生因子，並寫入 CSV 與 timestamp 檔。回傳摘要字典。"""
    print("perform_update: 開始建立資料...")
    k = _key(key)
    # decide symbol
    sym = symbol or "2330"
    base = build_symbol(sym, k)
    full = enrich(base, k)
    # decide output path
    outpath = OUTCSV if sym == "2330" else (DATA_DIR / f"{sym}_short_term_with_lag3.csv")
    full.to_csv(outpath, index=False, encoding="utf-8-sig")
    _write_timestamp_file(outpath)

    # 統計檢定
    stats = _stat_tests(full, target="收盤價(元)", lags=range(1, 6))
    # stats is already a list[dict]
    stats_json = stats

    # explicit lag features
    lag_features = [c for c in full.columns if 'lag' in c.lower()]

    summary = {
        "ok": True,
        "saved_to": str(OUTCSV),
        "rows": len(full),
        "cols": list(full.columns),
        "lag_features": lag_features,
        "stats": stats_json
    }
    print("perform_update: 完成。", summary)
    return summary


@app.get("/api/quick")
def api_quick(key: Optional[str] = None, symbol: Optional[str] = None):
    try:
        print("/api/quick 被呼叫")
        k = _key(key)
        summary = perform_update(k, symbol=symbol)
        print("/api/quick 執行完成，返回結果。")
        return summary
    except Exception as e:
        print(f"/api/quick 發生錯誤：{e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', '-s', help='Single Symbol to build (e.g. 2330 or MSFT).', default=None)
    p.add_argument('--symbols', help='Comma-separated symbols to batch build (e.g. 2330,MSFT,AAPL).', default=None)
    p.add_argument('--key', '-k', help='Twelve Data API key or set TWELVE_DATA_KEY env var', default=None)
    args = p.parse_args()
    k = args.key or os.getenv('TWELVE_DATA_KEY')
    # if symbols provided, run batch
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
        summary = {"ok": True, "results": {}}
        for s in syms:
            try:
                print(f"Building {s}...")
                res = perform_update(k, symbol=s)
                summary["results"][s] = {"ok": True, "saved_to": res.get("saved_to" if "saved_to" in res else str(DATA_DIR / f"{s}_short_term_with_lag3.csv")), "rows": res.get("rows")}
            except Exception as e:
                summary["results"][s] = {"ok": False, "error": str(e)}
        ts = int(time.time())
        outp = DATA_DIR / f"batch_{ts}_summary.json"
        with outp.open('w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print('Wrote batch summary to', outp)
        print(summary)
        raise SystemExit(0)
    else:
        # single symbol fallback (default behavior)
        print('Running perform_update for single symbol (or default 2330)')
        summary = perform_update(k, symbol=args.symbol)
        import json
        print(json.dumps(summary, ensure_ascii=False, indent=2))

@app.get("/download")
def download():
    return FileResponse(OUTCSV, media_type="text/csv", filename=OUTCSV.name) if OUTCSV.exists() \
           else JSONResponse({"ok": False, "error": "CSV 尚未產生"}, status_code=404)

@app.get("/api/where")
def where():
    return {"data_dir": str(DATA_DIR.resolve()), "outcsv": str(OUTCSV.resolve()), "exists": OUTCSV.exists()}

# ===== 自動更新 =====
TAIPEI = ZoneInfo("Asia/Taipei"); AUTO_TASK: asyncio.Task|None = None
async def _loop(key: str, interval_min: int):
    print(f"[auto loop] 啟動，interval_min={interval_min}")
    while True:
        try:
            print(f"[auto loop] 執行更新（下一次在 {interval_min} 分鐘後）...")
            # 使用 perform_update 以取得明確的回傳與時間檔
            await asyncio.to_thread(lambda: perform_update(key))
        except Exception as e:
            print(f"[auto loop] 更新時發生錯誤：{e}")
        await asyncio.sleep(max(1, int(interval_min))*60)

@app.get("/api/auto/start")
async def auto_start(interval: int = 5, key: Optional[str] = None):
    global AUTO_TASK
    if AUTO_TASK and not AUTO_TASK.done():
        return {"ok": True, "status": "already running"}
    k = _key(key); AUTO_TASK = asyncio.get_event_loop().create_task(_loop(k, interval))
    return {"ok": True, "status": "started", "interval_min": interval}

@app.get("/api/auto/stop")
async def auto_stop():
    global AUTO_TASK
    if AUTO_TASK and not AUTO_TASK.done():
        AUTO_TASK.cancel(); AUTO_TASK = None; return {"ok": True, "status": "stopped"}
    return {"ok": False, "error": "not running"}

# ===== 啟動時自動建檔 =====
@app.on_event("startup")
async def startup_event():
    try:
        k = _key(None)  # 用環境變數金鑰
        print("[啟動事件] 嘗試在啟動時建立資料檔")
        summary = await asyncio.to_thread(lambda: perform_update(k))
        print(f"[啟動完成] 已建立 {OUTCSV}；summary: {summary}")
    except Exception as e:
        print(f"[啟動警告] 無法自動建立 CSV：{e}")

# ===== 啟動 =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test:app", host="0.0.0.0", port=8000, reload=True)