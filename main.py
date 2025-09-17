# main.py — Main application script for the FastAPI application
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from stock import predict, MODELS_DIR
import joblib
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.tsa.stattools as tsstat
import re

app = FastAPI(title="股價之神")

# === 路徑設定 ===
ROOT = Path(__file__).parent
HTML = ROOT / "template2.html"
DATA = ROOT / "data" / "short_term_with_lag3.csv"

# serve any static assets placed in the project static/ directory (e.g. static/temple.jpg)
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
    return FileResponse(HTML) if HTML.exists() else JSONResponse({"error": "template2.html 不存在"}, status_code=404)

# === API: 抽籤（預測）=== 
@app.get("/api/draw")
def draw(model: str = "rf"):
    if not DATA.exists() or DATA.stat().st_size == 0:
        return JSONResponse({
            "error": "找不到有效的 short_term_with_lag3.csv，請先執行 test.py 建立資料",
            "fortune": {
                "title": "找不到資料",
                "text": ["請先執行 test.py 建立最新特徵資料。"],
                "advice": "請確認 test.py 有啟動成功並產生 CSV。"
            }
        }, status_code=404)

    # Only allow inference: require pre-trained pipeline and threshold to exist
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
        # artifacts exist -> safe to call predict (it will only load & infer)
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
        return {
            "ok": True,
            "model": out["model"],
            "threshold": out["threshold"],
            "proba": out["proba"],
            "label": out["label"],
            "fortune": fortune
        }
    except Exception as e:
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


@app.get("/api/diagnostics")
def diagnostics(n_bins: int = 20):
    """Return diagnostic information for frontend visualization:
    - latest_row: latest CSV row as dict
    - feature_stats: mean/std/min/max for numeric columns
    - histograms: simple histogram bins/counts for first few numeric features
    - models: list of persisted model pipeline files
    - thresholds: loaded thresholds if present
    """
    if not DATA.exists() or DATA.stat().st_size == 0:
        return JSONResponse({"error": "找不到資料 short_term_with_lag3.csv"}, status_code=404)

    try:
        df = pd.read_csv(DATA, encoding="utf-8-sig")
    except Exception as e:
        return JSONResponse({"error": f"讀取 CSV 失敗：{str(e)}"}, status_code=500)

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
def stat_tests(feature: str):
    """Run Shapiro normality, one-sample t-test (H0: mean=0), and ADF for the given numeric feature."""
    if not DATA.exists():
        return JSONResponse({"error": "資料不存在"}, status_code=404)
    df = pd.read_csv(DATA, encoding='utf-8-sig')
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
def lag_stats():
    """Compute detailed statistical inference for all lag features in the CSV.

    Returns a list of results for each lag feature (pearson r/p, linregress, shapiro, t-test, ADF).
    """
    if not DATA.exists():
        return JSONResponse({"error": "資料不存在"}, status_code=404)
    try:
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
def series(feature: str, n: int = 500):
    """Return last n values of a numeric feature for plotting."""
    if not DATA.exists():
        return JSONResponse({"error": "資料不存在"}, status_code=404)
    df = pd.read_csv(DATA, encoding='utf-8-sig')
    if feature not in df.columns:
        return JSONResponse({"error": f"找不到欄位 {feature}"}, status_code=404)
    arr = pd.to_numeric(df[feature], errors='coerce').dropna()
    vals = arr.tail(n).tolist()
    return {"feature": feature, "values": vals}


@app.get('/api/latest_features')
def latest_features(features: str | None = None, pattern: str | None = None, file: str | None = None, max_items: int = 100):
    """Return the latest row (or a selection of columns) from the CSV as JSON.

    - features: comma-separated column names to include (order preserved)
    - pattern: a regex to select column names (case-insensitive)
    - file: optional path (relative to project root) but must live under the data/ folder
    - max_items: maximum number of columns to return (safety)
    """
    # decide which CSV to read (restrict file to data/ folder)
    csv_path = DATA
    if file:
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

# === 執行啟動 ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)