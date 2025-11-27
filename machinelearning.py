"""machinelearning.py — 機器學習模型訓練與推論核心模組

從 stock.py 拆分出來的機器學習相關功能，便於獨立維護和更新。
包含模型訓練、閾值優化、特徵處理等核心邏輯。
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Dict, Any, Optional, Union
import json

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# ====== 超參數 ======
THRESH = 0.01
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

# 路徑設定
DEFAULT_CSV = Path(__file__).parent / "data" / "short_term_with_lag3.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.json"

# ====== 工具函式 ======
def _resolve_csv_path(csv_path: Optional[Union[str, Path]]) -> Path:
    """解析並驗證 CSV 路徑"""
    p = Path(csv_path) if csv_path else DEFAULT_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"找不到資料檔：{p.resolve()}\n"
            "請先在後端呼叫 /api/quick 或 /api/auto/start 生成 CSV。"
        )
    return p

def _build_y(df: pd.DataFrame, thresh: float | None) -> pd.Series:
    """建構標籤 y：根據明天收盤價相對今天的漲跌"""
    df = df.copy()
    df["明天收盤價"] = df["收盤價(元)"].shift(-1)
    if thresh is None:
        y = (df["明天收盤價"] > df["收盤價(元)"]).astype(float)
    else:
        ret1 = (df["明天收盤價"] - df["收盤價(元)"]) / df["收盤價(元)"]
        y = ret1.apply(lambda x: 1 if x > thresh else (0 if x < -thresh else np.nan))
    return y

def _best_threshold(model, X_val, y_val, grid=np.linspace(0.30, 0.70, 41)) -> tuple[float, float]:
    """尋找最佳機率閾值以最大化 F1 分數"""
    p = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_hat = (p >= t).astype(int)
        f1w = f1_score(y_val, y_hat, average="weighted")
        if f1w > best_f1:
            best_f1, best_t = f1w, t
    return best_t, best_f1

def _infer_symbol_from_path(path: Union[str, Path]) -> Optional[str]:
    """從檔案路徑推斷股票代碼"""
    try:
        p = Path(path)
        name = p.stem
        # expected format: <symbol>_short_term_with_lag3
        if name.endswith("_short_term_with_lag3"):
            return name.replace("_short_term_with_lag3", "")
    except Exception:
        pass
    return None

def _ensure_td():
    """延遲載入 twelvedata 模組"""
    global td
    try:
        from twelvedata import TDClient
        globals()['td'] = TDClient(apikey="38faf585444e44c7b076b781e8912d25")
    except Exception:
        import importlib
        td_module = importlib.import_module('twelvedata')
        globals()['td'] = td_module.TDClient(apikey="38faf585444e44c7b076b781e8912d25")

def _fetch_ohlcv_td(symbol: str, start: str = "2020-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """使用 twelvedata 獲取 OHLCV 數據"""
    _ensure_td()
    s = symbol.strip()
    # if symbol is digits only, treat as TW ticker
    if s.isdigit():
        ticker = f"{s}.TW"
    else:
        ticker = s
    
    client = globals().get('td')
    ts = client.time_series(
        symbol=ticker,
        interval="1day",
        outputsize=5000,
        start_date=start,
        end_date=end
    )
    hist = ts.as_pandas()
    
    if hist is None or hist.empty:
        raise RuntimeError(f"twelvedata 無法取得 {ticker}")
    
    hist = hist.reset_index()
    hist['date'] = pd.to_datetime(hist['datetime']).dt.strftime('%Y-%m-%d')
    hist = hist.rename(columns={
        'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
    })
    return hist[['date','open','high','low','close','volume']].sort_values('date').reset_index(drop=True)

def _build_from_td(symbol: str, out_csv: Path, start: str = "2020-01-01") -> pd.DataFrame:
    """使用 twelvedata 為指定股票建立特徵並保存 CSV"""
    df_ohlcv = _fetch_ohlcv_td(symbol, start=start)
    df = df_ohlcv.rename(columns={
        'date': '年月日','open':'開盤價(元)','high':'最高價(元)','low':'最低價(元)',
        'close':'收盤價(元)','volume':'成交量(千股)'
    }).copy()
    # twelvedata volume is raw shares; convert to thousands
    df['成交量(千股)'] = pd.to_numeric(df['成交量(千股)'], errors='coerce')/1000.0
    df['年月日'] = pd.to_datetime(df['年月日'])

    close = df['收盤價(元)']; prev = close.shift(1)
    df['報酬率％'] = close.pct_change()*100
    df['Gap％']    = (df['開盤價(元)']/prev-1)*100
    df['振幅％']   = ((df['最高價(元)']-df['最低價(元)'])/prev)*100
    for n in (5,10,20): df[f'MA{n}'] = close.rolling(n).mean()
    df['MA5差％']  = (close/df['MA5']-1)*100
    df['MA20差％'] = (close/df['MA20']-1)*100
    df['週幾']     = df['年月日'].dt.weekday
    df['成交量(千股)_log'] = np.log1p(df['成交量(千股)'])
    med20 = df['成交量(千股)'].rolling(20).median()
    df['成交量(千股)_rel20％'] = (df['成交量(千股)']/med20-1)*100

    # winsor and lags
    def _winsor_local(df_local, cols, lo=0.01, hi=0.99):
        out = df_local.copy()
        for c in cols:
            if c in out.columns:
                ql, qh = out[c].quantile([lo, hi])
                out[c] = out[c].clip(ql, qh)
        return out

    df = _winsor_local(df, ['報酬率％','Gap％','振幅％','MA5差％','MA20差％','成交量(千股)_rel20％'])
    lag_cols = ['收盤價(元)','成交量(千股)','報酬率％','Gap％','振幅％','MA5差％','MA20差％','成交量(千股)_log','成交量(千股)_rel20％']
    for c in lag_cols:
        if c in df.columns:
            for L in range(1,6):
                df[f"{c}_lag{L}"] = df[c].shift(L)

    df = df.dropna().reset_index(drop=True)
    df['年月日'] = df['年月日'].dt.strftime('%Y-%m-%d')

    # attempt to enrich with USD/TWD and TSM ADR
    try:
        client = globals().get('td')
        fx_ts = client.time_series(symbol='USD/TWD', interval='1day', outputsize=5000, start_date=start)
        fx = fx_ts.as_pandas()
        fx = fx.reset_index(); fx['date'] = pd.to_datetime(fx['datetime']).dt.strftime('%Y-%m-%d'); fx['USDTWD'] = pd.to_numeric(fx['close'], errors='coerce'); fx = fx[['date','USDTWD']]
        df = df.merge(fx, left_on='年月日', right_on='date', how='left').drop(columns=['date'])
        df[['USDTWD']] = df[['USDTWD']].ffill()
    except Exception:
        pass

    try:
        client = globals().get('td')
        adr_ts = client.time_series(symbol='TSM', interval='1day', outputsize=5000, start_date=start)
        adr = adr_ts.as_pandas()
        if adr is not None and not adr.empty:
            adr = adr.reset_index(); adr['date'] = pd.to_datetime(adr['datetime']).dt.strftime('%Y-%m-%d'); adr['TSM_ADR_close'] = pd.to_numeric(adr['close'], errors='coerce'); adr = adr[['date','TSM_ADR_close']]
            df = df.merge(adr, left_on='年月日', right_on='date', how='left').drop(columns=['date'])
            df[['TSM_ADR_close']] = df[['TSM_ADR_close']].ffill()
            df['TSM_ADR_ret1d'] = df['TSM_ADR_close'].pct_change()
            if 'USDTWD' in df.columns and 'TSM_ADR_ret1d' in df.columns:
                df['USDTWD_ret1d'] = df['USDTWD'].pct_change()
                df['TSM_ADR_gap％'] = ((1+df['TSM_ADR_ret1d'])*(1+df['USDTWD_ret1d'])-1)*100
    except Exception:
        pass

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return df

# ====== 主要預測函式 ======
def predict(csv_path: Optional[Union[str, Path]] = None,
            model: Literal["rf", "lr"] = "rf",
            symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    讀取 CSV 資料，訓練或載入模型，回傳最後一列對明天的預測。
    
    參數：
        csv_path: CSV 檔案路徑（可選）
        model: 模型類型 "rf" 或 "lr"
        symbol: 股票代碼（可選）
    
    回傳：
        {"label": "漲/跌", "proba": float, "threshold": float, "model": "rf/lr", "csv": "路徑", "symbol": str}
    """
    # If a symbol is provided, prefer symbol-specific CSV
    s = None
    if symbol:
        s = str(symbol).strip() or None

    # Determine model pipeline location
    model_name = f"{model}_pipeline.pkl"
    thr_name = f"{model}_threshold.pkl"
    model_path = MODELS_DIR / model_name
    thr_path = MODELS_DIR / thr_name

    expected = None
    # If pipeline exists, load it early to infer expected features
    if model_path.exists() and thr_path.exists():
        pipeline: Pipeline = joblib.load(model_path)
        # Determine the feature names the pipeline was trained on
        for step_name in ("scaler", "rf", "lr", "mdl"):
            if step_name in getattr(pipeline, 'named_steps', {}):
                step = pipeline.named_steps[step_name]
                if hasattr(step, 'feature_names_in_'):
                    expected = list(step.feature_names_in_)
                    break
        if expected is None and hasattr(pipeline, 'named_steps'):
            for step in pipeline.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    expected = list(step.feature_names_in_)
                    break
        if expected is None and hasattr(pipeline, 'feature_names_in_'):
            expected = list(pipeline.feature_names_in_)
        if expected is None and hasattr(pipeline, 'n_features_in_'):
            n = int(pipeline.n_features_in_)
            numeric_cols = None
            try:
                tmp_df = pd.read_csv(DEFAULT_CSV, encoding='utf-8-sig')
                numeric_cols = tmp_df.select_dtypes(include=[np.number]).columns.tolist()
            except Exception:
                numeric_cols = []
            expected = numeric_cols[-n:] if numeric_cols else None

        # persist expected feature names for reuse
        try:
            if expected:
                FEATURE_NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
                existing = None
                if FEATURE_NAMES_FILE.exists():
                    try:
                        existing = json.load(open(FEATURE_NAMES_FILE, 'r', encoding='utf-8'))
                    except Exception:
                        existing = None
                if existing != expected:
                    json.dump(expected, open(FEATURE_NAMES_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        except Exception:
            pass

    # If symbol requested, prepare symbol CSV path and build if missing
    if s:
        symbol_csv = DATA_DIR / f"{s}_short_term_with_lag3.csv"
        if csv_path is None:
            csv_path = symbol_csv
        if not Path(csv_path).exists():
            try:
                _ensure_td()
                _build_from_td(symbol=s, out_csv=Path(csv_path))
            except Exception:
                pass

    path = _resolve_csv_path(csv_path)
    df_raw = pd.read_csv(path, encoding="utf-8-sig")
    if "收盤價(元)" not in df_raw.columns:
        raise ValueError("找不到欄位『收盤價(元)』")

    # 1) 建 y，同時保留 X_all 供最後一列預測
    df = df_raw.copy()
    y_all = _build_y(df, THRESH)
    df["y_final"] = y_all
    drop_cols = ["年月日", "y_明天漲跌", "明天收盤價", "y_final"]
    X_all = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    # 轉數值
    for c in X_all.columns:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

    # 只取有 y 的列做訓練
    train_idx = df.index[df["y_final"].notna()]
    X_train_all_raw = X_all.loc[train_idx].reset_index(drop=True)
    y_train_all = df.loc[train_idx, "y_final"].astype(int).reset_index(drop=True)

    # 2) 時間切分
    split_idx = int(len(X_train_all_raw) * (1 - TEST_SPLIT))
    X_tr_raw, X_te_raw = X_train_all_raw.iloc[:split_idx].copy(), X_train_all_raw.iloc[split_idx:].copy()
    y_tr, y_te = y_train_all.iloc[:split_idx], y_train_all.iloc[split_idx:]

    # 缺值以訓練集中位數補
    medians = X_tr_raw.median(numeric_only=True)
    X_tr = X_tr_raw.fillna(medians)
    X_te = X_te_raw.fillna(medians)

    # 標準化
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # 3) 切驗證找閾值
    val_idx = int(len(X_tr_s) * (1 - VAL_SPLIT))
    X_core, X_val = X_tr_s[:val_idx], X_tr_s[val_idx:]
    y_core, y_val = y_tr.iloc[:val_idx], y_tr.iloc[val_idx:]

    # Check if a persisted model exists to avoid retraining
    if model_path.exists() and thr_path.exists():
        # load pipeline and threshold
        pipeline: Pipeline = joblib.load(model_path)
        thr = joblib.load(thr_path)

        # Determine the feature names the pipeline was trained on (again for safety)
        expected2 = None
        for step_name in ("scaler", "rf", "lr", "mdl"):
            if step_name in getattr(pipeline, 'named_steps', {}):
                step = pipeline.named_steps[step_name]
                if hasattr(step, 'feature_names_in_'):
                    expected2 = list(step.feature_names_in_)
                    break
        if expected2 is None and hasattr(pipeline, 'named_steps'):
            for step in pipeline.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    expected2 = list(step.feature_names_in_)
                    break
        if expected2 is None and hasattr(pipeline, 'feature_names_in_'):
            expected2 = list(pipeline.feature_names_in_)
        if expected2 is None and hasattr(pipeline, 'n_features_in_'):
            n = int(pipeline.n_features_in_)
            numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
            expected2 = numeric_cols[-n:]

        expected = expected2 or expected

        # persist expected feature names again (ensure file exists)
        try:
            if expected:
                FEATURE_NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
                json.dump(expected, open(FEATURE_NAMES_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        except Exception:
            pass

        # ensure expected columns exist in current CSV; if missing, create and fill with 0 or median
        missing = [c for c in expected if c not in X_all.columns]
        if missing:
            for c in missing:
                # add missing column with zeros (numeric) to avoid pipeline errors
                X_all[c] = 0.0

        # select and fill using medians for safety
        x_latest = X_all[expected].iloc[[-1]].fillna(medians)
        p = pipeline.predict_proba(x_latest)[:, 1][0]
        yhat = 1 if p >= thr else 0
    else:
        # 4) 建兩個模型並找最佳閾值，然後 persist
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=20, min_samples_split=10,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
        )

        # fit on core
        lr.fit(X_core, y_core)
        rf.fit(X_core, y_core)

        lr_t, _ = _best_threshold(lr, X_val, y_val)
        rf_t, _ = _best_threshold(rf, X_val, y_val)

        # retrain on full train set (core + val)
        lr.fit(X_tr_s, y_tr)
        rf.fit(X_tr_s, y_tr)

        # build pipelines that include scaler so inference can accept raw DataFrame
        lr_pipeline = Pipeline([("scaler", scaler), ("lr", lr)])
        rf_pipeline = Pipeline([("scaler", scaler), ("rf", rf)])

        if model == "rf":
            joblib.dump(rf_pipeline, model_path)
            joblib.dump(rf_t, thr_path)
            pipeline = rf_pipeline
            thr = rf_t
        else:
            joblib.dump(lr_pipeline, model_path)
            joblib.dump(lr_t, thr_path)
            pipeline = lr_pipeline
            thr = lr_t

        x_latest = X_all.iloc[[-1]].fillna(medians)
        p = pipeline.predict_proba(x_latest)[:, 1][0]
        yhat = 1 if p >= thr else 0

    return {
        "label": "漲" if yhat == 1 else "跌",
        "proba": float(p),
        "threshold": float(thr),
        "model": model,
        "csv": str(path.resolve()),
        "symbol": s if s else _infer_symbol_from_path(path)
    }