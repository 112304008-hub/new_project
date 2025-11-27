"""stock.py — 模型訓練與推論核心模組 (Traditional Chinese 說明)

功能總覽：
1. 單次訓練 / 推論：提供 predict(), predict_latest(), train() 等函式。
2. 支援兩種模型：
     - 隨機森林 (rf)
     - 邏輯迴歸 (lr)
3. 動態選擇與保存最佳機率閾值 (threshold) 以提升分類 F1 (weighted) 表現。
4. 支援符號 (symbol) 對應的獨立特徵 CSV 自動建構（若缺失則嘗試使用 twelvedata 抓取歷史價量）。
5. 以 Pipeline (scaler + model) 保存，利於後續推論不需重新擬合尺度。
6. 預期特徵名稱會保存於 feature_names.json，推論時若缺失欄位會自動補 0.0 避免錯誤。

CLI 使用方式：
    1) 直接預測（若已存在模型檔）：
             python stock.py --model rf
    2) 指定輸入 CSV：
             python stock.py --csv path/to/file.csv --model lr
    3) 訓練並儲存模型到 models/：
             python stock.py --model all --train   # 會訓練 rf + lr 兩者

注意：
    核心機器學習邏輯已拆分至 machinelearning.py，本檔案作為入口和封裝。
"""
# 用法1：python stock.py --model rf         # 直接讀 .\data\short_term_with_lag3.csv
# 用法2：python stock.py --csv 路徑 --model rf
# 用法3：from stock import predict, predict_latest; predict_latest(model="rf")

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Dict, Any, Optional, Union

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# 從 machinelearning.py 匯入核心函式
from machinelearning import predict as ml_predict

# ====== 超參數 ======
THRESH = 0.01
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

# 預設 CSV 位置（配合你的 test.py 輸出）
DEFAULT_CSV = Path(__file__).parent / "data" / "short_term_with_lag3.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ====== 工具函式 ======
def _resolve_csv_path(csv_path: Optional[Union[str, Path]]) -> Path:
    p = Path(csv_path) if csv_path else DEFAULT_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"找不到資料檔：{p.resolve()}\n"
            "請先在後端呼叫 /api/quick 或 /api/auto/start 生成 CSV。"
        )
    return p

def _build_y(df: pd.DataFrame, thresh: float | None) -> pd.Series:
    df = df.copy()
    df["明天收盤價"] = df["收盤價(元)"].shift(-1)
    if thresh is None:
        y = (df["明天收盤價"] > df["收盤價(元)"]).astype(float)
    else:
        ret1 = (df["明天收盤價"] - df["收盤價(元)"]) / df["收盤價(元)"]
        y = ret1.apply(lambda x: 1 if x > thresh else (0 if x < -thresh else np.nan))
    return y

def _best_threshold(model, X_val, y_val, grid=np.linspace(0.30, 0.70, 41)) -> tuple[float, float]:
    p = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_hat = (p >= t).astype(int)
        f1w = f1_score(y_val, y_hat, average="weighted")
        if f1w > best_f1:
            best_f1, best_t = f1w, t
    return best_t, best_f1

# ====== 主函式（可供 FastAPI 匯入）======
def predict(csv_path: Optional[Union[str, Path]] = None,
            model: Literal["rf", "lr"] = "rf",
            symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    讀取 csv_path（預設 data/short_term_with_lag3.csv），
    訓練模型並回傳「最後一列」對「明天」的上漲/下跌預測。
    回傳: {"label": "漲/跌", "proba": float, "threshold": float, "model": "rf/lr", "csv": "實際路徑"}
    
    注意：此函數現在調用 machinelearning.py 中的實現
    """
    return ml_predict(csv_path=csv_path, model=model, symbol=symbol)


def predict_latest(model: Literal["rf","lr"]="rf") -> Dict[str, Any]:
    """直接讀 data/short_term_with_lag3.csv 做預測"""
    return predict(DEFAULT_CSV, model=model)


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


def _build_from_td(symbol: str, out_csv: Path, start: str = "2020-01-01") -> pd.DataFrame:
    """使用 twelvedata 為指定股票建立特徵並保存 CSV"""
    from machinelearning import _fetch_ohlcv_td
    
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


def train(csv_path: Optional[Union[str, Path]] = None, models_to_train: Optional[Union[str, list]] = None) -> Dict[str, Any]:
    """Train and persist pipelines + thresholds for requested models.

    models_to_train: 'rf', 'lr', or ['rf','lr']
    Returns a dict with saved file paths and basic metrics.
    """
    path = _resolve_csv_path(csv_path)
    df_raw = pd.read_csv(path, encoding="utf-8-sig")
    df = df_raw.copy()
    y_all = _build_y(df, THRESH)
    df["y_final"] = y_all
    drop_cols = ["年月日", "y_明天漲跌", "明天收盤價", "y_final"]
    X_all = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    for c in X_all.columns:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

    train_idx = df.index[df["y_final"].notna()]
    X_train_all_raw = X_all.loc[train_idx].reset_index(drop=True)
    y_train_all = df.loc[train_idx, "y_final"].astype(int).reset_index(drop=True)

    split_idx = int(len(X_train_all_raw) * (1 - TEST_SPLIT))
    X_tr_raw, X_te_raw = X_train_all_raw.iloc[:split_idx].copy(), X_train_all_raw.iloc[split_idx:].copy()
    y_tr, y_te = y_train_all.iloc[:split_idx], y_train_all.iloc[split_idx:]

    medians = X_tr_raw.median(numeric_only=True)
    X_tr = X_tr_raw.fillna(medians)
    X_te = X_te_raw.fillna(medians)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    val_idx = int(len(X_tr_s) * (1 - VAL_SPLIT))
    X_core, X_val = X_tr_s[:val_idx], X_tr_s[val_idx:]
    y_core, y_val = y_tr.iloc[:val_idx], y_tr.iloc[val_idx:]

    saved = {}
    to_train = models_to_train
    if to_train is None:
        to_train = ["rf", "lr"]
    if isinstance(to_train, str):
        to_train = [to_train]

    # train requested models
    for m in to_train:
        if m == "lr":
            mdl = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
        else:
            mdl = RandomForestClassifier(
                n_estimators=400, max_depth=4, min_samples_leaf=20, min_samples_split=10,
                max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1
            )

        mdl.fit(X_core, y_core)
        t, f1 = _best_threshold(mdl, X_val, y_val)

        # retrain on full train
        mdl.fit(X_tr_s, y_tr)

        pipeline = Pipeline([("scaler", scaler), ("mdl", mdl)])
        model_file = MODELS_DIR / f"{m}_pipeline.pkl"
        thr_file = MODELS_DIR / f"{m}_threshold.pkl"
        joblib.dump(pipeline, model_file)
        joblib.dump(t, thr_file)

        saved[m] = {
            "model_file": str(model_file.resolve()),
            "threshold_file": str(thr_file.resolve()),
            "threshold": float(t),
            "val_f1": float(f1)
        }

    return saved

# ====== CLI ======
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="資料檔路徑，預設 data/short_term_with_lag3.csv")
    ap.add_argument("--model", choices=["rf","lr","all"], default="rf")
    ap.add_argument("--train", action="store_true", help="訓練並儲存模型到 models/ (若選 all 會訓練兩個模型)")
    args = ap.parse_args()

    if args.train:
        models = ["rf", "lr"] if args.model == "all" else [args.model]
        saved = train(args.csv, models_to_train=models)
        print("已儲存模型：")
        for k, v in saved.items():
            print(f"- {k}: model={v['model_file']} threshold={v['threshold']:.3f} val_f1={v['val_f1']:.4f}")
    else:
        out = predict(args.csv, model=args.model)
        print(f"讀取: {out['csv']}")
        print(f"模型: {out['model']} | 閾值: {out['threshold']:.3f}")
        print(f"機率(明天上漲): {out['proba']:.4f}")
        print(f"預測: {out['label']}")
