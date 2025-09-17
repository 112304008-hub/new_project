# stock.py
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

# ====== 超參數 ======
THRESH = 0.01
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

# 預設 CSV 位置（配合你的 test.py 輸出）
DEFAULT_CSV = Path(__file__).parent / "data" / "short_term_with_lag3.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
            model: Literal["rf", "lr"] = "rf") -> Dict[str, Any]:
    """
    讀取 csv_path（預設 data/short_term_with_lag3.csv），
    訓練模型並回傳「最後一列」對「明天」的上漲/下跌預測。
    回傳: {"label": "漲/跌", "proba": float, "threshold": float, "model": "rf/lr", "csv": "實際路徑"}
    """
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
    model_name = f"{model}_pipeline.pkl"
    thr_name = f"{model}_threshold.pkl"
    model_path = MODELS_DIR / model_name
    thr_path = MODELS_DIR / thr_name

    if model_path.exists() and thr_path.exists():
        # load pipeline and threshold
        pipeline: Pipeline = joblib.load(model_path)
        thr = joblib.load(thr_path)

        # Determine the feature names the pipeline was trained on.
        expected = None
        # First try common step names
        for step_name in ("scaler", "rf", "lr", "mdl"):
            if step_name in getattr(pipeline, 'named_steps', {}):
                step = pipeline.named_steps[step_name]
                if hasattr(step, 'feature_names_in_'):
                    expected = list(step.feature_names_in_)
                    break
        # fallback: look for any step with feature_names_in_
        if expected is None and hasattr(pipeline, 'named_steps'):
            for step in pipeline.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    expected = list(step.feature_names_in_)
                    break
        # fallback: pipeline-level attribute
        if expected is None and hasattr(pipeline, 'feature_names_in_'):
            expected = list(pipeline.feature_names_in_)
        # last resort: use n_features_in_ to pick numeric columns
        if expected is None and hasattr(pipeline, 'n_features_in_'):
            n = int(pipeline.n_features_in_)
            numeric_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
            expected = numeric_cols[-n:]

        if expected is None:
            # we couldn't infer training feature names; try to pass all numeric columns
            expected = X_all.select_dtypes(include=[np.number]).columns.tolist()

        # ensure expected columns exist in current CSV
        missing = [c for c in expected if c not in X_all.columns]
        if missing:
            raise ValueError(f"Model expects columns {expected} but these are missing from CSV: {missing}")

        # select and fill
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
        "csv": str(path.resolve())
    }

def predict_latest(model: Literal["rf","lr"]="rf") -> Dict[str, Any]:
    """直接讀 data/short_term_with_lag3.csv 做預測"""
    return predict(DEFAULT_CSV, model=model)


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
