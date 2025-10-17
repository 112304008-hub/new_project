"""GBDT.PY — 使用 Gradient Boosting 測試模型表現（不影響前端）

用途與重點：
- 讀取 data/ 內的特徵檔（預設 data/short_term_with_lag3.csv 或指定 symbol 對應檔）。
- 依照 stock.py 的做法建立 y（明日相對今日 ±THRESH 的二元分類）。
- 以時間切分成 train/test，再從訓練集切出一部份做 validation 尋找最佳 decision threshold（最大化 weighted F1）。
- 訓練 GradientBoostingClassifier（GBDT）並輸出 Accuracy / Precision / Recall / F1 / ROC-AUC 等指標。
- 可選擇加上 --compare-rf 同步訓練 RandomForestClassifier 當基準，比較差異。

注意：
- 本檔僅作離線測試，不會寫入 models/ 或影響 FastAPI；資料直接讀 data/。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	precision_recall_fscore_support,
	roc_auc_score,
	confusion_matrix,
	classification_report,
	f1_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif


# ===== 參數 =====
THRESH = 0.01           # 與 stock.py 一致：±1% 內視為不確定 -> y=NaN（不參與訓練）
VAL_SPLIT = 0.2         # 訓練集中再切 20% 做 validation（找最佳 threshold）
TEST_SPLIT = 0.2        # 整體資料尾端 20% 當作最終測試
RANDOM_STATE = 42

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV = DATA_DIR / "short_term_with_lag3.csv"
REPORT_DIR = ROOT / "data_work"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_csv_path(csv_path: Optional[Union[str, Path]], symbol: Optional[str]) -> Path:
	if symbol:
		p = DATA_DIR / f"{symbol}_short_term_with_lag3.csv"
		return p
	return Path(csv_path) if csv_path else DEFAULT_CSV


def _build_y(df: pd.DataFrame, thresh: float | None) -> pd.Series:
	df = df.copy()
	if "收盤價(元)" not in df.columns:
		raise ValueError("找不到欄位『收盤價(元)』，請確認 CSV 來源正確")
	df["明天收盤價"] = df["收盤價(元)"].shift(-1)
	if thresh is None:
		y = (df["明天收盤價"] > df["收盤價(元)"]).astype(float)
	else:
		ret1 = (df["明天收盤價"] - df["收盤價(元)"]) / df["收盤價(元)"]
		y = ret1.apply(lambda x: 1 if x > thresh else (0 if x < -thresh else np.nan))
	return y


def _best_threshold(model, X_val, y_val, grid=np.linspace(0.30, 0.70, 41), metric: str = "f1_weighted") -> tuple[float, float]:
	p = model.predict_proba(X_val)[:, 1]
	best_t, best_score = 0.5, -1.0
	for t in grid:
		y_hat = (p >= t).astype(int)
		if metric == "f1_macro":
			score = f1_score(y_val, y_hat, average="macro")
		else:
			score = f1_score(y_val, y_hat, average="weighted")
		if score > best_score:
			best_score, best_t = score, t
	return best_t, best_score


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
	# 與 stock.py 一致：移除日期與目標欄位
	drop_cols = ["年月日", "y_明天漲跌", "明天收盤價", "y_final"]
	X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
	# 轉數值
	for c in X.columns:
		X[c] = pd.to_numeric(X[c], errors="coerce")
	return X


def _time_split_indices(n_rows: int, test_ratio: float) -> tuple[int, range, range]:
	split_idx = int(n_rows * (1 - test_ratio))
	return split_idx, range(0, split_idx), range(split_idx, n_rows)


def _evaluate(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, Any]:
	y_pred = (proba >= thr).astype(int)
	acc = accuracy_score(y_true, y_pred)
	precision, recall, f1, support = precision_recall_fscore_support(
		y_true, y_pred, labels=[0, 1], zero_division=0
	)
	try:
		# 若 y_true 只有單一類別，roc_auc_score 會丟例外
		auc = roc_auc_score(y_true, proba)
	except Exception:
		auc = None
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
	report = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0)
	return {
		"threshold": float(thr),
		"accuracy": float(acc),
		"auc": (float(auc) if auc is not None else None),
		"precision": {"down": float(precision[0]), "up": float(precision[1])},
		"recall": {"down": float(recall[0]), "up": float(recall[1])},
		"f1": {"down": float(f1[0]), "up": float(f1[1])},
		"confusion_matrix": cm,
		"classification_report": report,
	}


def _augment_features_inplace(df_raw: pd.DataFrame, X_all: pd.DataFrame,
							  add_returns: bool, add_volatility: bool) -> None:
	"""基於收盤價新增簡單衍生特徵，直接寫入 X_all（與索引對齊）。"""
	try:
		close = pd.to_numeric(df_raw['收盤價(元)'], errors='coerce')
	except Exception:
		return
	# 日報酬
	ret1 = close.pct_change()
	if add_returns:
		for w in (3, 5, 10):
			X_all[f'ret_{w}'] = (close / close.shift(w) - 1).astype(float)
			# 動能（近 w 日報酬總和）
			X_all[f'mom_{w}'] = ret1.rolling(w).sum().astype(float)
	if add_volatility:
		for w in (10, 20):
			X_all[f'vol_{w}'] = (ret1.rolling(w).std()).astype(float)


def _compute_class_weights(y: pd.Series) -> dict:
	counts = y.value_counts().to_dict()
	total = sum(counts.values())
	n_classes = len(counts) if counts else 1
	weights = {cls: (total / (n_classes * cnt)) for cls, cnt in counts.items() if cnt > 0}
	return weights


def run_gbdt(csv_path: Path, quick: bool = False, compare_rf: bool = False,
			 tune: bool = False, calibrate: Optional[str] = None, balance: bool = False,
			 topk: int = 0, metric: str = "f1_weighted",
			 thr_grid: tuple[float, float, int] = (0.30, 0.70, 41),
			 add_returns: bool = False, add_volatility: bool = False) -> Dict[str, Any]:
	if not csv_path.exists():
		raise FileNotFoundError(f"找不到資料檔：{csv_path}")

	df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
	if df_raw.shape[0] < 100:
		print("[警告] 資料筆數較少，指標可能不穩定。")

	# 構建 y 與 X
	y_all = _build_y(df_raw, THRESH)
	df = df_raw.copy()
	df["y_final"] = y_all
	X_all = _select_features(df)
	# 選配：特徵擴充
	if add_returns or add_volatility:
		_augment_features_inplace(df_raw, X_all, add_returns=add_returns, add_volatility=add_volatility)

	# 僅使用 y 有值的列做訓練/測試
	mask = df["y_final"].notna()
	X_train_all = X_all.loc[mask].reset_index(drop=True)
	y_train_all = df.loc[mask, "y_final"].astype(int).reset_index(drop=True)
	if len(X_train_all) < 50:
		raise RuntimeError("有效訓練資料不足（<50 筆）")

	# 缺值以訓練集的中位數補齊
	medians = X_train_all.median(numeric_only=True)
	X_train_all = X_train_all.fillna(medians)

	# 時序切分 -> train/test
	split_idx, tr_idx, te_idx = _time_split_indices(len(X_train_all), TEST_SPLIT)
	X_tr = X_train_all.iloc[list(tr_idx)].copy()
	y_tr = y_train_all.iloc[list(tr_idx)].copy()
	X_te = X_train_all.iloc[list(te_idx)].copy()
	y_te = y_train_all.iloc[list(te_idx)].copy()

	# 再從訓練切出 validation
	val_idx = int(len(X_tr) * (1 - VAL_SPLIT))
	X_core, X_val = X_tr.iloc[:val_idx], X_tr.iloc[val_idx:]
	y_core, y_val = y_tr.iloc[:val_idx], y_tr.iloc[val_idx:]

	# 類別權重（可選）
	sample_weight_core = None
	sample_weight_full = None
	sample_weight_val = None
	if balance:
		cw_core = _compute_class_weights(y_core)
		sample_weight_core = y_core.map(cw_core).astype(float).to_numpy()
		cw_tr = _compute_class_weights(y_tr)
		sample_weight_full = y_tr.map(cw_tr).astype(float).to_numpy()
		sample_weight_val = y_val.map(_compute_class_weights(y_val)).astype(float).to_numpy()

	# ==== 訓練 GBDT ====
	# quick 模式用較小迭代數；否則給一個穩健的預設
	if quick:
		gb_params = dict(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.9,
						 random_state=RANDOM_STATE)
	else:
		gb_params = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
						 min_samples_leaf=20, max_features=None, random_state=RANDOM_STATE)
    
	# 可選：Top-K 特徵（用輕量模型估重要度）
	selected_cols = list(X_tr.columns)
	if isinstance(topk, int) and topk > 0 and topk < len(selected_cols):
		probe = GradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=2, subsample=0.9,
										   random_state=RANDOM_STATE)
		probe.fit(X_core, y_core, sample_weight=sample_weight_core)
		importances = getattr(probe, 'feature_importances_', None)
		if importances is not None:
			order = np.argsort(importances)[::-1][:topk]
			selected_cols = [X_tr.columns[i] for i in order]
			X_core, X_val = X_core[selected_cols], X_val[selected_cols]
			X_tr, X_te = X_tr[selected_cols], X_te[selected_cols]

	# 可選：超參數微調（在 core/val 上找最佳 threshold 分數）
	if tune:
		grid = [
			{
				'n_estimators': n,
				'learning_rate': lr,
				'max_depth': md,
				'subsample': ss,
				'min_samples_leaf': ms
			}
			for n in ([120, 180] if quick else [150, 250, 350])
			for lr in ([0.03, 0.05] if quick else [0.03, 0.05, 0.1])
			for md in [2, 3]
			for ss in [0.8, 0.9, 1.0]
			for ms in ([10, 20] if not quick else [10])
		]
		best_params, best_score, best_thr = gb_params.copy(), -1.0, 0.5
		lo, hi, steps = thr_grid
		thr_lin = np.linspace(float(lo), float(hi), int(steps))
		for p in grid:
			g = GradientBoostingClassifier(random_state=RANDOM_STATE, **p)
			g.fit(X_core, y_core, sample_weight=sample_weight_core)
			# 自訂 threshold 搜尋 + 指標
			t, s = _best_threshold(g, X_val, y_val, grid=thr_lin, metric=metric)
			if s > best_score:
				best_params, best_score, best_thr = p, s, t
		gb_params = best_params
    
	gb = GradientBoostingClassifier(**gb_params)
	gb.fit(X_core, y_core, sample_weight=sample_weight_core)

	lo, hi, steps = thr_grid
	thr_lin = np.linspace(float(lo), float(hi), int(steps))
	if calibrate in ("sigmoid", "isotonic"):
		# 以 core 訓練、以 val 做校準；不再重訓全資料，以避免洩漏。
		cal = CalibratedClassifierCV(gb, cv="prefit", method=calibrate)
		cal.fit(X_val, y_val, sample_weight=sample_weight_val)
		t_opt, val_score = _best_threshold(cal, X_val, y_val, grid=thr_lin, metric=metric)
		proba_gb = cal.predict_proba(X_te)[:, 1]
		thr_gb = t_opt
	else:
		# 找最佳 threshold（使用非校準模型）
		t_opt, val_score = _best_threshold(gb, X_val, y_val, grid=thr_lin, metric=metric)
		# 以 train(full) 重新訓練再測試
		gb.fit(X_tr, y_tr, sample_weight=sample_weight_full)
		proba_gb = gb.predict_proba(X_te)[:, 1]
		thr_gb = t_opt

	eval_gb = _evaluate(y_te.to_numpy(), proba_gb, thr_gb)
	eval_gb.update({
		"model": "gbdt",
		"val_score": float(val_score),
		"params": gb_params,
		"train_size": int(len(X_tr)),
		"val_size": int(len(X_val)),
		"test_size": int(len(X_te)),
		"metric": metric,
		"thr_grid": [float(lo), float(hi), int(steps)],
		"topk": int(topk or 0),
		"balanced": bool(balance),
		"calibrated": calibrate or None,
	})

	results: Dict[str, Any] = {"gbdt": eval_gb}

	# ==== 可選：RF 基準比較 ====
	if compare_rf:
		rf = RandomForestClassifier(
			n_estimators=(200 if quick else 400), max_depth=4, min_samples_leaf=20, min_samples_split=10,
			max_features="sqrt", class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
		)
		rf.fit(X_core, y_core)
		thr_rf, val_score_rf = _best_threshold(rf, X_val, y_val, grid=thr_lin, metric=metric)
		rf.fit(X_tr, y_tr)
		proba_rf = rf.predict_proba(X_te)[:, 1]
		eval_rf = _evaluate(y_te.to_numpy(), proba_rf, thr_rf)
		eval_rf.update({
			"model": "rf",
			"val_score": float(val_score_rf),
			"params": {
				"n_estimators": rf.n_estimators, "max_depth": rf.max_depth, "min_samples_leaf": rf.min_samples_leaf,
				"min_samples_split": rf.min_samples_split, "max_features": rf.max_features
			},
		})
		results["rf"] = eval_rf

	return results


def main():
	ap = argparse.ArgumentParser(description="用 Gradient Boosting 測試短期特徵的分類效果")
	ap.add_argument("--csv", default=str(DEFAULT_CSV), help="資料檔路徑（預設 data/short_term_with_lag3.csv）")
	ap.add_argument("--symbol", default=None, help="若提供，優先讀取 data/<symbol>_short_term_with_lag3.csv")
	ap.add_argument("--quick", action="store_true", help="快速模式（較少樹/較快）")
	ap.add_argument("--compare-rf", action="store_true", help="同時訓練 RF 做基準比較")
	ap.add_argument("--save", action="store_true", help="將結果另存 JSON 至 data_work/")
	# 進階強化選項
	ap.add_argument("--tune", action="store_true", help="在 core/val 上進行小型超參數搜尋")
	ap.add_argument("--calibrate", choices=["sigmoid", "isotonic"], default=None, help="使用校準機率（與 core/val 流程一致）")
	ap.add_argument("--balance", action="store_true", help="以 sample_weight 處理類別不平衡")
	ap.add_argument("--topk", type=int, default=0, help="僅使用重要度前 K 個特徵（0 表示不啟用）")
	ap.add_argument("--metric", choices=["f1_weighted", "f1_macro"], default="f1_weighted", help="最佳閾值的評分指標")
	ap.add_argument("--thr-grid", default="0.30,0.70,41", help="threshold grid: min,max,steps 例如 0.25,0.75,51")
	ap.add_argument("--feat-returns", action="store_true", help="加入多期報酬/動能特徵（基於收盤價）")
	ap.add_argument("--feat-volatility", action="store_true", help="加入多期波動度特徵（基於日報酬 std）")
	args = ap.parse_args()

	csv_path = _resolve_csv_path(args.csv, args.symbol)
	# parse thr grid
	try:
		parts = [p.strip() for p in str(args.__dict__["thr_grid"]).split(",")]
		tmin, tmax, tsteps = float(parts[0]), float(parts[1]), int(parts[2])
		thr_grid = (tmin, tmax, tsteps)
	except Exception:
		thr_grid = (0.30, 0.70, 41)

	results = run_gbdt(
		csv_path,
		quick=args.quick,
		compare_rf=args.compare_rf,
		tune=args.tune,
		calibrate=args.calibrate,
		balance=args.balance,
		topk=args.topk,
		metric=args.metric,
		thr_grid=thr_grid,
		add_returns=args.__dict__["feat_returns"],
		add_volatility=args.__dict__["feat_volatility"],
	)

	# 螢幕輸出（精簡摘要）
	def _fmt_eval(name: str, ev: Dict[str, Any]):
		print(f"\n=== {name.upper()} 結果 ===")
		print(f"threshold: {ev['threshold']:.3f}")
		print(f"accuracy : {ev['accuracy']:.4f}")
		print(f"auc      : {ev['auc']:.4f}" if ev['auc'] is not None else "auc      : n/a")
		print(f"f1(up)   : {ev['f1']['up']:.4f} | f1(down): {ev['f1']['down']:.4f}")
		print("confusion matrix [ [TN FP], [FN TP] ]:")
		print(ev["confusion_matrix"]) 
		print("\nclassification report:\n" + ev["classification_report"])

	for name, ev in results.items():
		_fmt_eval(name, ev)

	# 是否存檔
	if args.save:
		ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
		outp = REPORT_DIR / f"gbdt_report_{ts}.json"
		payload = {
			"csv": str(csv_path.resolve()),
			"threshold": THRESH,
			"splits": {"val_split": VAL_SPLIT, "test_split": TEST_SPLIT},
			"random_state": RANDOM_STATE,
			"results": results,
		}
		with outp.open("w", encoding="utf-8") as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)
		print(f"\n已寫入報告：{outp}")


if __name__ == "__main__":
	main()

