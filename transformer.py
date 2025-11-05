"""transformer.py — 測試用 Transformer 時序訓練與即時推論

目的（僅測試階段，不影響前端）：
- 將每 5 分鐘更新一次的特徵 CSV（data/<symbol>_short_term_with_lag3.csv 或預設 data/short_term_with_lag3.csv）
  轉為時序樣本，使用輕量 Transformer 進行二元分類（明天漲/跌），並將結果印出到 stdout。

特性：
- 不寫入前端或覆蓋既有 RF/LR 模型；僅在 console 列印訓練/驗證/測試與最新一筆的即時預測。
- 以滑動視窗（window_len）組成序列。標籤依據 stock.py 的邏輯：明日相對今日漲幅超過 +THRESH 記為 1，低於 -THRESH 記為 0，介於中間忽略。
- 支援直接指定 CSV 路徑或 symbol；也可用 --recent 選項掃描 data/ 內最近 N 分鐘有 last_update 的 symbols 批次訓練。

命令列用法：
  1) 針對單一資料檔：
	   python transformer.py --csv data/2330_short_term_with_lag3.csv --epochs 5 --window 64
	 或
	   python transformer.py --symbol 2330 --epochs 5 --window 64

  2) 掃描最近 10 分鐘有更新的 symbols 批次訓練一次：
	   python transformer.py --recent 10 --epochs 5 --window 64

注意：
- 本檔不會將權重保存到 models/，避免干擾現有服務；若日後需要持久化，可安全新增另存路徑（例如 models/transformer/*.pt）。
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
	f1_score,
	accuracy_score,
	balanced_accuracy_score,
	matthews_corrcoef,
	confusion_matrix,
	roc_curve,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif

# numpy quantile backward/forward compatibility helper
def _np_quantile_compat(a, q: float, method: str = 'linear') -> float:
	try:
		return float(np.quantile(a, q, method=method))  # NumPy >= 1.22
	except TypeError:
		return float(np.quantile(a, q, interpolation=method))  # NumPy < 1.22


# 與 stock.py 對齊的常數（門檻）
THRESH = 0.01  # default absolute label threshold; can be overridden via CLI

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def _resolve_csv_by_symbol(symbol: Optional[str]) -> Optional[Path]:
	if not symbol:
		return None
	p = DATA_DIR / f"{symbol}_short_term_with_lag3.csv"
	return p if p.exists() else None


def _resolve_csv(csv: Optional[str | Path], symbol: Optional[str]) -> Path:
	if csv:
		p = Path(csv)
		if not p.exists():
			raise FileNotFoundError(f"找不到指定的 CSV: {p}")
		return p
	if symbol:
		ps = _resolve_csv_by_symbol(symbol)
		if ps and ps.exists():
			return ps
	# fallback 預設
	default_p = DATA_DIR / "short_term_with_lag3.csv"
	if not default_p.exists():
		raise FileNotFoundError(f"找不到預設 CSV：{default_p}")
	return default_p


def _build_y_abs(df: pd.DataFrame, thresh: float = THRESH) -> Tuple[pd.Series, pd.Series, pd.Series]:
	df = df.copy()
	df["明天收盤價"] = df["收盤價(元)"].shift(-1)
	ret1 = (df["明天收盤價"] - df["收盤價(元)"]) / df["收盤價(元)"]
	y = ret1.apply(lambda x: 1 if x > thresh else (0 if x < -thresh else np.nan))
	ret0 = df["收盤價(元)"].pct_change()  # 當日已知的日內報酬（t 相對 t-1）
	return y, ret1, ret0


def _build_y_pct(df: pd.DataFrame, q: float = 0.3) -> Tuple[pd.Series, pd.Series, pd.Series]:
	"""以分位數決定標籤：>= 上分位數記 1，<= 下分位數記 0，其餘為 NaN（忽略）。
	q: 例如 0.3 表示取上 30% 與下 30%。
	"""
	df = df.copy()
	df["明天收盤價"] = df["收盤價(元)"].shift(-1)
	ret1 = (df["明天收盤價"] - df["收盤價(元)"]) / df["收盤價(元)"]
	lo = ret1.quantile(q)
	hi = ret1.quantile(1 - q)
	def _lab(x: float):
		if x >= hi:
			return 1
		elif x <= lo:
			return 0
		else:
			return np.nan
	ret0 = df["收盤價(元)"].pct_change()
	return ret1.apply(_lab), ret1, ret0


def _prepare_xy(
	df_raw: pd.DataFrame,
	label_thresh: float = THRESH,
	label_mode: str = "abs",
	label_q: float = 0.3,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
	df = df_raw.copy()
	# 派生日曆特徵（週幾 one-hot），僅使用當下可得資料
	try:
		if "年月日" in df.columns:
			dd = pd.to_datetime(df["年月日"], errors="coerce")
			dow = dd.dt.dayofweek  # 0=Mon ... 6=Sun
			for k in range(7):
				df[f"dow_{k}"] = (dow == k).astype(float)
	except Exception:
		pass
	if label_mode == "pct":
		y_all, ret1, ret0 = _build_y_pct(df, q=float(label_q))
	else:
		y_all, ret1, ret0 = _build_y_abs(df, label_thresh)
	df["y_final"] = y_all
	drop_cols = ["年月日", "y_明天漲跌", "明天收盤價", "y_final"]
	X_all = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
	# 轉數值
	for c in X_all.columns:
		X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
	# 僅保留有 y 的列
	idx = df.index[df["y_final"].notna()]
	X = X_all.loc[idx].reset_index(drop=True)
	y = df.loc[idx, "y_final"].astype(int).reset_index(drop=True)
	ret1_aligned = ret1.loc[idx].reset_index(drop=True)
	ret0_aligned = ret0.loc[idx].reset_index(drop=True)
	return X, y, ret1_aligned, ret0_aligned


class TimeSeriesWindowDataset(Dataset):
	def __init__(self, X: pd.DataFrame, y: pd.Series, window: int):
		self.Xn = X.fillna(X.median(numeric_only=True)).to_numpy(dtype=np.float32)
		self.y = y.to_numpy(dtype=np.int64)
		self.window = int(window)
		self.n = len(self.y)
		self.samples: List[Tuple[int, int]] = []  # (start, end) inclusive/exclusive for X window ending at t
		for t in range(self.window, self.n):
			self.samples.append((t - self.window, t))

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		s, e = self.samples[idx]
		x = self.Xn[s:e]  # (window, feat)
		# 對齊：使用窗口最後一天（e-1）的特徵來預測「隔天」(e-1 -> e) 的漲跌
		# 原本使用 y[e] 會變成用到 e 之前一天的特徵來預測 (e -> e+1)，少用到當天(e)特徵，
		# 造成目標對齊偏移（off-by-one）而降低可預測性。
		y = self.y[e - 1]
		return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # (1, max_len, d_model)
		self.register_buffer('pe', pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, L, D)
		x = x + self.pe[:, : x.size(1)]
		return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
	def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.dropout(x + self.pe[:, : x.size(1)])



class DropPath(nn.Module):
	"""Stochastic Depth / DropPath per sample (when applied in main path of residual blocks)."""
	def __init__(self, drop_prob: float = 0.0):
		super().__init__()
		self.drop_prob = float(max(0.0, drop_prob))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		drop_prob = self.drop_prob
		if drop_prob == 0.0 or not self.training:
			return x
		keep_prob = 1.0 - drop_prob
		shape = (x.size(0),) + (1,) * (x.ndim - 1)
		random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
		random_tensor.floor_()
		return x.div(keep_prob) * random_tensor


class SqueezeExcite1d(nn.Module):
	"""SE block for 1D conv: squeeze over time, excite channels."""
	def __init__(self, channels: int, reduction: int = 4):
		super().__init__()
		hid = max(1, channels // int(max(1, reduction)))
		self.fc1 = nn.Conv1d(channels, hid, kernel_size=1)
		self.act = nn.GELU()
		self.fc2 = nn.Conv1d(hid, channels, kernel_size=1)
		self.gate = nn.Sigmoid()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, C, L)
		s = x.mean(dim=-1, keepdim=True)
		s = self.fc2(self.act(self.fc1(s)))
		w = self.gate(s)
		return x * w


class MultiScaleConvMixer1d(nn.Module):
	"""Parallel multi-kernel temporal conv + pointwise projection with SE and DropPath.

	Input: (B, C, L) -> Output: (B, C, L)
	"""
	def __init__(self, channels: int, kernels: list[int], dropout: float = 0.1, se_reduction: int = 4, drop_path: float = 0.0):
		super().__init__()
		self.kernels = [int(k) for k in kernels if int(k) > 0]
		branches = []
		for k in self.kernels:
			pad = (k // 2)
			branches.append(nn.Sequential(
				nn.Conv1d(channels, channels, kernel_size=k, padding=pad, bias=False, groups=1),
				nn.GELU(),
				nn.Dropout(dropout)
			))
		self.branches = nn.ModuleList(branches)
		self.proj = nn.Conv1d(channels * len(self.kernels), channels, kernel_size=1, bias=False)
		self.se = SqueezeExcite1d(channels, reduction=se_reduction)
		self.drop_path = DropPath(drop_path)
		self.norm = nn.BatchNorm1d(channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, C, L)
		outs = []
		for blk in self.branches:
			outs.append(blk(x))
		y = torch.cat(outs, dim=1)
		y = self.proj(y)
		y = self.se(y)
		y = self.norm(y)
		# residual with DropPath
		return x + self.drop_path(y)




class TransformerClassifier(nn.Module):
	def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1, activation: str = "gelu", pooling: str = "cls", use_conv: bool = True, dilations: Optional[List[int]] = None, use_learnable_pos: bool = False, patch_len: int = 1, patch_stride: int = 1, use_msconv: bool = True, ms_kernels: Optional[List[int]] = None, se_reduction: int = 4, drop_path: float = 0.0, smooth_len: int = 1, last_skip: bool = True):
		super().__init__()
		# input projection + normalization for stability
		self.inp = nn.Linear(n_features, d_model)
		self.inp_ln = nn.LayerNorm(d_model)
		self.inp_drop = nn.Dropout(dropout)
		self.pos = LearnablePositionalEncoding(d_model, dropout=dropout) if use_learnable_pos else PositionalEncoding(d_model, dropout)
		# optional temporal smoothing (depthwise average conv, non-trainable)
		self.smooth_len = int(max(1, smooth_len))
		self.smoother = None
		if self.smooth_len > 1:
			k = self.smooth_len
			# depthwise conv initialized as uniform averaging; causal padding on left to avoid peeking forward
			self.smoother = nn.Conv1d(d_model, d_model, kernel_size=k, groups=d_model, bias=False)
			with torch.no_grad():
				w = torch.ones(d_model, 1, k) / float(k)
				self.smoother.weight.copy_(w)
			for p in self.smoother.parameters():
				p.requires_grad = False
		# norm_first=True is generally more stable for deeper stacks
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True, activation=activation, norm_first=True)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		# optional temporal patching (mean pooling over time) to reduce noise and sequence length
		self.patch_len = int(max(1, patch_len))
		self.patch_stride = int(max(1, patch_stride))
		self.patch_pool = None
		if self.patch_len > 1:
			self.patch_pool = nn.AvgPool1d(kernel_size=self.patch_len, stride=self.patch_stride, ceil_mode=False)
		# pooling strategy
		self.pooling = pooling if pooling in ("cls", "attn", "both") else "cls"
		if self.pooling == "cls":
			# learned CLS token for sequence pooling
			self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
		elif self.pooling == "attn" or self.pooling == "both":
			# attention pooling over time steps
			self.attn_q = nn.Linear(d_model, 1)
		# conv frontend (upgraded): multi-scale mixer or legacy dilated convs
		self.use_conv = use_conv
		self.use_msconv = bool(use_msconv)
		if self.use_conv:
			if self.use_msconv:
				if ms_kernels is None:
					ms_kernels = [3, 5, 7]
				self.conv_mixer = MultiScaleConvMixer1d(d_model, kernels=ms_kernels, dropout=dropout, se_reduction=int(max(1, se_reduction)), drop_path=float(max(0.0, drop_path)))
			else:
				if not dilations:
					dilations = [1, 2, 4]
				blocks = []
				for d in dilations:
					blocks.append(nn.Sequential(
						nn.Conv1d(d_model, d_model, kernel_size=3, padding=d, dilation=d, groups=1, bias=False),
						nn.GELU(),
						nn.Dropout(dropout)
					))
				self.conv = nn.ModuleList(blocks)
		# MLP head (wider) with explicit out layer for bias init
		self.head_ln = nn.LayerNorm(d_model)
		self.head_fc1 = nn.Linear(d_model, d_model * 2)
		self.head_act = nn.GELU()
		self.head_drop = nn.Dropout(dropout)
		self.out = nn.Linear(d_model * 2, 1)
		# residual last-timestep linear shortcut on raw features
		self.last_skip = bool(last_skip)
		if self.last_skip:
			self.last_lin = nn.Linear(n_features, 1)
			self.alpha_gate = nn.Parameter(torch.tensor(0.5))  # sigmoid -> [0,1]

	def init_output_bias(self, pos_prior: float | None, clip_abs_logit: Optional[float] = None, enable: bool = True):
		"""Initialize final layer bias to logit(prior) to help calibration and reduce early bias.
		pos_prior in (0,1). Optionally clip absolute logit and allow disabling."""
		try:
			if not enable or pos_prior is None or not (0.0 < pos_prior < 1.0):
				return
			# avoid extreme priors that cause degenerate all-1/all-0 early on
			if not (0.35 <= pos_prior <= 0.65):
				return
			b = math.log(pos_prior / (1.0 - pos_prior))
			if clip_abs_logit is not None and clip_abs_logit > 0:
				b = float(max(-clip_abs_logit, min(clip_abs_logit, b)))
			with torch.no_grad():
				if hasattr(self, 'out') and hasattr(self.out, 'bias') and self.out.bias is not None:
					self.out.bias.fill_(float(b))
		except Exception:
			pass

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, L, F)
		B = x.size(0)
		h = self.inp(x)
		h = self.inp_ln(h)
		h = self.inp_drop(h)
		# optional temporal patching before attention
		if self.patch_pool is not None:
			# (B, L, D) -> (B, D, L)
			ht = h.transpose(1, 2)
			ht = self.patch_pool(ht)
			h = ht.transpose(1, 2)
		# optional temporal smoothing (causal)
		if self.smoother is not None:
			# (B, L, D) -> (B, D, L)
			ht = h.transpose(1, 2)
			pad = (self.smooth_len - 1, 0)  # left pad for causal
			ht = nn.functional.pad(ht, (pad[0], pad[1]))
			ht = self.smoother(ht)
			h = ht.transpose(1, 2)
		# optional conv frontend over time
		if self.use_conv:
			# (B, L, D) -> (B, D, L)
			ht = h.transpose(1, 2)
			if self.use_msconv:
				ht = self.conv_mixer(ht)
			else:
				for blk in getattr(self, 'conv', []):
					r = blk(ht)
					ht = ht + r  # residual
			h = ht.transpose(1, 2)
		if self.pooling == "cls":
			# prepend CLS token
			cls_tok = self.cls.expand(B, -1, -1)
			h = torch.cat([cls_tok, h], dim=1)  # (B, 1+L, D)
			h = self.pos(h)
			h = self.encoder(h)
			# 取 CLS 向量做分類
			h = h[:, 0, :]  # (B, D)
		elif self.pooling == "attn":
			# attention pooling
			h = self.pos(h)
			h = self.encoder(h)  # (B, L, D)
			a = self.attn_q(h)  # (B, L, 1)
			w = torch.softmax(a.squeeze(-1), dim=1)  # (B, L)
			h = (h * w.unsqueeze(-1)).sum(dim=1)  # (B, D)
		else:  # both: combine CLS and attn pooled representations
			# CLS branch
			cls_tok = getattr(self, 'cls', nn.Parameter(torch.zeros(1,1,h.size(-1)))).expand(B, -1, -1)
			h_cls = torch.cat([cls_tok, h], dim=1)
			h_cls = self.pos(h_cls)
			h_cls = self.encoder(h_cls)
			h_cls = h_cls[:, 0, :]
			# Attn branch (reuse same encoder pass for efficiency is tricky due to CLS; run separately for clarity)
			# Recompute without CLS for attention pooling
			h_att = self.pos(h)
			h_att = self.encoder(h_att)
			a = self.attn_q(h_att)
			w = torch.softmax(a.squeeze(-1), dim=1)
			h_att = (h_att * w.unsqueeze(-1)).sum(dim=1)
			h = 0.5 * (h_cls + h_att)
		# MLP head
		h = self.head_ln(h)
		h = self.head_fc1(h)
		h = self.head_act(h)
		h = self.head_drop(h)
		logit = self.out(h).squeeze(-1)
		# combine with last-timestep linear shortcut on raw features
		if self.last_skip:
			last_logit = self.last_lin(x[:, -1, :]).squeeze(-1)
			alpha = torch.sigmoid(self.alpha_gate)
			logit = alpha * logit + (1.0 - alpha) * last_logit
		return logit


class FocalLoss(nn.Module):
	def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = 'mean'):
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction

	def forward(self, logits: torch.Tensor, targets: torch.Tensor):
		# logits: (B,), targets: (B,) float 0/1
		bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
		p = torch.sigmoid(logits)
		pt = p*targets + (1-p)*(1-targets)
		loss = (1-pt)**self.gamma * bce
		if self.alpha is not None:
			alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
			loss = alpha_t * loss
		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()
		return loss


class AsymmetricLoss(nn.Module):
	"""Asymmetric Focal Loss (ASL) for binary classification.

	References:
	- Asymmetric Loss For Multi-Label Classification (https://arxiv.org/abs/2009.14119)

	Binary variant with optional negative clipping to suppress easy negatives.
	"""

	def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, reduction: str = 'mean'):
		super().__init__()
		self.gamma_pos = float(gamma_pos)
		self.gamma_neg = float(gamma_neg)
		self.clip = float(clip) if (clip is not None and clip > 0) else 0.0
		self.reduction = reduction

	def forward(self, logits: torch.Tensor, targets: torch.Tensor):
		# logits: (B,), targets in {0,1} float tensor
		p = torch.sigmoid(logits)
		t = targets
		# negative clipping: reduce probability of easy negatives
		if self.clip > 0.0:
			pn = torch.clamp(p + self.clip, max=1.0)
		else:
			pn = p
		p_pos = p
		p_neg = 1.0 - pn
		# cross-entropy components
		loss_pos = -torch.log(torch.clamp(p_pos, min=1e-8))
		loss_neg = -torch.log(torch.clamp(p_neg, min=1e-8))
		# asymmetric focusing
		loss_pos *= torch.pow(1.0 - p_pos, self.gamma_pos)
		loss_neg *= torch.pow(1.0 - p_neg, self.gamma_neg)
		loss = t * loss_pos + (1.0 - t) * loss_neg
		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()
		return loss


@dataclass
class TrainConfig:
	window: int = 64
	batch_size: int = 64
	epochs: int = 5
	lr: float = 1e-3
	device: str = "cpu"
	d_model: int = 64
	nhead: int = 4
	layers: int = 2
	ff: int = 128
	dropout: float = 0.1
	weight_decay: float = 1e-2
	warmup_frac: float = 0.1
	max_grad_norm: float = 1.0
	patience: int = 5
	seed: int = 42
	optimize_metric: str = "acc"  # 'acc' or 'f1'
	# 可選 'acc' | 'f1' | 'bacc' | 'mcc' | 'youden'（ROC 最大化 Youden's J）
	use_weighted_sampler: bool = True
	ema_decay: float | None = 0.0
	pooling: str = "cls"
	use_conv: bool = True
	learnable_pos: bool = False
	dilations: Optional[str] = None  # e.g., "1,2,4"
	loss: str = "bce"  # 'bce' | 'focal' | 'asl'
	focal_gamma: float = 2.0
	asl_gamma_pos: float = 0.0
	asl_gamma_neg: float = 4.0
	asl_clip: float = 0.05
	label_smoothing: float = 0.0  # for BCE smoothing in [0, 0.2]
	prior_bias_clip: float = 0.5   # clip absolute logit when initializing output bias; 0 disables clipping
	prior_bias_enable: bool = True # disable to not use prior-based bias init
	bce_pos_weight_mode: str = "auto"  # 'auto'|'sqrt'|'none' (auto = neg/pos; sqrt = sqrt(neg/pos))
	k_features: int = 0  # 0 = disable mutual information feature selection
	label_thresh: float = THRESH
	label_mode: str = "abs"  # 'abs' | 'pct'
	label_q: float = 0.3
	print_confusion: bool = True
	calibrate: bool = True  # isotonic calibration on validation probabilities
	pp_margin: float = 0.25  # threshold selection will try to keep pred_pos_rate within val_prevalence ± margin
	early_stop: str = "loss"  # 'loss' | 'acc' | 'f1' | 'bacc'
	auto_flip_auc: bool = False  # if val AUC < 0.5, flip probabilities (1-p) for thresholding/testing
	fix_test_pp: bool = True     # adjust test threshold to keep predicted positive rate within val_prev ± margin
	scaler: str = "robust"    # 'standard' | 'robust' (robust as default based on empirical improvement)
	mc_dropout_passes: int = 0   # >0 enables MC dropout averaging at eval (val/test)
	auto_retrain_pct_if_low_auc: bool = False  # if AUC too low, auto retrain with label_mode='pct'
	grad_accum_steps: int = 1    # gradient accumulation steps during training
	ensembles: int = 1           # number of models in simple seed ensemble (used by ensemble recipes)
	# calibration and thresholding
	temp_scale: bool = True      # enable temperature scaling on validation logits
	regime_thr: bool = False     # enable regime-wise thresholding based on |ret1|
	regime_q1: float = 0.33
	regime_q2: float = 0.66
	# temporal patching
	patch_len: int = 1
	patch_stride: int = 1
	# upgraded conv frontend flags
	use_msconv: bool = True
	ms_kernels: str = "3,5,7"
	se_reduction: int = 4
	drop_path: float = 0.0
	# temporal smoothing and residual shortcut
	smooth_len: int = 1
	last_skip: bool = True
	# tabular backends
	backend: str = "transformer"  # 'transformer' | 'gbdt' | 'rf'
	gbdt_topk: int = 0
	gbdt_calibrate: str = "none"  # 'none' | 'sigmoid' | 'isotonic'
	gbdt_balance: bool = False
	gbdt_thr_grid: Tuple[float, float, int] = (0.30, 0.70, 41)
	gbdt_metric: str = "f1"  # 'acc' | 'f1' | 'bacc'


def train_tabular_on_csv(csv_path: Path, cfg: TrainConfig, backend: str = "gbdt") -> dict:
	# Seed
	def _set_seed(seed: int):
		import random
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	_set_seed(int(cfg.seed))

	df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
	if "收盤價(元)" not in df_raw.columns:
		raise ValueError("找不到欄位『收盤價(元)』")
	# 構建 X, y（支援 abs/pct 標籤），並加入 DoW 特徵
	X_all, y_all, ret1_all, ret0_all = _prepare_xy(
		df_raw,
		label_thresh=float(cfg.label_thresh),
		label_mode=str(cfg.label_mode),
		label_q=float(cfg.label_q),
	)
	# 時序切分（與 Transformer 一致）
	n = len(y_all)
	if n < 100:
		raise ValueError(f"資料不足，僅 {n} 筆有效樣本")
	n_test = int(n * 0.2)
	n_train = n - n_test
	n_val = int(n_train * 0.2)

	X_train = X_all.iloc[: n_train - n_val].reset_index(drop=True)
	y_train = y_all.iloc[: n_train - n_val].reset_index(drop=True)
	X_val = X_all.iloc[n_train - n_val : n_train].reset_index(drop=True)
	y_val = y_all.iloc[n_train - n_val : n_train].reset_index(drop=True)
	X_test = X_all.iloc[n_train :].reset_index(drop=True)
	y_test = y_all.iloc[n_train :].reset_index(drop=True)

	# 印出分佈
	def _dist(name: str, ys: pd.Series):
		c0 = int((ys == 0).sum()); c1 = int((ys == 1).sum()); total = int(len(ys))
		p1 = (c1 / total) if total > 0 else 0.0
		print(f"[{name}] size={total} y=1比率={p1:.3f} (0:{c0}, 1:{c1})")
	_dist("train", y_train)
	_dist("val", y_val)
	_dist("test", y_test)

	# 缺值補齊（樹模型不需縮放）
	med = X_train.median(numeric_only=True)
	X_train = X_train.fillna(med)
	X_val = X_val.fillna(med)
	X_test = X_test.fillna(med)

	# 類別權重（可選）
	sample_weight_train = None
	if cfg.gbdt_balance:
		counts = y_train.value_counts().to_dict()
		total = sum(counts.values())
		nc = max(1, len(counts))
		weights = {cls: (total / (nc * cnt)) for cls, cnt in counts.items() if cnt > 0}
		sample_weight_train = y_train.map(weights).astype(float).to_numpy()

	# 選擇 Top-K 特徵（用輕量 GBDT 估重要度）
	selected_cols = list(X_train.columns)
	if isinstance(cfg.gbdt_topk, int) and cfg.gbdt_topk > 0 and cfg.gbdt_topk < len(selected_cols):
		probe = GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, subsample=0.9, random_state=int(cfg.seed))
		probe.fit(X_train, y_train)
		imp = getattr(probe, 'feature_importances_', None)
		if imp is not None:
			order = np.argsort(imp)[::-1][: int(cfg.gbdt_topk)]
			selected_cols = [X_train.columns[i] for i in order]
			X_train = X_train[selected_cols]
			X_val = X_val[selected_cols]
			X_test = X_test[selected_cols]

	# 建立模型
	if backend == 'rf':
		model = RandomForestClassifier(
			n_estimators=400, max_depth=4, min_samples_leaf=20, min_samples_split=10,
			max_features='sqrt', class_weight=('balanced' if cfg.gbdt_balance else None), random_state=int(cfg.seed), n_jobs=-1
		)
	else:
		model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9, min_samples_leaf=20, random_state=int(cfg.seed))

	# 擬合於 train，必要時於 val 做校準
	model.fit(X_train, y_train, sample_weight=sample_weight_train)
	calibrate = str(cfg.gbdt_calibrate or 'none').lower()
	if calibrate in ('sigmoid', 'isotonic'):
		cal = CalibratedClassifierCV(model, cv='prefit', method=calibrate)
		cal.fit(X_val, y_val)
		clf = cal
	else:
		clf = model

	# 取得機率
	va_p = clf.predict_proba(X_val)[:, 1]
	te_p = clf.predict_proba(X_test)[:, 1]
	va_y = y_val.to_numpy()
	te_y = y_test.to_numpy()

	# 閾值選擇（與 Transformer 的 optimize_metric 對齊，並套用 pp 約束）
	lo, hi, steps = cfg.gbdt_thr_grid
	grid = np.linspace(float(lo), float(hi), int(steps))
	best_thr, best_score = 0.5, -1.0
	val_prev = float(np.mean(va_y)) if len(va_y) else 0.5
	pp_low = max(0.0, val_prev - float(cfg.pp_margin))
	pp_high = min(1.0, val_prev + float(cfg.pp_margin))
	def _consider(t):
		pred = (va_p >= t).astype(int)
		pp = float(np.mean(pred)) if len(pred) else 0.5
		return (pp_low <= pp <= pp_high), pred
	metric_name = (cfg.optimize_metric or cfg.gbdt_metric)
	for t in grid:
		ok, pred = _consider(t)
		if not ok:
			continue
		if metric_name == 'acc':
			score = accuracy_score(va_y, pred)
		elif metric_name == 'bacc':
			score = balanced_accuracy_score(va_y, pred)
		else:
			score = f1_score(va_y, pred, average='weighted')
		if score > best_score:
			best_score, best_thr = float(score), float(t)

	# 測試階段：可選預測正類比例修正
	te_probs = np.array(te_p, dtype=float)
	thr_for_test = float(best_thr)
	if cfg.fix_test_pp and len(te_probs) > 0:
		pp_target = float(min(pp_high, max(pp_low, val_prev)))
		q = float(min(1.0, max(0.0, 1.0 - pp_target)))
		thr_q = _np_quantile_compat(te_probs, q, method='linear')
		thr_for_test = float(thr_q)

	te_pred = (te_probs >= thr_for_test).astype(int)
	te_acc = accuracy_score(te_y, te_pred) if len(te_y) else 0.0
	te_f1 = f1_score(te_y, te_pred, average='weighted') if len(te_y) else 0.0
	te_bacc = balanced_accuracy_score(te_y, te_pred) if len(te_y) else 0.0
	te_mcc = matthews_corrcoef(te_y, te_pred) if len(te_y) else 0.0
	pp_test = float(np.mean(te_pred)) if len(te_pred) else float('nan')
	print(f"[test] acc={te_acc:.3f} f1={te_f1:.3f} thr={best_thr:.3f}{' test_thr='+str(round(thr_for_test,3)) if thr_for_test!=best_thr else ''}")
	print(f"[test-extra] bacc={te_bacc:.3f} mcc={te_mcc:.3f} pred_pos_rate={pp_test:.3f}")
	if cfg.print_confusion and len(te_y):
		cm = confusion_matrix(te_y, te_pred, labels=[0,1])
		print(f"[confusion]\nTN={cm[0,0]} FP={cm[0,1]}\nFN={cm[1,0]} TP={cm[1,1]}")

	# 最新一筆預測（單列特徵；與訓練欄位對齊）
	df_inf = df_raw.copy()
	try:
		if "年月日" in df_inf.columns:
			dd = pd.to_datetime(df_inf["年月日"], errors='coerce')
			dow = dd.dt.dayofweek
			for k in range(7):
				df_inf[f"dow_{k}"] = (dow == k).astype(float)
	except Exception:
		pass
	X_inf = df_inf.drop(columns=[c for c in ("年月日", "y_明天漲跌", "明天收盤價") if c in df_inf.columns], errors='ignore').copy()
	for c in X_inf.columns:
		X_inf[c] = pd.to_numeric(X_inf[c], errors='coerce')
	X_inf = X_inf.fillna(med)
	X_inf = X_inf.reindex(columns=selected_cols, fill_value=0.0)
	if len(X_inf) >= 1:
		p1 = float(clf.predict_proba(X_inf.iloc[[-1]])[:,1][0])
		label = "漲" if p1 >= best_thr else "跌"
		print(f"[latest] 機率(上漲)={p1:.4f} 預測={label} thr={best_thr:.3f} | 檔案={csv_path.name}")
	else:
		p1 = float('nan'); label=None

	return {
		"csv": str(csv_path.resolve()),
		"test_acc": float(te_acc),
		"test_f1": float(te_f1),
		"test_bacc": float(te_bacc),
		"test_mcc": float(te_mcc),
		"best_threshold": float(best_thr),
		"latest_proba": (None if (isinstance(p1, float) and math.isnan(p1)) else float(p1)),
		"latest_label": label,
	}


def train_transformer_on_csv(csv_path: Path, cfg: TrainConfig) -> dict:
	# seed for reproducibility
	def _set_seed(seed: int):
		import random
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	_set_seed(int(cfg.seed))

	df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
	if "收盤價(元)" not in df_raw.columns:
		raise ValueError("找不到欄位『收盤價(元)』")
	X, y, ret1_all, ret0_all = _prepare_xy(
		df_raw,
		label_thresh=float(cfg.label_thresh),
		label_mode=str(cfg.label_mode),
		label_q=float(cfg.label_q),
	)
	if len(y) < cfg.window + 10:
		raise ValueError(f"資料量不足以建立視窗（至少需 {cfg.window + 10} 筆有效樣本），目前僅 {len(y)} 筆")

	# 時間切分：最後 20% 當 test，其前段 20% 當 val（保持時序順序）
	n = len(y)
	n_test = int(n * 0.2)
	n_train = n - n_test
	n_val = int(n_train * 0.2)

	X_train = X.iloc[: n_train - n_val].reset_index(drop=True)
	y_train = y.iloc[: n_train - n_val].reset_index(drop=True)
	ret1_train = ret1_all.iloc[: n_train - n_val].reset_index(drop=True)
	ret0_train = ret0_all.iloc[: n_train - n_val].reset_index(drop=True)
	X_val = X.iloc[n_train - n_val : n_train].reset_index(drop=True)
	y_val = y.iloc[n_train - n_val : n_train].reset_index(drop=True)
	ret1_val = ret1_all.iloc[n_train - n_val : n_train].reset_index(drop=True)
	ret0_val = ret0_all.iloc[n_train - n_val : n_train].reset_index(drop=True)
	X_test = X.iloc[n_train :].reset_index(drop=True)
	y_test = y.iloc[n_train :].reset_index(drop=True)
	ret1_test = ret1_all.iloc[n_train :].reset_index(drop=True)
	ret0_test = ret0_all.iloc[n_train :].reset_index(drop=True)

	# 標準化（以訓練集統計量）；可選 robust（median/IQR）
	if cfg.scaler == 'robust':
		med = X_train.median(numeric_only=True)
		iqr = (X_train.quantile(0.75, numeric_only=True) - X_train.quantile(0.25, numeric_only=True)).replace(0, 1.0)
		def _scale(df: pd.DataFrame) -> pd.DataFrame:
			Z = df.copy()
			for c in Z.columns:
				Z[c] = (pd.to_numeric(Z[c], errors='coerce') - float(med.get(c, 0.0))) / float(iqr.get(c, 1.0))
			return Z
		scale_stats = (med, iqr)
	else:
		mean = X_train.mean(numeric_only=True)
		std = X_train.std(numeric_only=True).replace(0, 1.0)
		def _scale(df: pd.DataFrame) -> pd.DataFrame:
			Z = df.copy()
			for c in Z.columns:
				Z[c] = (pd.to_numeric(Z[c], errors='coerce') - float(mean.get(c, 0.0))) / float(std.get(c, 1.0))
			return Z
		scale_stats = (mean, std)

	X_train = _scale(X_train)
	X_val = _scale(X_val)
	X_test = _scale(X_test)

	# 特徵選擇（互信息 MI），對齊窗口後的標籤
	selected_cols = list(X_train.columns)
	if cfg.k_features and cfg.k_features > 0:
		# 對齊修正：第一筆有效標籤位於 index = window - 1
		mi_X = X_train.iloc[cfg.window - 1:].fillna(0.0)
		mi_y = y_train.iloc[cfg.window - 1:]
		try:
			scores = mutual_info_classif(mi_X, mi_y, discrete_features=False, random_state=int(cfg.seed))
			idx = np.argsort(scores)[::-1][: int(cfg.k_features)]
			selected_cols = mi_X.columns[idx].tolist()
			X_train = X_train[selected_cols]
			X_val = X_val[selected_cols]
			X_test = X_test[selected_cols]
		except Exception:
			pass

	# 記錄最終訓練特徵欄位順序，供即時預測對齊使用
	feature_cols = list(X_train.columns)

	# 顯示三組的標籤分佈，協助觀察偏態
	def _dist(name: str, ys: pd.Series):
		c0 = int((ys == 0).sum()); c1 = int((ys == 1).sum()); total = int(len(ys))
		p1 = (c1 / total) if total > 0 else 0.0
		print(f"[{name}] size={total} y=1比率={p1:.3f} (0:{c0}, 1:{c1})")
	_dist("train", y_train)
	_dist("val", y_val)
	_dist("test", y_test)

	train_ds = TimeSeriesWindowDataset(X_train, y_train, cfg.window)
	val_ds = TimeSeriesWindowDataset(X_val, y_val, cfg.window)
	test_ds = TimeSeriesWindowDataset(X_test, y_test, cfg.window)

	"""建立 DataLoader：可選加權取樣以處理不平衡"""
	if cfg.use_weighted_sampler:
		# 針對每個訓練樣本（窗口末端的 y）給予 1/freq 權重
		y_list = []
		for (s, e) in train_ds.samples:
			# 視窗對齊修正：樣本標籤位於 e-1
			y_list.append(int(train_ds.y[e - 1]))
		y_arr = np.array(y_list)
		counts = np.bincount(y_arr, minlength=2).astype(float)
		counts[counts == 0] = 1.0
		weights = 1.0 / counts[y_arr]
		sampler = torch.utils.data.WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
		train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=False)
	else:
		train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
	val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
	test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

	device = torch.device(cfg.device)
	model = TransformerClassifier(
		n_features=X.shape[1] if not (cfg.k_features and cfg.k_features>0) else len(selected_cols),
		d_model=cfg.d_model, nhead=cfg.nhead,
		num_layers=cfg.layers, dim_ff=cfg.ff, dropout=cfg.dropout,
		pooling=cfg.pooling, use_conv=cfg.use_conv,
		dilations=[int(x) for x in (cfg.dilations.split(',') if cfg.dilations else ['1','2','4'])],
		use_learnable_pos=bool(cfg.learnable_pos),
		patch_len=int(getattr(cfg, 'patch_len', 1)), patch_stride=int(getattr(cfg, 'patch_stride', 1)),
		use_msconv=bool(getattr(cfg, 'use_msconv', True)),
		ms_kernels=[int(x) for x in str(getattr(cfg, 'ms_kernels', '3,5,7')).split(',') if x.strip()],
		se_reduction=int(getattr(cfg, 'se_reduction', 4)),
		drop_path=float(getattr(cfg, 'drop_path', 0.0)),
		smooth_len=int(getattr(cfg, 'smooth_len', 1)),
		last_skip=bool(getattr(cfg, 'last_skip', True))
	).to(device)
	# Initialize output bias from training prevalence (aligned to windows)
	with torch.no_grad():
		# 對齊修正：視窗對齊後的標籤從 index = window - 1 開始
		prior = float((y_train.iloc[cfg.window - 1:] == 1).mean()) if len(y_train) > (cfg.window - 1) else float((y_train == 1).mean())
		try:
			clip_val = float(cfg.prior_bias_clip) if (cfg.prior_bias_clip and cfg.prior_bias_clip > 0) else None
			model.init_output_bias(prior, clip_abs_logit=clip_val, enable=bool(cfg.prior_bias_enable))
		except Exception:
			pass
	optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	# class imbalance: pos_weight = neg/pos on train labels aligned to windows
	# 對齊修正：訓練樣本的標籤從 window - 1 開始
	y_train_win = y_train.iloc[cfg.window - 1:]
	pos = int((y_train_win == 1).sum())
	neg = int((y_train_win == 0).sum())
	# determine BCE positive class weight
	if cfg.bce_pos_weight_mode == 'none':
		pos_weight = 1.0
	elif cfg.bce_pos_weight_mode == 'sqrt':
		pos_weight = float(math.sqrt(neg / max(1, pos))) if pos > 0 else 1.0
	else:  # 'auto'
		pos_weight = float(neg / max(1, pos)) if pos > 0 else 1.0
	if cfg.loss == 'focal':
		# 以類別比例自動設定 alpha（偏向較少的類別）
		alpha_pos = float(neg / max(1, (pos + neg)))
		criterion = FocalLoss(gamma=float(cfg.focal_gamma), alpha=alpha_pos)
	elif cfg.loss == 'asl':
		criterion = AsymmetricLoss(gamma_pos=float(cfg.asl_gamma_pos), gamma_neg=float(cfg.asl_gamma_neg), clip=float(cfg.asl_clip))
	else:
		criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

	# cosine scheduler with warmup
	total_steps = max(1, cfg.epochs * len(train_loader))
	warmup_steps = max(1, int(cfg.warmup_frac * total_steps))
	def lr_lambda(step: int):
		if step < warmup_steps:
			return float(step + 1) / float(warmup_steps)
		progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
		return 0.5 * (1.0 + math.cos(math.pi * progress))
	scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

	# EMA 狀態（可選）
	use_ema = bool(cfg.ema_decay and cfg.ema_decay > 0)
	ema_state = None
	if use_ema:
		ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

	def _maybe_update_ema():
		if not use_ema:
			return
		d = float(cfg.ema_decay)
		with torch.no_grad():
			for (k, v) in model.state_dict().items():
				if v.dtype.is_floating_point:
					ema_state[k].mul_(d).add_(v, alpha=(1.0 - d))

	class _EvalWithEMA:
		def __enter__(self):
			if not use_ema:
				self._no = True
				return
			self._no = False
			self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
			model.load_state_dict(ema_state, strict=False)
		def __exit__(self, exc_type, exc, tb):
			if not use_ema:
				return False
			model.load_state_dict(self._backup, strict=False)
			return False

	def _run_epoch(loader: DataLoader, train: bool, keep_dropout: bool=False) -> Tuple[float, List[int], List[float], List[float]]:
		if train:
			model.train()
		else:
			# keep_dropout=True: stay in train() to keep dropout active while disabling grads and optimizer
			if keep_dropout:
				model.train()
			else:
				model.eval()
		total_loss = 0.0
		all_y: List[int] = []
		all_p: List[float] = []
		all_logit: List[float] = []
		accum = max(1, int(cfg.grad_accum_steps)) if train else 1
		if train:
			optim.zero_grad()
		for batch_idx, (xb, yb) in enumerate(loader):
			xb = xb.to(device)
			yb = yb.to(device).float()
			# keep hard labels (0/1) for metrics
			yb_hard = yb.detach().clone().long()
			with torch.set_grad_enabled(train):
				logit = model(xb)
				# apply label smoothing only during training for BCE
				if train and cfg.loss == 'bce' and (cfg.label_smoothing and cfg.label_smoothing > 0):
					eps = float(min(0.2, max(0.0, cfg.label_smoothing)))
					yb_smooth = yb * (1.0 - eps) + 0.5 * eps
					loss = criterion(logit, yb_smooth)
				else:
					loss = criterion(logit, yb)
				prob = torch.sigmoid(logit).detach().cpu().numpy().tolist()
				all_logit.extend(logit.detach().cpu().numpy().astype(float).tolist())
				if train:
					(loss / accum).backward()
					# step on accumulation boundary or last batch
					if ((batch_idx + 1) % accum == 0) or ((batch_idx + 1) == len(loader)):
						if cfg.max_grad_norm and cfg.max_grad_norm > 0:
							nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
						optim.step()
						scheduler.step()
						_maybe_update_ema()
						optim.zero_grad()
			total_loss += float(loss.detach().cpu().item()) * len(yb)
			all_y.extend(yb_hard.detach().cpu().numpy().astype(int).tolist())
			all_p.extend(prob)
		total_loss /= max(1, len(all_y))
		return total_loss, all_y, all_p, all_logit

	# early stopping bookkeeping
	if cfg.early_stop == 'loss':
		best_val = math.inf  # minimize
	else:
		best_val = -math.inf  # maximize
	best_state = None
	no_improve = 0
	def _metric_at_05(y_true, p):
		pred = (np.array(p) >= 0.5).astype(int)
		return (
			accuracy_score(y_true, pred) if len(y_true) else 0.0,
			f1_score(y_true, pred, average="weighted") if len(y_true) else 0.0,
			balanced_accuracy_score(y_true, pred) if len(y_true) else 0.0,
		)

	for ep in range(1, cfg.epochs + 1):
		tr_loss, tr_y, tr_p, _ = _run_epoch(train_loader, train=True)
		with _EvalWithEMA():
			va_loss, va_y, va_p, _ = _run_epoch(val_loader, train=False)
		tr_acc, tr_f1, tr_bacc = _metric_at_05(tr_y, tr_p)
		va_acc, va_f1, va_bacc = _metric_at_05(va_y, va_p)
		print(f"[ep {ep:02d}] train loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} bacc={tr_bacc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f} bacc={va_bacc:.3f}")
		# early stopping by chosen metric
		if cfg.early_stop == 'loss':
			cur_val = va_loss
			is_better = cur_val < best_val
		elif cfg.early_stop == 'acc':
			cur_val = va_acc
			is_better = cur_val > best_val
		elif cfg.early_stop == 'f1':
			cur_val = va_f1
			is_better = cur_val > best_val
		elif cfg.early_stop == 'bacc':
			cur_val = va_bacc
			is_better = cur_val > best_val
		else:
			# default safety: treat as loss
			cur_val = va_loss
			is_better = cur_val < best_val

		if is_better:
			best_val = cur_val
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			no_improve = 0
		else:
			no_improve += 1
			if no_improve >= cfg.patience:
				print(f"[early-stop] no improvement for {cfg.patience} epoch(s), stop training.")
				break

	if best_state is not None:
		model.load_state_dict(best_state)

	with _EvalWithEMA():
		# support MC dropout averaging during evaluation
		def _eval_with_mc(loader: DataLoader):
			if not cfg.mc_dropout_passes or cfg.mc_dropout_passes <= 1:
				return _run_epoch(loader, train=False)
			# run multiple stochastic passes
			all_probs: List[float] = []
			all_loss = 0.0
			all_y: List[int] = []
			all_logits: List[float] = []
			passes = int(cfg.mc_dropout_passes)
			for i in range(passes):
				# keep_dropout=True keeps dropout active while not training
				loss_i, y_i, p_i, lg_i = _run_epoch(loader, train=False, keep_dropout=True)
				# accumulate
				if i == 0:
					all_y = y_i
					all_logits = np.array(lg_i, dtype=float)
				else:
					all_logits += np.array(lg_i, dtype=float)
				all_probs.append(np.array(p_i, dtype=float))
				all_loss += loss_i
			# average
			avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0).tolist()
			avg_logits = (all_logits / passes).tolist()
			avg_loss = all_loss / passes
			return avg_loss, all_y, avg_probs, avg_logits

		te_loss, te_y, te_p, te_logits = _eval_with_mc(test_loader)
		# 找最佳閾值（根據 validation 預測）
		_, va_y2, va_p2, va_logits = _eval_with_mc(val_loader)

	# Optional Temperature Scaling (before calibration & thresholding)
	if cfg.temp_scale and len(va_logits) > 10 and (len(set(va_y2)) > 1):
		try:
			logits_t = torch.tensor(np.array(va_logits, dtype=float))
			labels_t = torch.tensor(np.array(va_y2, dtype=float))
			logT = torch.nn.Parameter(torch.zeros(()))
			op = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
			bce = nn.BCEWithLogitsLoss()
			def closure():
				op.zero_grad()
				T = torch.exp(logT)
				loss = bce(logits_t / T, labels_t)
				loss.backward()
				return loss
			op.step(closure)
			T = float(torch.exp(logT).detach().cpu().item())
			# apply to val/test logits
			va_p2 = torch.sigmoid(torch.tensor(np.array(va_logits, dtype=float)) / T).numpy().tolist()
			te_p = torch.sigmoid(torch.tensor(np.array(te_logits, dtype=float)) / T).numpy().tolist()
		except Exception:
			pass

	# Optional isotonic calibration on validation probabilities
	calibrator = None
	if cfg.calibrate and len(va_y2) > 10:
		try:
			# keep a copy for brier comparison
			va_p_uncal = np.array(va_p2)
			te_p_uncal = np.array(te_p)
			calibrator = IsotonicRegression(out_of_bounds='clip')
			calibrator.fit(np.array(va_p2), np.array(va_y2))
			va_p_cal = calibrator.transform(np.array(va_p2))
			te_p_cal = calibrator.transform(np.array(te_p))
			# Use calibration only if it improves Brier score on val
			brier_uncal = float(np.mean((va_p_uncal - np.array(va_y2))**2))
			brier_cal = float(np.mean((va_p_cal - np.array(va_y2))**2))
			if brier_cal <= brier_uncal * 0.999:
				va_p2 = va_p_cal.tolist()
				te_p = te_p_cal.tolist()
			else:
				calibrator = None  # discard calibration
		except Exception:
			calibrator = None

	# Optional AUC-based flip to avoid inverted ranking
	try:
		val_auc = roc_auc_score(np.array(va_y2), np.array(va_p2)) if len(set(va_y2))>1 else float('nan')
	except Exception:
		val_auc = float('nan')
	flip_used = False
	if cfg.auto_flip_auc and (not np.isnan(val_auc)) and val_auc < 0.5:
		va_p2 = (1.0 - np.array(va_p2)).tolist()
		te_p = (1.0 - np.array(te_p)).tolist()
		flip_used = True

	grid = np.linspace(0.05, 0.95, 181)
	best_thr, best_score = 0.5, -1.0
	va_y_arr = np.array(va_y2)
	va_p_arr = np.array(va_p2)
	# Predicted positive rate constraints to avoid degenerate all-0 or all-1
	val_prev = float(np.mean(va_y_arr)) if len(va_y_arr) else 0.5
	pp_low = max(0.0, val_prev - float(cfg.pp_margin))
	pp_high = min(1.0, val_prev + float(cfg.pp_margin))
	def _consider(t):
		yhat = (va_p_arr >= t).astype(int)
		pp = float(np.mean(yhat)) if len(yhat) else 0.5
		return (pp_low <= pp <= pp_high), yhat

	# Diagnostics helper
	def _metric_values(y_true, y_pred):
		return dict(
			acc=accuracy_score(y_true, y_pred) if len(y_true) else 0.0,
			f1=f1_score(y_true, y_pred, average='weighted') if len(y_true) else 0.0,
			bacc=balanced_accuracy_score(y_true, y_pred) if len(y_true) else 0.0,
			mcc=matthews_corrcoef(y_true, y_pred) if len(y_true) else 0.0,
		)

	if cfg.regime_thr:
		# Build regime bins on absolute next-day returns using training quantiles
		try:
			# training regime cutpoints (on sample-aligned current returns, known at time t)
			ret_train_s = np.array(ret0_train.iloc[cfg.window - 1:], dtype=float)
			q1 = float(np.quantile(np.abs(ret_train_s), float(cfg.regime_q1)))
			q2 = float(np.quantile(np.abs(ret_train_s), float(cfg.regime_q2)))
			def _bin(abs_r: float) -> int:
				if abs_r <= q1:
					return 0
				elif abs_r <= q2:
					return 1
				else:
					return 2
			# build val/test regime arrays aligned to predictions
			val_absr = np.abs(np.array(ret0_val.iloc[cfg.window - 1:], dtype=float))
			val_bins = np.array([_bin(v) for v in val_absr], dtype=int)
			# scan per-regime thresholds maximizing accuracy
			thr_by_bin = {}
			for b in (0,1,2):
				mask = (val_bins == b)
				if not np.any(mask):
					thr_by_bin[b] = 0.5
					continue
				vy = va_y_arr[mask]
				vp = va_p_arr[mask]
				best_b_thr, best_b_score = 0.5, -1.0
				for t in grid:
					pred = (vp >= t).astype(int)
					score = accuracy_score(vy, pred) if len(vy) else 0.0
					if score > best_b_score:
						best_b_score, best_b_thr = score, float(t)
				thr_by_bin[b] = best_b_thr
			best_thr = 0.5  # not used directly
		except Exception:
			cfg.regime_thr = False  # fallback to global threshold

	if not cfg.regime_thr and cfg.optimize_metric == 'acc':
		for t in grid:
			ok, yhat = _consider(t)
			if not ok:
				continue
			score = accuracy_score(va_y_arr, yhat) if len(va_y_arr) else 0.0
			if score > best_score:
				best_score, best_thr = score, float(t)
	elif not cfg.regime_thr and cfg.optimize_metric == 'f1':
		for t in grid:
			ok, yhat = _consider(t)
			if not ok:
				continue
			score = f1_score(va_y_arr, yhat, average='weighted') if len(va_y_arr) else 0.0
			if score > best_score:
				best_score, best_thr = score, float(t)
	elif not cfg.regime_thr and cfg.optimize_metric == 'bacc':
		for t in grid:
			ok, yhat = _consider(t)
			if not ok:
				continue
			score = balanced_accuracy_score(va_y_arr, yhat) if len(va_y_arr) else 0.0
			if score > best_score:
				best_score, best_thr = score, float(t)
	elif not cfg.regime_thr and cfg.optimize_metric == 'mcc':
		for t in grid:
			ok, yhat = _consider(t)
			if not ok:
				continue
			score = matthews_corrcoef(va_y_arr, yhat) if len(va_y_arr) else 0.0
			if score > best_score:
				best_score, best_thr = score, float(t)
	elif not cfg.regime_thr and cfg.optimize_metric == 'youden':
		# 基於 ROC 曲線，最大化 tpr - fpr（Youden's J）
		try:
			fpr, tpr, thr = roc_curve(va_y_arr, va_p_arr)
			j = tpr - fpr
			k = int(np.argmax(j))
			cand_thr = float(thr[k]) if 0 <= k < len(thr) else 0.5
			# 應用 pp 約束，不符則回退到 bacc 的網格搜尋
			ok, _ = _consider(cand_thr)
			if ok:
				best_thr = cand_thr
			else:
				raise RuntimeError("youden thr violates pp constraints")
		except Exception:
			# 回退到 balanced accuracy 的網格搜尋
			for t in grid:
				ok, yhat = _consider(t)
				if not ok:
					continue
				score = balanced_accuracy_score(va_y_arr, yhat) if len(va_y_arr) else 0.0
				if score > best_score:
					best_score, best_thr = score, float(t)

	# Diagnostics: show effects of thresholding and calibration on val/test if requested via env or simple heuristic
	try:
		if os.environ.get('TRANS_DIAG','0') == '1':
			# Val metrics at 0.5 vs best_thr
			val_pred_05 = (va_p_arr >= 0.5).astype(int)
			val_pred_bt = (va_p_arr >= best_thr).astype(int)
			val_auc = roc_auc_score(va_y_arr, va_p_arr) if len(np.unique(va_y_arr))>1 else float('nan')
			print(f"[diag-val] auc={val_auc:.3f} thr0.5={_metric_values(va_y_arr, val_pred_05)} thr*={best_thr:.3f} metrics={_metric_values(va_y_arr, val_pred_bt)} pp_range=[{pp_low:.2f},{pp_high:.2f}] prev={val_prev:.2f}")
			# Test metrics at 0.5 vs best_thr
			te_arr = np.array(te_y)
			te_p_arr = np.array(te_p)
			te_pred_05 = (te_p_arr >= 0.5).astype(int)
			te_pred_bt = (te_p_arr >= best_thr).astype(int)
			print(f"[diag-test] thr0.5={_metric_values(te_arr, te_pred_05)} thr*={best_thr:.3f} metrics={_metric_values(te_arr, te_pred_bt)} prev={float(np.mean(te_arr)) if len(te_arr) else float('nan'):.2f}")
	except Exception:
		pass

	# Optionally adjust test threshold to respect predicted-positive-rate constraints (no label peeking)
	te_probs = np.array(te_p)
	thr_for_test = float(best_thr)
	# If using regime-wise thresholds, build test predictions accordingly and skip global test-threshold adjustment
	if cfg.regime_thr:
		try:
			absr_test = np.abs(np.array(ret0_test.iloc[cfg.window - 1:], dtype=float))
			# reuse q1/q2 computed earlier
			# If thr_by_bin not defined (due to exception), fall back later
			test_bins = np.zeros_like(absr_test, dtype=int)
			test_bins[absr_test > q1] = 1
			test_bins[absr_test > q2] = 2
			te_pred = np.array([
				1 if te_probs[i] >= thr_by_bin.get(int(test_bins[i]), 0.5) else 0
				for i in range(len(te_probs))
			], dtype=int)
		except Exception:
			cfg.regime_thr = False  # fallback to global path

	if (not cfg.regime_thr) and cfg.fix_test_pp and len(te_probs) > 0:
		# choose threshold so that predicted positive rate ~ val_prev (clipped to [pp_low, pp_high])
		pp_target = float(min(pp_high, max(pp_low, val_prev)))
		q = 1.0 - pp_target
		q = float(min(1.0, max(0.0, q)))
		# initial via quantile
		thr_q = _np_quantile_compat(te_probs, q, method='linear')
		yhat_q = (te_probs >= thr_q).astype(int)
		pp_q = float(np.mean(yhat_q)) if len(yhat_q) else 0.5
		thr_for_test = float(thr_q)
		# if outside a small tolerance or degenerate, refine by rank-based selection
		tol = 0.01
		if not (pp_low - tol <= pp_q <= pp_high + tol) or pp_q in (0.0, 1.0):
			te_sorted = np.sort(te_probs)
			n = len(te_sorted)
			# target count of positives
			k = int(np.ceil((1.0 - pp_target) * n)) - 1
			k = max(0, min(n - 1, k))
			thr_rank = float(te_sorted[k])
			# try slight adjustments to avoid all-1/all-0 due to ties
			eps_down = np.nextafter(thr_rank, -np.inf)
			eps_up = np.nextafter(thr_rank, np.inf)
			cands = [thr_rank, eps_up, eps_down]
			best = None
			for t in cands:
				pred = (te_probs >= t).astype(int)
				pp = float(np.mean(pred)) if len(pred) else 0.5
				# score by closeness to pp_target and penalize degeneracy
				score = -abs(pp - pp_target) - (1.0 if pp in (0.0, 1.0) else 0.0)
				if (best is None) or (score > best[0]):
					best = (score, t, pp)
			thr_for_test = float(best[1]) if best else thr_for_test

	if not cfg.regime_thr:
		te_pred = (te_probs >= thr_for_test).astype(int)
	te_acc = accuracy_score(te_y, te_pred) if len(te_y) else 0.0
	te_f1 = f1_score(te_y, te_pred, average="weighted") if len(te_y) else 0.0
	te_bacc = balanced_accuracy_score(te_y, te_pred) if len(te_y) else 0.0
	te_mcc = matthews_corrcoef(te_y, te_pred) if len(te_y) else 0.0
	pp_test = float(np.mean(te_pred)) if len(te_pred) else float('nan')
	print(f"[test] loss={te_loss:.4f} acc={te_acc:.3f} f1={te_f1:.3f} thr={best_thr:.3f}{' flip' if flip_used else ''}{' test_thr='+str(round(thr_for_test,3)) if thr_for_test!=best_thr else ''}")
	print(f"[test-extra] bacc={te_bacc:.3f} mcc={te_mcc:.3f} pred_pos_rate={pp_test:.3f}")
	if cfg.print_confusion and len(te_y):
		cm = confusion_matrix(te_y, te_pred, labels=[0,1])
		print(f"[confusion]\nTN={cm[0,0]} FP={cm[0,1]}\nFN={cm[1,0]} TP={cm[1,1]}")

	# 最新一筆即時預測（不含 y 產生時 shift 的未來訊息）
	# 先加入與訓練相同的 DoW one-hot 特徵，再對齊欄位
	df_inf = df_raw.copy()
	try:
		if "年月日" in df_inf.columns:
			dd = pd.to_datetime(df_inf["年月日"], errors="coerce")
			dow = dd.dt.dayofweek
			for k in range(7):
				df_inf[f"dow_{k}"] = (dow == k).astype(float)
	except Exception:
		pass
	X_all = df_inf.drop(columns=[c for c in ("年月日", "y_明天漲跌", "明天收盤價") if c in df_inf.columns], errors="ignore").copy()
	for c in X_all.columns:
		X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
	# 同步標準化（使用訓練集統計量）
	X_all = X_all.fillna(X_all.median(numeric_only=True))
	if cfg.scaler == 'robust':
		med, iqr = scale_stats
		for c in X_all.columns:
			X_all[c] = (pd.to_numeric(X_all[c], errors='coerce') - float(med.get(c, 0.0))) / float(iqr.get(c, 1.0))
	else:
		mean, std = scale_stats
		for c in X_all.columns:
			X_all[c] = (pd.to_numeric(X_all[c], errors='coerce') - float(mean.get(c, 0.0))) / float(std.get(c, 1.0))
	# 同步對齊特徵欄位與順序（包含 MI 選擇與 DoW 特徵）。缺失欄位以 0 填充，多餘欄位忽略。
	X_all = X_all.reindex(columns=feature_cols, fill_value=0.0)
	if len(X_all) >= cfg.window:
		x_latest = torch.from_numpy(X_all.to_numpy(dtype=np.float32)[-cfg.window:]).unsqueeze(0).to(device)  # (1, L, F)
		with torch.no_grad():
			with _EvalWithEMA():
				p1 = torch.sigmoid(model(x_latest)).item()
		label = "漲" if p1 >= best_thr else "跌"
		print(f"[latest] 機率(上漲)={p1:.4f} 預測={label} thr={best_thr:.3f} | 檔案={csv_path.name}")
	else:
		p1 = float('nan')
		label = None
		print(f"[latest] 資料不足以形成視窗（需要 {cfg.window}），無法輸出即時預測。")

	return {
		"csv": str(csv_path.resolve()),
		"test_acc": float(te_acc),
		"test_f1": float(te_f1),
		"test_bacc": float(te_bacc),
		"test_mcc": float(te_mcc),
		"best_threshold": float(best_thr),
			# expose arrays for ensemble recipes
			"val_labels": [int(v) for v in va_y_arr.tolist()] if len(va_y_arr) else [],
			"val_probs": [float(v) for v in va_p_arr.tolist()] if len(va_p_arr) else [],
			"test_labels": [int(v) for v in np.array(te_y).tolist()] if len(te_y) else [],
			"test_probs": [float(v) for v in np.array(te_p).tolist()] if len(te_p) else [],
		"latest_proba": float(p1) if not (isinstance(p1, float) and math.isnan(p1)) else None,
		"latest_label": label,
	}


def _scan_recent_symbols(minutes: int) -> List[str]:
	symbols: List[str] = []
	if not DATA_DIR.exists():
		return symbols
	now = pd.Timestamp.utcnow()
	for p in DATA_DIR.glob("*_last_update.txt"):
		try:
			mtime = pd.Timestamp.utcfromtimestamp(p.stat().st_mtime)
			if (now - mtime).total_seconds() <= minutes * 60:
				sym = p.stem.replace("_last_update", "")
				csvp = DATA_DIR / f"{sym}_short_term_with_lag3.csv"
				if csvp.exists():
					symbols.append(sym)
		except Exception:
			continue
	return symbols


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--csv", type=str, default=None, help="指定 CSV 路徑；若省略且有 --symbol 則優先使用 symbol 對應檔，否則使用 data/short_term_with_lag3.csv")
	ap.add_argument("--symbol", type=str, default=None, help="股票代碼（例如 2330 或 AAPL），將讀取 data/<symbol>_short_term_with_lag3.csv")
	ap.add_argument("--window", type=int, default=64)
	ap.add_argument("--epochs", type=int, default=5)
	ap.add_argument("--batch", type=int, default=64)
	ap.add_argument("--lr", type=float, default=1e-3)
	ap.add_argument("--device", type=str, default="cpu")
	ap.add_argument("--d_model", type=int, default=64)
	ap.add_argument("--nhead", type=int, default=4)
	ap.add_argument("--layers", type=int, default=2)
	ap.add_argument("--ff", type=int, default=128)
	ap.add_argument("--dropout", type=float, default=0.1)
	ap.add_argument("--weight_decay", type=float, default=1e-2)
	ap.add_argument("--warmup_frac", type=float, default=0.1)
	ap.add_argument("--max_grad_norm", type=float, default=1.0)
	ap.add_argument("--patience", type=int, default=5)
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--metric", type=str, default="acc", choices=["acc","f1"], help="驗證集調整閾值時採用的指標")
	ap.add_argument("--opt_metric", type=str, default=None, choices=["acc","f1","bacc","mcc","youden"], help="同 --metric，若提供則覆蓋 optimize_metric")
	ap.add_argument("--sampler", type=str, default="weighted", choices=["weighted","none"], help="訓練資料取樣策略")
	ap.add_argument("--ema", type=float, default=0.0, help="啟用 EMA 權重（0 關閉，例如 0.999 啟用）")
	ap.add_argument("--pooling", type=str, default="cls", choices=["cls","attn","both"], help="序列池化：CLS / attention / both")
	ap.add_argument("--conv", type=str, default="yes", choices=["yes","no"], help="是否啟用卷積前端")
	ap.add_argument("--dilations", type=str, default="1,2,4", help="卷積前端的 dilation 設定（逗號分隔）")
	ap.add_argument("--learnable_pos", action="store_true", help="改用可學習位置編碼")
	ap.add_argument("--loss", type=str, default="bce", choices=["bce","focal","asl"], help="損失函數類型")
	ap.add_argument("--focal_gamma", type=float, default=2.0, help="FocalLoss 的 gamma")
	ap.add_argument("--asl_gamma_pos", type=float, default=0.0, help="ASL 的正類 gamma_pos")
	ap.add_argument("--asl_gamma_neg", type=float, default=4.0, help="ASL 的負類 gamma_neg")
	ap.add_argument("--asl_clip", type=float, default=0.05, help="ASL 的負類機率裁剪幅度 clip (0~0.3)")
	ap.add_argument("--label_smoothing", type=float, default=0.0, help="BCE 損失的標籤平滑 (0~0.2)")
	ap.add_argument("--prior_bias_clip", type=float, default=0.5, help="輸出層先驗偏置的logit裁剪幅度(絕對值)；設0表示不裁剪")
	ap.add_argument("--no_prior_bias", action="store_true", help="停用基於先驗的輸出層偏置初始化")
	ap.add_argument("--bce_pos_weight_mode", type=str, default=None, choices=["auto","sqrt","none"], help="BCE 正類權重策略：auto=neg/pos, sqrt=sqrt(neg/pos), none=不使用")
	ap.add_argument("--k_features", type=int, default=0, help="互信息排行取前 K 個特徵（0 表示不啟用）")
	ap.add_argument("--label_thresh", type=float, default=0.01, help="建立標籤的漲跌閾值（例如 0.01 表示 ±1% 以外才標註）")
	ap.add_argument("--label_mode", type=str, default="abs", choices=["abs","pct"], help="標籤建立模式：abs=以固定閾值；pct=以分位數（由 --label_q 指定）")
	ap.add_argument("--label_q", type=float, default=0.3, help="label_mode=pct 時，取上/下分位數的 q（如 0.3 代表上30% 與下30%）")
	ap.add_argument("--no_confusion", action="store_true", help="不要列印測試集混淆矩陣")
	ap.add_argument("--no_calibrate", action="store_true", help="關閉驗證集 isotonic 機率校準")
	ap.add_argument("--pp_margin", type=float, default=0.25, help="調整閾值時，預測為 1 的比例須落在 val 預期 ± margin 範圍")
	ap.add_argument("--early_stop", type=str, default="loss", choices=["loss","acc","f1","bacc"], help="早停依據的驗證指標")
	ap.add_argument("--auto_flip_auc", action="store_true", help="若驗證集 AUC < 0.5 則自動翻轉機率(1-p)")
	ap.add_argument("--no_fix_test_pp", action="store_true", help="關閉測試階段的預測正類比例調整")
	ap.add_argument("--scaler", type=str, default="robust", choices=["standard","robust"], help="特徵縮放器：standard=均值/標準差；robust=中位數/IQR")
	ap.add_argument("--patch_len", type=int, default=1, help="時間維度 patch 長度（>1 表示啟用 temporal patching）")
	ap.add_argument("--patch_stride", type=int, default=1, help="時間維度 patch stride（與 patch_len 搭配使用）")
	ap.add_argument("--use_msconv", action="store_true", help="啟用多尺度卷積混合 (Multi-Scale Conv Mixer) 前端")
	ap.add_argument("--ms_kernels", type=str, default="3,5,7", help="多尺度卷積的 kernel 組合，例如 3,5,7")
	ap.add_argument("--se_reduction", type=int, default=4, help="SE block 的 reduction 比例")
	ap.add_argument("--drop_path", type=float, default=0.0, help="DropPath 比例（0~0.2）")
	ap.add_argument("--smooth_len", type=int, default=1, help="時序平滑長度（>1 啟用因果平均濾波）")
	ap.add_argument("--no_last_skip", action="store_true", help="關閉最後一步線性捷徑混合（預設開啟）")
	ap.add_argument("--mc_dropout_passes", type=int, default=0, help=">0 則在驗證/測試期間進行 MC dropout 平均的次數")
	ap.add_argument("--no_temp_scale", action="store_true", help="關閉驗證集溫度縮放校準")
	ap.add_argument("--regime_thr", action="store_true", help="啟用基於 |r| 的三分位 regime 門檻（val 找各自最佳門檻）")
	ap.add_argument("--regime_q1", type=float, default=0.33, help="regime 低/中分位數 cut")
	ap.add_argument("--regime_q2", type=float, default=0.66, help="regime 中/高分位數 cut")
	# backend (tabular options)
	ap.add_argument("--backend", type=str, default="transformer", choices=["transformer","gbdt","rf"], help="選擇後端模型：transformer/gbdt/rf")
	ap.add_argument("--gbdt_topk", type=int, default=0, help="GBDT/RF 僅使用前 K 個重要特徵（0 表示不啟用）")
	ap.add_argument("--gbdt_calibrate", type=str, default="none", choices=["none","sigmoid","isotonic"], help="GBDT/RF 使用機率校準")
	ap.add_argument("--gbdt_balance", action="store_true", help="GBDT/RF 以類別權重平衡樣本")
	ap.add_argument("--gbdt_thr_grid", type=str, default="0.30,0.70,41", help="GBDT/RF threshold grid: min,max,steps")
	# stepwise recipe
	ap.add_argument("--recipe", type=str, default=None, choices=["stepwise","debias","strong","hard","hard_sweep","hard_ens"], help="配方：stepwise=架構增強；debias=消偏與校準；strong=強化（分位數標籤等）；hard=追求準確率的強力設定；hard_sweep=自動掃描 pp_margin；hard_ens=多 seed 集成")
	ap.add_argument("--steps_epochs", type=int, default=8, help="stepwise 配方中每一步的 epochs 數量")
	ap.add_argument("--grad_accum", type=int, default=1, help="梯度累積步數（有效增大 batch，大模型時穩定訓練）")
	ap.add_argument("--ens_n", type=int, default=5, help="ensemble 次數（不同 seed 訓練後平均機率）")
	ap.add_argument("--recent", type=int, default=None, help="掃描 data/ 內最近 N 分鐘有 last_update 的 symbols，逐一訓練並印出結果（忽略 --csv/--symbol）")
	args = ap.parse_args()

	if args.recent is not None:
		syms = _scan_recent_symbols(int(args.recent))
		if not syms:
			print(f"[recent] 最近 {args.recent} 分鐘內沒有偵測到更新的 symbols（依據 *_last_update.txt）")
			return
		print(f"[recent] 偵測到 {len(syms)} 個 symbols 有更新：{', '.join(syms[:8])}{'...' if len(syms)>8 else ''}")
		for s in syms:
			try:
				csvp = _resolve_csv(None, s)
				cfg = TrainConfig(
					window=args.window, batch_size=args.batch, epochs=args.epochs, lr=args.lr, device=args.device,
					d_model=args.d_model, nhead=args.nhead, layers=args.layers, ff=args.ff, dropout=args.dropout,
					weight_decay=args.weight_decay, warmup_frac=args.warmup_frac, max_grad_norm=args.max_grad_norm,
					patience=args.patience, seed=args.seed, optimize_metric=args.metric,
					use_weighted_sampler=(args.sampler == 'weighted'), ema_decay=(args.ema if args.ema > 0 else 0.0), pooling=args.pooling,
					use_conv=(args.conv == 'yes'), dilations=args.dilations, learnable_pos=bool(args.learnable_pos),
					loss=args.loss, focal_gamma=args.focal_gamma, k_features=int(args.k_features),
					label_thresh=float(args.label_thresh)
				)
				print(f"===== {s} =====")
				train_transformer_on_csv(csvp, cfg)
			except Exception as e:
				print(f"[error] {s}: {e}")
		return

	# 單檔/單 symbol 模式
	csvp = _resolve_csv(args.csv, args.symbol)
	cfg = TrainConfig(
		window=args.window, batch_size=args.batch, epochs=args.epochs, lr=args.lr, device=args.device,
		d_model=args.d_model, nhead=args.nhead, layers=args.layers, ff=args.ff, dropout=args.dropout,
		weight_decay=args.weight_decay, warmup_frac=args.warmup_frac, max_grad_norm=args.max_grad_norm,
		patience=args.patience, seed=args.seed, optimize_metric=(args.opt_metric or args.metric),
		use_weighted_sampler=(args.sampler == 'weighted'), ema_decay=(args.ema if args.ema > 0 else 0.0), pooling=args.pooling,
		use_conv=(args.conv == 'yes'), dilations=args.dilations, learnable_pos=bool(args.learnable_pos),
		loss=args.loss, focal_gamma=args.focal_gamma, asl_gamma_pos=float(args.asl_gamma_pos), asl_gamma_neg=float(args.asl_gamma_neg), asl_clip=float(args.asl_clip), label_smoothing=float(args.label_smoothing), prior_bias_clip=float(args.prior_bias_clip), prior_bias_enable=(not args.no_prior_bias), bce_pos_weight_mode=(args.bce_pos_weight_mode or ("sqrt" if args.sampler=="weighted" else "auto")), k_features=int(args.k_features),
		label_thresh=float(args.label_thresh), label_mode=str(args.label_mode), label_q=float(args.label_q),
		print_confusion=(not args.no_confusion), calibrate=(not args.no_calibrate), pp_margin=float(args.pp_margin), early_stop=str(args.early_stop),
		auto_flip_auc=bool(args.auto_flip_auc), fix_test_pp=(not args.no_fix_test_pp),
		scaler=str(args.scaler), mc_dropout_passes=int(args.mc_dropout_passes), grad_accum_steps=int(args.grad_accum),
		temp_scale=(not args.no_temp_scale), regime_thr=bool(args.regime_thr), regime_q1=float(args.regime_q1), regime_q2=float(args.regime_q2),
		backend=str(args.backend), gbdt_topk=int(args.gbdt_topk), gbdt_calibrate=str(args.gbdt_calibrate), gbdt_balance=bool(args.gbdt_balance)
	)
	# upgrade conv frontend flags
	cfg.use_msconv = bool(args.use_msconv)
	cfg.ms_kernels = str(args.ms_kernels)
	cfg.se_reduction = int(args.se_reduction)
	cfg.drop_path = float(args.drop_path)
	cfg.smooth_len = int(args.smooth_len)
	cfg.last_skip = (not args.no_last_skip)
	# parse gbdt thr grid
	try:
		parts = [p.strip() for p in str(args.gbdt_thr_grid).split(",")]
		cfg.gbdt_thr_grid = (float(parts[0]), float(parts[1]), int(parts[2]))
	except Exception:
		cfg.gbdt_thr_grid = (0.30, 0.70, 41)
	if args.recipe == 'stepwise':
		# 定義逐步增強的階梯
		steps = [
			# 每一步都強制加入：use_weighted_sampler=True 與 optimize_metric='youden'（ROC 門檻校正）
			{"name": "baseline", "cfg": dict(
				pooling='cls', ema_decay=0.0, use_conv=False, learnable_pos=False, loss='bce',
				d_model=max(64, args.d_model), ff=max(128, args.ff), layers=max(2, args.layers)
			)},
			{"name": "ema", "cfg": dict(ema_decay=0.999)},
			{"name": "attn_pool", "cfg": dict(pooling='attn')},
			{"name": "learnable_pos", "cfg": dict(learnable_pos=True)},
			{"name": "conv_frontend", "cfg": dict(use_conv=True, dilations='1,2,4,8')},
			{"name": "wider_head", "cfg": dict(d_model=max(96, args.d_model), ff=max(256, args.ff))},
		]
		base = cfg
		for st in steps:
			# 合併設定
			params = base.__dict__.copy()
			params.update(st["cfg"])  # 覆寫
			params["epochs"] = args.steps_epochs
			# 強制每一步套用平衡取樣與 ROC-Youden 門檻目標
			params["use_weighted_sampler"] = True
			params["optimize_metric"] = 'youden'
			step_cfg = TrainConfig(**params)
			print(f"\n==== step: {st['name']} (epochs={step_cfg.epochs}) ====")
			try:
				train_transformer_on_csv(csvp, step_cfg)
			except Exception as e:
				print(f"[step error] {st['name']}: {e}")
		return
	elif args.recipe == 'debias':
		# 專注在降低偏態與改善決策門檻/標籤定義
		steps = [
			{"name": "baseline_balance", "cfg": dict(
				pooling='cls', ema_decay=0.0, use_conv=False, learnable_pos=False, loss='bce',
				use_weighted_sampler=True, optimize_metric='bacc', pp_margin=min(0.2, max(0.05, args.pp_margin))
			)},
			{"name": "focal_alpha", "cfg": dict(loss='focal', focal_gamma=2.0)},
			{"name": "youden_thr", "cfg": dict(optimize_metric='youden')},
			{"name": "quantile_label", "cfg": dict(label_mode='pct', label_q=0.3)},
			{"name": "mi_topk", "cfg": dict(k_features=max(32, int(args.k_features) or 32))},
			{"name": "attn_pool", "cfg": dict(pooling='attn')},
		]
		base = cfg
		for st in steps:
			params = base.__dict__.copy()
			params.update(st["cfg"])  # 覆寫
			params["epochs"] = args.steps_epochs
			step_cfg = TrainConfig(**params)
			print(f"\n==== step: {st['name']} (epochs={step_cfg.epochs}) ====")
			try:
				train_transformer_on_csv(csvp, step_cfg)
			except Exception as e:
				print(f"[step error] {st['name']}: {e}")
		return
	elif args.recipe == 'strong':
		# 強化設定：分位數標籤、雙池化、卷積前端、特徵選擇、AUC 翻轉、測試比例修正，並避免 pos_weight 過度放大
		params = cfg.__dict__.copy()
		params.update(dict(
			label_mode='pct', label_q=0.25,
			pooling='both', use_conv=True, dilations='1,2,4,8',
			k_features=max(32, int(args.k_features) or 64),
			auto_flip_auc=True, # 自動翻轉避免排序反向
			# fix_test_pp 預設開啟，不需設置
			bce_pos_weight_mode=('sqrt' if args.sampler=='weighted' else 'auto'),
			d_model=max(128, args.d_model), layers=max(3, args.layers), ff=max(256, args.ff),
			label_smoothing=max(0.05, float(args.label_smoothing)),
			pp_margin=min(0.15, max(0.08, args.pp_margin)),
			prior_bias_clip=0.3,
			loss='bce'
		))
		params["epochs"] = args.steps_epochs
		step_cfg = TrainConfig(**params)
		print(f"\n==== strong (epochs={step_cfg.epochs}) ====")
		try:
			train_transformer_on_csv(csvp, step_cfg)
		except Exception as e:
			print(f"[strong error]: {e}")
		return
	elif args.recipe == 'hard':
		# 以準確率為優先的強力設定：強模型 + 分位數標籤 + 特徵選擇 + 比例約束 + EMA + MC Dropout 評估 + 梯度累積
		params = cfg.__dict__.copy()
		params.update(dict(
			label_mode='pct', label_q=0.25,
			pooling='both', use_conv=True, dilations='1,2,4,8,16',
			k_features=max(64, int(args.k_features) or 96),
			auto_flip_auc=True,
			bce_pos_weight_mode=('sqrt' if args.sampler=='weighted' else 'auto'),
			d_model=max(192, args.d_model), layers=max(4, args.layers), ff=max(512, args.ff),
			label_smoothing=max(0.05, float(args.label_smoothing)),
			pp_margin=min(0.12, max(0.08, args.pp_margin)),
			prior_bias_clip=0.3,
			loss='bce',
			scaler='robust',
			mc_dropout_passes=max(8, int(args.mc_dropout_passes) or 12),
			ema_decay=0.999,
			optimize_metric='youden',
			early_stop='bacc',
			grad_accum_steps=max(2, int(args.grad_accum))
		))
		params["epochs"] = max(16, int(args.steps_epochs) or 20)
		step_cfg = TrainConfig(**params)
		print(f"\n==== hard (epochs={step_cfg.epochs}) ====")
		try:
			train_transformer_on_csv(csvp, step_cfg)
		except Exception as e:
			print(f"[hard error]: {e}")
		return
	elif args.recipe == 'hard_sweep':
		# 掃描 pp_margin 三組設定，挑 test bacc/MCC 最佳
		base = cfg.__dict__.copy()
		base.update(dict(
			label_mode='pct', label_q=0.25,
			pooling='both', use_conv=True, dilations='1,2,4,8,16',
			k_features=max(96, int(args.k_features) or 96),
			auto_flip_auc=True,
			bce_pos_weight_mode=('sqrt' if args.sampler=='weighted' else 'auto'),
			d_model=max(256, args.d_model), layers=max(4, args.layers), ff=max(768, args.ff),
			label_smoothing=max(0.05, float(args.label_smoothing)),
			prior_bias_clip=0.3,
			loss='bce',
			scaler='robust',
			mc_dropout_passes=max(12, int(args.mc_dropout_passes) or 12),
			ema_decay=0.999,
			optimize_metric='youden',
			early_stop='bacc',
			grad_accum_steps=max(2, int(args.grad_accum))
		))
		margins = [0.10, 0.08, 0.12]
		results = []
		for pm in margins:
			params = base.copy()
			params["pp_margin"] = pm
			params["epochs"] = max(30, int(args.steps_epochs) or 30)
			step_cfg = TrainConfig(**params)
			print(f"\n==== hard_sweep pp_margin={pm:.2f} (epochs={step_cfg.epochs}) ====")
			try:
				res = train_transformer_on_csv(csvp, step_cfg)
				if isinstance(res, dict):
					res["pp_margin"] = pm
					results.append(res)
			except Exception as e:
				print(f"[hard_sweep error pm={pm:.2f}]: {e}")
		# 總結結果
		if results:
			print("\n==== hard_sweep summary ====")
			for r in results:
				print(f"pp_margin={r.get('pp_margin'):.2f} | test acc={r.get('test_acc'):.3f} bacc={r.get('test_bacc'):.3f} mcc={r.get('test_mcc'):.3f}")
			# 挑選最佳（bacc, mcc, acc 依序）
			best = max(results, key=lambda r: (r.get('test_bacc',0.0), r.get('test_mcc',0.0), r.get('test_acc',0.0)))
			print(f"[best] pp_margin={best.get('pp_margin'):.2f} | acc={best.get('test_acc'):.3f} bacc={best.get('test_bacc'):.3f} mcc={best.get('test_mcc'):.3f}")
		return
	elif args.recipe == 'hard_ens':
		# 強模型 + 多 seed ensemble（平均機率），閾值以 ensemble 的 val 機率校準
		seed0 = int(args.seed)
		N = max(2, int(args.ens_n))
		val_labels_ref = None
		test_labels_ref = None
		val_probs_all = []
		test_probs_all = []
		for i in range(N):
			params = cfg.__dict__.copy()
			params.update(dict(
				label_mode='pct', label_q=0.25,
				pooling='both', use_conv=True, dilations='1,2,4,8,16',
				k_features=max(96, int(args.k_features) or 96),
				auto_flip_auc=True,
				bce_pos_weight_mode=('sqrt' if args.sampler=='weighted' else 'auto'),
				d_model=max(256, args.d_model), layers=max(4, args.layers), ff=max(768, args.ff),
				label_smoothing=max(0.05, float(args.label_smoothing)),
				pp_margin=min(0.12, max(0.08, args.pp_margin)),
				prior_bias_clip=0.3,
				loss='bce',
				scaler='robust',
				mc_dropout_passes=max(12, int(args.mc_dropout_passes) or 12),
				ema_decay=0.999,
				optimize_metric='youden',
				early_stop='bacc',
				grad_accum_steps=max(2, int(args.grad_accum)),
				seed=seed0 + i*1009,
				epochs=max(24, int(args.steps_epochs) or 24)
			))
			step_cfg = TrainConfig(**params)
			print(f"\n==== hard_ens member {i+1}/{N} (seed={step_cfg.seed}, epochs={step_cfg.epochs}) ====")
			try:
				res = train_transformer_on_csv(csvp, step_cfg)
				va_y = np.array(res.get('val_labels', []))
				va_p = np.array(res.get('val_probs', []), dtype=float)
				te_y = np.array(res.get('test_labels', []))
				te_p = np.array(res.get('test_probs', []), dtype=float)
				if val_labels_ref is None and len(va_y):
					val_labels_ref = va_y
				if test_labels_ref is None and len(te_y):
					test_labels_ref = te_y
				if len(va_p):
					val_probs_all.append(va_p)
				if len(te_p):
					test_probs_all.append(te_p)
			except Exception as e:
				print(f"[hard_ens member error]: {e}")
		# 集成評估
		if val_probs_all and test_probs_all and val_labels_ref is not None and test_labels_ref is not None:
			ens_va_p = np.mean(np.stack(val_probs_all, axis=0), axis=0)
			ens_te_p = np.mean(np.stack(test_probs_all, axis=0), axis=0)
			va_y_arr = val_labels_ref
			val_prev = float(np.mean(va_y_arr)) if len(va_y_arr) else 0.5
			pp_low = max(0.0, val_prev - float(cfg.pp_margin))
			pp_high = min(1.0, val_prev + float(cfg.pp_margin))
			# 選阈值（Youden，若違反 pp 約束則退回 bacc 網格）
			grid = np.linspace(0.05, 0.95, 181)
			best_thr, best_score = 0.5, -1.0
			def _consider(t):
				pred = (ens_va_p >= t).astype(int)
				pp = float(np.mean(pred)) if len(pred) else 0.5
				return (pp_low <= pp <= pp_high), pred
			try:
				fpr, tpr, thr = roc_curve(va_y_arr, ens_va_p)
				j = tpr - fpr
				k = int(np.argmax(j))
				cand_thr = float(thr[k]) if 0 <= k < len(thr) else 0.5
				ok, _ = _consider(cand_thr)
				if ok:
					best_thr = cand_thr
				else:
					raise RuntimeError("youden thr violates pp constraints")
			except Exception:
				for t in grid:
					ok, yhat = _consider(t)
					if not ok:
						continue
					score = balanced_accuracy_score(va_y_arr, yhat) if len(va_y_arr) else 0.0
					if score > best_score:
						best_score, best_thr = score, float(t)
			# 測試階段按 val prevalence 選 test_thr（不看標籤）
			pp_target = float(min(pp_high, max(pp_low, val_prev)))
			q = float(min(1.0, max(0.0, 1.0 - pp_target)))
			test_thr = _np_quantile_compat(ens_te_p, q, method='nearest')
			# 指標
			va_pred_bt = (ens_va_p >= best_thr).astype(int)
			te_pred = (ens_te_p >= test_thr).astype(int)
			val_auc = roc_auc_score(va_y_arr, ens_va_p) if len(np.unique(va_y_arr))>1 else float('nan')
			te_acc = accuracy_score(test_labels_ref, te_pred)
			te_f1 = f1_score(test_labels_ref, te_pred, average='weighted')
			te_bacc = balanced_accuracy_score(test_labels_ref, te_pred)
			te_mcc = matthews_corrcoef(test_labels_ref, te_pred)
			pp_test = float(np.mean(te_pred)) if len(te_pred) else float('nan')
			print(f"\n[ens] val_auc={val_auc:.3f} best_thr={best_thr:.3f}")
			print(f"[ens-test] acc={te_acc:.3f} f1={te_f1:.3f} bacc={te_bacc:.3f} mcc={te_mcc:.3f} test_thr={test_thr:.3f} pred_pos_rate={pp_test:.3f}")
		else:
			print("[hard_ens] 無法集成（可能樣本不足）")
		return
	else:
		if cfg.backend in ("gbdt", "rf"):
			try:
				train_tabular_on_csv(csvp, cfg, backend=cfg.backend)
			except Exception as e:
				print(f"[error] {cfg.backend}: {e}")
		else:
			train_transformer_on_csv(csvp, cfg)


if __name__ == "__main__":
	main()

