import os
import sys
from pathlib import Path
import pandas as pd
import pytest

"""conftest.py — pytest 共用設定與治具（fixtures）說明（繁體中文）

提供內容：
    tmp_workspace    建立隔離的臨時工作目錄 (含 data/ models/)
    tmp_csv          建立最小可用特徵 CSV（含收盤價欄位）
    prepared_models  在臨時資料集上訓練一個輕量模型 (lr) 供推論端點使用
    client           綁定上述資源並回傳 TestClient (FastAPI)

目的：確保測試間不互相汙染，並且能在無網路或無外部依賴的環境下進行。
"""
# Ensure project root is importable (so `import main` works when running pytest from repo root)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# FastAPI test client
from fastapi.testclient import TestClient

import importlib
import stock as stock_mod


@pytest.fixture(scope="session")
def tmp_workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("ws")
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "models").mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session")
def tmp_csv(tmp_workspace: Path):
    # Create a deterministic CSV with clear >1% daily moves so y isn't all NaN
    import pandas as pd
    import numpy as np
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Alternate +3% and -3% daily to ensure labels exist with THRESH=1%
    steps = np.tile([1.03, 0.97], (n // 2) + 1)[:n]
    prices = 100 * np.cumprod(steps)
    df = pd.DataFrame({
        "年月日": dates,
        "收盤價(元)": prices.round(2),
    })
    df.to_csv(tmp_workspace / "data" / "short_term_with_lag3.csv", index=False, encoding="utf-8-sig")
    return tmp_workspace / "data" / "short_term_with_lag3.csv"


@pytest.fixture(scope="session")
def prepared_models(tmp_workspace: Path, tmp_csv: Path):
    # Point stock module to temp dirs and train a tiny model once
    stock_mod.MODELS_DIR = tmp_workspace / "models"
    stock_mod.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stock_mod.DATA_DIR = tmp_workspace / "data"
    # Train a lightweight model (lr) on the small CSV
    stock_mod.train(str(tmp_csv), models_to_train=["lr"])  # creates lr_pipeline.pkl + lr_threshold.pkl
    return stock_mod.MODELS_DIR


@pytest.fixture()
def client(tmp_workspace: Path, tmp_csv: Path, prepared_models: Path):
    # 確保 stock 指向臨時 models/data 目錄（需在 import main 之前）
    stock_mod.MODELS_DIR = tmp_workspace / "models"
    stock_mod.DATA_DIR = tmp_workspace / "data"

    # 延遲 import main，確保 `from stock import MODELS_DIR` 會拿到上面設定
    app_main = importlib.import_module("main")

    # 設定 main 的 data 目錄（本版 API 依據 symbol 從 data/ 取檔）
    app_main.DATA_DIR = tmp_workspace / "data"
    app_main.DATA_WRITE_DIR = tmp_workspace / "data"
    # 關閉 API key 保護
    app_main.API_KEY = None

    # 準備幾個 symbol 的 CSV 用於測試（包含 >1% 的日變動，避免全為 NaN 標籤）
    import numpy as np
    for sym in ("AAPL", "MSFT"):
        n = 60
        dates = pd.date_range("2024-02-01", periods=n, freq="D")
        steps = np.tile([1.03, 0.97], (n // 2) + 1)[:n]
        prices = 120 * np.cumprod(steps)
        df = pd.DataFrame({
            "年月日": dates,
            "收盤價(元)": prices.round(2),
        })
        df.to_csv(tmp_workspace / "data" / f"{sym}_short_term_with_lag3.csv", index=False, encoding="utf-8-sig")

    # 以 AAPL 的資料重新訓練一次輕量 lr 模型，使推論使用到的特徵與該 CSV 對齊
    stock_mod.train(str(tmp_workspace / "data" / "AAPL_short_term_with_lag3.csv"), models_to_train=["lr"])

    return TestClient(app_main.app)
