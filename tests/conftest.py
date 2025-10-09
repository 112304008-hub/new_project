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

import main as app_main
import stock as stock_mod


@pytest.fixture(scope="session")
def tmp_workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("ws")
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "models").mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session")
def tmp_csv(tmp_workspace: Path):
    # Create a minimal but valid CSV with required columns
    import pandas as pd
    import numpy as np
    n = 60
    df = pd.DataFrame({
        "年月日": pd.date_range("2024-01-01", periods=n, freq="D"),
        "收盤價(元)": pd.Series(100 + np.cumsum(np.random.randn(n))).round(2),
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
    # Monkeypatch main module paths to use temp workspace
    app_main.DATA = tmp_csv
    app_main.DATA_DIR = tmp_workspace / "data"
    app_main.DATA_WRITE_DIR = tmp_workspace / "data"
    app_main.MODELS_DIR = prepared_models
    # Ensure API key is disabled for tests
    app_main.API_KEY = None
    # Also ensure stock module points to our prepared models
    stock_mod.MODELS_DIR = prepared_models
    # Create a couple of symbol CSVs to test list_symbols
    for sym in ("AAPL", "MSFT"):
        df = pd.DataFrame({
            "年月日": pd.date_range("2024-02-01", periods=10, freq="D"),
            "收盤價(元)": [150 + i for i in range(10)],
        })
        df.to_csv(tmp_workspace / "data" / f"{sym}_short_term_with_lag3.csv", index=False, encoding="utf-8-sig")
    return TestClient(app_main.app)
