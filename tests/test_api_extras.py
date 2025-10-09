"""test_api_extras.py — 進階/附加 API 端點測試（繁體中文說明）

涵蓋：
    - /version /metrics
    - /api/series /api/latest_features 邊界與錯誤
    - /api/build_symbol 模擬建置
    - /api/list_symbols 寫入目錄優先權
    - 模型缺失 / API key 保護錯誤路徑

目的：針對非最核心但常見輔助端點做回歸驗證。
"""
import os
from pathlib import Path
import time

import pytest


def test_version_endpoint(client):
    r = client.get("/version")
    assert r.status_code == 200
    j = r.json()
    assert "git_sha" in j
    assert "python" in j


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    # prometheus_client CONTENT_TYPE_LATEST
    ctype = r.headers.get("content-type", "")
    assert ctype.startswith("text/plain")


def test_series_happy_path(client):
    r = client.get("/api/series", params={"feature": "收盤價(元)", "n": 5})
    assert r.status_code == 200
    j = r.json()
    assert j["feature"] == "收盤價(元)"
    assert isinstance(j["values"], list)
    assert 0 < len(j["values"]) <= 5


def test_series_missing_feature_returns_404(client):
    r = client.get("/api/series", params={"feature": "NOT_EXISTS"})
    assert r.status_code == 404
    assert "找不到欄位" in r.text


def test_latest_features_defaults_and_regex_errors(client):
    # defaults should include at least the numeric column when no lag features exist
    r = client.get("/api/latest_features")
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j.get("selected_columns"), list)
    assert len(j["selected_columns"]) >= 1
    # bad regex should return 400
    r2 = client.get("/api/latest_features", params={"pattern": "(oops"})
    assert r2.status_code == 400


def test_build_symbol_param_validation(client):
    # missing required query param -> FastAPI validation error 422
    r = client.get("/api/build_symbol")
    assert r.status_code == 422


def test_build_symbol_with_mocked_builder(client, tmp_path, monkeypatch):
    import main as app_main
    # route writes to DATA_WRITE_DIR via _symbol_csv_write_path; ensure it's writable
    app_main.DATA_WRITE_DIR = tmp_path

    def fake_build(symbol: str, out_csv: Path):
        import pandas as pd
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"年月日": ["2024-01-01"], "收盤價(元)": [123.45]}).to_csv(out_csv, index=False, encoding="utf-8-sig")

    monkeypatch.setattr(app_main, "_ensure_yf", lambda: None)
    monkeypatch.setattr(app_main, "_build_from_yfinance", lambda symbol, out_csv: fake_build(symbol, out_csv))

    r = client.get("/api/build_symbol", params={"symbol": "ZZZ"})
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    p = Path(j["csv"])
    assert p.exists() and p.read_text(encoding="utf-8-sig").strip() != ""


def test_list_symbols_prefers_write_dir(client, tmp_path):
    import main as app_main
    # Arrange: create duplicates in data (read-only base) and data_write; write_dir should be preferred
    app_main.DATA_WRITE_DIR = tmp_path
    p_write = app_main.DATA_WRITE_DIR / "DUP_short_term_with_lag3.csv"
    p_read = app_main.DATA_DIR / "DUP_short_term_with_lag3.csv"
    p_write.parent.mkdir(parents=True, exist_ok=True)
    p_write.write_text("col\n1\n", encoding="utf-8-sig")
    p_read.write_text("col\n2\n", encoding="utf-8-sig")

    r = client.get("/api/list_symbols")
    assert r.status_code == 200
    j = r.json()
    found = [s for s in j["symbols"] if s["symbol"] == "DUP"]
    assert len(found) == 1
    # ensure path points to write dir copy
    assert str(p_write.resolve()) == found[0]["csv"]


def test_draw_missing_model_returns_404(client):
    # choose a non-existing model name to trigger the 404 error path
    r = client.get("/api/draw", params={"model": "no_such_model"})
    assert r.status_code == 404
    assert "未發現已訓練的模型" in r.text


def test_api_key_enforcement_on_api_routes(client):
    import main as app_main
    app_main.API_KEY = "k"
    # protected route without key
    r = client.get("/api/list_symbols")
    assert r.status_code == 401
    # with key
    r2 = client.get("/api/list_symbols", headers={"x-api-key": "k"})
    assert r2.status_code == 200
    app_main.API_KEY = None
