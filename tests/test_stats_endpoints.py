"""test_stats_endpoints.py — 統計分析相關端點測試（繁體中文說明）

涵蓋：
    - /api/stattests 正常與欄位缺失情形
    - /api/lag_stats 自訂 CSV 情境
    - /api/series 搭配 symbol CSV

目的：確保統計檢定與時間序列特徵查詢端點在典型與邊界案例下都能回傳可解析 JSON。
"""
import re
from pathlib import Path


def test_stattests_happy_path(client):
    # 需指定 symbol；使用 conftest 建立的 AAPL
    r = client.get("/api/stattests", params={"feature": "收盤價(元)", "symbol": "AAPL"})
    assert r.status_code == 200
    j = r.json()
    assert j["feature"] == "收盤價(元)"
    # keys present (values may be None for some stats depending on data)
    assert set(j.keys()) >= {"shapiro", "ttest", "adf"}


def test_stattests_missing_feature_404(client):
    r = client.get("/api/stattests", params={"feature": "NOT_EXISTS", "symbol": "AAPL"})
    assert r.status_code == 404


def test_lag_stats_with_custom_csv(client, tmp_path, monkeypatch):
    import main as app_main
    import pandas as pd

    df = pd.DataFrame({
        "年月日": pd.date_range("2024-01-01", periods=30, freq="D"),
        "收盤價(元)": [100 + i for i in range(30)],
        "lag1": [i for i in range(30)],
        "lag2": [i * 0.5 for i in range(30)],
    })
    app_main.DATA_DIR = tmp_path
    p = tmp_path / "CUS_short_term_with_lag3.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    r = client.get("/api/lag_stats", params={"symbol": "CUS"})
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j.get("count", 0) >= 2
    feats = {e.get("feature") for e in j["results"]}
    assert {"lag1", "lag2"}.issubset(feats)


def test_series_with_symbol_uses_symbol_csv(client, tmp_path):
    import main as app_main
    app_main.DATA_DIR = tmp_path
    p = app_main.DATA_DIR / "SYM1_short_term_with_lag3.csv"
    p.write_text("年月日,收盤價(元)\n2024-01-01,99\n2024-01-02,101\n", encoding="utf-8-sig")

    r = client.get("/api/series", params={"feature": "收盤價(元)", "symbol": "SYM1", "n": 2})
    assert r.status_code == 200
    j = r.json()
    assert j["feature"] == "收盤價(元)"
    assert j["values"][-1] in (101, 101.0)
