import re
from pathlib import Path


def test_stattests_happy_path(client):
    # use default tmp_csv from conftest (has 收盤價(元))
    r = client.get("/api/stattests", params={"feature": "收盤價(元)"})
    assert r.status_code == 200
    j = r.json()
    assert j["feature"] == "收盤價(元)"
    # keys present (values may be None for some stats depending on data)
    assert set(j.keys()) >= {"shapiro", "ttest", "adf"}


def test_stattests_missing_feature_404(client):
    r = client.get("/api/stattests", params={"feature": "NOT_EXISTS"})
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
    p = tmp_path / "short_term_with_lag3.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    app_main.DATA = p

    r = client.get("/api/lag_stats")
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j.get("count", 0) >= 2
    feats = {e.get("feature") for e in j["results"]}
    assert {"lag1", "lag2"}.issubset(feats)


def test_series_with_symbol_uses_symbol_csv(client, tmp_path):
    import main as app_main
    p = tmp_path / "SYM1_short_term_with_lag3.csv"
    p.write_text("年月日,收盤價(元)\n2024-01-01,99\n2024-01-02,101\n", encoding="utf-8-sig")
    app_main.DATA_WRITE_DIR = tmp_path

    r = client.get("/api/series", params={"feature": "收盤價(元)", "symbol": "SYM1", "n": 2})
    assert r.status_code == 200
    j = r.json()
    assert j["feature"] == "收盤價(元)"
    assert j["values"][-1] in (101, 101.0)
