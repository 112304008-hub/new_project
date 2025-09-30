import json
import time


def test_auto_start_index_and_registry(client, tmp_path, monkeypatch):
    import main as app_main

    # Use temp registry files and avoid network in index loop
    app_main.DATA_WRITE_DIR = tmp_path
    app_main.AUTO_REG_FILE = tmp_path / "auto_registry.json"
    app_main.INDEX_AUTO_REG_FILE = tmp_path / "index_auto_registry.json"

    # Monkeypatch fetch index tickers to a small static list
    monkeypatch.setattr(app_main, "_fetch_index_tickers", lambda idx: ["AAA", "BBB"])
    # Avoid real network/file ops inside loop
    monkeypatch.setattr(app_main, "_ensure_yf", lambda: None)
    monkeypatch.setattr(app_main, "_build_from_yfinance", lambda symbol, out_csv: None)

    r = client.get("/api/auto/start_index", params={"index": "sp500", "interval": 1, "concurrency": 2})
    assert r.status_code == 200
    # registry should persist
    assert app_main.INDEX_AUTO_REG_FILE.exists()
    reg = json.loads(app_main.INDEX_AUTO_REG_FILE.read_text(encoding="utf-8"))
    assert "sp500" in reg

    # stop the index loop
    s = client.get("/api/auto/stop_index", params={"index": "sp500"})
    assert s.status_code == 200


def test_auto_start_existing_csvs_enumerates(client, tmp_path, monkeypatch):
    import main as app_main
    app_main.DATA_DIR = tmp_path
    # create a couple CSVs
    for sym in ("AAA", "BBB"):
        (tmp_path / f"{sym}_short_term_with_lag3.csv").write_text("x\n1\n", encoding="utf-8-sig")
    # avoid actual building in loop creation
    monkeypatch.setattr(app_main, "_ensure_yf", lambda: None)
    monkeypatch.setattr(app_main, "_build_from_yfinance", lambda symbol, out_csv: None)
    r = client.get("/api/auto/start_existing_csvs", params={"interval": 1})
    assert r.status_code == 200
    j = r.json()
    assert j.get("count", 0) >= 2
