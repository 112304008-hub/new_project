import time
from pathlib import Path


def test_latest_features_file_safety(client, tmp_path):
    # Trying to read outside data/ should be rejected
    outside = tmp_path / "evil.csv"
    outside.write_text("x\n1\n", encoding="utf-8-sig")
    r = client.get("/api/latest_features", params={"file": str(outside)})
    assert r.status_code in (400, 404)
    assert "data/" in r.text or "必須位於" in r.text


def test_bulk_build_start_and_status_monkeypatched(client, tmp_path, monkeypatch):
    import main as app_main

    # Route writes to temp workspace and avoid network calls
    app_main.DATA_WRITE_DIR = tmp_path

    def fake_build(symbol: str, out_csv: Path):
        import pandas as pd
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"年月日": ["2024-01-01"], "收盤價(元)": [100.0]}).to_csv(out_csv, index=False, encoding="utf-8-sig")

    monkeypatch.setattr(app_main, "_ensure_yf", lambda: None)
    monkeypatch.setattr(app_main, "_build_from_yfinance", lambda symbol, out_csv: fake_build(symbol, out_csv))

    r = client.get("/api/bulk_build_start", params={"symbols": "AAA,BBB", "concurrency": 2})
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True and j.get("task_id")
    task_id = j["task_id"]

    # Poll for completion briefly
    for _ in range(50):
        s = client.get("/api/bulk_build_status", params={"task_id": task_id})
        assert s.status_code == 200
        sj = s.json()["task"]
        if sj.get("status") == "completed":
            assert sj.get("done") == sj.get("total") == 2
            break
        time.sleep(0.05)
    else:
        raise AssertionError("bulk build did not complete in time")


def test_auto_start_symbol_persists_registry(client, tmp_path, monkeypatch):
    import main as app_main

    # Redirect registry to tmp dir and avoid actual builds
    app_main.DATA_WRITE_DIR = tmp_path
    app_main.AUTO_REG_FILE = tmp_path / "auto_registry.json"
    app_main.INDEX_AUTO_REG_FILE = tmp_path / "index_auto_registry.json"
    monkeypatch.setattr(app_main, "_ensure_yf", lambda: None)
    monkeypatch.setattr(app_main, "_build_from_yfinance", lambda symbol, out_csv: None)

    r = client.get("/api/auto/start_symbol", params={"symbol": "ZZZ", "interval": 5})
    assert r.status_code == 200
    # Registry file should be written
    assert app_main.AUTO_REG_FILE.exists()
    data = app_main.AUTO_REG_FILE.read_text(encoding="utf-8")
    assert "ZZZ" in data
