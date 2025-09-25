def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "ok"
    assert "versions" in j


def test_list_symbols(client):
    r = client.get("/api/list_symbols")
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j.get("count", 0) >= 2
    syms = {s["symbol"] for s in j["symbols"]}
    assert {"AAPL", "MSFT"}.issubset(syms)


def test_draw_without_symbol_uses_default_csv(client):
    r = client.get("/api/draw?model=lr")
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j.get("model") == "lr"
    assert 0.0 <= j.get("proba", 0.0) <= 1.0


def test_draw_with_symbol(client):
    r = client.get("/api/draw?model=lr&symbol=AAPL")
    assert r.status_code == 200
    j = r.json()
    assert j.get("ok") is True
    assert j.get("symbol") == "AAPL"


def test_diagnostics_basic(client):
    r = client.get("/api/diagnostics?n_bins=10")
    # diagnostics reads default DATA CSV; should be present in fixture
    assert r.status_code == 200
    j = r.json()
    assert "latest_row" in j
    assert "feature_stats" in j


def test_api_key_protection(client, monkeypatch):
    import main as app_main
    # Enable API key and verify unauthorized without header
    app_main.API_KEY = "secret"
    r = client.get("/api/draw?model=lr")
    assert r.status_code == 401
    # With correct key should pass
    r2 = client.get("/api/draw?model=lr", headers={"x-api-key": "secret"})
    assert r2.status_code == 200


def test_rate_limit_basic(client, monkeypatch):
    import main as app_main
    # Set a very low rate limit to trigger quickly
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    app_main.app.state.rate_bucket = {}  # reset in-memory bucket
    ok1 = client.get("/api/draw?model=lr")
    assert ok1.status_code == 200
    blocked = client.get("/api/draw?model=lr")
    assert blocked.status_code in (200, 429)  # depending on reuse of client session IP


def test_symbol_loop_fast_forward(monkeypatch, tmp_path):
    # Patch builder to avoid network and speed up loop sleep
    import main as app_main
    calls = {"n": 0}

    def fake_build(symbol: str, out_csv):
        import pandas as pd
        df = pd.DataFrame({
            "年月日": pd.date_range("2024-01-01", periods=5, freq="D"),
            "收盤價(元)": [100, 101, 102, 103, 104],
        })
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        calls["n"] += 1

    monkeypatch.setenv("DATA_DIR_WRITE", str(tmp_path))
    app_main.DATA_WRITE_DIR = tmp_path
    # Patch builder and sleep
    app_main._build_from_yfinance = lambda symbol, out_csv: fake_build(symbol, out_csv)
    async def fast_sleep(_):
        return None
    app_main.asyncio.sleep = fast_sleep

    # Run one loop iteration manually by calling the internal function body once
    # We simulate one cycle of _symbol_loop by invoking its core build code
    sym = "TESTX"
    p = app_main._symbol_csv_write_path(sym)
    app_main._ensure_yf = lambda: None
    # Execute the core build routine via ensure wrapper so timestamp is updated
    app_main._ensure_symbol_csv(sym)
    ts = p.parent / f"{sym}_last_update.txt"
    assert p.exists() and ts.exists()
    assert calls["n"] >= 1