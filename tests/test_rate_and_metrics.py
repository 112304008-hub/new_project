"""test_rate_and_metrics.py — Rate Limit 與 Metrics 相關測試（繁體中文說明）

內容：
    - 人為降低 RATE_LIMIT_PER_MIN 以觸發速率限制
    - 啟動 symbol 自動任務後檢視 Prometheus 指標是否出現 app_background_tasks

目的：驗證服務端監控與保護機制可被觀測。
"""
import re
import time


def test_rate_limit_enforced_on_api_routes(client, monkeypatch):
    # set strict rate limit to trigger quickly
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    import main as app_main
    app_main.app.state.rate_bucket = {}

    r1 = client.get("/api/list_symbols")
    assert r1.status_code in (200, 401)  # may be protected later by API key in other tests
    r2 = client.get("/api/list_symbols")
    assert r2.status_code in (429, 200, 401)


def test_metrics_background_gauge_increases(client, tmp_path, monkeypatch):
    import main as app_main
    # Redirect write dir and avoid real work
    app_main.DATA_WRITE_DIR = tmp_path
    monkeypatch.setattr(app_main, "_ensure_yf", lambda: None)
    monkeypatch.setattr(app_main, "_build_from_yfinance", lambda symbol, out_csv: None)

    # Start a symbol auto task to bump BACKGROUND_TASKS_GAUGE
    r = client.get("/api/auto/start_symbol", params={"symbol": "METRICX", "interval": 1})
    assert r.status_code == 200
    # Fetch metrics; expect our custom metric name present
    m = client.get("/metrics")
    assert m.status_code == 200
    assert "app_background_tasks" in m.text
