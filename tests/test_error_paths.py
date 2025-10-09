"""test_error_paths.py — 錯誤處理與異常路徑測試（繁體中文說明）

測試焦點：
    - 模板缺失時首頁 404
    - diagnostics：檔案缺失 / 損毀 (非 CSV) 回應碼
    - build_symbols 混合成功與失敗情境
    - bulk_build_status 不存在 task
    - latest_features 指定 symbol CSV
    - /metrics 是否包含自訂計數器

確保例外情況回傳合理 HTTP 狀態與訊息。
"""
import uuid
from pathlib import Path


def test_home_missing_template_returns_404(client, monkeypatch, tmp_path):
    import main as app_main
    app_main.HTML = tmp_path / "no_template.html"  # ensure missing
    r = client.get("/")
    assert r.status_code == 404


def test_diagnostics_missing_csv_returns_404(client, monkeypatch, tmp_path):
    import main as app_main
    app_main.DATA = tmp_path / "no.csv"
    r = client.get("/api/diagnostics")
    assert r.status_code == 404


def test_diagnostics_corrupted_csv_returns_500(client, monkeypatch, tmp_path):
    import main as app_main
    bad = tmp_path / "short_term_with_lag3.csv"
    bad.write_text("\x00\x00not a csv", encoding="utf-8", errors="ignore")
    app_main.DATA = bad
    r = client.get("/api/diagnostics")
    assert r.status_code == 500


def test_build_symbols_mixed_results(client, monkeypatch, tmp_path):
    import main as app_main
    app_main.DATA_WRITE_DIR = tmp_path

    def fake_ok(symbol: str, out_csv: Path):
        import pandas as pd
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"年月日": ["2024-01-01"], "收盤價(元)": [1]}).to_csv(out_csv, index=False, encoding="utf-8-sig")

    def fake_build(symbol: str, out_csv: Path):
        if symbol == "OK":
            return fake_ok(symbol, out_csv)
        raise RuntimeError("boom")

    monkeypatch.setattr(app_main, "_ensure_yf", lambda: None)
    monkeypatch.setattr(app_main, "_build_from_yfinance", lambda symbol, out_csv: fake_build(symbol, out_csv))

    r = client.get("/api/build_symbols", params={"symbols": "OK,FAIL"})
    assert r.status_code == 200
    j = r.json()["results"]
    assert j["OK"]["ok"] is True and "csv" in j["OK"]
    assert j["FAIL"]["ok"] is False and "error" in j["FAIL"]


def test_bulk_build_status_not_found(client):
    r = client.get("/api/bulk_build_status", params={"task_id": str(uuid.uuid4())})
    assert r.status_code == 404


def test_latest_features_symbol_path(client, tmp_path, monkeypatch):
    import main as app_main
    app_main.DATA_WRITE_DIR = tmp_path
    p = app_main.DATA_WRITE_DIR / "ZZX_short_term_with_lag3.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("年月日,收盤價(元)\n2024-01-01,123.4\n", encoding="utf-8-sig")
    r = client.get("/api/latest_features", params={"symbol": "ZZX"})
    assert r.status_code == 200
    j = r.json()
    assert j["source"].endswith("ZZX_short_term_with_lag3.csv")


def test_metrics_contains_counters(client):
    # hit an endpoint to increment counters
    client.get("/health")
    m = client.get("/metrics")
    assert m.status_code == 200
    body = m.text
    assert "app_http_requests_total" in body
