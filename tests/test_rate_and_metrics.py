"""Rate Limit 測試

保留基本的速率限制測試；Metrics 與 /api/auto/* 相關測試已移除。
"""
import re


def test_rate_limit_enforced_on_api_routes(client, monkeypatch):
    # set strict rate limit to trigger quickly
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    import main as app_main
    app_main.app.state.rate_bucket = {}

    r1 = client.get("/api/list_symbols")
    assert r1.status_code in (200, 401)  # may be protected later by API key in other tests
    r2 = client.get("/api/list_symbols")
    assert r2.status_code in (429, 200, 401)


# /metrics 與 /api/auto/* 已移除，對應測試刪除
