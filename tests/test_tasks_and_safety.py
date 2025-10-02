import time
from pathlib import Path
import pandas as pd


def test_bulk_build_start_and_status_monkeypatched(client, tmp_path, monkeypatch):
    import main as app_main

    # 把資料寫到 pytest 的暫存資料夾
    app_main.DATA_WRITE_DIR = tmp_path

    def fake_build(symbol: str, out_csv: Path):
        print(f"[FAKE BUILD] called with {symbol} -> {out_csv}")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"年月日": ["2024-01-01"], "收盤價(元)": [100.0]}
        ).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 🔑 patch 掉真正會被呼叫的函數
    monkeypatch.setattr("main._build_from_yfinance", fake_build)
    monkeypatch.setattr("main._ensure_yf", lambda: None)

    # 🔑 用 async def 包 fake worker，讓 loop.create_task() 不會噴錯
    async def fake_bulk_worker(symbols, concurrency, task_id):
        for s in symbols:
            fake_build(s, app_main.DATA_WRITE_DIR / f"{s}_short_term_with_lag3.csv")
        app_main.BULK_TASKS[task_id]["done"] = len(symbols)
        app_main.BULK_TASKS[task_id]["status"] = "completed"
        app_main.BULK_TASKS[task_id]["finished_at"] = time.time()

    monkeypatch.setattr("main._bulk_build_worker", fake_bulk_worker)

    # 啟動 bulk build
    r = client.get("/api/bulk_build_start", params={"symbols": "AAA,BBB", "concurrency": 2})
    assert r.status_code == 200
    task_id = r.json()["task_id"]

    # 立刻查詢狀態 → 應該已經 completed
    s = client.get("/api/bulk_build_status", params={"task_id": task_id})
    assert s.status_code == 200
    sj = s.json()["task"]
    assert sj.get("status") == "completed"
    assert sj.get("done") == sj.get("total") == 2

    # 驗證檔案真的被 fake_build 建出來
    for sym in ["AAA", "BBB"]:
        f = tmp_path / f"{sym}_short_term_with_lag3.csv"
        assert f.exists()
