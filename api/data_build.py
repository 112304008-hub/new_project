"""資料建置 API 

本模組處理：
1. 單一/批次建置 CSV
2. 列出現有 CSV
3. 手動觸發全域更新（立即執行所有 CSV 的覆寫更新）
4. 背景自動更新循環（每 5 分鐘）
"""
import time as _time
import asyncio
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api", tags=["data"])


async def _all_symbols_loop(interval_min: int = 5, concurrency: int = 4, backoff_factor: float = 2.0, max_backoff_min: int = 30):
    """定期掃描 data/ 並受控併發更新；失敗採指數退避。"""
    # 延遲導入避免循環依賴
    import main
    from datetime import datetime
    from stock import _ensure_td, _build_from_td
    
    log = logging.getLogger("app")
    log.info("[global auto] start every %s min (concurrency=%s, backoff=%s, max=%sm)", interval_min, concurrency, backoff_factor, max_backoff_min)
    consecutive_failures, base = 0, max(1, int(interval_min))
    while True:
        cycle_failed = False
        try:
            syms = list(dict.fromkeys(p.stem.replace('_short_term_with_lag3','').upper() for p in main.DATA_DIR.glob('*_short_term_with_lag3.csv')))
        except Exception as e:
            log.warning("[global auto] 掃描失敗：%s", e); syms = []; cycle_failed = True
        if not syms:
            log.info("[global auto] 尚無可更新之 CSV，稍後再試")
        sem = asyncio.Semaphore(max(1, int(concurrency)))
        async def _build_one(sym: str):
            async with sem:
                p = main._symbol_csv_write_path(sym)
                try:
                    await asyncio.to_thread(lambda: (_ensure_td(), _build_from_td(symbol=sym, out_csv=p)))
                    try: (p.parent / f"{sym}_last_update.txt").write_text(datetime.now().isoformat(), encoding="utf-8")
                    except Exception: pass
                    return None
                except Exception as e:
                    log.warning("[global auto] 更新 %s 失敗: %s", sym, e); return e
        if syms:
            res = await asyncio.gather(*[asyncio.create_task(_build_one(s)) for s in syms])
            cycle_failed = any(res)
        if cycle_failed:
            consecutive_failures += 1
            sleep_for = min(int(max_backoff_min), int(base * (backoff_factor ** consecutive_failures)))
            log.info("[global auto] 失敗 backoff %s 分鐘 (連續 %s)", sleep_for, consecutive_failures)
        else:
            if consecutive_failures: log.info("[global auto] 復原成功，重置 backoff")
            consecutive_failures, sleep_for = 0, base
        try: await asyncio.sleep(sleep_for * 60)
        except asyncio.CancelledError: log.info("[global auto] loop cancelled"); raise


@router.get('/build_symbol')
def build_symbol(symbol: str):
    """按需建置單一 symbol 的 CSV，成功則回傳路徑，否則回傳錯誤。
    
    - 若 CSV 不存在：從 twelvedata 抓取資料並建立檔案
    - 若 CSV 存在：直接回傳現有檔案路徑（不覆寫）
    """
    # 延遲導入避免循環依賴
    import main
    
    start_ts = _time.time()
    print(f"[build_symbol] trigger symbol={symbol!r}")
    
    if not symbol:
        print("[build_symbol] error: missing symbol param")
        return JSONResponse({"ok": False, "error": "請提供 symbol 參數"}, status_code=400)
    
    try:
        csv_path = main._ensure_symbol_csv(symbol).resolve()
        size = csv_path.stat().st_size if csv_path.exists() else 0
        elapsed = _time.time() - start_ts
        print(f"[build_symbol] success symbol={symbol} csv={csv_path} size={size}B elapsed={elapsed:.3f}s")
        return {"ok": True, "symbol": symbol, "csv": str(csv_path)}
    except Exception as e:
        elapsed = _time.time() - start_ts
        print(f"[build_symbol] failed symbol={symbol} error={e} elapsed={elapsed:.3f}s")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get('/build_symbols')
def build_symbols(symbols: str):
    """一次建置多個 symbol 的 CSV（以逗號分隔，例如 '2330,2317,AAPL'）。
    
    每個 symbol 會依序處理：
    - 不存在的會從 twelvedata 抓取並建立
    - 已存在的直接回傳路徑
    """
    # 延遲導入避免循環依賴
    import main
    
    if not symbols:
        return JSONResponse({"ok": False, "error": "請提供 symbols 參數"}, status_code=400)
    
    def _one(s: str):
        try:
            return {"ok": True, "csv": str(main._ensure_symbol_csv(s).resolve())}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    symbol_list = [x.strip() for x in symbols.split(',') if x.strip()]
    return {"ok": True, "results": {s: _one(s) for s in symbol_list}}


@router.get('/list_symbols')
def list_symbols():
    """列出存在於 data/ 中的 symbol CSV。
    
    回傳格式：
    {
        "ok": True,
        "count": 數量,
        "symbols": [
            {"symbol": "AAPL", "csv": "絕對路徑", "size": 檔案大小(bytes)},
            ...
        ]
    }
    """
    # 延遲導入避免循環依賴
    import main
    
    out = []
    seen = set()
    
    try:
        # 優先從可寫目錄，再從唯讀 data 目錄讀取
        for dir_ in (main.DATA_WRITE_DIR, main.DATA_DIR):
            if not dir_.exists():
                continue
            for p in dir_.glob('*_short_term_with_lag3.csv'):
                sym = p.stem.replace('_short_term_with_lag3', '').upper()
                if sym in seen:
                    continue
                seen.add(sym)
                out.append({
                    "symbol": sym,
                    "csv": str(p.resolve()),
                    "size": p.stat().st_size
                })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    
    return {"ok": True, "count": len(out), "symbols": out}


@router.get('/refresh_all')
async def refresh_all():
    """手動觸發批次更新：立即掃描 data/ 並覆寫所有現有 CSV。
    
    與自動每五分鐘更新的差異：
    - 自動更新：背景執行，每 5 分鐘一次
    - 手動更新：立即執行一次，適合需要即時更新的場景
    
    更新流程：
    1. 掃描 data/ 找出所有現有的 CSV
    2. 從 twelvedata 抓取最新資料
    3. 覆寫對應的 CSV 檔案
    4. 更新 last_update.txt 時間戳
    """
    # 延遲導入避免循環依賴
    import main
    from datetime import datetime
    from stock import _ensure_td, _build_from_td
    
    start_time = _time.time()
    
    # 掃描現有 CSV
    try:
        symbols = list(dict.fromkeys(
            p.stem.replace('_short_term_with_lag3', '').upper() 
            for p in main.DATA_DIR.glob('*_short_term_with_lag3.csv')
        ))
    except Exception as e:
        return JSONResponse({
            "ok": False, 
            "error": f"掃描失敗：{e}"
        }, status_code=500)
    
    if not symbols:
        return {
            "ok": True,
            "message": "沒有需要更新的 CSV",
            "updated": 0,
            "failed": 0
        }
    
    # 並發更新（限制同時執行數量避免 API rate limit）
    concurrency = 4
    sem = asyncio.Semaphore(concurrency)
    results = {"success": [], "failed": []}
    
    async def _update_one(sym: str):
        async with sem:
            csv_path = main._symbol_csv_write_path(sym)
            try:
                # 在背景執行緒中執行 I/O 操作
                await asyncio.to_thread(
                    lambda: (_ensure_td(), _build_from_td(symbol=sym, out_csv=csv_path))
                )
                # 更新時間戳
                try:
                    (csv_path.parent / f"{sym}_last_update.txt").write_text(
                        datetime.now().isoformat(), 
                        encoding="utf-8"
                    )
                except Exception:
                    pass
                results["success"].append(sym)
                return None
            except Exception as e:
                results["failed"].append({"symbol": sym, "error": str(e)})
                return e
    
    # 執行更新
    await asyncio.gather(*[asyncio.create_task(_update_one(s)) for s in symbols])
    
    elapsed = _time.time() - start_time
    
    return {
        "ok": True,
        "total": len(symbols),
        "updated": len(results["success"]),
        "failed": len(results["failed"]),
        "success_symbols": results["success"],
        "failed_details": results["failed"],
        "elapsed_seconds": round(elapsed, 2),
        "message": f"已更新 {len(results['success'])}/{len(symbols)} 個 CSV"
    }


@router.get('/updater_status')
def updater_status():
    """查詢自動更新器狀態。
    
    回傳資訊：
    - 是否啟用
    - 更新間隔（分鐘）
    - 並發數
    - 目前狀態（運行中/停止）
    """
    # 延遲導入避免循環依賴
    import main
    
    running = (
        main.GLOBAL_UPDATE_TASK is not None 
        and not main.GLOBAL_UPDATE_TASK.done()
    )
    
    return {
        "ok": True,
        "enabled": main.ENABLE_GLOBAL_UPDATER,
        "running": running,
        "interval_minutes": main.GLOBAL_UPDATE_INTERVAL_MIN,
        "concurrency": main.GLOBAL_UPDATE_CONCURRENCY,
        "description": "背景自動更新器會每 5 分鐘掃描並覆寫 data/ 中的所有 CSV"
    }