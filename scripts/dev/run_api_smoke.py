"""run_api_smoke.py — API 輕量冒煙測試（繁體中文說明）

目的：
    直接在同一個 Python 行程中呼叫 `main.py` 提供的函式 (build_symbol, lag_stats) 以檢查：
        1. 匯入是否正常
        2. 基本資料建構與統計端點邏輯是否可執行

使用：
    python -m scripts.dev.run_api_smoke

注意：
    - 這不是 HTTP 層測試，不會涵蓋 FastAPI 路由層中間件。
    - 若需要完整路由/權限/Rate Limit 驗證，使用 pytest 測試套件。
"""
from main import build_symbol, lag_stats

if __name__ == '__main__':
    sym = '2330'
    print('Building symbol CSV for', sym)
    r = build_symbol(sym)
    print('build_symbol result:', r)
    print('Computing lag_stats for symbol', sym)
    out = lag_stats(symbol=sym)
    print('lag_stats returned count:', out.get('count'))
    if out.get('count'):
        print('Top feature:', out['results'][0]['feature'])
