"""start_first50.py — 啟動 S&P500 前 50 檔批次背景建置（繁體中文說明）

流程：
    1. 呼叫 _fetch_index_tickers('sp500') 取得完整清單
    2. 切片 [:50] 取前 50 檔
    3. 呼叫 bulk_build_start(symbols=..., concurrency=4)

使用：
    python -m scripts.batch.start_first50

差異：
    - 與 fetch_sp500_github.py 不同：此腳本直接使用現有內部抓取邏輯，不依賴 GitHub Raw。
"""
from main import _fetch_index_tickers, bulk_build_start

if __name__ == '__main__':
    print('Fetching S&P500 tickers...')
    try:
        tickers = _fetch_index_tickers('sp500')
        first50 = tickers[:50]
        print('First 50 tickers count:', len(first50))
        print(first50)
        syms = ','.join(first50)
        print('Starting bulk build for first 50 with concurrency=4...')
        r = bulk_build_start(symbols=syms, concurrency=4)
        print('bulk_build_start returned:', r)
    except Exception as e:
        print('Failed to start bulk build:', e)
