"""
批次工具：從 GitHub raw 抓取 S&P 500 成分股清單，並啟動第一批（前 50 檔）資料建置。

用途
- 下載 `constituents.csv` 並解析 Symbol 欄位
- 為避免一次觸發過多任務，僅取前 50 檔交由 `main.bulk_build_start(concurrency=4)` 背景建置

執行方式（請在專案根目錄執行）
- `python -m scripts.batch.fetch_sp500_github`
- 或 `python .\scripts\batch\fetch_sp500_github.py`

前置條件
- 需要可連外的網路環境
- 本腳本直接呼叫 `main.py` 的函式，不需先啟動 Web 服務
- `bulk_build_start` 為 async 函式，腳本內會使用 `asyncio.run(...)` 啟動事件迴圈

注意
- Ticker 中的「.」會轉為「-」（較符合 Yahoo Finance 符號）
- 若要調整數量，可修改 `tickers[:50]` 的切片
"""

import asyncio
import requests
from main import bulk_build_start

URL = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'

if __name__ == '__main__':
    print('Fetching S&P500 list from GitHub raw...')
    try:
        r = requests.get(URL, timeout=20)
        r.raise_for_status()
        txt = r.text
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        header = lines[0].split(',')
        idx_symbol = header.index('Symbol')
        tickers = [line.split(',')[idx_symbol].strip().replace('.', '-') for line in lines[1:]]
        print('Total tickers fetched:', len(tickers))
        first50 = tickers[:50]
        print('First 10:', first50[:10])
        syms = ','.join(first50)
        print('Starting bulk build for first 50 (concurrency=4)...')
        # bulk_build_start 是 async，使用 asyncio.run 以正確取得回傳值
        r2 = asyncio.run(bulk_build_start(symbols=syms, concurrency=4))
        print('bulk_build_start returned:', r2)
    except Exception as e:
        print('Failed to fetch or start:', e)
