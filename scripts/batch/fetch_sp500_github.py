"""fetch_sp500_github.py — 從 GitHub 擷取 S&P500 清單並啟動前 50 檔批次建置（繁體中文說明）

功能：
    - 下載 `constituents.csv` → 解析 Symbol 欄位 → 正規化（將 '.' 換成 '-'）
    - 取前 50 檔以避免同時過度壓力，再呼叫 bulk_build_start(concurrency=4) 背景建置

執行（於專案根目錄）：
    python -m scripts.batch.fetch_sp500_github
    或 python scripts/batch/fetch_sp500_github.py

重要：
    - bulk_build_start 為 async；此腳本用 asyncio.run() 以取得回傳值
    - 調整批次大小：修改 first50 切片範圍

適合於：
    - 初次啟動環境快速生成部分基礎特徵檔
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
