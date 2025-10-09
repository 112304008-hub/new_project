"""start_next50.py — 從 GitHub Raw 取 S&P500 清單並建置第 51~100 檔（繁體中文說明）

邏輯：
    - 下載 constituents.csv
    - 取索引 50:100 (即第 51~100) 的 ticker
    - 呼叫 bulk_build_start(concurrency=4)

使用：
    python -m scripts.batch.start_next50

調整：
    - 想改範圍可調整 next50 = tickers[50:100]
"""
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
        # select next 50 (51-100), indexes 50..99
        next50 = tickers[50:100]
        print('Next 50 tickers count:', len(next50))
        syms = ','.join(next50)
        print('Starting bulk build for next 50 (concurrency=4)...')
        r2 = bulk_build_start(symbols=syms, concurrency=4)
        print('bulk_build_start returned:', r2)
    except Exception as e:
        print('Failed to fetch or start:', e)
