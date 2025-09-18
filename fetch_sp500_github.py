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
        r2 = bulk_build_start(symbols=syms, concurrency=4)
        print('bulk_build_start returned:', r2)
    except Exception as e:
        print('Failed to fetch or start:', e)
