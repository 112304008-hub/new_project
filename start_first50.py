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
