# quick smoke script to call build_symbol and lag_stats functions
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
