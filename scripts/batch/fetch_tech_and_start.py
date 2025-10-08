import requests
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import stock

WIKI = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
GITHUB = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'

def get_tech_symbols():
    try:
        tables = pd.read_html(WIKI)
        df = tables[0]
        # try various possible column names for sector
        sector_col = None
        for c in df.columns:
            if 'sector' in c.lower() or 'gics sector' in c.lower():
                sector_col = c
                break
        if sector_col is None:
            raise RuntimeError('No sector column')
        tech = df[df[sector_col].str.contains('Information Technology', case=False, na=False)]
        syms = tech['Symbol'].astype(str).str.replace('.', '-', regex=False).tolist()
        return syms
    except Exception as e:
        print('Wiki fetch failed:', e)
        # fallback to GitHub CSV (no sector info) -> return empty
        try:
            df = pd.read_csv(GITHUB)
            # find any column that looks like a sector column (e.g., 'Sector', 'GICS Sector')
            sector_col = None
            for c in df.columns:
                if 'sector' in c.lower():
                    sector_col = c
                    break
            if sector_col is not None:
                tech_df = df[df[sector_col].astype(str).str.contains('Information Technology', case=False, na=False)]
                # try common symbol column names
                sym_col = None
                for sc in ['Symbol', 'Symbol', 'Ticker', 'Ticker symbol', 'Security']:
                    if sc in df.columns:
                        sym_col = sc
                        break
                if sym_col is None:
                    # fallback to first column
                    sym_col = df.columns[0]
                syms = df.loc[tech_df.index, sym_col].astype(str).str.replace('.', '-', regex=False).tolist()
                return syms
        except Exception as e2:
            print('GitHub fallback failed:', e2)
        return []

if __name__ == '__main__':
    syms = get_tech_symbols()
    print('Found tech count:', len(syms))
    if len(syms) == 0:
        print('No tech tickers found; aborting')
        raise SystemExit(1)
    print('Starting local build for tech symbols, count=', len(syms))
    out_dir = Path(__file__).parent.parent / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'total': len(syms), 'success': [], 'failed': {}}

    def _build(s):
        try:
            p = out_dir / f"{s}_short_term_with_lag3.csv"
            stock._ensure_yf()
            stock._build_from_yfinance(symbol=s, out_csv=p)
            return (s, True, str(p.resolve()))
        except Exception as e:
            return (s, False, str(e))

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_build, s): s for s in syms}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                sym, ok, info = fut.result()
                if ok:
                    summary['success'].append(sym)
                else:
                    summary['failed'][sym] = info
            except Exception as e:
                summary['failed'][s] = str(e)

    summary_path = out_dir / 'tech_bulk_summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Done. summary written to', summary_path)
