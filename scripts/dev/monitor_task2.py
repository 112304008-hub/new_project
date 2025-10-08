import time
from pathlib import Path
import main

TASK_ID = '0bbe5a32-b164-4448-8f14-c923c7899dad'
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'

print(f'Monitoring bulk task {TASK_ID} (polling in-process BULK_TASKS)...')
start = time.time()
while True:
    info = main.BULK_TASKS.get(TASK_ID)
    if info is None:
        print('Task id not found in BULK_TASKS. Exiting.')
        break
    status = info.get('status')
    done = info.get('done', 0)
    total = info.get('total', 0)
    progress = done/total if total else 1.0
    print(f'status={status} done={done}/{total} progress={progress:.0%}')
    if status != 'running':
        print('Task completed with status:', status)
        break
    # timeout after 6 hours
    if time.time() - start > 6*3600:
        print('Timeout waiting for task to finish (6h).')
        break
    time.sleep(5)

# summarize
info = main.BULK_TASKS.get(TASK_ID) or {}
errors = info.get('errors', {})
started_at = info.get('started_at', 0)
print('\nTask started_at:', started_at)

# find CSVs created/modified since started_at - 300s buffer
found = []
for p in DATA_DIR.glob('*_short_term_with_lag3.csv'):
    try:
        m = p.stat().st_mtime
    except Exception:
        m = 0
    if m >= (started_at - 300):
        sym = p.stem.replace('_short_term_with_lag3', '')
        found.append({'symbol': sym, 'csv': str(p.resolve()), 'size': p.stat().st_size, 'mtime': m})

found_symbols = {f['symbol']: f for f in found}

# successes are found symbols excluding error keys
failed_symbols = set(errors.keys())
succeeded = {s: found_symbols[s] for s in found_symbols.keys() if s not in failed_symbols}

print('\nSummary:')
print(' Total requested:', info.get('total'))
print(' Done counted:', info.get('done'))
print(' Errors count:', len(failed_symbols))
if failed_symbols:
    print('\nFailed symbols and errors:')
    for s in sorted(failed_symbols):
        print('-', s, '->', errors.get(s))

print('\nSucceeded symbols (sample up to 50):')
for s, meta in list(succeeded.items())[:50]:
    print('-', s, meta['csv'], f"size={meta['size']}")

print('\nIf you want, I can upload a CSV report of successes/failures or continue with the next batch.')
