"""start_and_monitor_batch3.py — 啟動並監控 S&P500 第 101~150 檔批次（繁體中文說明）

功能：
    1. 擷取清單 → 取 100:150 區間（實際顯示為 101~150）。
    2. 呼叫 bulk_build_start 啟動背景任務。
    3. 輪詢 BULK_TASKS 直到完成或逾時（10 分鐘）。
    4. 蒐集成功建立的 CSV 路徑並寫入 data/bulk_task_<task_id>_summary.json。

使用：
    python -m scripts.batch.start_and_monitor_batch3
"""
import time
import requests
from .fetch_sp500_github import URL  # reuse URL constant
from main import bulk_build_start, BULK_TASKS, DATA_DIR
from pathlib import Path
import json

print('Fetching S&P500 tickers from GitHub...')
r = requests.get(URL, timeout=20)
r.raise_for_status()
txt = r.text
lines = [l.strip() for l in txt.splitlines() if l.strip()]
header = lines[0].split(',')
idx_symbol = header.index('Symbol')
tickers = [line.split(',')[idx_symbol].strip().replace('.', '-') for line in lines[1:]]
batch = tickers[100:150]
print('Selected batch (101-150) size:', len(batch))
syms = ','.join(batch)
print('Starting bulk_build_start for batch...')
res = bulk_build_start(symbols=syms, concurrency=4)
print('bulk_build_start ->', res)
if not res.get('ok'):
    raise SystemExit('bulk_build_start failed: ' + str(res))

task_id = res['task_id']
print('Task ID:', task_id)
# poll for up to 10 minutes but report progress every 5s
start = time.time()
summary = {'task_id': task_id, 'started_at': BULK_TASKS[task_id]['started_at'], 'total': BULK_TASKS[task_id]['total'], 'done': 0, 'errors': {}}
while True:
    info = BULK_TASKS.get(task_id)
    if not info:
        print('Task disappeared from BULK_TASKS')
        break
    print(f"progress: {info['done']}/{info['total']} (errors={len(info.get('errors',{}))})")
    if info.get('status') == 'completed':
        summary['done'] = info.get('done')
        summary['errors'] = info.get('errors', {})
        summary['finished_at'] = info.get('finished_at')
        break
    if time.time() - start > 60*10:  # 10 minute timeout for this interactive run
        print('Timeout waiting for batch (10 min)')
        break
    time.sleep(5)

# enumerate created CSVs
created = []
for s in batch:
    p = Path(DATA_DIR) / f"{s}_short_term_with_lag3.csv"
    if p.exists():
        created.append(str(p.resolve()))

summary['created_csvs'] = created
# write summary json
outp = Path(DATA_DIR) / f"bulk_task_{task_id}_summary.json"
with outp.open('w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print('Wrote summary to', outp)
print('Summary:', summary)
