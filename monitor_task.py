import time
from main import bulk_build_status, list_symbols

task_id = '0bbe5a32-b164-4448-8f14-c923c7899dad'

print(f'Monitoring bulk task {task_id}...')
start = time.time()
while True:
    s = bulk_build_status(task_id)
    if not s.get('ok'):
        print('Error fetching status:', s)
        break
    task = s['task']
    print(f"status={task.get('status')} done={task.get('done')}/{task.get('total')} progress={task.get('progress'):.2%}")
    if task.get('status') != 'running':
        print('Task finished with status:', task.get('status'))
        break
    if time.time() - start > 60*60:  # 1 hour timeout
        print('Timeout waiting for task')
        break
    time.sleep(5)

# After finish, summarize
print('\nSummary:')
info = task
errors = info.get('errors', {}) if info else {}
print('Errors count:', len(errors))
if errors:
    for sym, err in errors.items():
        print('-', sym, '->', err)

# list CSVs
ls = list_symbols()
print('\nCurrently available symbol CSVs:', ls.get('count'))
for s in ls.get('symbols', [])[:20]:
    print('-', s['symbol'], s['csv'])

print('\nMonitor script done.')
