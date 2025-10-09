"""run_bulk_task_test.py — 啟動並輪詢批次背景任務（繁體中文說明）

流程：
    1. 呼叫 bulk_build_start(symbols=..., concurrency=2)
    2. 每秒輪詢 bulk_build_status(task_id) 最多 30 次或直到非 running
    3. 輸出任務完成狀態

使用：
    python -m scripts.dev.run_bulk_task_test

適用：
    需要快速確定背景批次任務行為是否正常（不使用 pytest）
"""
from main import bulk_build_start, bulk_build_status
import time

if __name__ == '__main__':
    print('Starting bulk build for AAPL,MSFT')
    r = bulk_build_start(symbols='AAPL,MSFT', concurrency=2)
    print('start returned:', r)
    tid = r.get('task_id')
    for i in range(30):
        s = bulk_build_status(tid)
        print('status:', s)
        if s['task']['status'] != 'running':
            break
        time.sleep(1)
    print('done')
