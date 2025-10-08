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
