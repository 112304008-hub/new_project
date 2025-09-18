from fastapi.testclient import TestClient
import main

client = TestClient(main.app)
resp = client.get('/api/bulk_build_start?symbols=AAPL&concurrency=2')
print('status', resp.status_code)
print(resp.text)
print('json:', resp.json() if resp.status_code==200 else '')
