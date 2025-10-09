"""run_test_bulk.py — 使用 TestClient 測試 /api/bulk_build_start 端點（繁體中文說明）

內容：
	建立 FastAPI 測試用客戶端，呼叫 bulk_build_start API（symbols=AAPL）並輸出回應。

使用：
	python -m scripts.dev.run_test_bulk

說明：
	- 這是簡化版的端點測試，僅示範啟動背景任務的回應格式。
	- 更完整的行為（進度輪詢、錯誤處理）由 pytest 測試檔涵蓋。
"""
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)
resp = client.get('/api/bulk_build_start?symbols=AAPL&concurrency=2')
print('status', resp.status_code)
print(resp.text)
print('json:', resp.json() if resp.status_code==200 else '')
