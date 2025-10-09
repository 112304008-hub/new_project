"""(已棄用) 靜態資源快速檢查腳本 — 已由 tests/ 與 /health 端點取代（繁體中文說明）

說明：
  早期用來手動檢查 static/ 目錄是否可透過 FastAPI 提供；現已由自動化測試與健康檢查覆蓋。

替代方式：
  1. 執行單元測試：pytest -q tests\
  2. 健康檢查：curl http://localhost:8000/health
  3. 手動取檔：curl http://localhost:8000/static/<檔名>

現在執行本腳本只會列印提示並結束。
"""

import sys

if __name__ == "__main__":
  print("[deprecated] 靜態資源檢查已被自動化測試與 /health 取代，請參考 README。")
  sys.exit(0)
