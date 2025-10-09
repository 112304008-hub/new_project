"""(已棄用) 臨時 API 檢查工具 — 已由 tests/ 內的 pytest 測試與 run_api_smoke 取代（繁體中文說明）

替代方案：
  - 自動化測試：pytest -q tests\
  - 輕量冒煙測試：python -m scripts.dev.run_api_smoke

本檔現在僅輸出提醒訊息後立即結束，避免混淆。
"""

import sys

if __name__ == "__main__":
  print("[deprecated] 請改用 `pytest -q tests/` 或 `python -m scripts.dev.run_api_smoke`。")
  sys.exit(0)
