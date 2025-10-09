"""run_bulk_build.py — 批次觸發與檢視 symbols 建置結果（繁體中文說明）

流程：
    1. 呼叫 build_symbols('2330,2317') 建立/更新兩檔特徵 CSV。
    2. 立即呼叫 list_symbols() 查看現有清單與檔案大小。

使用：
    python -m scripts.dev.run_bulk_build

適合情境：
    - 快速在本地驗證多檔是否能成功抓資料 + 產生特徵
    - 不需要背景任務與非同步監控的情況
"""
from main import build_symbols, list_symbols

if __name__ == '__main__':
    print('Building symbols: 2330,2317')
    r = build_symbols('2330,2317')
    print(r)
    print('Listing symbols...')
    print(list_symbols())
