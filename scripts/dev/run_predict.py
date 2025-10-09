"""run_predict.py — 呼叫模型推論的命令列工具（繁體中文說明）

功能：
    透過 stock.predict() 對指定 symbol 執行推論（若有模型檔），輸出返回欄位與預測結果。

使用範例：
    python -m scripts.dev.run_predict --symbol 2330 --model rf
    python scripts/dev/run_predict.py -s AAPL -m lr

參數：
    --symbol / -s  目標代號（台股數字 / 美股代碼），預設 2330。
    --model / -m   rf 或 lr。

錯誤處理：
    若缺少模型檔（*_pipeline.pkl / *_threshold.pkl），stock.predict 可能重新訓練或報錯，視現有邏輯而定。
"""
from stock import predict

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', '-s', default='2330', help='Symbol to predict (e.g., 2330 or AAPL)')
    ap.add_argument('--model', '-m', default='rf', choices=['rf','lr'], help='Model to use')
    args = ap.parse_args()
    sym = args.symbol
    print(f'Testing predict for symbol {sym}')
    try:
        r = predict(model=args.model, symbol=sym)
        print('Returned keys:', sorted(list(r.keys())))
        print('symbol:', r.get('symbol'))
        print('csv:', r.get('csv'))
        print('label, proba:', r.get('label'), r.get('proba'))
    except Exception as e:
        print('predict failed:', type(e).__name__, e)

if __name__ == '__main__':
    main()
