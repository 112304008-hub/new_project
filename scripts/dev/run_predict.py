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
