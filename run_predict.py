# small helper to exercise stock.predict for a non-TSMC symbol
from stock import predict

if __name__ == '__main__':
    sym = '2330'
    print(f'Testing predict for symbol {sym}')
    try:
        r = predict(model='rf', symbol=sym)
        print('Returned keys:', sorted(list(r.keys())))
        print('symbol:', r.get('symbol'))
        print('csv:', r.get('csv'))
        print('label, proba:', r.get('label'), r.get('proba'))
    except Exception as e:
        print('predict failed:', type(e).__name__, e)
