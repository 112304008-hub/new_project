from stock import predict

if __name__ == '__main__':
    print('Predicting MSFT...')
    try:
        out = predict(model='rf', symbol='MSFT')
        print('predict returned:', out)
    except Exception as e:
        print('predict failed:', type(e).__name__, e)
