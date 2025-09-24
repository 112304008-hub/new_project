import urllib.request, json
#TODO: 
def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = r.read()
            return json.loads(data)
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    base = 'http://127.0.0.1:8000'
    print('Fetching /api/latest_features')
    print(fetch(base + '/api/latest_features?max_items=200'))
    print('\nFetching /api/lag_stats (may take a few seconds)')
    print(fetch(base + '/api/lag_stats'))
