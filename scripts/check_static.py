import urllib.request

url = 'http://127.0.0.1:8000/static/ComfyUI_00012_.png'
print('Requesting', url)
try:
    with urllib.request.urlopen(url, timeout=10) as r:
        print('Status:', r.getcode())
        info = r.info()
        print('Content-Type:', info.get_content_type())
        data = r.read()
        print('Bytes:', len(data))
except Exception as e:
    print('Error:', e)
