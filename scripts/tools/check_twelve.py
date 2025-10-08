# check_twelve.py
import requests

import os

API_KEY = os.environ.get("TWELVE_API_KEY")  # set this in your environment
BASE_URL = "https://api.twelvedata.com"

def check_usage():
    url = f"{BASE_URL}/usage"
    if not API_KEY:
        print("❌ 未設定環境變數 TWELVE_API_KEY，請在 shell 中執行：\n$env:TWELVE_API_KEY='你的_api_key' (PowerShell) 或 export TWELVE_API_KEY=... (bash)")
        return
    params = {"apikey": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "message" in data:
            print("❌ 錯誤訊息：", data["message"])
            return
        print("✅ Twelve Data 使用狀況：")
        print(f"- 訂閱方案: {data.get('plan', '未知')}")
        print(f"- 每分鐘限制: {data.get('rate_limit_per_min', '?')} 次")
        print(f"- 每日限制: {data.get('rate_limit_per_day', '?')} 次")
        print(f"- 已用次數 (今日): {data.get('requests_today', '?')} 次")
        print(f"- 剩餘次數 (今日): {data.get('remaining_requests', '?')} 次")
    except Exception as e:
        print("❌ 無法取得使用狀況：", e)

def test_sample_call():
    print("\n📡 測試取得 TSM（ADR）股價 ...")
    url = f"{BASE_URL}/time_series"
    params = {
        "symbol": "TSM",
        "interval": "1day",
        "apikey": API_KEY,
        "outputsize": 1
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "status" in data and data["status"] == "error":
            print("❌ 錯誤訊息：", data.get("message", "未知錯誤"))
        else:
            print("✅ 成功！取回價格：", data["values"][0])
    except Exception as e:
        print("❌ 無法取得股價資料：", e)

if __name__ == "__main__":
    check_usage()
    test_sample_call()
