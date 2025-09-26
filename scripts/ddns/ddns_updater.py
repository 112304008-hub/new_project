import os
import time
import json
from typing import Optional

import requests


def get_public_ipv4(timeout: float = 5.0) -> Optional[str]:
    try:
        r = requests.get("https://api.ipify.org", params={"format": "text"}, timeout=timeout)
        if r.ok:
            ip = r.text.strip()
            # very basic sanity
            if ip and 6 <= len(ip) <= 45:
                return ip
    except Exception:
        pass
    return None


def update_duckdns(domain: str, token: str, ip: str) -> bool:
    url = "https://www.duckdns.org/update"
    params = {"domains": domain, "token": token, "ip": ip}
    try:
        r = requests.get(url, params=params, timeout=10)
        body = r.text.strip().lower()
        ok = r.ok and ("ok" in body)
        print(f"ddns[duckdns] status_code={r.status_code} body={body}")
        return ok
    except Exception as e:
        print(f"ddns[duckdns] error: {e}")
        return False


def _cf_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _cf_get_zone_id(api_token: str, zone_name: str) -> Optional[str]:
    try:
        resp = requests.get(
            "https://api.cloudflare.com/client/v4/zones",
            params={"name": zone_name, "status": "active"},
            headers=_cf_headers(api_token),
            timeout=10,
        )
        data = resp.json()
        if data.get("success") and data.get("result"):
            return data["result"][0]["id"]
    except Exception as e:
        print(f"ddns[cloudflare] get_zone_id error: {e}")
    return None


def _cf_get_or_create_record(api_token: str, zone_id: str, record_name: str) -> Optional[str]:
    # Try get existing A record
    try:
        resp = requests.get(
            f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records",
            params={"type": "A", "name": record_name},
            headers=_cf_headers(api_token),
            timeout=10,
        )
        data = resp.json()
        if data.get("success") and data.get("result"):
            return data["result"][0]["id"]
    except Exception as e:
        print(f"ddns[cloudflare] get_record error: {e}")

    # Create if missing
    try:
        payload = {"type": "A", "name": record_name, "content": "127.0.0.1", "ttl": 60, "proxied": False}
        resp = requests.post(
            f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records",
            headers=_cf_headers(api_token),
            data=json.dumps(payload),
            timeout=10,
        )
        data = resp.json()
        if data.get("success"):
            return data["result"]["id"]
        else:
            print(f"ddns[cloudflare] create_record failed: {data}")
    except Exception as e:
        print(f"ddns[cloudflare] create_record error: {e}")
    return None


def update_cloudflare(api_token: str, zone_name: str, record_name: str, ip: str) -> bool:
    zone_id = _cf_get_zone_id(api_token, zone_name)
    if not zone_id:
        print("ddns[cloudflare] zone not found; check CF_ZONE_NAME and token permissions")
        return False
    record_id = _cf_get_or_create_record(api_token, zone_id, record_name)
    if not record_id:
        print("ddns[cloudflare] record not found and failed to create; check CF_RECORD_NAME")
        return False
    try:
        payload = {"type": "A", "name": record_name, "content": ip, "ttl": 60, "proxied": False}
        resp = requests.put(
            f"https://api.cloudflare.com/client/v4/zones/{zone_id}/dns_records/{record_id}",
            headers=_cf_headers(api_token),
            data=json.dumps(payload),
            timeout=10,
        )
        data = resp.json()
        ok = bool(data.get("success"))
        print(f"ddns[cloudflare] update status={ok} resp={data}")
        return ok
    except Exception as e:
        print(f"ddns[cloudflare] update error: {e}")
        return False


def main():
    provider = (os.getenv("DDNS_PROVIDER") or "").strip().lower()
    interval = int(os.getenv("DDNS_INTERVAL_SECONDS", "300") or 300)

    if provider not in {"duckdns", "cloudflare"}:
        print("ddns: DDNS_PROVIDER not set (duckdns|cloudflare); exiting")
        return

    last_ip: Optional[str] = None
    print(f"ddns: starting provider={provider} interval={interval}s")

    while True:
        ip = get_public_ipv4()
        if not ip:
            print("ddns: unable to determine public IPv4; retrying later")
            time.sleep(interval)
            continue

        if ip == last_ip:
            print(f"ddns: ip unchanged {ip}; skipping update")
            time.sleep(interval)
            continue

        ok = False
        if provider == "duckdns":
            domain = os.getenv("DUCKDNS_DOMAIN", "").strip()
            token = os.getenv("DUCKDNS_TOKEN", "").strip()
            if not domain or not token:
                print("ddns[duckdns]: DUCKDNS_DOMAIN or DUCKDNS_TOKEN missing; exiting")
                return
            ok = update_duckdns(domain, token, ip)
        elif provider == "cloudflare":
            api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
            zone_name = os.getenv("CF_ZONE_NAME", "").strip()
            record_name = os.getenv("CF_RECORD_NAME", "").strip()
            if not api_token or not zone_name or not record_name:
                print("ddns[cloudflare]: CLOUDFLARE_API_TOKEN, CF_ZONE_NAME, or CF_RECORD_NAME missing; exiting")
                return
            ok = update_cloudflare(api_token, zone_name, record_name, ip)

        if ok:
            last_ip = ip
            print(f"ddns: updated {provider} record to {ip}")
        else:
            print("ddns: update failed; will retry next interval")

        time.sleep(interval)


if __name__ == "__main__":
    main()
