# DNS Setup

- Create an A record for your DOMAIN pointing to your server public IP.
- Example for Namecheap / Cloudflare:
  - Name/Host: app
  - Type: A
  - Value: 203.0.113.10 (your server IP)
  - TTL: Auto / 5 min

If you use Cloudflare and want Caddy to handle TLS directly, set the record to DNS only (grey cloud) or configure Caddy for Cloudflare DNS challenge.
