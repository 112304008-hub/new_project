# 環境變數設定說明

## 📍 .env 檔案位置

本專案支援兩個位置的 `.env` 檔案：

1. **根目錄** `fortune-ticker/.env` - 主要配置檔（建議維護此檔）
2. **compose 目錄** `fortune-ticker/compose/.env` - Docker Compose 專用

### 為什麼需要兩個位置？

不同版本的 Docker Compose 對 `.env` 檔案的讀取位置有不同行為：

- **舊版 Docker Compose**：會從執行目錄或專案根目錄讀取
- **新版 Docker Compose v2+**：優先讀取與 `docker-compose.yml` 同目錄的 `.env`

## 🔄 部署到新電腦時

### 方案 1：複製檔案（建議）

```powershell
# 在專案根目錄執行
Copy-Item .env compose\.env -Force
```

### 方案 2：使用符號連結（需管理員權限）

```powershell
# 以管理員身份執行 PowerShell
cd compose
New-Item -ItemType SymbolicLink -Path .env -Target ..\.env -Force
```

### 方案 3：只維護一份（適合新電腦）

如果你確定只會在 `compose/` 目錄下執行 Docker Compose：

```powershell
# 將根目錄的 .env 移動到 compose/
Move-Item .env compose\.env -Force
```

## 📝 .env 檔案內容範本

```dotenv
# Domain 設定（Caddy HTTPS）
DOMAIN=your-domain.example.com
ACME_EMAIL=your-email@example.com

# API 保護（可選）
API_KEY=your-secret-key

# 速率限制
RATE_LIMIT_PER_MIN=200

# 應用程式埠（主機端）
APP_PORT=8001

# 全域自動更新
ENABLE_GLOBAL_UPDATER=true

# DDNS 設定（如使用 DuckDNS）
DDNS_PROVIDER=duckdns
DDNS_INTERVAL_SECONDS=300
DUCKDNS_DOMAIN=your-subdomain
DUCKDNS_TOKEN=your-token
DDNS_STATIC_IP=your-static-ip
DDNS_ONESHOT=true
```

## 🔍 驗證配置

檢查 `.env` 是否被正確讀取：

```powershell
# 查看 compose 目錄的 .env
Get-Content compose\.env

# 啟動服務並檢查環境變數
docker compose -f compose/docker-compose.prod.yml config
```

## ⚠️ 注意事項

1. `.env` 檔案包含敏感資訊，已加入 `.gitignore`，不會被 Git 追蹤
2. 修改任一 `.env` 後，記得同步到另一個位置（如使用方案 1）
3. 部署前務必檢查 `DOMAIN` 和 `ACME_EMAIL` 設定
4. 如不需要 HTTPS，可留空 `DOMAIN` 和 `ACME_EMAIL`

## 🚀 快速部署指令

```powershell
# 確保 .env 在兩個位置都存在
Copy-Item .env compose\.env -Force

# 執行部署腳本（會自動建置並啟動）
powershell -File .\compose\run_all.ps1
```
