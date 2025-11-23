# run_all.ps1 - 一鍵建置並啟動 web + caddy（正式環境 HTTPS）
# 可在任何目錄執行：powershell -File .\infra\compose\run_all.ps1
# 若遇到執行原則問題：powershell -ExecutionPolicy Bypass -File .\infra\compose\run_all.ps1

param(
    [string]$ImageTag = "new_project:latest",
    [string]$Domain = $env:DOMAIN,
    [string]$AcmeEmail = $env:ACME_EMAIL,
    [string]$ApiKey = $env:API_KEY,
    [int]$AppPort = $(if ($env:APP_PORT) { [int]$env:APP_PORT } else { 8001 })
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg){ Write-Host "[run_all] $msg" -ForegroundColor Cyan }
function Write-Ok($msg){ Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg){ Write-Host "[WARN] $msg" -ForegroundColor Yellow }

# Resolve repo root from script location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir '..' '..' '..')
Set-Location $repoRoot
Write-Info "Repo root: $repoRoot"

if (-not (Test-Path "$repoRoot/Dockerfile")) { Write-Warn "Dockerfile not found at repo root." }

# Pull base image (optional but recommended on first run)
Write-Info "Pulling base image (may skip if already present)"
try { docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime | Out-Null } catch { Write-Warn "Base image pull failed: $_" }

# Build image
Write-Info "Building image $ImageTag"
docker build -t $ImageTag .
Write-Ok "Image built: $ImageTag"

# Ensure compose directory
$composeDir = Join-Path $repoRoot 'infra' 'compose'
Set-Location $composeDir
Write-Info "Compose directory: $composeDir"

# Inject env vars if provided
if ($Domain) { $env:DOMAIN = $Domain }
if ($AcmeEmail) { $env:ACME_EMAIL = $AcmeEmail }
if ($ApiKey) { $env:API_KEY = $ApiKey }
$env:APP_PORT = $AppPort

Write-Info "Starting services (web + caddy)"
docker compose -f docker-compose.prod.yml up -d
Write-Ok "Compose up initiated"

# Health check via domain (if domain provided) else localhost APP_PORT
$healthUrl = if ($Domain) { "http://$Domain/health" } else { "http://localhost:$AppPort/health" }
Write-Info "Health check: $healthUrl"
for ($i=0; $i -lt 10; $i++) {
    try {
        $r = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 3
        if ($r.StatusCode -eq 200) { Write-Ok "Health OK"; break }
    } catch { Start-Sleep -Seconds 2 }
    if ($i -eq 9) { Write-Warn "Health endpoint not ready after retries; check logs." }
}

Write-Info "Tail logs (Ctrl+C to stop)"
docker compose -f docker-compose.prod.yml logs -f web
