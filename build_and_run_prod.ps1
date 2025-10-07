Param(
  [string]$ImageName = "new_project",
  [switch]$SkipRun
)

Write-Host "==> Determining git revision..." -ForegroundColor Cyan
$sha = (git rev-parse --short HEAD) 2>$null
if (-not $sha) { $sha = "nogit" }
$ts = (Get-Date -Format o)

# Ensure BuildKit is enabled for faster builds and cache mounts
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

Write-Host "==> Building image ${ImageName}:${sha}" -ForegroundColor Cyan
$buildCmd = @(
  "docker","build",
  "--build-arg","APP_GIT_SHA=$sha",
  "--build-arg","APP_BUILD_TIME=$ts",
  "-t","${ImageName}:${sha}",
  "-t","${ImageName}:latest",
  "."
) -join ' '
Invoke-Expression $buildCmd
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }

if ($SkipRun) { Write-Host "--SkipRun specified, exiting after build."; exit 0 }

Write-Host "==> Starting container via docker-compose.prod.yml" -ForegroundColor Cyan
# Ensure any previous container is stopped
try { docker compose -f docker-compose.prod.yml down } catch {}
$env:API_KEY = $env:API_KEY  # pass through if set
$runCmd = "docker compose -f docker-compose.prod.yml up -d"
Invoke-Expression $runCmd
if ($LASTEXITCODE -ne 0) { Write-Error "Compose up failed"; exit 1 }

Start-Sleep -Seconds 3
Write-Host "==> Health check" -ForegroundColor Cyan
# Since web is only exposed within the compose network (expose: 8000),
# check via Caddy on localhost:80 with Host header = DOMAIN. Retry for up to ~60s.
$domain = $env:DOMAIN
if (-not $domain) {
  $envFile = Join-Path $PSScriptRoot '.env'
  if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
      if ($line -match '^\s*DOMAIN\s*=\s*(.+)$') { $domain = $matches[1].Trim(); break }
    }
  }
}
if (-not $domain) { $domain = 'localhost' }

$maxAttempts = 20
$ok = $false
for ($i = 1; $i -le $maxAttempts; $i++) {
  try {
    $resp = Invoke-WebRequest -Uri http://localhost/health -Headers @{ Host = $domain } -UseBasicParsing -TimeoutSec 5
    if ($resp.StatusCode -eq 200) {
      Write-Host "Health OK (attempt #$i)" -ForegroundColor Green
      Write-Host $resp.Content
      $ok = $true
      break
    }
    else {
      Write-Host "Attempt #${i}: HTTP $($resp.StatusCode) — retrying..."
    }
  }
  catch {
    Write-Host "Attempt #${i}: waiting for Caddy/web to be ready..."
  }
  Start-Sleep -Seconds 3
}

if (-not $ok) {
  Write-Warning "Health check failed via Caddy (Host=$domain). Containers may still be starting or port 80 may be blocked."
  Write-Warning "Tips: ensure Docker Desktop is running and port 80/443 are free (IIS/Hyper-V off)."
  exit 1
}

Write-Host "==> Done: image=${ImageName}:${sha}" -ForegroundColor Green
