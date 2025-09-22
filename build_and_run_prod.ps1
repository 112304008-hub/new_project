Param(
  [string]$ImageName = "new_project",
  [switch]$SkipRun
)

Write-Host "==> Determining git revision..." -ForegroundColor Cyan
$sha = (git rev-parse --short HEAD) 2>$null
if (-not $sha) { $sha = "nogit" }
$ts = (Get-Date -Format o)

Write-Host "==> Building image $ImageName:$sha" -ForegroundColor Cyan
$buildCmd = @(
  "docker","build",
  "--build-arg","APP_GIT_SHA=$sha",
  "--build-arg","APP_BUILD_TIME=$ts",
  "-t","$ImageName:$sha",
  "-t","$ImageName:latest",
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
try {
  $resp = Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing -TimeoutSec 5
  Write-Host $resp.Content
} catch {
  Write-Warning "Health check failed: $_"
  exit 1
}

Write-Host "==> Done: image=$ImageName:$sha" -ForegroundColor Green
