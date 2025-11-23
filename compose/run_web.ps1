<#
 run_web.ps1 - Build image from project root and start web-only compose service with health check.
 Safe to run from any directory.
#>

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path "$PSScriptRoot\..\..\").Path  # new-project root
Write-Host "[run_web] project root: $root"
$dockerfile = Join-Path $root 'Dockerfile'
if (-not (Test-Path $dockerfile)) { throw "Dockerfile not found at $dockerfile" }

Write-Host "[run_web] pulling base image (optional, ignore errors)..."
try { docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime | Out-Null } catch { Write-Host "[run_web] pull skipped: $($_.Exception.Message)" }

Write-Host "[run_web] building image new_project:latest using -f Dockerfile"
 docker build -f $dockerfile -t new_project:latest $root

Write-Host "[run_web] starting compose web service"
 Push-Location $PSScriptRoot
 docker compose -f docker-compose.prod.yml up -d web
 Pop-Location

$port = if ($env:APP_PORT) { $env:APP_PORT } else { 8001 }
Write-Host "[run_web] health check on http://localhost:$port/health"
for ($i=0; $i -lt 10; $i++) {
  try {
    $r = Invoke-WebRequest -Uri "http://localhost:$port/health" -TimeoutSec 4
    Write-Host "[run_web] OK status=$($r.StatusCode) body=$($r.Content)"
    break
  } catch {
    Write-Host "[run_web] retry $($i+1) ..."; Start-Sleep -Seconds 2
  }
}
