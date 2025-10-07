# Run-DDNS.ps1 - Update DuckDNS/Cloudflare DNS using .env configuration
# Usage:
#   pwsh -File .\scripts\Run-DDNS.ps1
#
# This script loads .env variables into current process and runs the Python DDNS updater.

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repoRoot '.env'
# Build paths in a PS 5.1-compatible way (Join-Path supports only -Path and -ChildPath)
$pythonExe = Join-Path (Join-Path (Join-Path $repoRoot '.venv') 'Scripts') 'python.exe'
$ddnsPy = Join-Path (Join-Path (Join-Path $repoRoot 'scripts') 'ddns') 'ddns_updater.py'

if (-not (Test-Path $envFile)) { throw ".env not found at $envFile" }
if (-not (Test-Path $pythonExe)) { throw "Python venv not found at $pythonExe. Run scripts/Setup-Env.ps1 first." }
if (-not (Test-Path $ddnsPy)) { throw "DDNS updater not found at $ddnsPy" }

Write-Host "[ddns] Loading environment variables from .env"
Get-Content $envFile | ForEach-Object {
  if ($_ -and ($_ -notmatch '^\s*#') -and ($_ -match '^\s*([^=]+)=(.*)$')) {
    $key = $matches[1].Trim(); $val = $matches[2].Trim()
    [System.Environment]::SetEnvironmentVariable($key, $val, 'Process')
  }
}

Write-Host "[ddns] Running updater via $pythonExe"
& $pythonExe $ddnsPy