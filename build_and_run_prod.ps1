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

Write-Host "==> Starting container via docker-compose.prod.yml (+ override if present)" -ForegroundColor Cyan
${composeProd} = Join-Path $PSScriptRoot 'docker-compose.prod.yml'
${composeOverride} = Join-Path $PSScriptRoot 'docker-compose.override.yml'
${composeGenerated} = Join-Path $PSScriptRoot 'docker-compose.override.local.generated.yml'

# Helper: pick a free TCP port on localhost, preferring common alt ports
function Get-FreeTcpPort {
  param([int[]]$Preferred = @(18080, 28080, 38080))
  foreach ($p in $Preferred) {
    try {
      $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p)
      $l.Start(); $l.Stop(); return $p
    } catch {}
  }
  $l2 = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
  $l2.Start(); $p2 = $l2.LocalEndpoint.Port; $l2.Stop(); return $p2
}

# If 80/8080 are busy and a static override exists, we still allow a generated override to win last in merge order
$useGenerated = $false
try {
  # Check if localhost:18080 is busy; if so, generate a new one
  $test = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 18080)
  $test.Start(); $test.Stop()
} catch {
  $useGenerated = $true
}

if ($useGenerated) {
  $freePort = Get-FreeTcpPort
  @(
    "services:",
    "  caddy:",
    "    ports:",
    "      - ${freePort}:80",
    "    volumes:",
    "      - ./infra/caddy/conf/Caddyfile.local:/etc/caddy/Caddyfile:ro"
  ) | Set-Content -Encoding UTF8 ${composeGenerated}
  Write-Host "Using generated override: ${composeGenerated} (HTTP on localhost:$freePort)" -ForegroundColor DarkCyan
}
# Ensure any previous container is stopped
try {
  if (Test-Path ${composeGenerated}) {
    docker compose -f ${composeProd} -f ${composeOverride} -f ${composeGenerated} down
  } elseif (Test-Path ${composeOverride}) {
    docker compose -f ${composeProd} -f ${composeOverride} down
  } else {
    docker compose -f ${composeProd} down
  }
} catch {}
$env:API_KEY = $env:API_KEY  # pass through if set
$runCmd = if (Test-Path ${composeGenerated}) {
  "docker compose -f `"${composeProd}`" -f `"${composeOverride}`" -f `"${composeGenerated}`" up -d"
} elseif (Test-Path ${composeOverride}) {
  "docker compose -f `"${composeProd}`" -f `"${composeOverride}`" up -d"
} else {
  "docker compose -f `"${composeProd}`" up -d"
}
Write-Host "Run: $runCmd" -ForegroundColor DarkCyan
Invoke-Expression $runCmd
if ($LASTEXITCODE -ne 0) { Write-Error "Compose up failed"; exit 1 }

Start-Sleep -Seconds 3
Write-Host "==> Health check" -ForegroundColor Cyan
# Since web is only exposed within the compose network (expose: 8000),
# check via Caddy on localhost with Host header. Retry for up to ~60s.
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
    if (Test-Path ${composeGenerated}) {
      # parse generated override to get chosen port (unquoted mapping)
      $portLine = (Get-Content ${composeGenerated} | Where-Object { $_ -match '^[\s-]*-\s*(\d+):80' } | Select-Object -First 1)
      if ($portLine -and $portLine -match '([0-9]+):80') { $port = [int]$matches[1] } else { $port = 18080 }
      $healthUrl = "http://localhost:$port/health"; $hostHeader = 'localhost'
    } elseif (Test-Path ${composeOverride}) {
      $healthUrl = 'http://localhost:18080/health'; $hostHeader = 'localhost'
    } else {
      $healthUrl = 'http://localhost/health'; $hostHeader = $domain
    }
    Write-Host "Health URL: $healthUrl (Host=$hostHeader)" -ForegroundColor DarkCyan
    $resp = Invoke-WebRequest -Uri $healthUrl -Headers @{ Host = $hostHeader } -UseBasicParsing -TimeoutSec 5
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
  Write-Warning "Health check failed via Caddy (Host=$domain or localhost). Containers may still be starting or host ports may be blocked."
  Write-Warning "Tips: ensure Docker Desktop is running and local firewall/VPN allows localhost access."
  # Final fallback: check from inside web container
  try {
    Write-Host "==> Fallback: in-container health probe (web:8000)" -ForegroundColor Cyan
    docker exec newproject_web_prod python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"
    if ($LASTEXITCODE -eq 0) {
      Write-Host "Internal health OK. App is running; host access may be blocked. You can visit http://localhost:PORT/health replacing PORT with the port in use." -ForegroundColor Yellow
      exit 0
    }
  } catch {}
  exit 1
}

Write-Host "==> Done: image=${ImageName}:${sha}" -ForegroundColor Green
