param(
    [switch]$SkipDeps,
    [switch]$NoPush,
    [switch]$Latest,
    [switch]$Test
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to project root
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
Set-Location $projectRoot

# Basics
$owner = "112304008-hub"
$repo  = "fortune-ticker"
$registry = "ghcr.io/$owner/$repo"

# Compute tags
$reqHash = ((Get-FileHash "$projectRoot/requirements.txt" -Algorithm SHA256).Hash.Substring(0,12).ToLower())
$gitSha  = (git --no-pager rev-parse --short=12 HEAD).Trim()
$buildTime = (Get-Date -Format o)

$depsTag = "$registry/py311-deps:$reqHash"
$appTag  = "$registry/app:$gitSha"

Write-Host "[info] projectRoot=$projectRoot"
Write-Host "[info] reqHash=$reqHash gitSha=$gitSha"

# Determine push behavior
$doPush = -not $NoPush

# Optional: login if CR_PAT available
if ($env:CR_PAT) {
    try {
        Write-Host "[login] Attempting docker login ghcr.io as $owner (env:CR_PAT found)"
        $env:CR_PAT | docker login ghcr.io -u $owner --password-stdin | Out-Null
    } catch {
        Write-Warning "[login] docker login failed: $_"
    }
} else {
    Write-Host "[login] Skipping docker login (set $env:CR_PAT to login automatically)"
}

# Build deps image (unless skipped)
if (-not $SkipDeps) {
    Write-Host "[build:deps] Building $depsTag"
    docker build -f Dockerfile.deps --build-arg REQUIREMENTS_SHA=$reqHash -t $depsTag .
    if ($doPush) {
        Write-Host "[push:deps] Pushing $depsTag"
        docker push $depsTag
    }
} else {
    Write-Host "[build:deps] Skipped by flag"
}

# Build app image (using deps as base)
Write-Host "[build:app] Building $appTag"
docker build -f Dockerfile `
  --build-arg BASE_IMAGE=$depsTag `
  --build-arg SKIP_PIP_INSTALL=true `
  --build-arg APP_GIT_SHA=$gitSha `
  --build-arg APP_BUILD_TIME=$buildTime `
  -t $appTag .

if ($doPush) {
    Write-Host "[push:app] Pushing $appTag"
    docker push $appTag
}

if ($Latest) {
    $latestTag = "$registry/app:latest"
    Write-Host "[tag] Tagging $latestTag"
    docker tag $appTag $latestTag
    if ($doPush) {
        Write-Host "[push] Pushing $latestTag"
        docker push $latestTag
    }
}

if ($Test) {
    Write-Host "[test] Running quick healthcheck on $appTag"
    $name = "fortune-ticker_smoke_" + ($gitSha.Substring(0,6))
    docker rm -f $name 2>$null | Out-Null
    docker run -d --name $name -p 8000:8000 $appTag | Out-Null
    try {
        Start-Sleep -Seconds 5
        $res = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5
        Write-Host ("[test] GET /health -> {0} {1}" -f $res.StatusCode, $res.StatusDescription)
    } catch {
        Write-Warning "[test] Healthcheck failed: $_"
    } finally {
        docker rm -f $name | Out-Null
    }
}

Write-Host "[done] Build/push flow completed."