param(
  [string]$Registry = "ghcr.io",
  [string]$Owner = "112304008-hub",
  [string]$Repo = "new_project",
  [string]$AppTag = "dev",
  [switch]$UseLatestTag
)

$ErrorActionPreference = 'Stop'

# Compute requirements fingerprint (must match CI deps image tag)
if (-not (Test-Path -Path ./requirements.txt)) {
  Write-Error "requirements.txt not found in current directory."
}
$reqHash = (Get-FileHash ./requirements.txt -Algorithm SHA256).Hash.Substring(0,12)

$depsImage = '{0}/{1}/{2}/py311-deps:{3}' -f $Registry, $Owner, $Repo, $reqHash
$appImage = ('{0}:{1}' -f $Repo, $AppTag)

Write-Host "[info] Using deps image: $depsImage"

# Pull the prebuilt deps image from GHCR (may require docker login ghcr.io if private)
try {
  docker pull $depsImage | Out-Host
} catch {
  Write-Warning "Failed to pull $depsImage. If the package is private, run: docker login ghcr.io"
  throw
}

# Build the app using the deps image and skip reinstalling dependencies
$buildArgs = @(
  "--build-arg", "BASE_IMAGE=$depsImage",
  "--build-arg", "SKIP_PIP_INSTALL=true",
  "-t", $appImage,
  "."
)

Write-Host "[build] docker build $($buildArgs -join ' ')"
docker build @buildArgs | Out-Host

if ($UseLatestTag) {
  $latest = ('{0}:latest' -f $Repo)
  Write-Host "[tag] docker tag $appImage $latest"
  docker tag $appImage $latest | Out-Host
}

Write-Host "[done] Built image: $appImage"
Write-Host "Run: docker run --rm -p 8000:8000 $appImage"
