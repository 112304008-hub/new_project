param(
    [switch]$Reinstall
)
$ErrorActionPreference = 'Stop'

Write-Host "[env] Using PowerShell $($PSVersionTable.PSVersion) on $([System.Environment]::OSVersion.VersionString)"

$venvPath = Join-Path $PSScriptRoot '..' '.venv' | Resolve-Path -ErrorAction SilentlyContinue
if (-not $venvPath) {
    $venvPath = Join-Path $PSScriptRoot '..' '.venv'
}

# Create venv if missing or reinstall requested
if ($Reinstall -or -not (Test-Path $venvPath)) {
    Write-Host "[env] Creating virtual environment at $venvPath"
    python -m venv $venvPath
}

$pythonExe = Join-Path $venvPath 'Scripts' 'python.exe'
if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe. Ensure Python is installed and try again."
}

Write-Host "[env] Installing/Updating dependencies via $pythonExe"
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r (Join-Path $PSScriptRoot '..' 'requirements.txt')

Write-Host "[env] Done. To activate the venv in this shell, run:`n  . .\.venv\Scripts\Activate.ps1"