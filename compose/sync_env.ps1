# sync_env.ps1 - Sync .env files between root and compose directory
# Usage: .\sync_env.ps1 [-Direction root-to-compose|compose-to-root|check]

param(
    [ValidateSet('root-to-compose', 'compose-to-root', 'check')]
    [string]$Direction = 'root-to-compose'
)

$ErrorActionPreference = 'Stop'

$rootEnv = Join-Path (Join-Path $PSScriptRoot '..') '.env'
$composeEnv = Join-Path $PSScriptRoot '.env'

function Write-Info($msg) { Write-Host "[sync_env] $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Check if root .env exists
if (-not (Test-Path $rootEnv)) {
    Write-Err "Root .env not found: $rootEnv"
    Write-Info "Please create .env file first (refer to .env.example)"
    exit 1
}

switch ($Direction) {
    'root-to-compose' {
        Write-Info "Copying root/.env -> compose/.env"
        Copy-Item $rootEnv $composeEnv -Force
        Write-Ok "Synced to compose/.env"
        
        $rootSize = (Get-Item $rootEnv).Length
        $composeSize = (Get-Item $composeEnv).Length
        Write-Info "File sizes: root=$rootSize bytes, compose=$composeSize bytes"
    }
    
    'compose-to-root' {
        if (-not (Test-Path $composeEnv)) {
            Write-Err "compose/.env not found"
            exit 1
        }
        Write-Warn "This will overwrite root .env!"
        $confirm = Read-Host "Continue? (y/N)"
        if ($confirm -ne 'y') {
            Write-Info "Cancelled"
            exit 0
        }
        Copy-Item $composeEnv $rootEnv -Force
        Write-Ok "Synced to root/.env"
    }
    
    'check' {
        Write-Info "Checking .env file status"
        
        Write-Host "`nRoot .env:"
        if (Test-Path $rootEnv) {
            $rootItem = Get-Item $rootEnv
            Write-Host "  Exists: $($rootItem.FullName)"
            Write-Host "  Size: $($rootItem.Length) bytes"
            Write-Host "  Modified: $($rootItem.LastWriteTime)"
        } else {
            Write-Warn "  Not found"
        }
        
        Write-Host "`nCompose .env:"
        if (Test-Path $composeEnv) {
            $composeItem = Get-Item $composeEnv
            Write-Host "  Exists: $($composeItem.FullName)"
            Write-Host "  Size: $($composeItem.Length) bytes"
            Write-Host "  Modified: $($composeItem.LastWriteTime)"
        } else {
            Write-Warn "  Not found"
        }
        
        # Compare content
        if ((Test-Path $rootEnv) -and (Test-Path $composeEnv)) {
            $rootContent = Get-Content $rootEnv -Raw
            $composeContent = Get-Content $composeEnv -Raw
            
            if ($rootContent -eq $composeContent) {
                Write-Ok "`n[OK] Files are identical"
            } else {
                Write-Warn "`n[WARN] Files differ!"
                Write-Info "Run: .\sync_env.ps1 -Direction root-to-compose"
            }
        }
    }
}

Write-Host ""
