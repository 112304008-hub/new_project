<#!
.SYNOPSIS
  開發快捷指令集合（PowerShell）
.DESCRIPTION
  為 Windows 開發者提供與 Makefile 對應的一組函式 / 指令。
.USAGE
  dot-source 之後呼叫對應函式，例如：
    . .\scripts\win\dev_shortcuts.ps1
    Start-Dev
#>

<#! 內部變數設定：使用 $PSScriptRoot 推導專案根目錄 #>
if (-not $PSScriptRoot) { $PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..' '..')).Path  # 原本多上一層，修正為往上兩層取得專案根目錄
$script:VenvDir  = Join-Path $script:RepoRoot '.venv'

function _Resolve-Python {
  if (Test-Path (Join-Path $script:VenvDir 'Scripts/python.exe')) { return (Join-Path $script:VenvDir 'Scripts/python.exe') }
  $sys = (Get-Command python -ErrorAction SilentlyContinue)?.Source
  return $sys
}

function _Ensure-Venv {
  if (-not (Test-Path $script:VenvDir)) {
    Write-Host '[dev_shortcuts] 建立虛擬環境 .venv ...' -ForegroundColor Yellow
    python -m venv $script:VenvDir
  }
}

_Ensure-Venv
$script:PythonExe = _Resolve-Python
$script:PythonAvailable = $true
if (-not $script:PythonExe) {
  Write-Warning '[dev_shortcuts] 無法找到 python；請安裝 Python 3.11+ 或建立 .venv 後重新 dot-source。'
  $script:PythonAvailable = $false
}

# Pytest 使用 -m 執行；若未安裝 pytest，Run-Tests 會提示安裝。
$script:PytestCmd = @($script:PythonExe,'-m','pytest') -join ' '

# 在專案根目錄內執行命令的輔助函式（確保相對路徑正確）
function Invoke-InRepo {
  param(
    [Parameter(Mandatory)][ScriptBlock]$Script
  )
  Push-Location $script:RepoRoot
  try { & $Script } finally { Pop-Location }
}

function Enter-Venv {
  <#
  .SYNOPSIS
    載入 .venv Python 環境。
  .DESCRIPTION
    嘗試執行 .venv\Scripts\Activate.ps1，若不存在則提出警告。
  #>
  $activate = Join-Path $script:VenvDir 'Scripts/Activate.ps1'
  if (Test-Path $activate) {
    . $activate
    Write-Information "[dev_shortcuts] 已載入: $script:VenvDir" -InformationAction Continue
  } else {
    Write-Warning '[dev_shortcuts] 找不到 .venv'
  }
}

function Update-Requirements {
  <#
  .SYNOPSIS
    安裝/更新專案所需的 Python 套件。
  .DESCRIPTION
    讀取 requirements.txt 並以 pip 安裝；在專案根目錄內執行以保證路徑正確。
  #>
  if (-not $script:PythonAvailable) { throw 'Python 不可用' }
  Invoke-InRepo { & $script:PythonExe -m pip install -r (Join-Path $script:RepoRoot 'requirements.txt') }
}

function Start-DevServer {
  <#
  .SYNOPSIS
    啟動開發伺服器 (自動 reload)。
  .DESCRIPTION
    以 uvicorn 啟動 main:app，監聽 0.0.0.0:8000，啟用 --reload。
  #>
  if (-not $script:PythonAvailable) { throw 'Python 不可用' }
  Invoke-InRepo { & $script:PythonExe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 --app-dir $script:RepoRoot }
}

function Start-AppServer {
  <#
  .SYNOPSIS
    啟動生產伺服器 (無 reload)。
  .DESCRIPTION
    以 uvicorn 在固定埠啟動，關閉自動重新載入以提升穩定性。
  #>
  if (-not $script:PythonAvailable) { throw 'Python 不可用' }
  Invoke-InRepo { & $script:PythonExe -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir $script:RepoRoot }
}

function Start-ModelTraining {
  <#
  .SYNOPSIS
    執行模型訓練。
  .PARAMETER Model
    指定模型種類：rf / all。預設 all。
  .EXAMPLE
    Start-ModelTraining -Model rf
  .EXAMPLE
    Start-ModelTraining  # 等同 all
  #>
  [CmdletBinding()]
  param(
    [ValidateSet('rf','all')][string]$Model='all'
  )
  if (-not $script:PythonAvailable) { throw 'Python 不可用' }
  Invoke-InRepo { & $script:PythonExe (Join-Path $script:RepoRoot 'stock.py') --train --model $Model }
}

function Update-SP500Data {
  <#
  .SYNOPSIS
    抓取並更新 S&P500 成份資料。
  .DESCRIPTION
    透過 scripts.batch.fetch_sp500_github 模組更新最新列表與相關檔案。
  #>
  if (-not $script:PythonAvailable) { throw 'Python 不可用' }
  Invoke-InRepo { & $script:PythonExe -m scripts.batch.fetch_sp500_github }
}

function Test-Project {
  <#
  .SYNOPSIS
    執行測試 (安靜模式)。
  .DESCRIPTION
    使用 pytest -q 在 tests 目錄執行所有測試。
  #>
  if (-not $script:PythonAvailable) { throw 'Python 不可用' }
  Invoke-InRepo { & $script:PythonExe -m pytest -q (Join-Path $script:RepoRoot 'tests') }
}

function Get-CoverageReport {
  <#
  .SYNOPSIS
    產生 coverage HTML 報告。
  .DESCRIPTION
    執行 pytest --cov 產出 htmlcov 資料夾並提示檔案位置。
  #>
  if (-not $script:PythonAvailable) { throw 'Python 不可用' }
  Invoke-InRepo { & $script:PythonExe -m pytest --cov=. --cov-report=html (Join-Path $script:RepoRoot 'tests') }
  Write-Information ('[dev_shortcuts] Coverage: ' + (Join-Path $script:RepoRoot 'htmlcov/index.html')) -InformationAction Continue
}

function Get-RepoSummary {
  <#
  .SYNOPSIS
    快速列出關鍵資源摘要。
  .DESCRIPTION
    列出 Python 可執行路徑、models 資料夾檔案清單與 data 前 10 筆項目。
  #>
  Write-Output ('Python: ' + $script:PythonExe)
  Write-Output 'Models:'
  Get-ChildItem (Join-Path $script:RepoRoot 'models') -ErrorAction SilentlyContinue
  Write-Output 'Data (Top 10):'
  Get-ChildItem (Join-Path $script:RepoRoot 'data') -ErrorAction SilentlyContinue | Select-Object -First 10
}

# ------------------------------
# Backward compatibility wrappers (舊名稱維持可用)
# ------------------------------
function Use-Venv { [CmdletBinding()] param(); Enter-Venv }
function Install-Dependencies { [CmdletBinding()] param(); Update-Requirements }
function Start-Dev { [CmdletBinding()] param(); Start-DevServer }
function Start-Prod { [CmdletBinding()] param(); Start-AppServer }
function Invoke-TrainRF { [CmdletBinding()] param(); Start-ModelTraining -Model rf }
function Invoke-TrainAll { [CmdletBinding()] param(); Start-ModelTraining -Model all }
function Invoke-BulkSP500 { [CmdletBinding()] param(); Update-SP500Data }
function Invoke-Tests { [CmdletBinding()] param(); Test-Project }
function Invoke-CoverageReport { [CmdletBinding()] param(); Get-CoverageReport }
function Get-Summary { [CmdletBinding()] param(); Get-RepoSummary }

# 便利 alias（短名）
Set-Alias enterVenv Enter-Venv -ErrorAction SilentlyContinue
Set-Alias upreq Update-Requirements -ErrorAction SilentlyContinue
Set-Alias dev Start-DevServer -ErrorAction SilentlyContinue
Set-Alias prod Start-AppServer -ErrorAction SilentlyContinue
Set-Alias trainrf Invoke-TrainRF -ErrorAction SilentlyContinue   # 舊行為 rf
Set-Alias trainall Invoke-TrainAll -ErrorAction SilentlyContinue # 舊行為 all
Set-Alias sp500 Update-SP500Data -ErrorAction SilentlyContinue
Set-Alias runtests Test-Project -ErrorAction SilentlyContinue
Set-Alias covreport Get-CoverageReport -ErrorAction SilentlyContinue
Set-Alias summary Get-RepoSummary -ErrorAction SilentlyContinue
# 保留舊 alias 'install' 指向新函式
Set-Alias install Update-Requirements -ErrorAction SilentlyContinue
# Dot-source 之後即可使用： . .\scripts\win\dev_shortcuts.ps1; Start-Dev

