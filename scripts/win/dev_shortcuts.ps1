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

function Install-Deps { pip install -r requirements.txt }
function Start-Dev { uvicorn main:app --reload --host 0.0.0.0 --port 8000 }
function Start-Prod { uvicorn main:app --host 0.0.0.0 --port 8000 }
function Train-RF { python stock.py --train --model rf }
function Train-All { python stock.py --train --model all }
function Bulk-SP500 { python -m scripts.batch.fetch_sp500_github }
function Run-Tests { pytest -q tests }
function Cov-Report { pytest --cov=. --cov-report=html tests; Write-Host '查看 htmlcov/index.html' }
function Summary { Write-Host 'Models:'; Get-ChildItem models -ErrorAction SilentlyContinue; Write-Host 'Data (Top 10):'; Get-ChildItem data -ErrorAction SilentlyContinue | Select-Object -First 10 }

Export-ModuleMember -Function Install-Deps,Start-Dev,Start-Prod,Train-RF,Train-All,Bulk-SP500,Run-Tests,Cov-Report,Summary
