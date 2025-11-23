<#
 cleanup_containers.ps1 - Stop & remove old containers whose names contain new_project or newproject
 Usage:
   Set-Location infra/compose
   ./cleanup_containers.ps1
#>

Write-Host "[cleanup] scanning for matching containers..."
$patterns = @('new_project','newproject')
$all = docker ps -a --format '{{.Names}}'
if (-not $all) { Write-Host '[cleanup] no containers found.'; exit 0 }
$targets = @()
foreach ($p in $patterns) {
  $targets += ($all | Select-String $p | ForEach-Object { $_.Line })
}
$targets = $targets | Sort-Object -Unique
if ($targets.Count -eq 0) { Write-Host '[cleanup] no matching containers.'; exit 0 }
Write-Host "[cleanup] found: $($targets -join ', ')"
foreach ($name in $targets) {
  Write-Host "[cleanup] stopping $name ..."
  docker stop $name 2>$null | Out-Null
  Write-Host "[cleanup] removing $name ..."
  docker rm $name 2>$null | Out-Null
}
Write-Host "[cleanup] done."
