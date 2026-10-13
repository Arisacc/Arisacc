$runtimeDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "local_tools\cache\client_runtime"
$pidFile = Join-Path $runtimeDir "backend.pid"

if (!(Test-Path $pidFile)) {
  Write-Host "No backend process was recorded by the client launcher."
  exit 0
}

$processId = [int](Get-Content $pidFile -Raw).Trim()
$process = Get-Process -Id $processId -ErrorAction SilentlyContinue
if ($null -ne $process) {
  Stop-Process -Id $processId -Force
  Write-Host "Stopped client backend process: $processId"
} else {
  Write-Host "Backend process already exited: $processId"
}
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
