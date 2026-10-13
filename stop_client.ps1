$runtimeDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "local_tools\cache\client_runtime"
$pidFile = Join-Path $runtimeDir "backend.pid"
$workerPidFile = Join-Path $runtimeDir "worker.pid"

if (Test-Path $pidFile) {
  $processId = [int](Get-Content $pidFile -Raw).Trim()
  $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
  if ($null -ne $process) {
    Stop-Process -Id $processId -Force
    Write-Host "Stopped client backend process: $processId"
  } else {
    Write-Host "Backend process already exited: $processId"
  }
} else {
  Write-Host "No backend process was recorded by the client launcher."
}
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue

if (Test-Path $workerPidFile) {
  $workerId = [int](Get-Content $workerPidFile -Raw).Trim()
  $worker = Get-Process -Id $workerId -ErrorAction SilentlyContinue
  if ($null -ne $worker) {
    Stop-Process -Id $workerId -Force
    Write-Host "Stopped worker process: $workerId"
  }
  Remove-Item $workerPidFile -Force -ErrorAction SilentlyContinue
}
