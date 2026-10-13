$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$bundledPython = Join-Path $root "local_tools\venv\python.exe"
$projectPython = Join-Path $root ".venv\Scripts\python.exe"
$python = if (Test-Path $bundledPython) { $bundledPython } elseif (Test-Path $projectPython) { $projectPython } else { "python" }
$frontendDist = Join-Path $root "frontend\dist"
$runtimeDir = Join-Path $root "local_tools\cache\client_runtime"
$backendLog = Join-Path $runtimeDir "backend.out.log"
$backendErrorLog = Join-Path $runtimeDir "backend.err.log"
$workerLog = Join-Path $runtimeDir "worker.out.log"
$workerErrorLog = Join-Path $runtimeDir "worker.err.log"
$pidFile = Join-Path $runtimeDir "backend.pid"
$workerPidFile = Join-Path $runtimeDir "worker.pid"
$backendUrl = "http://127.0.0.1:8000"
$clientUrl = "$backendUrl/pipeline"

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

if ($python -eq "python" -and $null -eq (Get-Command python -ErrorAction SilentlyContinue)) {
  throw "Python was not found. Run setup_windows.bat first."
}

if (!(Test-Path (Join-Path $frontendDist "index.html"))) {
  $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
  if ($null -eq $npm) {
    throw "frontend/dist is missing and npm is unavailable. Install Node.js or build the frontend first."
  }
  Push-Location (Join-Path $root "frontend")
  try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) {
      throw "Frontend build failed."
    }
  } finally {
    Pop-Location
  }
}

function Test-HttpEndpoint([string] $url) {
  try {
    $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
    return $response.StatusCode -ge 200 -and $response.StatusCode -lt 400
  } catch {
    return $false
  }
}

$backendReady = Test-HttpEndpoint "$backendUrl/hello"
if (!$backendReady) {
  $process = Start-Process `
    -FilePath $python `
    -ArgumentList @(
      "-m", "audio_dataset_agent.web.fastapi_server",
      "--host", "127.0.0.1",
      "--port", "8000",
      "--frontend-dir", "frontend/dist"
    ) `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $backendLog `
    -RedirectStandardError $backendErrorLog `
    -PassThru
  Set-Content -Path $pidFile -Value $process.Id -Encoding ascii
}

$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and !(Test-HttpEndpoint "$backendUrl/hello")) {
  Start-Sleep -Milliseconds 500
}

if (!(Test-HttpEndpoint "$backendUrl/hello")) {
  throw "Backend startup failed. See: $backendErrorLog"
}

if (!(Test-HttpEndpoint $clientUrl)) {
  throw "Port 8000 is occupied by an old service without the frontend. Stop it and run start_client.bat again."
}

$workerProcess = $null
if (Test-Path $workerPidFile) {
  $workerProcess = Get-Process -Id ([int](Get-Content $workerPidFile -Raw).Trim()) -ErrorAction SilentlyContinue
}
if ($null -eq $workerProcess) {
  $workerProcess = Start-Process `
    -FilePath $python `
    -ArgumentList @(
      "-m", "audio_dataset_agent.web.worker",
      "--runs-root", "local_tools/cache/audio_web_runs"
    ) `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $workerLog `
    -RedirectStandardError $workerErrorLog `
    -PassThru
  Set-Content -Path $workerPidFile -Value $workerProcess.Id -Encoding ascii
}

Start-Process $clientUrl
Write-Host "Audio dataset client started: $clientUrl"
Write-Host "Worker started: $($workerProcess.Id)"
