@echo off
cd /d %~dp0..
if "%~1"=="" (
  echo Usage: scripts\run_cli_smoke.bat ^<audio-directory^>
  exit /b 1
)
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else if exist "local_tools\venv\python.exe" (
  set "PYTHON=local_tools\venv\python.exe"
) else (
  set "PYTHON=python"
)
%PYTHON% -m audio_dataset_agent.cli --input-dir "%~1" --output-dir local_tools\cache\manual_run_work --export-dir local_tools\cache\manual_run_exports --target-policy voice_clone --workflow repair --vad fixed --normalizer ffmpeg --sample-materializer ffmpeg --audio-metrics wave --asr none
