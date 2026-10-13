@echo off
cd /d %~dp0
if exist "local_tools\venv\python.exe" (
  set "PYTHON=local_tools\venv\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)
%PYTHON% -m audio_dataset_agent.web.fastapi_server --host 127.0.0.1 --port 8000 --frontend-dir frontend/dist
