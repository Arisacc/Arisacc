@echo off
cd /d %~dp0
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)
%PYTHON% -m audio_dataset_agent.web.local_server --host 127.0.0.1 --port 8000 --frontend-dir frontend/dist
