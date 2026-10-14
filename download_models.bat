@echo off
chcp 65001 >nul
cd /d %~dp0
if exist "local_tools\venv\python.exe" (
  set "PYTHON=local_tools\venv\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)
%PYTHON% local_tools\download_models.py
pause
