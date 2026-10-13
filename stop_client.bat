@echo off
cd /d %~dp0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_client.ps1"
pause
