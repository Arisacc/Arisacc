@echo off
cd /d %~dp0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_client.ps1"
if errorlevel 1 pause
