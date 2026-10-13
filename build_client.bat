@echo off
cd /d %~dp0frontend
npm run build
if errorlevel 1 pause
