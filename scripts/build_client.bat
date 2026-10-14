@echo off
cd /d %~dp0..\frontend
npm run build
if errorlevel 1 pause
