@echo off
chcp 65001 >nul
setlocal
cd /d %~dp0

where python >nul 2>nul
if errorlevel 1 (
  echo 未找到 Python，请先安装 Python 3.11 或更高版本。
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 exit /b 1
)

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

where npm.cmd >nul 2>nul
if errorlevel 1 (
  if not exist "frontend\dist\index.html" (
    echo 未找到 Node.js，且仓库中没有前端构建产物。
    exit /b 1
  )
) else (
  pushd frontend
  call npm.cmd ci
  if errorlevel 1 (
    popd
    exit /b 1
  )
  call npm.cmd run build
  if errorlevel 1 (
    popd
    exit /b 1
  )
  popd
)

echo 初始化完成。运行 start_client.bat 启动项目。
endlocal
