@echo off
chcp 65001 >nul
cd /d %~dp0
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)
%PYTHON% -m audio_dataset_agent.cli --input-dir "%~1" --output-dir local_tools\cache\manual_run_work --export-dir local_tools\cache\manual_run_exports --target-policy voice_clone --workflow repair --vad whole-file --normalizer ffmpeg --sample-materializer ffmpeg --audio-metrics wave --asr none
