@echo off
chcp 65001 >nul
cd /d %~dp0
local_tools\venv\python.exe -m audio_dataset_agent.cli --input-dir datasets\smoke_input --output-dir local_tools\cache\manual_run_work --export-dir local_tools\cache\manual_run_exports --target-policy voice_clone --workflow repair --vad fixed --normalizer ffmpeg --sample-materializer ffmpeg --audio-metrics wave --asr none
