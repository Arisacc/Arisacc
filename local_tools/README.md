# 本地工具与模型说明

此目录保存音频数据集 Agent 的本地运行环境和外部工具。发布版不包含模型权重和运行缓存；代码通过适配器调用第三方工具，不直接把第三方命令混入 Agent 决策逻辑。

## 目录约定

- `venv/`：项目专用 Python 运行环境。
- `models/`：可选模型目录，发布版默认不提供，需要模型推理时手动放入。
- `deepfilternet/`：可选的 DeepFilterNet CLI。
- `ffmpeg/`：手工放置的 FFmpeg 发行版；当前优先使用 `imageio-ffmpeg` 提供的可执行文件。
- `downloads/`、`python_packages/`：离线安装包或下载中间文件。
- `cache/`：运行时自动创建的任务产物缓存，不作为项目源码阅读内容。

## Python 环境

项目命令统一使用：

```powershell
.\local_tools\venv\python.exe
```

已验证的核心依赖包括：`torch`、`torchaudio`、`numpy`、`soundfile`、`silero-vad`、`faster-whisper`、`imageio-ffmpeg`、`transformers`、`accelerate` 和 `safetensors`。

## 已接入工具

### FFmpeg

- 用途：格式标准化、候选片段裁剪、响度标准化、静音裁剪和轻量降噪。
- 当前来源：`imageio-ffmpeg`。
- 当前可执行文件通常位于：

```text
E:\Fin\3\项目源码\local_tools\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe
```

代码会先尝试系统 `ffmpeg`，找不到时自动回退到该本地可执行文件。

### Silero VAD

- 用途：检测语音活动区间，生成候选样本边界。
- Python 包：`silero-vad`。
- 验证导入：

```powershell
.\local_tools\venv\python.exe -c "from silero_vad import load_silero_vad, read_audio, get_speech_timestamps; print('silero import ok')"
```

### faster-whisper

- 用途：生成 ASR 候选文本和置信度；TTS、ASR 用途会把该结果作为质量证据。
- Python 包：`faster-whisper`。
- 发布版未包含 faster-whisper 模型权重；运行 ASR 前需要自行放入模型目录。

```powershell
.\local_tools\venv\python.exe -c "from faster_whisper import WhisperModel; print('faster-whisper import ok')"
```

### DeepFilterNet

- 用途：对高噪声但有保留价值的样本执行神经网络语音增强。
- 运行形式：可选 CLI，目标路径为 `local_tools\deepfilternet\deep-filter.exe`。
- 接入边界：只由 Repair Executor 执行；修复后必须经 Repair Evaluator 评估，不能直接替换原音频。

下载完成后验证：

```powershell
.\local_tools\deepfilternet\deep-filter.exe --help
```

当前 Python 版 `deepfilternet` 不作为默认方案，因为它和此环境的新版 `torchaudio` API 不兼容。没有 CLI 时，相关样本会保留修复失败记录并进入既有决策路径，不影响整个批次导出。

### 本地 Qwen

- 用途：为 Strategy Agent、Curator Agent 和 Report Agent 提供可选 LLM 推理。
- 前端默认模型路径：`local_tools\models\qwen\Qwen3-4B-Instruct-2507`。
- 启用后仍受数据用途画像、工具注册表、结构化输出校验和规则 fallback 约束。

## 工具接入规范

新增音频工具时遵循以下步骤：

1. 下载到 `local_tools/`，先独立验证命令可用。
2. 在项目源码内为它创建适配器类。
3. 把原始输出转换为项目的 `AnalysisResult`、`RepairExecution` 等内部数据结构。
4. 在工具注册表声明 `tool_id`、输入输出产物、适用场景和限制。
5. 由 Strategy Agent 选择工具，由 Executor 调用工具，由 Evaluator 判断产物是否可采用。

这样可以保持“Agent 负责决策、工具负责执行、状态文件负责审计”的边界。
