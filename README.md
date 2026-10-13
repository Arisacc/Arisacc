# 音频数据集 Agent 工作台

面向 ASR、TTS 和声音克隆数据准备的本地优先音频治理流水线。项目把音频分析、处理工具、Agent 决策和最终样本选择拆成独立阶段，输出可检查的样本决策、质量报告和处理追溯信息。

## 项目能力

- 递归扫描本地音频目录，并匹配同名文本、CSV、JSONL 元数据。
- 使用 FFmpeg 生成统一格式的分析副本，不覆盖原始音频。
- 支持整段音频、能量 VAD，以及可选 Silero VAD。
- 对音频计算时长、响度、静音比例、削波、SNR 和质量分等证据。
- Strategy Agent 根据质量问题、数据用途和工具注册表提出处理策略。
- Curator Agent 判断样本 `keep / reject / repair / review`；默认使用可复现的规则基线。
- 可选地接入本地 Qwen、faster-whisper、Silero VAD 和 DeepFilterNet。
- 对修复结果重新分析和评估，导出 manifest、报告、工具版本和 Agent Graph 追溯。
- 提供 React 本地工作台，用于配置任务、查看质量概览、筛选样本和人工复核。
- 使用 Harness Runtime 管理 Run 状态、SQLite 事件日志、检查点和导出产物索引。

## 处理流程

```text
选择本地音频目录与数据用途
  -> 扫描文件和匹配元数据
  -> FFmpeg 标准化工作副本
  -> VAD 检测或整段处理
  -> 可选 ASR 候选文本
  -> Inspector 生成质量证据
  -> Strategy Agent 提出策略和工具建议
  -> Curator Agent 决定样本去留
  -> 可选修复并重新分析
  -> 最终样本选择
  -> 导出 manifest、质量报告和决策追溯
```

Agent 输出是受结构化数据模型和可用工具约束的；Agent 负责分析和决策，实际音频变换由工具执行，修复结果需经过复评。规则基线无需下载模型即可运行。

## 轻量仓库边界

仓库不包含用户音频、模型权重、Python 虚拟环境、前端 `node_modules` 和运行缓存。克隆后按下方步骤安装轻量默认依赖即可运行规则基线。Qwen、faster-whisper、Silero VAD、Torch 和 DeepFilterNet 是可选扩展，不是默认流程的运行前提。

| 能力 | 默认状态 | 额外要求 |
| --- | --- | --- |
| 目录扫描、规则 Agent、质量分析、报告导出 | 可用 | Python 3.11+ |
| FFmpeg 格式标准化、片段处理 | 可用 | 安装 `imageio-ffmpeg`，首次运行时下载 FFmpeg wheel |
| Web 工作台 | 可用 | FastAPI 使用仓库自带的 `frontend/dist`；修改前端才需要 Node.js |
| faster-whisper ASR | 可选 | 安装 `requirements-optional.txt` 中相关依赖并准备模型 |
| Silero VAD | 可选 | 安装 PyTorch、torchaudio、silero-vad |
| Qwen 本地 Agent | 可选 | 安装 Transformers 依赖并自行准备兼容模型 |
| DeepFilterNet 神经降噪 | 可选 | 按 `local_tools/README.md` 下载 CLI |

## Windows 快速开始

环境要求：Windows 10/11、Python 3.11 或更高版本。若需要从前端源码重新构建，再安装 Node.js 20 或更高版本。

```powershell
git clone <你的 GitHub 仓库地址>
cd <仓库目录>
.\setup_windows.bat
```

初始化完成后启动客户端：

```powershell
.\start_client.bat
```

客户端会启动仅监听本机回环地址的 FastAPI Harness 后端，并打开 `http://127.0.0.1:8000/pipeline`。在网页中选择本地音频目录，选择 TTS、ASR 或声音克隆用途，按需调整参数后点击“开始处理”。默认关闭 ASR，默认 Agent 为规则基线。

Harness API 还提供 `GET /audio-dataset/jobs/{job_id}/events` 和 `GET /audio-dataset/jobs/{job_id}/artifacts`，可查看一次 Run 的状态迁移、检查点和导出产物。SQLite 数据库默认位于 `local_tools/cache/audio_web_runs/harness.sqlite3`。

停止后端：

```powershell
.\stop_client.bat
```

如果只需要 API 服务，可以单独运行 `start_backend.bat`。开发前端时运行 `start_frontend.bat`，页面地址为 `http://localhost:5184`。

## 命令行示例

```powershell
python -m audio_dataset_agent.cli `
  --input-dir "D:\audio_samples" `
  --output-dir "local_tools\cache\work" `
  --export-dir "local_tools\cache\exports" `
  --target-policy voice_clone `
  --workflow basic `
  --vad whole-file `
  --normalizer ffmpeg `
  --sample-materializer ffmpeg `
  --audio-metrics wave `
  --asr none
```

运行产物写入 `local_tools/cache/`，该目录已加入 `.gitignore`，不会被提交。

## 可选模型能力

默认安装只覆盖轻量规则流程。需要本地模型功能时，先阅读 `local_tools/README.md`，再安装可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

可选依赖包含 PyTorch 和 Transformers，体积较大且需要根据显卡、驱动和模型许可证单独评估。不要把模型权重提交到 GitHub；模型路径、下载和硬件配置由使用者自行管理。

## 仓库结构

```text
audio_dataset_agent/    Python 核心流水线、Agent、工具适配器和 Web API
frontend/src/           React + TypeScript 前端源码
frontend/dist/          已构建前端，供本地客户端直接启动
knowledge_base/         音频工具与数据集规范知识库
local_tools/            可选模型/工具下载脚本和使用说明
tests/                  单元测试与集成测试
setup_windows.bat       创建虚拟环境、安装依赖并构建前端
start_client.bat        一键启动本地客户端
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

测试套件会自动跳过需要额外安装 `torch`、`soundfile` 和 `silero-vad` 的适配器测试。安装 `requirements-optional.txt` 后重新运行，可以补充验证 Silero VAD 运行时。

前端构建：

```powershell
cd frontend
npm ci
npm run build
```

## 隐私与安全

- 后端默认绑定 `127.0.0.1`，不对局域网或公网开放。
- 输入音频不会随仓库发布；输出文件仅保存在本机运行缓存中。
- 不需要 API Key；不要提交 `.env`、访问令牌、个人路径、私有数据或模型权重。
- 数据和阈值是工程默认配置，真实业务使用前应按目标数据分布校准。

