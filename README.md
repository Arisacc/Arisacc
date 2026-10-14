# 声音数据处理工作台

面向 `voice_clone` 的本地优先音频数据治理工具。它把音频扫描、预处理、质量分析、Agent 决策、人工复核和数据导出串成一条可追溯流水线。

## 核心能力

- 本地音频目录选择与递归扫描
- FFmpeg 音频标准化
- 整段处理、能量 VAD 和候选片段生成
- 音频时长、静音、削波、响度、SNR 和质量分分析
- Strategy Agent 生成处理建议
- Curator Agent 生成 `keep / reject / repair / review` 决策
- 修复后重新分析和评估
- 人工复核后将样本归档到 `keep` 或 `reject`
- 下载人工保留音频数据包
- FastAPI + SQLite + Worker + SSE + checkpoint Harness
- 可选 MSST/pymss 去混响预处理

## 处理流程

```text
选择音频目录
  -> voice_clone 配置
  -> 可选 MSST 去混响
  -> FFmpeg 标准化
  -> VAD / 整段处理
  -> Inspector 质量分析
  -> Strategy Agent 建议
  -> Curator Agent 决策
  -> 修复与复评
  -> 人工保留/拒绝
  -> 导出 manifest、报告和保留音频包
```

项目不负责 SoVITS 训练，只输出可供后续训练使用的音频和追溯文件。

## 轻量版边界

仓库不包含：

- 用户音频
- 模型权重
- Python 虚拟环境
- `node_modules`
- 运行缓存和 SQLite 数据库

源码仓库包含已构建的 `frontend/dist`，克隆后可以直接启动。MSST 模型和运行时需要使用者按需安装。

## Windows 启动

环境要求：Windows 10/11、Python 3.11+。如果需要重新构建前端，再安装 Node.js 20+。

```powershell
git clone <你的 GitHub 仓库地址>
cd <仓库目录>
.\setup_windows.bat
.\start_client.bat
```

启动后浏览器会打开：

```text
http://127.0.0.1:8000/pipeline
```

停止服务：

```powershell
.\stop_client.bat
```

`start_client.bat` 会同时启动 FastAPI 网关和 SQLite Worker。只启动后端而不启动 Worker 时，任务不会被消费。

## MSST 去混响

当前只支持：

```text
deverb_bs_roformer_8_256dim_8depth
```

模型目录约定：

```text
local_tools/models/msst/deverb_bs_roformer_8_256dim_8depth/
├── deverb_bs_roformer_8_256dim_8depth.ckpt
└── deverb_bs_roformer_8_256dim_8depth.ckpt.yaml
```

轻量版不携带模型。正式版已经验证过该模型和 `pymss==2.1.8` 的 GPU 推理。

按需安装 MSST 运行时：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

启用 MSST 后，模型会在项目 Worker 进程内直接加载，不需要另外启动 MSST-WebUI。
该可选依赖会安装 `pymss` 及其音频处理依赖；PyTorch 体积较大。若使用 NVIDIA GPU，请先按本机 CUDA/驱动选择并安装匹配的 PyTorch 版本，再安装该文件中的依赖。仓库不包含 MSST 权重文件。

## 一键下载模型

双击项目根目录的 `download_models.bat`，可交互选择：

| 模型 | 保存目录 | 用途 | 预计大小 |
| --- | --- | --- | ---: |
| Qwen3-4B-Instruct | `local_tools/models/qwen/Qwen3-4B-Instruct-2507` | 本地 Agent 推理 | 约 7.5 GiB |
| faster-whisper-small | `local_tools/models/faster-whisper/small` | ASR 候选文本 | 约 486 MB |
| MSST BS-Roformer | `local_tools/models/msst/deverb_bs_roformer_8_256dim_8depth` | 去混响 | 约 163MB |

也可以命令行选择：

```powershell
.\.venv\Scripts\python.exe local_tools\download_models.py --model msst --yes
.\.venv\Scripts\python.exe local_tools\download_models.py --model qwen --provider modelscope --yes
.\.venv\Scripts\python.exe local_tools\download_models.py --model faster-whisper --yes
```

模型权重不会提交到 GitHub。脚本可以重复执行，已经存在的文件会跳过。

模型缺失策略：

- `阻断任务并报错`：严格模式，缺失模型或运行时就停止。
- `跳过 MSST，继续处理`：不执行去混响，继续后续标准化和质量治理。

## 导出结果

报告页提供质量报告、manifest、Agent Graph 追溯、工具链信息和“下载保留音频”按钮。

下载包只包含人工保留音频：

```text
manual_groups/keep/
```

不包含拒绝、修复或待复核音频。

## 项目结构

```text
audio_dataset_agent/    Python 流水线、Agent、工具适配器和 Web API
frontend/src/           React + TypeScript 前端源码
frontend/dist/          可直接启动的前端构建产物
knowledge_base/         本地音频工具和数据规范知识库
local_tools/            工具说明、模型目录和辅助脚本
tests/                  单元测试和集成测试
docs/                   使用说明
scripts/                开发与调试辅助脚本
start_client.bat        一键启动客户端
stop_client.bat         一键停止客户端
download_models.bat     可选模型下载入口
docs/运行说明.md        本地运行说明
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

前端构建：

```powershell
cd frontend
npm ci
npm run build
```

## API 和数据安全

- 默认只绑定 `127.0.0.1`。
- 音频数据不上传外部服务。
- SQLite 只保存 Run、Task、事件、Checkpoint 和产物索引。
- 不要提交 `.env`、个人路径、音频文件、模型权重或访问令牌。

