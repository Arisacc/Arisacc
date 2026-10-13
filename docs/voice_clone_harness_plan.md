# Voice Clone Harness 正式版实施计划

## 目标

将项目收敛为只处理 `voice_clone` 音频数据的本地优先质量治理系统，并打通 FastAPI、SQLite 任务队列、独立 Worker、Harness 事件流、checkpoint、前端实时状态和真实端到端验证。

## 技术边界

- 后端：FastAPI、Pydantic、SQLite WAL、标准库线程 Worker。
- 处理：FFmpeg、VAD、Inspector、Strategy、Curator、Repair、Manifest/报告导出。
- 前端：React 19、TypeScript、Vite、React Router、Fetch/SSE、原生 CSS。
- 不接入 SoVITS 训练；输出可直接交给后续训练系统的音频数据包和追溯文件。
- 本阶段不引入 PostgreSQL、Redis、Celery、Docker，避免为单机/小规模云服务器增加非必要运维面。

## 分阶段闭环

1. Run/Task 状态闭环：创建 Run 与任务、领取租约、成功/失败/重试、终态一致。
2. Harness 生命周期：追加事件、Worker 状态事件、checkpoint、重启后可重新领取未完成任务。
3. 数据流契约：验证 raw -> normalized -> VAD sample -> inspection -> strategy/curator -> repair -> export 的 sample_id、路径和产物不丢失。
4. 前端收口：只展示 voice_clone、已接通的 VAD/流程/阈值/目录选择和真实 Run 状态，移除假 ASR/模型/Agent 配置。
5. 实时交互：SSE 优先、轮询兜底，显示当前阶段、Worker、checkpoint 和失败原因。
6. E2E 验证：生成临时 WAV，提交 Run，Worker 执行，断言 SQLite、manifest、报告、checkpoint、事件和前端页面一致。
7. 发布同步：正式版通过验证后，清理缓存/测试数据/虚拟环境，再同步到 GitHub 轻量版。
