---
title: DeepFilterNet
tool_ids: repair.deepfilternet_denoise
profiles: voice_clone,tts,asr
---
# DeepFilterNet

## 用途

DeepFilterNet 用于对含噪语音执行神经网络语音增强。当轻量 FFmpeg 降噪不足且样本仍有保留价值时，可以作为 Repair 工具。

## 声音克隆风险

神经降噪可能改变说话人音色，也可能产生过度抑制和伪影。voice_clone 样本必须比较修复前后指标，必要时进行人工试听，不能仅凭“噪声降低”自动采用。

## 项目中的位置

Strategy Agent 只能推荐 `repair.deepfilternet_denoise`，Repair Executor 才负责执行，Repair Evaluator 决定是否采用输出。缺少 CLI 或执行失败时，样本保留失败记录并进入复核路径。
