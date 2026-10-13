---
title: faster-whisper
tool_ids: asr.faster_whisper
profiles: asr,tts
---
# faster-whisper

## 用途

faster-whisper 是基于 CTranslate2 的 Whisper 推理实现，用于批量生成 ASR 候选文本和置信度。项目将结果写入 `transcript`、`transcript_source` 和 `asr_confidence`。

## 数据质量边界

ASR 输出是候选证据，不是无条件真值。低置信度、空文本或与人工参考文本差异较大的样本应进入人工复核。TTS 可以使用 CER/WER 检查文本和音频配对；ASR 没有参考文本时也可以独立生成候选转写。

## 运行约束

模型、设备、量化方式和语言应写入运行配置。批量处理前应确认本地模型存在，并限制推理超时和失败重试次数。
