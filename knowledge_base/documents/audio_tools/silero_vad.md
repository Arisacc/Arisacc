---
title: Silero VAD
tool_ids: vad.silero
profiles: asr,tts,voice_clone
---
# Silero VAD

## 用途

Silero VAD 用于检测语音活动区间，输出候选片段的起止时间。它适合批量语音切分和去除无信息静音，不负责判断文本是否正确，也不负责最终样本去留。

## 项目中的位置

Normalizer 生成工作音频后，VAD 产生时间段，Sample Materializer 再把时间段裁剪成实际音频文件。TTS/ASR 通常使用切分结果，voice_clone 可以选择较长片段或 whole-file 模式。

## 限制

VAD 的边界受阈值、噪声和录音环境影响。切分后的文本如果没有片段级标注，不能把整段参考文本直接当作每个候选片段的精确文本。
