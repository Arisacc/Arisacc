---
title: FFmpeg 音频工具
tool_ids: normalizer.ffmpeg,materializer.ffmpeg,metrics.ffmpeg_astats,metrics.ffmpeg_silencedetect,repair.ffmpeg_loudness,repair.ffmpeg_silence_trim,repair.ffmpeg_denoise
profiles: asr,tts,voice_clone
---
# FFmpeg 音频工具

## 用途

FFmpeg 用于格式标准化、采样率和声道转换、按时间戳裁剪、响度标准化、静音裁剪和轻量降噪。项目通过适配器调用 FFmpeg，不让 LLM 直接生成命令。

## 处理边界

格式转换和音频滤镜属于确定性执行步骤。轻量降噪可能产生伪影，响度标准化可能同时放大噪声，静音裁剪可能移除有意停顿；这些输出都需要经过修复后指标评估。

## 质量指标

FFmpeg astats 可以提供 RMS、峰值、噪声底、动态范围和直流偏移；silencedetect 可以辅助统计静音区间。指标只作为 Inspector 证据，不能单独替代人耳或下游模型评估。
