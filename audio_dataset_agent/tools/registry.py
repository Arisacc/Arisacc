# 工具注册表：给 Strategy Agent 暴露可用处理工具及其适用场景。
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCapability:
    """面向 Agent 的确定性音频工具能力描述。"""

    tool_id: str
    family: str
    display_name: str
    implementation: str
    description: str
    input_artifacts: list[str] = field(default_factory=list)
    output_artifacts: list[str] = field(default_factory=list)
    best_for: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    configurable_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRegistry:
    """Agent 选择处理动作前可查看的小型工具目录。"""

    capabilities: list[ToolCapability]

    def by_family(self, family: str) -> list[ToolCapability]:
        return [tool for tool in self.capabilities if tool.family == family]

    def get(self, tool_id: str) -> ToolCapability | None:
        for tool in self.capabilities:
            if tool.tool_id == tool_id:
                return tool
        return None


def default_tool_registry() -> ToolRegistry:
    """返回第一版项目使用的内置工具目录。"""
    return ToolRegistry(
        capabilities=[
            ToolCapability(
                tool_id="normalizer.copy",
                family="normalizer",
                display_name="复制型音频标准化器",
                implementation="CopyAudioNormalizer",
                description="创建工作副本，不修改原始音频内容。",
                input_artifacts=["RawAsset.source_path"],
                output_artifacts=["RawAsset.work_path"],
                best_for=["安全基线运行", "测试", "已标准化的 WAV 文件"],
                limitations=["不重采样", "不转换声道数"],
            ),
            ToolCapability(
                tool_id="normalizer.ffmpeg",
                family="normalizer",
                display_name="FFmpeg 音频标准化器",
                implementation="FFmpegAudioNormalizer",
                description="使用 FFmpeg 创建标准 WAV 工作副本。",
                input_artifacts=["RawAsset.source_path"],
                output_artifacts=["RawAsset.work_path"],
                best_for=["格式转换", "采样率标准化", "单声道转换"],
                limitations=["需要 FFmpeg 可执行文件", "不负责判断质量问题"],
                dependencies=["ffmpeg"],
                configurable_parameters={"sample_rate": 16000, "channels": 1},
            ),
            ToolCapability(
                tool_id="vad.fixed",
                family="vad",
                display_name="固定结果 VAD",
                implementation="FixedVADAnalyzer",
                description="为确定性演示和测试返回预设语音片段。",
                input_artifacts=["RawAsset.work_path"],
                output_artifacts=["SpeechSegment"],
                best_for=["单元测试", "流水线冒烟测试"],
                limitations=["不是真实语音检测"],
            ),
            ToolCapability(
                tool_id="vad.wave_energy",
                family="vad",
                display_name="波形能量 VAD",
                implementation="WaveEnergyVADAnalyzer",
                description="在 PCM WAV 文件上通过帧级 RMS 能量检测类语音区域。",
                input_artifacts=["RawAsset.work_path"],
                output_artifacts=["SpeechSegment"],
                best_for=["轻量本地 VAD", "快速验证", "无模型依赖"],
                limitations=["仅支持 PCM WAV", "对噪声底敏感", "不区分说话人"],
                configurable_parameters={
                    "frame_ms": 30,
                    "energy_threshold_dbfs": -40.0,
                    "min_duration_seconds": 0.2,
                },
            ),
            ToolCapability(
                tool_id="vad.silero",
                family="vad",
                display_name="Silero VAD 语音检测器",
                implementation="SileroVADAnalyzer",
                description="使用 Silero VAD 神经网络模型检测语音片段。",
                input_artifacts=["RawAsset.work_path"],
                output_artifacts=["SpeechSegment"],
                best_for=["真实语音切分", "本地 CPU 推理"],
                limitations=["需要 torch 和 silero-vad", "模型效果受音频领域影响"],
                dependencies=["torch", "silero-vad"],
                configurable_parameters={
                    "sampling_rate": 16000,
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                },
            ),
            ToolCapability(
                tool_id="materializer.copy",
                family="materializer",
                display_name="复制型样本物化器",
                implementation="CopySampleMaterializer",
                description="通过复制标准化工作音频创建样本文件。",
                input_artifacts=["CandidateSample.source_path"],
                output_artifacts=["CandidateSample.sample_path"],
                best_for=["测试", "无需精确裁剪的演示运行"],
                limitations=["不按片段时间戳裁剪"],
            ),
            ToolCapability(
                tool_id="materializer.ffmpeg",
                family="materializer",
                display_name="FFmpeg 样本物化器",
                implementation="FFmpegSampleMaterializer",
                description="使用 FFmpeg 按 VAD 片段时间戳裁剪候选样本。",
                input_artifacts=["CandidateSample.source_path", "start_time", "end_time"],
                output_artifacts=["CandidateSample.sample_path"],
                best_for=["真实样本提取", "基于时间戳的裁剪"],
                limitations=["需要 FFmpeg 可执行文件"],
                dependencies=["ffmpeg"],
            ),
            ToolCapability(
                tool_id="asr.fixed",
                family="asr",
                display_name="固定结果转写器",
                implementation="FixedTranscriber",
                description="为确定性测试和演示返回预设转写文本。",
                input_artifacts=["CandidateSample.sample_path"],
                output_artifacts=["transcript", "transcript_source", "asr_confidence"],
                best_for=["单元测试", "流水线冒烟测试"],
                limitations=["不是真实转写"],
            ),
            ToolCapability(
                tool_id="asr.faster_whisper",
                family="asr",
                display_name="faster-whisper ASR 转写器",
                implementation="FasterWhisperTranscriber",
                description="运行 faster-whisper，生成候选转写文本及置信度。",
                input_artifacts=["CandidateSample.sample_path"],
                output_artifacts=["transcript", "transcript_source", "asr_confidence"],
                best_for=["真实本地 ASR", "TTS 数据集候选文本", "人工复核队列"],
                limitations=["ASR 文本可能错误", "低置信度应触发人工复核"],
                dependencies=["faster-whisper", "local whisper model"],
                configurable_parameters={
                    "model": "small",
                    "device": "cpu",
                    "compute_type": "int8",
                },
            ),
            ToolCapability(
                tool_id="metrics.wave",
                family="audio_metrics",
                display_name="Wave 音频指标分析器",
                implementation="WaveAudioMetricsAnalyzer",
                description="从 PCM WAV 文件测量 RMS、峰值、削波比例、静音比例、动态范围和直流偏移。",
                input_artifacts=["CandidateSample.sample_path"],
                output_artifacts=["audio_metrics"],
                best_for=["质量证据", "修复效果评估", "审核智能体输入"],
                limitations=["最适合可读取的 WAV 文件", "不是感知质量模型"],
            ),
            ToolCapability(
                tool_id="metrics.ffmpeg_astats",
                family="audio_metrics",
                display_name="FFmpeg AStats 指标分析器",
                implementation="FFmpegAudioMetricsAnalyzer",
                description="使用 FFmpeg astats 提取 RMS、峰值、噪声底、动态范围和直流偏移证据。",
                input_artifacts=["CandidateSample.sample_path"],
                output_artifacts=[
                    "rms_dbfs",
                    "peak_dbfs",
                    "noise_floor_dbfs",
                    "dynamic_range_db",
                    "dc_offset",
                ],
                best_for=[
                    "audio_noise_high",
                    "audio_too_quiet",
                    "audio_clipping",
                    "tts_quality_gate",
                ],
                limitations=[
                    "需要 FFmpeg 可执行文件",
                    "噪声底是启发式指标，修复后应重新评估",
                ],
                dependencies=["ffmpeg"],
            ),
            ToolCapability(
                tool_id="metrics.ffmpeg_silencedetect",
                family="audio_metrics",
                display_name="FFmpeg 静音检测器",
                implementation="FFmpegAudioMetricsAnalyzer",
                description="使用 FFmpeg silencedetect 测量静音区间，支持静音比例质量标签。",
                input_artifacts=["CandidateSample.sample_path"],
                output_artifacts=["silence_ratio", "silence_regions"],
                best_for=["audio_high_silence_ratio", "trim_silence", "tts_quality_gate"],
                limitations=[
                    "需要 FFmpeg 可执行文件",
                    "对阈值敏感，应与数据集用途策略一致",
                ],
                dependencies=["ffmpeg"],
                configurable_parameters={"silence_threshold_dbfs": -50.0},
            ),
            ToolCapability(
                tool_id="repair.ffmpeg_loudness",
                family="repair",
                display_name="FFmpeg 响度标准化",
                implementation="FFmpegRepairExecutor",
                description="使用 FFmpeg loudnorm 改善音量偏低的样本。",
                input_artifacts=["RepairPlan", "CandidateSample.sample_path"],
                output_artifacts=["RepairExecution.output_path"],
                best_for=["audio_too_quiet", "normalize_loudness"],
                limitations=["可能同时放大噪声", "需要修复后评估"],
                dependencies=["ffmpeg"],
                configurable_parameters={"target_lufs": -23.0},
            ),
            ToolCapability(
                tool_id="repair.ffmpeg_silence_trim",
                family="repair",
                display_name="FFmpeg 静音裁剪",
                implementation="FFmpegRepairExecutor",
                description="使用 FFmpeg 静音移除功能减少过长静音区间。",
                input_artifacts=["RepairPlan", "CandidateSample.sample_path"],
                output_artifacts=["RepairExecution.output_path"],
                best_for=["audio_high_silence_ratio", "trim_silence"],
                limitations=["可能移除有意停顿", "需要修复后评估"],
                dependencies=["ffmpeg"],
                configurable_parameters={
                    "silence_threshold_dbfs": -50.0,
                    "min_silence_duration_ms": 200,
                },
            ),
            ToolCapability(
                tool_id="repair.ffmpeg_denoise",
                family="repair",
                display_name="FFmpeg 轻量降噪",
                implementation="FFmpegRepairExecutor",
                description="使用 FFmpeg afftdn 进行轻量级 FFT 降噪。",
                input_artifacts=["RepairPlan", "CandidateSample.sample_path"],
                output_artifacts=["RepairExecution.output_path"],
                best_for=["audio_noise_high", "denoise_light"],
                limitations=[
                    "不是神经网络语音增强模型",
                    "过强使用可能产生伪影",
                    "需要修复后评估",
                ],
                dependencies=["ffmpeg"],
                configurable_parameters={"noise_reduction_db": 12.0},
            ),
            ToolCapability(
                tool_id="repair.deepfilternet_denoise",
                family="repair",
                display_name="DeepFilterNet 神经网络降噪",
                implementation="DeepFilterNetRepairExecutor",
                description="可选的神经网络语音增强降噪工具，适用于轻量 FFmpeg 降噪不足的高噪声语音样本。",
                input_artifacts=["RepairPlan", "CandidateSample.sample_path"],
                output_artifacts=["RepairExecution.output_path"],
                best_for=["audio_noise_high", "neural_denoise", "speech_enhancement"],
                limitations=[
                    "需要 DeepFilterNet CLI；存在时会自动识别 local_tools/deepfilternet/deep-filter.exe",
                    "可能改变说话人音色，需要修复后评估",
                    "过度抑制的输出必须进入复核，不能自动采用",
                ],
                dependencies=["DeepFilterNet"],
                configurable_parameters={"strength": "default"},
            ),
        ]
    )


def tool_capability_to_dict(capability: ToolCapability) -> dict[str, Any]:
    return {
        "tool_id": capability.tool_id,
        "family": capability.family,
        "display_name": capability.display_name,
        "implementation": capability.implementation,
        "description": capability.description,
        "input_artifacts": list(capability.input_artifacts),
        "output_artifacts": list(capability.output_artifacts),
        "best_for": list(capability.best_for),
        "limitations": list(capability.limitations),
        "dependencies": list(capability.dependencies),
        "configurable_parameters": dict(capability.configurable_parameters),
    }


def tool_registry_to_dict(registry: ToolRegistry) -> dict[str, Any]:
    return {
        "capability_count": len(registry.capabilities),
        "capabilities": [
            tool_capability_to_dict(capability)
            for capability in registry.capabilities
        ],
    }
