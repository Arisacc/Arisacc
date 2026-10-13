"""数据用途画像。

同一批音频用于 ASR、TTS、voice_clone 时，文本可靠性、噪声容忍度和修复偏好不同。DatasetProfile 把这些差异集中定义，Inspector、Strategy、Curator 和 Report Agent 都读取同一个 profile，避免各模块各自写一套判断口径。
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetProfile:
    """面向下游任务的质量要求集合。

    requires_* 决定流程是否必须有文本；allow_* 决定文本问题是否阻塞；preferred_denoise_operation 会影响 Strategy Agent 在噪声样本上推荐轻量降噪还是神经网络降噪。
    """

    name: str
    requires_asr: bool
    requires_text: bool
    text_reliability_required: bool
    allow_missing_transcript: bool
    allow_low_asr_confidence: bool
    strict_audio_cleanliness: bool
    moderate_noise_tolerated: bool
    preferred_denoise_operation: str
    min_duration_seconds: float
    max_duration_seconds: float | None
    min_asr_confidence: float
    min_rms_dbfs: float
    max_clipping_ratio: float
    max_silence_ratio: float
    max_noise_floor_dbfs: float
    requires_reference_text: bool
    text_cer_pass: float | None
    text_cer_review: float | None
    report_focus: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)

    @property
    def text_issue_tags_are_blocking(self) -> bool:
        return self.text_reliability_required and not self.allow_missing_transcript


_PROFILES = {
    "asr": DatasetProfile(
        name="asr",
        requires_asr=True,
        requires_text=True,
        text_reliability_required=True,
        allow_missing_transcript=False,
        allow_low_asr_confidence=False,
        strict_audio_cleanliness=False,
        moderate_noise_tolerated=True,
        preferred_denoise_operation="denoise_light",
        min_duration_seconds=0.4,
        max_duration_seconds=30.0,
        min_asr_confidence=0.75,
        min_rms_dbfs=-50.0,
        max_clipping_ratio=0.02,
        max_silence_ratio=0.75,
        max_noise_floor_dbfs=-28.0,
        requires_reference_text=False,
        text_cer_pass=0.15,
        text_cer_review=0.35,
        report_focus=[
            "text_reliability",
            "recognizable_speech",
            "noise_distribution",
        ],
        risk_notes=[
            "低置信度文本应进入人工复核。",
            "语音仍可识别时，可容忍中等程度噪声。",
        ],
    ),
    "tts": DatasetProfile(
        name="tts",
        requires_asr=True,
        requires_text=True,
        text_reliability_required=True,
        allow_missing_transcript=False,
        allow_low_asr_confidence=False,
        strict_audio_cleanliness=True,
        moderate_noise_tolerated=False,
        preferred_denoise_operation="denoise_light",
        min_duration_seconds=1.0,
        max_duration_seconds=15.0,
        min_asr_confidence=0.85,
        min_rms_dbfs=-40.0,
        max_clipping_ratio=0.005,
        max_silence_ratio=0.35,
        max_noise_floor_dbfs=-40.0,
        requires_reference_text=True,
        text_cer_pass=0.08,
        text_cer_review=0.25,
        report_focus=[
            "text_audio_pairing",
            "audio_cleanliness",
            "silence",
            "loudness",
        ],
        risk_notes=[
            "文本与音频不匹配会污染监督式 TTS 训练数据。",
            "噪声、削波和过长静音应修复或人工复核。",
        ],
    ),
    "voice_clone": DatasetProfile(
        name="voice_clone",
        requires_asr=False,
        requires_text=False,
        text_reliability_required=False,
        allow_missing_transcript=True,
        allow_low_asr_confidence=True,
        strict_audio_cleanliness=True,
        moderate_noise_tolerated=False,
        preferred_denoise_operation="neural_denoise",
        min_duration_seconds=1.5,
        max_duration_seconds=30.0,
        min_asr_confidence=0.8,
        min_rms_dbfs=-45.0,
        max_clipping_ratio=0.005,
        max_silence_ratio=0.45,
        max_noise_floor_dbfs=-40.0,
        requires_reference_text=False,
        text_cer_pass=None,
        text_cer_review=None,
        report_focus=[
        "speaker_timbre",
        "speaker_consistency",
            "noise",
            "clipping",
            "silence",
        ],
        risk_notes=[
            "纯声音克隆数据准备中，文本是可选项。",
            "降噪可能改变说话人音色，应在修复后评估。",
        ],
    ),
}


def get_dataset_profile(target_policy: str | DatasetProfile) -> DatasetProfile:
    if isinstance(target_policy, DatasetProfile):
        return target_policy
    try:
        return _PROFILES[target_policy]
    except KeyError as exc:
        raise ValueError(f"未知数据用途画像：{target_policy}") from exc
