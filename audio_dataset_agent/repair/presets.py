"""修复预设：按数据用途提供可复现的默认修复参数。"""
from dataclasses import dataclass

from audio_dataset_agent.profiles.policy import DatasetProfile, get_dataset_profile


@dataclass(frozen=True)
class RepairPreset:
    """同一修复操作在不同下游任务中的默认参数。"""

    target_policy: str
    target_lufs: float
    silence_threshold_dbfs: float
    min_silence_duration_ms: int
    noise_reduction_db: float


_PRESETS = {
    "tts": RepairPreset(
        target_policy="tts",
        target_lufs=-23.0,
        silence_threshold_dbfs=-50.0,
        min_silence_duration_ms=200,
        noise_reduction_db=12.0,
    ),
    "asr": RepairPreset(
        target_policy="asr",
        target_lufs=-20.0,
        silence_threshold_dbfs=-45.0,
        min_silence_duration_ms=250,
        noise_reduction_db=8.0,
    ),
    "voice_clone": RepairPreset(
        target_policy="voice_clone",
        target_lufs=-23.0,
        silence_threshold_dbfs=-50.0,
        min_silence_duration_ms=200,
        noise_reduction_db=10.0,
    ),
}


def get_repair_preset(
    target_policy: str | DatasetProfile,
) -> RepairPreset:
    """返回目标用途对应的修复预设。"""
    profile = get_dataset_profile(target_policy)
    return _PRESETS[profile.name]
