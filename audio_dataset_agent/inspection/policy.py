# 质量标签规则：把指标转换成 too_short、high_silence 等问题标签。
from dataclasses import dataclass

from audio_dataset_agent.profiles.policy import DatasetProfile, get_dataset_profile


@dataclass(frozen=True)
class InspectionPolicy:
    min_duration_seconds: float = 1.0
    max_duration_seconds: float | None = None
    min_asr_confidence: float = 0.8
    min_rms_dbfs: float = -45.0
    max_clipping_ratio: float = 0.01
    max_silence_ratio: float = 0.6
    max_noise_floor_dbfs: float = -35.0

    @classmethod
    def from_profile(
        cls,
        target_policy: str | DatasetProfile,
    ) -> "InspectionPolicy":
        """从 DatasetProfile 构造 Inspector 使用的质量阈值。"""
        profile = get_dataset_profile(target_policy)
        return cls(
            min_duration_seconds=profile.min_duration_seconds,
            max_duration_seconds=profile.max_duration_seconds,
            min_asr_confidence=profile.min_asr_confidence,
            min_rms_dbfs=profile.min_rms_dbfs,
            max_clipping_ratio=profile.max_clipping_ratio,
            max_silence_ratio=profile.max_silence_ratio,
            max_noise_floor_dbfs=profile.max_noise_floor_dbfs,
        )

    def to_dict(self) -> dict[str, float | None]:
        return {
            "min_duration_seconds": self.min_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "min_asr_confidence": self.min_asr_confidence,
            "min_rms_dbfs": self.min_rms_dbfs,
            "max_clipping_ratio": self.max_clipping_ratio,
            "max_silence_ratio": self.max_silence_ratio,
            "max_noise_floor_dbfs": self.max_noise_floor_dbfs,
        }
