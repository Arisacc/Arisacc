"""声音克隆场景的轻量批次声学一致性检查。"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from audio_dataset_agent.inspection.audio_metrics import AudioMetrics, AudioMetricsAnalyzer


@dataclass(frozen=True)
class AcousticConsistencyPolicy:
    """控制同批样本声学特征离散度的提醒阈值。"""

    max_rms_range_db: float = 12.0
    max_silence_range: float = 0.35
    max_noise_floor_range_db: float = 18.0
    max_dynamic_range_db: float = 35.0


@dataclass(frozen=True)
class AcousticConsistencyResult:
    sample_count: int
    compared_sample_count: int
    rms_range_db: float
    silence_range: float
    noise_floor_range_db: float | None
    dynamic_range_range_db: float | None
    warning: bool
    warning_reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "compared_sample_count": self.compared_sample_count,
            "rms_range_db": self.rms_range_db,
            "silence_range": self.silence_range,
            "noise_floor_range_db": self.noise_floor_range_db,
            "dynamic_range_range_db": self.dynamic_range_range_db,
            "warning": self.warning,
            "warning_reasons": list(self.warning_reasons),
            "scope": "轻量声学一致性提醒，不等同于声纹身份验证",
        }


class ConsistencyAnalyzer(Protocol):
    def analyze(self, audio_paths: list[Path]) -> AcousticConsistencyResult:
        """分析同一批候选音频的声学特征离散度。"""


class BatchAcousticConsistencyAnalyzer:
    """通过基础音频指标检查批次内部的声学环境差异。"""

    def __init__(
        self,
        metrics_analyzer: AudioMetricsAnalyzer,
        policy: AcousticConsistencyPolicy | None = None,
    ) -> None:
        self.metrics_analyzer = metrics_analyzer
        self.policy = policy or AcousticConsistencyPolicy()

    def analyze(self, audio_paths: list[Path]) -> AcousticConsistencyResult:
        metrics: list[AudioMetrics] = []
        for path in audio_paths:
            try:
                metrics.append(self.metrics_analyzer.analyze(path))
            except (OSError, ValueError, RuntimeError):
                continue
        rms_range = _range(item.rms_dbfs for item in metrics)
        silence_range = _range(item.silence_ratio for item in metrics)
        noise_values = [
            item.noise_floor_dbfs
            for item in metrics
            if item.noise_floor_dbfs is not None
        ]
        dynamic_values = [
            item.dynamic_range_db
            for item in metrics
            if item.dynamic_range_db is not None
        ]
        noise_range = _range(noise_values) if noise_values else None
        dynamic_range = _range(dynamic_values) if dynamic_values else None
        reasons = []
        if rms_range > self.policy.max_rms_range_db:
            reasons.append("批次样本响度差异较大")
        if silence_range > self.policy.max_silence_range:
            reasons.append("批次样本静音比例差异较大")
        if noise_range is not None and noise_range > self.policy.max_noise_floor_range_db:
            reasons.append("批次样本噪声底差异较大")
        if dynamic_range is not None and dynamic_range > self.policy.max_dynamic_range_db:
            reasons.append("批次样本动态范围差异较大")
        return AcousticConsistencyResult(
            sample_count=len(audio_paths),
            compared_sample_count=len(metrics),
            rms_range_db=round(rms_range, 6),
            silence_range=round(silence_range, 6),
            noise_floor_range_db=_round_optional(noise_range),
            dynamic_range_range_db=_round_optional(dynamic_range),
            warning=bool(reasons),
            warning_reasons=reasons,
        )


def _range(values) -> float:
    values = [float(value) for value in values if value is not None]
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
