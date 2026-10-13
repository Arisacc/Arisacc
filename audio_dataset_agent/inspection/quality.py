"""把 Inspector 的基础音频指标转换成可解释的质量评估结果。"""
import math
from dataclasses import dataclass

from audio_dataset_agent.inspection.audio_metrics import AudioMetrics


@dataclass(frozen=True)
class AudioQualityAssessment:
    """单个样本的估算 SNR 和质量分，不代表模型准确率。"""

    snr_db: float | None
    quality_score: float | None
    version: str = "rule-v1"


def estimate_snr_db(metrics: AudioMetrics) -> float | None:
    """使用 RMS 与噪声底之差估算 SNR；缺少任一有限值时返回空值。"""
    if not math.isfinite(metrics.rms_dbfs):
        return None
    if metrics.noise_floor_dbfs is None or not math.isfinite(metrics.noise_floor_dbfs):
        return None
    return round(metrics.rms_dbfs - metrics.noise_floor_dbfs, 3)


def assess_audio_quality(
    duration: float,
    metrics: AudioMetrics,
) -> AudioQualityAssessment:
    """根据音频证据计算 0 到 1 的规则质量分。"""
    snr_db = estimate_snr_db(metrics)
    components: list[tuple[float, float]] = []

    if snr_db is not None:
        # 5dB 以下视为低质量，25dB 及以上达到该维度满分。
        components.append((0.40, _scale(snr_db, 5.0, 25.0)))

    if math.isfinite(metrics.silence_ratio):
        components.append((0.25, 1.0 - _clamp(metrics.silence_ratio / 0.5)))

    if math.isfinite(metrics.clipping_ratio):
        components.append((0.15, 1.0 - _clamp(metrics.clipping_ratio / 0.01)))

    if math.isfinite(metrics.rms_dbfs):
        # -24dBFS 是参考中心，偏离 24dB 以上时该维度降为 0。
        components.append((0.15, 1.0 - _clamp(abs(metrics.rms_dbfs + 24.0) / 24.0)))

    if math.isfinite(duration) and duration >= 0:
        components.append((0.05, _clamp(duration / 0.6)))

    if not components:
        return AudioQualityAssessment(snr_db=snr_db, quality_score=None)

    weight_total = sum(weight for weight, _ in components)
    score = sum(weight * value for weight, value in components) / weight_total
    return AudioQualityAssessment(
        snr_db=snr_db,
        quality_score=round(_clamp(score), 4),
    )


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
