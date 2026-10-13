# 修复评估器：比较修复前后指标，判断是否采纳修复结果。
from typing import Protocol

from audio_dataset_agent.core.models import RepairEvaluation, RepairExecution
from audio_dataset_agent.inspection.audio_metrics import AudioMetricsAnalyzer


class RepairEvaluator(Protocol):
    # RepairEvaluator 判断修复输出是否比输入更适合作为训练样本。
    def evaluate(
        self,
        execution: RepairExecution,
        analyzer: AudioMetricsAnalyzer,
    ) -> RepairEvaluation:
        """评估一次修复执行结果。"""


class BaselineRepairEvaluator:
    """用于修复后音频质量的规则评估器。"""

    def evaluate(
        self,
        execution: RepairExecution,
        analyzer: AudioMetricsAnalyzer,
    ) -> RepairEvaluation:
        before = analyzer.analyze(execution.input_path).to_dict()
        after = analyzer.analyze(execution.output_path).to_dict()

        improved, reason = self._compare(execution.operation, before, after)
        return RepairEvaluation(
            sample_id=execution.sample_id,
            operation=execution.operation,
            before_metrics=before,
            after_metrics=after,
            improved=improved,
            recommendation="adopt_repaired" if improved else "keep_original",
            reason=reason,
        )

    def _compare(
        self,
        operation: str,
        before: dict[str, float],
        after: dict[str, float],
    ) -> tuple[bool, str]:
        if operation == "loudness_normalization":
            rms_improved = after["rms_dbfs"] > before["rms_dbfs"]
            clipping_safe = (
                after["clipping_ratio"] <= before["clipping_ratio"] + 0.001
            )
            return (
                rms_improved and clipping_safe,
                "均方根响度提升，且削波没有明显增加。",
            )

        if operation == "silence_trimming":
            silence_improved = after["silence_ratio"] < before["silence_ratio"]
            return (
                silence_improved,
                "裁剪后静音比例下降。",
            )

        if operation in {"noise_reduction", "neural_noise_reduction"}:
            before_floor = before.get("noise_floor_dbfs")
            after_floor = after.get("noise_floor_dbfs")
            if before_floor is None or after_floor is None:
                return False, "降噪后缺少噪声底指标，无法判断效果。"
            over_suppressed, suppression_reason = self._over_suppressed(
                before,
                after,
            )
            if over_suppressed:
                return False, suppression_reason
            noise_floor_improved = after_floor < before_floor - 3.0
            clipping_safe = (
                after["clipping_ratio"] <= before["clipping_ratio"] + 0.001
            )
            if not clipping_safe:
                return False, "降噪后削波比例上升。"
            if not noise_floor_improved:
                return False, "噪声底改善未达到采用阈值。"
            return True, "噪声底下降，且削波没有明显增加。"

        return False, "不支持自动评估该修复操作，需要人工复核。"

    def _over_suppressed(
        self,
        before: dict[str, float],
        after: dict[str, float],
    ) -> tuple[bool, str]:
        before_silence = before.get("silence_ratio", 0.0)
        after_silence = after.get("silence_ratio", 0.0)
        after_rms = after.get("rms_dbfs", 0.0)

        if after_silence >= 0.95:
            return True, "降噪过度，输出音频接近静音。"
        if after_silence > before_silence + 0.5:
            return True, "降噪过度，静音比例显著上升。"
        if after_rms < -70.0:
            return True, "降噪过度，音频响度过低。"
        return False, ""
