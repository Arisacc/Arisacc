# 修复规划器：把 Strategy 建议转换成可执行修复计划。
from typing import Protocol

from audio_dataset_agent.core.models import CuratorDecision, DecisionType, RepairPlan
from audio_dataset_agent.profiles.policy import DatasetProfile, get_dataset_profile
from audio_dataset_agent.repair.presets import RepairPreset, get_repair_preset
from audio_dataset_agent.repair.validation import validate_repair_plan


class RepairPlanner(Protocol):
    # RepairPlanner 把 Curator 的 recommended_action 转成后续 Executor 可执行的计划。
    def plan(self, decision: CuratorDecision) -> RepairPlan | None:
        """为修复决策返回修复计划；无须修复时返回 None。"""


class BaselineRepairPlanner:
    """根据用途预设生成受约束 RepairPlan 的确定性规划器。"""

    def __init__(self, target_policy: str | DatasetProfile = "tts") -> None:
        self.profile = get_dataset_profile(target_policy)
        self.preset: RepairPreset = get_repair_preset(self.profile)

    def plan(self, decision: CuratorDecision) -> RepairPlan | None:
        if decision.decision != DecisionType.REPAIR:
            return None

        if decision.recommended_action == "normalize_loudness":
            return self._validated_plan(
                sample_id=decision.sample_id,
                operation="loudness_normalization",
                parameters={"target_lufs": self.preset.target_lufs},
                reason=decision.reason,
            )

        if decision.recommended_action == "trim_silence":
            return self._validated_plan(
                sample_id=decision.sample_id,
                operation="silence_trimming",
                parameters={
                    "silence_threshold_dbfs": self.preset.silence_threshold_dbfs,
                    "min_silence_duration_ms": self.preset.min_silence_duration_ms,
                },
                reason=decision.reason,
            )

        if decision.recommended_action == "denoise_light":
            return self._validated_plan(
                sample_id=decision.sample_id,
                operation="noise_reduction",
                parameters={"noise_reduction_db": self.preset.noise_reduction_db},
                reason=decision.reason,
            )

        if decision.recommended_action == "neural_denoise":
            return self._validated_plan(
                sample_id=decision.sample_id,
                operation="neural_noise_reduction",
                parameters={"post_filter": True},
                reason=decision.reason,
            )

        return self._validated_plan(
            sample_id=decision.sample_id,
            operation="manual_repair_review",
            parameters={"recommended_action": decision.recommended_action},
            reason=decision.reason,
        )

    @staticmethod
    def _validated_plan(
        sample_id: str,
        operation: str,
        parameters: dict[str, object],
        reason: str,
    ) -> RepairPlan:
        plan = RepairPlan(
            sample_id=sample_id,
            operation=operation,
            parameters=parameters,
            reason=reason,
        )
        validate_repair_plan(plan)
        return plan
