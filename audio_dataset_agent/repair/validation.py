"""修复计划校验：阻止越界参数进入音频执行器。"""
from numbers import Real

from audio_dataset_agent.core.models import RepairPlan


def validate_repair_plan(plan: RepairPlan) -> None:
    """验证 RepairPlan 的操作类型、必填参数和安全范围。"""
    if plan.operation == "loudness_normalization":
        _require_number_in_range(
            plan.parameters,
            "target_lufs",
            -30.0,
            -16.0,
            "目标响度",
        )
        return

    if plan.operation == "silence_trimming":
        _require_number_in_range(
            plan.parameters,
            "silence_threshold_dbfs",
            -70.0,
            -30.0,
            "静音阈值",
        )
        _require_number_in_range(
            plan.parameters,
            "min_silence_duration_ms",
            80.0,
            1500.0,
            "最短静音时长",
        )
        return

    if plan.operation == "noise_reduction":
        _require_number_in_range(
            plan.parameters,
            "noise_reduction_db",
            0.0,
            24.0,
            "降噪强度",
        )
        return

    if plan.operation == "neural_noise_reduction":
        post_filter = plan.parameters.get("post_filter", False)
        if not isinstance(post_filter, bool):
            raise ValueError("神经降噪参数 post_filter 必须为布尔值")
        return

    if plan.operation == "manual_repair_review":
        return

    raise ValueError(f"不支持的修复操作：{plan.operation}")


def _require_number_in_range(
    parameters: dict[str, object],
    key: str,
    minimum: float,
    maximum: float,
    label: str,
) -> None:
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}必须是数值")
    if value < minimum or value > maximum:
        raise ValueError(f"{label}必须在 {minimum} 到 {maximum} 之间")
