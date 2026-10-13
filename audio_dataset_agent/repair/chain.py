"""受控的多步音频修复链。"""
from dataclasses import replace
from uuid import uuid4

from audio_dataset_agent.core.models import CuratorDecision, RepairChain, RepairPlan
from audio_dataset_agent.profiles.policy import DatasetProfile, get_dataset_profile
from audio_dataset_agent.repair.planner import BaselineRepairPlanner
from audio_dataset_agent.repair.validation import validate_repair_plan


MAX_REPAIR_CHAIN_STEPS = 3
_OPERATION_ORDER = {
    "silence_trimming": 1,
    "noise_reduction": 2,
    "neural_noise_reduction": 2,
    "loudness_normalization": 3,
    "manual_repair_review": 4,
}


def build_repair_chain(
    decision: CuratorDecision,
    target_policy: str | DatasetProfile,
    mode: str = "single",
    max_steps: int = MAX_REPAIR_CHAIN_STEPS,
) -> RepairChain:
    """将一个 repair 决策转换成单步或自动多步修复链。"""
    if mode not in {"single", "auto"}:
        raise ValueError(f"不支持的修复链模式：{mode}")
    if max_steps < 1 or max_steps > MAX_REPAIR_CHAIN_STEPS:
        raise ValueError(f"修复链最多允许 {MAX_REPAIR_CHAIN_STEPS} 步")

    base_plan = BaselineRepairPlanner(target_policy).plan(decision)
    if base_plan is None:
        raise ValueError("只有 repair 决策可以生成修复链")

    plans = [base_plan]
    if mode == "auto":
        if base_plan.operation in {"noise_reduction", "neural_noise_reduction", "silence_trimming"}:
            profile = get_dataset_profile(target_policy)
            plans.append(
                RepairPlan(
                    sample_id=decision.sample_id,
                    operation="loudness_normalization",
                    parameters={"target_lufs": _target_lufs(profile.name)},
                    reason="修复链后续步骤：统一修复结果的目标响度",
                )
            )
    chain_id = f"chain_{uuid4().hex[:12]}"
    numbered_plans = [
        replace(plan, chain_id=chain_id, step_index=index)
        for index, plan in enumerate(plans[:max_steps], start=1)
    ]
    chain = RepairChain(
        chain_id=chain_id,
        sample_id=decision.sample_id,
        plans=numbered_plans,
        max_steps=max_steps,
    )
    validate_repair_chain(chain)
    return chain


def validate_repair_chain(chain: RepairChain) -> None:
    """校验步数、样本归属、操作顺序及每一步参数。"""
    if not chain.plans:
        raise ValueError("修复链不能为空")
    if len(chain.plans) > chain.max_steps or len(chain.plans) > MAX_REPAIR_CHAIN_STEPS:
        raise ValueError(f"修复链最多允许 {min(chain.max_steps, MAX_REPAIR_CHAIN_STEPS)} 步")
    previous_order = 0
    for expected_index, plan in enumerate(chain.plans, start=1):
        if plan.sample_id != chain.sample_id:
            raise ValueError("修复链中的样本 ID 不一致")
        if plan.chain_id != chain.chain_id:
            raise ValueError("修复链中的 chain_id 不一致")
        if plan.step_index != expected_index:
            raise ValueError("修复链步骤编号必须连续")
        order = _OPERATION_ORDER.get(plan.operation)
        if order is None:
            raise ValueError(f"修复链包含不支持的操作：{plan.operation}")
        if order < previous_order:
            raise ValueError("修复链操作顺序不合法")
        previous_order = order
        validate_repair_plan(plan)


def _target_lufs(profile_name: str) -> float:
    return {"tts": -23.0, "asr": -20.0, "voice_clone": -23.0}[profile_name]
