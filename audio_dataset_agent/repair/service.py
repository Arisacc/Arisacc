"""修复服务：串联 repair plan、execute、evaluate 三个阶段。"""
from dataclasses import replace
from datetime import datetime, timezone

from audio_dataset_agent.core.models import Batch, CandidateSample, RepairChain
from audio_dataset_agent.core.run_context import RunContext
from audio_dataset_agent.inspection.audio_metrics import AudioMetricsAnalyzer
from audio_dataset_agent.repair.evaluator import (
    BaselineRepairEvaluator,
    RepairEvaluator,
)
from audio_dataset_agent.repair.executor import CompositeRepairExecutor, RepairExecutor
from audio_dataset_agent.repair.planner import BaselineRepairPlanner, RepairPlanner
from audio_dataset_agent.repair.chain import build_repair_chain, validate_repair_chain


def plan_repairs(
    batch: Batch,
    planner: RepairPlanner | None = None,
    chain_mode: str | None = None,
) -> Batch:
    """根据 Curator 的 repair 决策创建修复计划。

    这里只规划，不执行。这样可以在 state/repair_plans.json 中清楚看到 Agent 为什么认为某些样本需要 loudness、silence trim 或 neural denoise。
    """
    active_planner = planner or BaselineRepairPlanner(
        target_policy=batch.target_policy,
    )
    repair_plans = []
    repair_chains: list[RepairChain] = []
    active_chain_mode = chain_mode or str(batch.run_config.get("repair_chain", "single"))

    # 只有 Curator 判为 repair 的样本才会被 planner 转成 RepairPlan。
    for decision in batch.curator_decisions:
        if decision.decision.value != "repair":
            continue
        if isinstance(active_planner, BaselineRepairPlanner):
            chain = build_repair_chain(
                decision,
                target_policy=batch.target_policy,
                mode=active_chain_mode,
            )
            repair_chains.append(chain)
            repair_plans.extend(chain.plans)
            continue
        plan = active_planner.plan(decision)
        if plan is not None:
            repair_plans.append(plan)
            repair_chains.append(
                RepairChain(
                    chain_id=plan.chain_id or f"chain_{plan.sample_id}",
                    sample_id=plan.sample_id,
                    plans=[replace(plan, chain_id=plan.chain_id or f"chain_{plan.sample_id}")],
                )
            )

    return replace(
        batch,
        updated_at=datetime.now(timezone.utc),
        repair_plans=repair_plans,
        repair_chains=repair_chains,
    )


def execute_repairs(
    batch: Batch,
    executor: RepairExecutor | None = None,
) -> Batch:
    """执行已规划的修复，并记录输出文件。"""
    # CompositeRepairExecutor 会根据 operation 在 FFmpeg 与 DeepFilterNet 之间路由。
    active_executor = executor or CompositeRepairExecutor()
    samples_by_id = {sample.sample_id: sample for sample in batch.candidate_samples}
    repairs_dir = RunContext.from_batch(batch).repairs_dir
    executions = []

    chains = batch.repair_chains or [
        RepairChain(
            chain_id=plan.chain_id or f"chain_{plan.sample_id}",
            sample_id=plan.sample_id,
            plans=[replace(plan, chain_id=plan.chain_id or f"chain_{plan.sample_id}")],
        )
        for plan in batch.repair_plans
    ]
    for chain in chains:
        validate_repair_chain(chain)
        sample = samples_by_id.get(chain.sample_id)
        if sample is None:
            continue
        current_sample = sample
        for plan in chain.plans:
            execution = active_executor.execute(plan, current_sample, repairs_dir)
            if execution is None:
                continue
            executions.append(execution)
            if execution.status != "succeeded":
                break
            current_sample = replace(current_sample, sample_path=execution.output_path)

    return replace(
        batch,
        updated_at=datetime.now(timezone.utc),
        repair_executions=executions,
    )


def evaluate_repairs(
    batch: Batch,
    audio_metrics_analyzer: AudioMetricsAnalyzer,
    evaluator: RepairEvaluator | None = None,
) -> Batch:
    """使用修复前后音频指标评估成功执行的修复。"""
    active_evaluator = evaluator or BaselineRepairEvaluator()
    evaluations = []

    # 评估阶段只比较成功生成的修复文件；失败的执行会保留错误信息，交给报告展示。
    for execution in batch.repair_executions:
        if execution.status != "succeeded":
            continue
        evaluation = active_evaluator.evaluate(execution, audio_metrics_analyzer)
        evaluations.append(
            replace(
                evaluation,
                chain_id=execution.chain_id,
                step_index=execution.step_index,
            )
        )

    return replace(
        batch,
        updated_at=datetime.now(timezone.utc),
        repair_evaluations=evaluations,
    )
