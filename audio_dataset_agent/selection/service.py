"""最终选择服务：把 Curator 和 Repair 结果汇总成导出状态。

keep 采用原始切片，reject 不导出，review 进入人工复核；repair 只有在执行成功且评估建议采用修复版时，才选择 repaired output。
"""
from dataclasses import replace
from datetime import datetime, timezone

from audio_dataset_agent.core.models import (
    Batch,
    CandidateSample,
    CuratorDecision,
    DecisionType,
    FinalSelection,
    RepairEvaluation,
    RepairExecution,
)


def select_final_samples(batch: Batch) -> Batch:
    """为每个候选样本选择最终训练音频路径。

    keep/reject/review 是直接分支；repair 分支还要继续看执行是否成功，以及评估器是否建议采用修复版。
    """
    # 这里先把各阶段结果按 sample_id 建索引，避免用列表顺序隐式关联样本。
    decisions = {decision.sample_id: decision for decision in batch.curator_decisions}
    executions_by_sample = _group_by_sample(batch.repair_executions)
    evaluations_by_sample = _group_by_sample(batch.repair_evaluations)

    selections = [
        _select_sample(
            sample,
            decisions.get(sample.sample_id),
            executions_by_sample.get(sample.sample_id, []),
            evaluations_by_sample.get(sample.sample_id, []),
        )
        for sample in batch.candidate_samples
    ]

    return replace(
        batch,
        updated_at=datetime.now(timezone.utc),
        final_selections=selections,
    )


def _select_sample(
    sample: CandidateSample,
    decision: CuratorDecision | None,
    executions: list[RepairExecution],
    evaluations: list[RepairEvaluation],
) -> FinalSelection:
    original_path = sample.sample_path or sample.source_path

    if decision is None:
        return FinalSelection(
            sample_id=sample.sample_id,
            status="needs_review",
            selected_audio_path=None,
            reason="missing curator decision",
        )

    # keep/reject/review 是直接分支；repair 分支还要继续看执行和评估结果。
    if decision.decision == DecisionType.KEEP:
        return FinalSelection(
            sample_id=sample.sample_id,
            status="selected_original",
            selected_audio_path=original_path,
            reason="selected original sample from keep decision",
        )

    if decision.decision == DecisionType.REJECT:
        return FinalSelection(
            sample_id=sample.sample_id,
            status="excluded",
            selected_audio_path=None,
            reason=decision.reason,
        )

    if decision.decision == DecisionType.REVIEW:
        return FinalSelection(
            sample_id=sample.sample_id,
            status="needs_review",
            selected_audio_path=None,
            reason=decision.reason,
        )

    if (
        decision.decision == DecisionType.REPAIR
        and executions
        and len(evaluations) == len(executions)
        and all(execution.status == "succeeded" for execution in executions)
        and all(evaluation.improved for evaluation in evaluations)
        and all(
            evaluation.recommendation == "adopt_repaired"
            for evaluation in evaluations
        )
    ):
        execution = sorted(executions, key=lambda item: item.step_index)[-1]
        evaluation = sorted(evaluations, key=lambda item: item.step_index)[-1]
        return FinalSelection(
            sample_id=sample.sample_id,
            status="selected_repaired",
            selected_audio_path=execution.output_path,
            reason=evaluation.reason,
        )

    if decision.decision == DecisionType.REPAIR:
        return FinalSelection(
            sample_id=sample.sample_id,
            status="needs_review",
            selected_audio_path=original_path,
            reason="repair did not produce an adopted output",
        )

    return FinalSelection(
        sample_id=sample.sample_id,
        status="needs_review",
        selected_audio_path=None,
        reason="unsupported final selection state",
    )


def _group_by_sample(items):
    grouped = {}
    for item in items:
        grouped.setdefault(item.sample_id, []).append(item)
    for values in grouped.values():
        values.sort(key=lambda item: item.step_index)
    return grouped
