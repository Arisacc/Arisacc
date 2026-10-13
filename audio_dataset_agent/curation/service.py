# Curator 服务：批量执行样本级 keep/reject/repair/review 决策。
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

from audio_dataset_agent.core.models import (
    AnalysisResult,
    Batch,
    BatchPhase,
    CuratorDecision,
    ProcessingStrategy,
)
from audio_dataset_agent.curation.policy import BaselineCurationPolicy


class CurationPolicy(Protocol):
    """规则策略和 LLM Curator Agent 共用的决策接口。"""

    def decide(
        self,
        result: AnalysisResult,
        strategy: ProcessingStrategy | None = None,
    ) -> CuratorDecision:
        """根据分析结果和 Strategy 建议生成一个 Curator 决策。"""


def curate_candidate_samples(
    batch: Batch,
    policy: CurationPolicy | None = None,
) -> Batch:
    """根据分析结果批量生成 Curator 决策。"""
    active_policy = policy or BaselineCurationPolicy(
        target_policy=batch.target_policy,
    )
    strategy_by_sample = {
        strategy.sample_id: strategy
        for strategy in batch.processing_strategies
    }
    decisions = [
        _decide_with_optional_strategy(
            active_policy,
            result,
            strategy_by_sample.get(result.sample_id),
        )
        for result in batch.analysis_results
    ]

    return replace(
        batch,
        phase=BatchPhase.CURATED,
        updated_at=datetime.now(timezone.utc),
        curator_decisions=decisions,
    )


def _decide_with_optional_strategy(
    policy: CurationPolicy,
    result: AnalysisResult,
    strategy: ProcessingStrategy | None,
) -> CuratorDecision:
    """兼容旧版自定义策略，同时把新 Strategy 建议传给内置 Curator。"""
    if strategy is None:
        return policy.decide(result)

    try:
        decision = policy.decide(result, strategy)
    except TypeError as exc:
        # 旧的扩展策略只接受一个参数；只在签名不兼容时回退，避免吞掉策略内部错误。
        if "positional argument" not in str(exc) and "unexpected keyword" not in str(exc):
            raise
        decision = policy.decide(result)

    if decision.strategy_consistency == "not_applicable":
        return _annotate_strategy_consistency(decision, strategy)
    return decision


def _annotate_strategy_consistency(
    decision: CuratorDecision,
    strategy: ProcessingStrategy,
) -> CuratorDecision:
    from dataclasses import replace

    consistency = (
        "accepted"
        if decision.recommended_action == strategy.recommended_operation
        else "overridden"
    )
    return replace(decision, strategy_consistency=consistency)
