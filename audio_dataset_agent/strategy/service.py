# 策略服务：批量调用 Strategy Agent 并保存每个样本的处理建议。
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

from audio_dataset_agent.core.models import AnalysisResult, Batch, ProcessingStrategy
from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
from audio_dataset_agent.tools.registry import ToolRegistry


class ProcessingStrategyAgent(Protocol):
    def decide(self, result: AnalysisResult) -> ProcessingStrategy:
        """根据一条分析结果创建一个处理策略。"""


def select_processing_strategies(
    batch: Batch,
    tool_registry: ToolRegistry,
    agent: ProcessingStrategyAgent | None = None,
) -> Batch:
    """将处理策略决策挂接到 batch。"""
    active_agent = agent or BaselineProcessingStrategyAgent(
        tool_registry,
        target_policy=batch.target_policy,
    )
    strategies = [active_agent.decide(result) for result in batch.analysis_results]

    return replace(
        batch,
        updated_at=datetime.now(timezone.utc),
        processing_strategies=strategies,
    )
