"""贯穿整条流水线的核心数据结构。"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# 批次级状态机：每完成一个主流程阶段就推进一次，方便报告和前端判断进度。
class BatchPhase(str, Enum):
    CREATED = "created"
    IMPORTED = "imported"
    NORMALIZED = "normalized"
    SEGMENTED = "segmented"
    TRANSCRIBED = "transcribed"
    ANALYZED = "analyzed"
    CURATED = "curated"
    EXPORTED = "exported"
    FAILED = "failed"


# 原始资产状态：RawAsset 始终代表输入目录里的原文件
class RawAssetStatus(str, Enum):
    IMPORTED = "imported"
    NORMALIZED = "normalized"
    SKIPPED = "skipped"
    FAILED = "failed"


# 候选样本状态：样本由 VAD 产生，之后会依次经历转写和分析。
class CandidateSampleStatus(str, Enum):
    CREATED = "created"
    TRANSCRIBED = "transcribed"
    ANALYZED = "analyzed"
    FAILED = "failed"


# Curator 的固定决策集合；后续选择、修复和导出都依赖这些值。
class DecisionType(str, Enum):
    KEEP = "keep"
    REJECT = "reject"
    REPAIR = "repair"
    REVIEW = "review"


@dataclass(frozen=True)
class RawAsset:
    # RawAsset 保存输入音频的登记信息；source_path 不被覆盖，work_path 保存标准化工作副本。
    asset_id: str
    batch_id: str
    source_path: Path
    file_name: str
    file_size: int
    extension: str
    work_path: Path | None = None
    status: RawAssetStatus = RawAssetStatus.IMPORTED
    reference_text: str | None = None
    reference_text_source: str | None = None
    language: str | None = None
    speaker_id: str | None = None
    metadata_status: str = "missing"
    metadata_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateSample:
    # CandidateSample 是从工作音频中切出的候选片段；sample_path 会在物化阶段补齐。
    sample_id: str
    batch_id: str
    asset_id: str
    source_path: Path
    start_time: float
    end_time: float
    duration: float
    sample_path: Path | None = None
    transcript: str | None = None
    transcript_source: str | None = None
    asr_confidence: float | None = None
    status: CandidateSampleStatus = CandidateSampleStatus.CREATED
    reference_text: str | None = None
    reference_text_source: str | None = None
    language: str | None = None
    speaker_id: str | None = None
    metadata_status: str = "missing"


@dataclass(frozen=True)
class AnalysisResult:
    # AnalysisResult 是 Inspector 输出：issue_tags 给 Agent 快速判断，evidence 给报告和复核解释原因。
    sample_id: str
    duration: float
    transcript_length: int
    asr_confidence: float | None
    issue_tags: list[str] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CuratorDecision:
    # CuratorDecision 决定样本进入 keep/reject/repair/review 哪条分支。
    sample_id: str
    decision: DecisionType
    reason: str
    confidence: float
    recommended_action: str
    policy_name: str = "baseline"
    # 记录 Curator 是否接受了 Strategy 的建议，便于审计两个 Agent 的协作结果。
    strategy_consistency: str = "not_applicable"


@dataclass(frozen=True)
class ProcessingStrategy:
    sample_id: str
    action: str
    recommended_operation: str
    selected_tool_ids: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    agent_name: str = "baseline_processing_strategy"


@dataclass(frozen=True)
class RepairPlan:
    # RepairPlan 只描述要做什么修复
    sample_id: str
    operation: str
    parameters: dict[str, object] = field(default_factory=dict)
    reason: str = ""
    status: str = "planned"
    chain_id: str | None = None
    step_index: int = 1


@dataclass(frozen=True)
class RepairChain:
    """一个样本的有序修复步骤集合。"""

    chain_id: str
    sample_id: str
    plans: list[RepairPlan] = field(default_factory=list)
    max_steps: int = 3
    status: str = "planned"


@dataclass(frozen=True)
class RepairExecution:
    # RepairExecution 记录工具执行结果
    sample_id: str
    operation: str
    input_path: Path
    output_path: Path
    status: str
    error_message: str | None = None
    chain_id: str | None = None
    step_index: int = 1


@dataclass(frozen=True)
class RepairEvaluation:
    # RepairEvaluation 比较修复前后指标
    sample_id: str
    operation: str
    before_metrics: dict[str, float]
    after_metrics: dict[str, float]
    improved: bool
    recommendation: str
    reason: str = ""
    chain_id: str | None = None
    step_index: int = 1


@dataclass(frozen=True)
class FinalSelection:
    # FinalSelection 是最终导出状态：采用原切片、采用修复版、排除或进入复核。
    sample_id: str
    status: str
    selected_audio_path: Path | None
    reason: str


@dataclass
class Batch:
    # Batch 是一次批处理任务的总状态容器，所有阶段都通过 replace 写入新的阶段结果。
    batch_id: str
    input_dir: Path
    output_dir: Path
    target_policy: str
    # 保存本次运行使用的工具、模型和流程参数，随批次导出用于复现。
    run_config: dict[str, Any] = field(default_factory=dict)
    metadata_summary: dict[str, Any] = field(default_factory=dict)
    phase: BatchPhase = BatchPhase.CREATED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_assets: list[RawAsset] = field(default_factory=list)
    candidate_samples: list[CandidateSample] = field(default_factory=list)
    analysis_results: list[AnalysisResult] = field(default_factory=list)
    processing_strategies: list[ProcessingStrategy] = field(default_factory=list)
    curator_decisions: list[CuratorDecision] = field(default_factory=list)
    repair_plans: list[RepairPlan] = field(default_factory=list)
    repair_chains: list[RepairChain] = field(default_factory=list)
    repair_executions: list[RepairExecution] = field(default_factory=list)
    repair_evaluations: list[RepairEvaluation] = field(default_factory=list)
    final_selections: list[FinalSelection] = field(default_factory=list)
    export_paths: dict[str, str] = field(default_factory=dict)

    @property
    def raw_asset_count(self) -> int:
        return len(self.raw_assets)

    @property
    def candidate_sample_count(self) -> int:
        return len(self.candidate_samples)

    @property
    def analysis_result_count(self) -> int:
        return len(self.analysis_results)

    @property
    def curator_decision_count(self) -> int:
        return len(self.curator_decisions)

    @property
    def processing_strategy_count(self) -> int:
        return len(self.processing_strategies)

    @property
    def repair_plan_count(self) -> int:
        return len(self.repair_plans)

    @property
    def repair_chain_count(self) -> int:
        return len(self.repair_chains)

    @property
    def repair_execution_count(self) -> int:
        return len(self.repair_executions)

    @property
    def repair_evaluation_count(self) -> int:
        return len(self.repair_evaluations)

    @property
    def final_selection_count(self) -> int:
        return len(self.final_selections)
