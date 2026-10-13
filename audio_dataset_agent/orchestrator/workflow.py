import time
from pathlib import Path
from typing import Any

from audio_dataset_agent.analyzers.vad import VADAnalyzer
from audio_dataset_agent.core.models import Batch
from audio_dataset_agent.curation.service import CurationPolicy
from audio_dataset_agent.curation.service import curate_candidate_samples
from audio_dataset_agent.exporters.service import ReportAgent
from audio_dataset_agent.exporters.service import export_curated_batch
from audio_dataset_agent.importer.service import create_import_batch
from audio_dataset_agent.importer.metadata import MetadataImportConfig
from audio_dataset_agent.inspection.audio_metrics import AudioMetricsAnalyzer
from audio_dataset_agent.inspection.consistency import ConsistencyAnalyzer
from audio_dataset_agent.inspection.service import inspect_candidate_samples
from audio_dataset_agent.orchestrator.agent_graph import (
    AgentGraphNodeTrace,
    AgentGraphRecorder,
    attach_agent_graph_trace,
    batch_summary,
)
from audio_dataset_agent.processing.normalizer import AudioNormalizer
from audio_dataset_agent.processing.service import normalize_batch
from audio_dataset_agent.repair.evaluator import RepairEvaluator
from audio_dataset_agent.repair.executor import RepairExecutor
from audio_dataset_agent.repair.service import (
    evaluate_repairs,
    execute_repairs,
    plan_repairs,
)
from audio_dataset_agent.selection.service import select_final_samples
from audio_dataset_agent.segmentation.materialization_service import (
    materialize_candidate_samples,
)
from audio_dataset_agent.segmentation.materializer import SampleMaterializer
from audio_dataset_agent.segmentation.service import generate_candidate_samples
from audio_dataset_agent.strategy.service import ProcessingStrategyAgent
from audio_dataset_agent.strategy.service import select_processing_strategies
from audio_dataset_agent.tools.registry import ToolRegistry, default_tool_registry
from audio_dataset_agent.transcription.service import transcribe_candidate_samples
from audio_dataset_agent.transcription.transcriber import Transcriber


def run_batch_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    export_dir: Path | str,
    vad_analyzer: VADAnalyzer,
    transcriber: Transcriber,
    normalizer: AudioNormalizer | None = None,
    sample_materializer: SampleMaterializer | None = None,
    audio_metrics_analyzer: AudioMetricsAnalyzer | None = None,
    curator_policy: CurationPolicy | None = None,
    processing_strategy_agent: ProcessingStrategyAgent | None = None,
    report_agent: ReportAgent | None = None,
    tool_registry: ToolRegistry | None = None,
    target_policy: str = "voice_clone",
    run_asr: bool = True,
    run_config: dict[str, Any] | None = None,
    metadata_config: MetadataImportConfig | None = None,
    acoustic_consistency_analyzer: ConsistencyAnalyzer | None = None,
    repair_chain_mode: str = "single",
) -> Batch:
    """运行从导入到导出的确定性流程。"""
    # graph 记录每一步的输入输出摘要，不参与业务判断，但会用于前端展示 Agent Graph。
    graph = AgentGraphRecorder()
    batch = _create_import_batch_with_trace(
        graph,
        input_dir,
        output_dir,
        target_policy,
        run_config,
        metadata_config,
    )
    # 1. 标准化：生成分析用工作副本，不覆盖原始音频。
    normalized = graph.run_node(
        "normalize",
        "Normalizer",
        batch,
        lambda current: normalize_batch(current, normalizer=normalizer),
        uses_tool=normalizer is not None,
        notes="创建用于分析的工作音频，不覆盖原始文件。",
    )
    # 2. VAD：把工作音频切成语音时间段，得到候选样本。
    segmented = graph.run_node(
        "vad",
        "VAD Analyzer",
        normalized,
        lambda current: generate_candidate_samples(current, vad_analyzer),
        uses_tool=True,
        notes="检测语音区域，并生成候选样本。",
    )
    # 3. 物化：把候选时间段裁剪成真实 wav，方便 ASR、指标分析和播放。
    materialized = graph.run_node(
        "materialize",
        "Sample Materializer",
        segmented,
        lambda current: materialize_candidate_samples(current, sample_materializer),
        uses_tool=sample_materializer is not None,
        notes="生成供下游工具使用的实际样本文件。",
    )
    # 4. ASR：按 profile/参数决定是否生成候选文本。voice_clone 可以关闭这一步。
    transcribed = graph.run_node(
        "asr",
        "ASR Transcriber",
        materialized,
        lambda current: (
            transcribe_candidate_samples(current, transcriber)
            if run_asr
            else current
        ),
        uses_tool=run_asr,
        notes="当前用途需要文本时，为样本补充候选转写。",
    )
    # 5. Inspector：把音频指标和文本信息整理成 issue_tags/evidence。
    analyzed = graph.run_node(
        "inspect",
        "Inspector",
        transcribed,
        lambda current: inspect_candidate_samples(
            current,
            audio_metrics_analyzer=audio_metrics_analyzer,
            acoustic_consistency_analyzer=acoustic_consistency_analyzer,
        ),
        uses_tool=audio_metrics_analyzer is not None,
        notes="将文本和音频证据转换为问题标签。",
    )
    # 6. Strategy Agent：选择处理意图和候选工具，但不执行工具。
    strategized = graph.run_node(
        "strategy",
        "Strategy Agent",
        analyzed,
        lambda current: select_processing_strategies(
            current,
            tool_registry=tool_registry or default_tool_registry(),
            agent=processing_strategy_agent,
        ),
        uses_agent=True,
        notes="根据 Inspector 证据选择处理方向和候选工具。",
    )
    # 7. Curator Agent：做样本级 keep/reject/repair/review 决策。
    curated = graph.run_node(
        "curate",
        "Curator Agent",
        strategized,
        lambda current: curate_candidate_samples(current, policy=curator_policy),
        uses_agent=True,
        notes="做出 keep/reject/repair/review 样本决策。",
    )
    # 8. Repair Planner：把 repair 决策变成可执行计划。
    planned = graph.run_node(
        "repair_plan",
        "Repair Planner",
        curated,
        lambda current: plan_repairs(current, chain_mode=repair_chain_mode),
        notes="将修复决策转换为可执行的修复计划。",
    )
    # 9. Final Selector：结合决策和修复结果确定最终导出状态。
    selected = graph.run_node(
        "final_select",
        "Final Selector",
        planned,
        select_final_samples,
        notes="选择原始音频、修复音频、排除或复核等最终状态。",
    )
    # 10. Archivist：写 manifest、state、报告和复核队列。
    exported = graph.run_node(
        "export",
        "Archivist",
        selected,
        lambda current: export_curated_batch(
            current,
            export_dir,
            report_agent=report_agent,
        ),
        notes="导出 manifest、状态文件、质量报告和复核队列。",
    )
    return attach_agent_graph_trace(exported, graph)


def run_repair_enabled_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    export_dir: Path | str,
    vad_analyzer: VADAnalyzer,
    transcriber: Transcriber,
    audio_metrics_analyzer: AudioMetricsAnalyzer,
    normalizer: AudioNormalizer | None = None,
    sample_materializer: SampleMaterializer | None = None,
    curator_policy: CurationPolicy | None = None,
    processing_strategy_agent: ProcessingStrategyAgent | None = None,
    report_agent: ReportAgent | None = None,
    tool_registry: ToolRegistry | None = None,
    repair_executor: RepairExecutor | None = None,
    repair_evaluator: RepairEvaluator | None = None,
    target_policy: str = "voice_clone",
    run_asr: bool = True,
    run_config: dict[str, Any] | None = None,
    metadata_config: MetadataImportConfig | None = None,
    acoustic_consistency_analyzer: ConsistencyAnalyzer | None = None,
    repair_chain_mode: str = "single",
) -> Batch:
    """运行包含修复执行和评估的完整流程。"""
    graph = AgentGraphRecorder()
    batch = _create_import_batch_with_trace(
        graph,
        input_dir,
        output_dir,
        target_policy,
        run_config,
        metadata_config,
    )
    normalized = graph.run_node(
        "normalize",
        "Normalizer",
        batch,
        lambda current: normalize_batch(current, normalizer=normalizer),
        uses_tool=normalizer is not None,
    )
    segmented = graph.run_node(
        "vad",
        "VAD Analyzer",
        normalized,
        lambda current: generate_candidate_samples(current, vad_analyzer),
        uses_tool=True,
    )
    materialized = graph.run_node(
        "materialize",
        "Sample Materializer",
        segmented,
        lambda current: materialize_candidate_samples(current, sample_materializer),
        uses_tool=sample_materializer is not None,
    )
    transcribed = graph.run_node(
        "asr",
        "ASR Transcriber",
        materialized,
        lambda current: (
            transcribe_candidate_samples(current, transcriber)
            if run_asr
            else current
        ),
        uses_tool=run_asr,
    )
    analyzed = graph.run_node(
        "inspect",
        "Inspector",
        transcribed,
        lambda current: inspect_candidate_samples(
            current,
            audio_metrics_analyzer=audio_metrics_analyzer,
            acoustic_consistency_analyzer=acoustic_consistency_analyzer,
        ),
        uses_tool=True,
    )
    strategized = graph.run_node(
        "strategy",
        "Strategy Agent",
        analyzed,
        lambda current: select_processing_strategies(
            current,
            tool_registry=tool_registry or default_tool_registry(),
            agent=processing_strategy_agent,
        ),
        uses_agent=True,
    )
    curated = graph.run_node(
        "curate",
        "Curator Agent",
        strategized,
        lambda current: curate_candidate_samples(current, policy=curator_policy),
        uses_agent=True,
    )
    planned = graph.run_node(
        "repair_plan",
        "Repair Planner",
        curated,
        lambda current: plan_repairs(current, chain_mode=repair_chain_mode),
    )
    executed = graph.run_node(
        "repair_execute",
        "Repair Executor",
        planned,
        lambda current: execute_repairs(current, executor=repair_executor),
        uses_tool=True,
    )
    evaluated = graph.run_node(
        "repair_evaluate",
        "Repair Evaluator",
        executed,
        lambda current: evaluate_repairs(
            current,
            audio_metrics_analyzer=audio_metrics_analyzer,
            evaluator=repair_evaluator,
        ),
        uses_tool=True,
    )
    selected = graph.run_node(
        "final_select",
        "Final Selector",
        evaluated,
        select_final_samples,
    )
    exported = graph.run_node(
        "export",
        "Archivist",
        selected,
        lambda current: export_curated_batch(
            current,
            export_dir,
            report_agent=report_agent,
        ),
    )
    return attach_agent_graph_trace(exported, graph)


def _create_import_batch_with_trace(
    graph: AgentGraphRecorder,
    input_dir: Path | str,
    output_dir: Path | str,
    target_policy: str,
    run_config: dict[str, Any] | None = None,
    metadata_config: MetadataImportConfig | None = None,
) -> Batch:
    started = time.perf_counter()
    batch = create_import_batch(
        input_dir,
        output_dir,
        target_policy=target_policy,
        run_config=run_config,
        metadata_config=metadata_config,
    )
    duration_ms = (time.perf_counter() - started) * 1000
    graph.nodes.append(
        AgentGraphNodeTrace(
            node_name="import",
            agent_name="Data Intake",
            input_summary={
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "target_policy": target_policy,
            },
            output_summary=batch_summary(batch),
            duration_ms=round(duration_ms, 3),
            notes="扫描输入目录并登记原始音频资产。",
        )
    )
    return batch
