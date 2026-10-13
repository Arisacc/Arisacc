# Supervisor 抽象：统一表达工作流节点的输入输出与执行顺序。
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from audio_dataset_agent.analyzers.vad import VADAnalyzer
from audio_dataset_agent.core.models import Batch
from audio_dataset_agent.core.serialization import batch_to_dict
from audio_dataset_agent.curation.service import CurationPolicy
from audio_dataset_agent.curation.service import curate_candidate_samples
from audio_dataset_agent.exporters.service import export_curated_batch
from audio_dataset_agent.exporters.service import ReportAgent
from audio_dataset_agent.importer.service import create_import_batch
from audio_dataset_agent.importer.metadata import MetadataImportConfig
from audio_dataset_agent.inspection.audio_metrics import AudioMetricsAnalyzer
from audio_dataset_agent.inspection.consistency import ConsistencyAnalyzer
from audio_dataset_agent.inspection.service import inspect_candidate_samples
from audio_dataset_agent.processing.normalizer import AudioNormalizer
from audio_dataset_agent.processing.service import normalize_batch
from audio_dataset_agent.repair.evaluator import RepairEvaluator
from audio_dataset_agent.repair.executor import RepairExecutor
from audio_dataset_agent.repair.service import evaluate_repairs, execute_repairs, plan_repairs
from audio_dataset_agent.selection.service import select_final_samples
from audio_dataset_agent.segmentation.materialization_service import (
    materialize_candidate_samples,
)
from audio_dataset_agent.segmentation.materializer import SampleMaterializer
from audio_dataset_agent.segmentation.service import generate_candidate_samples
from audio_dataset_agent.strategy.service import select_processing_strategies
from audio_dataset_agent.strategy.service import ProcessingStrategyAgent
from audio_dataset_agent.tools.registry import ToolRegistry, default_tool_registry
from audio_dataset_agent.transcription.service import transcribe_candidate_samples
from audio_dataset_agent.transcription.transcriber import Transcriber


@dataclass(frozen=True)
class SupervisorTraceStep:
    """Supervisor 记录的一次可审计交接。"""

    step_index: int
    agent_name: str
    phase: str
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class SupervisorRunResult:
    """最终 batch 以及生成它的编排轨迹。"""

    batch: Batch
    trace: list[SupervisorTraceStep]


class AudioDatasetSupervisor:
    """协调音频数据集各工作节点，并保留交接轨迹。"""

    def __init__(
        self,
        vad_analyzer: VADAnalyzer,
        transcriber: Transcriber,
        normalizer: AudioNormalizer | None = None,
        sample_materializer: SampleMaterializer | None = None,
        audio_metrics_analyzer: AudioMetricsAnalyzer | None = None,
        curator_policy: CurationPolicy | None = None,
        repair_executor: RepairExecutor | None = None,
        repair_evaluator: RepairEvaluator | None = None,
        processing_strategy_agent: ProcessingStrategyAgent | None = None,
        report_agent: ReportAgent | None = None,
        tool_registry: ToolRegistry | None = None,
        target_policy: str = "voice_clone",
        run_asr: bool = True,
        run_config: dict[str, Any] | None = None,
        metadata_config: MetadataImportConfig | None = None,
        acoustic_consistency_analyzer: ConsistencyAnalyzer | None = None,
        repair_chain_mode: str = "single",
    ) -> None:
        self.vad_analyzer = vad_analyzer
        self.transcriber = transcriber
        self.normalizer = normalizer
        self.sample_materializer = sample_materializer
        self.audio_metrics_analyzer = audio_metrics_analyzer
        self.curator_policy = curator_policy
        self.repair_executor = repair_executor
        self.repair_evaluator = repair_evaluator
        self.processing_strategy_agent = processing_strategy_agent
        self.report_agent = report_agent
        self.tool_registry = tool_registry or default_tool_registry()
        self.target_policy = target_policy
        self.run_asr = run_asr
        self.run_config = dict(run_config or {})
        self.metadata_config = metadata_config
        self.acoustic_consistency_analyzer = acoustic_consistency_analyzer
        self.repair_chain_mode = repair_chain_mode

    def run_basic(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        export_dir: Path | str,
    ) -> SupervisorRunResult:
        """运行稳定的导入到导出流程，并收集轨迹。"""
        trace: list[SupervisorTraceStep] = []

        batch = self._run_import(input_dir, output_dir, trace)
        batch = self._run_batch_step(
            trace,
            "Normalizer",
            "normalize",
            batch,
            lambda current: normalize_batch(current, normalizer=self.normalizer),
            "创建用于分析的工作副本，不覆盖原始音频。",
        )
        batch = self._run_batch_step(
            trace,
            "VAD Segmenter",
            "segment",
            batch,
            lambda current: generate_candidate_samples(current, self.vad_analyzer),
            "检测可作为训练候选的语音区域。",
        )
        batch = self._run_batch_step(
            trace,
            "Sample Materializer",
            "materialize",
            batch,
            lambda current: materialize_candidate_samples(
                current,
                self.sample_materializer,
            ),
            "生成供下游工具使用的实际样本文件。",
        )
        batch = self._run_transcription_step(trace, batch)
        batch = self._run_batch_step(
            trace,
            "Inspector",
            "inspect",
            batch,
            lambda current: inspect_candidate_samples(
                current,
                audio_metrics_analyzer=self.audio_metrics_analyzer,
                acoustic_consistency_analyzer=self.acoustic_consistency_analyzer,
            ),
            "将原始字段和音频指标转换为问题标签与证据。",
        )
        batch = self._run_batch_step(
            trace,
            "Strategy Agent",
            "strategy",
            batch,
            lambda current: select_processing_strategies(
                current,
                tool_registry=self.tool_registry,
                agent=self.processing_strategy_agent,
            ),
            "根据 Inspector 证据选择处理方向和候选工具。",
        )
        batch = self._run_batch_step(
            trace,
            "Curator",
            "curate",
            batch,
            lambda current: curate_candidate_samples(
                current,
                policy=self.curator_policy,
            ),
            "根据 Inspector 证据做出 keep/reject/repair/review 决策。",
        )
        batch = self._run_batch_step(
            trace,
            "Repair Planner",
            "repair_plan",
            batch,
            lambda current: plan_repairs(current, chain_mode=self.repair_chain_mode),
            "将修复决策转换为可执行的修复计划。",
        )
        batch = self._run_batch_step(
            trace,
            "Final Selector",
            "final_select",
            batch,
            select_final_samples,
            "选择最终音频路径，或将样本标记为排除/复核。",
        )
        batch = self._run_batch_step(
            trace,
            "Archivist",
            "export",
            batch,
            lambda current: export_curated_batch(
                current,
                export_dir,
                report_agent=self.report_agent,
            ),
            "写入 manifest、报告和状态文件。",
        )
        batch = _attach_trace_artifact(batch, trace)

        return SupervisorRunResult(batch=batch, trace=trace)

    def run_repair(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        export_dir: Path | str,
    ) -> SupervisorRunResult:
        """运行启用修复的流程，并记录执行和评估轨迹。"""
        if self.audio_metrics_analyzer is None:
            raise ValueError("run_repair 需要 audio_metrics_analyzer")

        result = self.run_basic_until_repair_plan(input_dir, output_dir)
        trace = list(result.trace)
        batch = result.batch
        batch = self._run_batch_step(
            trace,
            "Repair Executor",
            "repair_execute",
            batch,
            lambda current: execute_repairs(
                current,
                executor=self.repair_executor,
            ),
            "调用修复工具并记录生成的音频路径。",
        )
        batch = self._run_batch_step(
            trace,
            "Repair Evaluator",
            "repair_evaluate",
            batch,
            lambda current: evaluate_repairs(
                current,
                audio_metrics_analyzer=self.audio_metrics_analyzer,
                evaluator=self.repair_evaluator,
            ),
            "比较修复前后指标，判断修复音频是否改善。",
        )
        batch = self._run_batch_step(
            trace,
            "Final Selector",
            "final_select",
            batch,
            select_final_samples,
            "选择原始、修复、排除或复核输出。",
        )
        batch = self._run_batch_step(
            trace,
            "Archivist",
            "export",
            batch,
            lambda current: export_curated_batch(
                current,
                export_dir,
                report_agent=self.report_agent,
            ),
            "写入 manifest、报告和状态文件。",
        )
        batch = _attach_trace_artifact(batch, trace)
        return SupervisorRunResult(batch=batch, trace=trace)

    def run_basic_until_repair_plan(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
    ) -> SupervisorRunResult:
        """在最终选择和导出前运行共享准备阶段。"""
        trace: list[SupervisorTraceStep] = []
        batch = self._run_import(input_dir, output_dir, trace)
        for agent_name, phase, runner, notes in [
            (
                "Normalizer",
                "normalize",
                lambda current: normalize_batch(current, normalizer=self.normalizer),
                "创建用于分析的工作副本，不覆盖原始音频。",
            ),
            (
                "VAD Segmenter",
                "segment",
                lambda current: generate_candidate_samples(current, self.vad_analyzer),
                "检测可作为训练候选的语音区域。",
            ),
            (
                "Sample Materializer",
                "materialize",
                lambda current: materialize_candidate_samples(
                    current,
                    self.sample_materializer,
                ),
                "生成供下游工具使用的实际样本文件。",
            ),
            self._transcription_step_definition(),
            (
                "Inspector",
                "inspect",
                lambda current: inspect_candidate_samples(
                    current,
                    audio_metrics_analyzer=self.audio_metrics_analyzer,
                    acoustic_consistency_analyzer=self.acoustic_consistency_analyzer,
                ),
                "将原始字段和音频指标转换为问题标签与证据。",
            ),
            (
                "Strategy Agent",
                "strategy",
                lambda current: select_processing_strategies(
                    current,
                    tool_registry=self.tool_registry,
                    agent=self.processing_strategy_agent,
                ),
                "根据 Inspector 证据选择处理方向和候选工具。",
            ),
            (
                "Curator",
                "curate",
                lambda current: curate_candidate_samples(
                    current,
                    policy=self.curator_policy,
                ),
                "根据 Inspector 证据做出 keep/reject/repair/review 决策。",
            ),
            (
                "Repair Planner",
                "repair_plan",
                lambda current: plan_repairs(current, chain_mode=self.repair_chain_mode),
                "将修复决策转换为可执行的修复计划。",
            ),
        ]:
            batch = self._run_batch_step(
                trace,
                agent_name,
                phase,
                batch,
                runner,
                notes,
            )
        return SupervisorRunResult(batch=batch, trace=trace)

    def _run_transcription_step(
        self,
        trace: list[SupervisorTraceStep],
        batch: Batch,
    ) -> Batch:
        agent_name, phase, runner, notes = self._transcription_step_definition()
        return self._run_batch_step(trace, agent_name, phase, batch, runner, notes)

    def _transcription_step_definition(
        self,
    ) -> tuple[str, str, Callable[[Batch], Batch], str]:
        if self.run_asr:
            return (
                "ASR Transcriber",
                "transcribe",
                lambda current: transcribe_candidate_samples(current, self.transcriber),
                "为每个样本补充候选文本和 ASR 置信度。",
            )
        return (
            "ASR Transcriber",
            "transcribe",
            lambda current: current,
            "当前数据用途不需要文本，因此跳过 ASR。",
        )

    def _run_import(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        trace: list[SupervisorTraceStep],
    ) -> Batch:
        input_summary = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "target_policy": self.target_policy,
        }
        batch = create_import_batch(
            input_dir,
            output_dir,
            target_policy=self.target_policy,
            run_config=self.run_config,
            metadata_config=self.metadata_config,
        )
        self._append_trace(
            trace,
            "Data Intake",
            "import",
            input_summary,
            _batch_summary(batch),
            "扫描输入目录并登记原始音频资产。",
        )
        return batch

    def _run_batch_step(
        self,
        trace: list[SupervisorTraceStep],
        agent_name: str,
        phase: str,
        batch: Batch,
        runner: Callable[[Batch], Batch],
        notes: str,
    ) -> Batch:
        input_summary = _batch_summary(batch)
        next_batch = runner(batch)
        self._append_trace(
            trace,
            agent_name,
            phase,
            input_summary,
            _batch_summary(next_batch),
            notes,
        )
        return next_batch

    def _append_trace(
        self,
        trace: list[SupervisorTraceStep],
        agent_name: str,
        phase: str,
        input_summary: dict[str, Any],
        output_summary: dict[str, Any],
        notes: str,
    ) -> None:
        trace.append(
            SupervisorTraceStep(
                step_index=len(trace) + 1,
                agent_name=agent_name,
                phase=phase,
                input_summary=input_summary,
                output_summary=output_summary,
                notes=notes,
            )
        )


def supervisor_trace_step_to_dict(step: SupervisorTraceStep) -> dict[str, Any]:
    return {
        "step_index": step.step_index,
        "agent_name": step.agent_name,
        "phase": step.phase,
        "input_summary": dict(step.input_summary),
        "output_summary": dict(step.output_summary),
        "notes": step.notes,
    }


def supervisor_run_result_to_dict(result: SupervisorRunResult) -> dict[str, Any]:
    return {
        "batch": batch_to_dict(result.batch),
        "trace": [supervisor_trace_step_to_dict(step) for step in result.trace],
    }


def _attach_trace_artifact(
    batch: Batch,
    trace: list[SupervisorTraceStep],
) -> Batch:
    report_path = batch.export_paths.get("report")
    if report_path is None:
        return batch

    trace_path = Path(report_path).parent / "supervisor_trace.json"
    trace_payload = [supervisor_trace_step_to_dict(step) for step in trace]
    trace_path.write_text(
        json.dumps(trace_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_paths = dict(batch.export_paths)
    export_paths["supervisor_trace"] = str(trace_path)
    return replace(batch, export_paths=export_paths)


def _batch_summary(batch: Batch) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "phase": batch.phase.value,
        "raw_assets": batch.raw_asset_count,
        "candidate_samples": batch.candidate_sample_count,
        "analysis_results": batch.analysis_result_count,
        "processing_strategies": batch.processing_strategy_count,
        "curator_decisions": batch.curator_decision_count,
        "repair_plans": batch.repair_plan_count,
        "repair_executions": batch.repair_execution_count,
        "repair_evaluations": batch.repair_evaluation_count,
        "final_selections": batch.final_selection_count,
        "exported": bool(batch.export_paths),
    }
