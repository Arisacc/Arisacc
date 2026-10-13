from __future__ import annotations
"""Web 任务编排层。

这里把前端表单参数转换成后台 job，再把真正耗时的音频 pipeline 放到线程中执行。
Web 层不直接做音频分析；它负责准备工具实例、记录任务状态、生成前端 summary，
并生成本地任务摘要。
"""

import inspect
import threading
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from audio_dataset_agent.analyzers.vad import (
    FixedVADAnalyzer,
    SpeechSegment,
    WaveEnergyVADAnalyzer,
    WholeFileVADAnalyzer,
)
from audio_dataset_agent.cli import resolve_ffmpeg_path
from audio_dataset_agent.harness.runtime import HarnessRuntime
from audio_dataset_agent.curation.agent import LLMCuratorAgent
from audio_dataset_agent.curation.policy import BaselineCurationPolicy
from audio_dataset_agent.curation.qwen_client import TransformersQwenClient
from audio_dataset_agent.inspection.audio_metrics import WaveAudioMetricsAnalyzer
from audio_dataset_agent.inspection.consistency import BatchAcousticConsistencyAnalyzer
from audio_dataset_agent.knowledge.retriever import LocalKnowledgeRetriever
from audio_dataset_agent.importer.metadata import MetadataImportConfig
from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline, run_repair_enabled_pipeline
from audio_dataset_agent.portfolio.context import build_local_portfolio_research
from audio_dataset_agent.portfolio.renderer import write_portfolio_artifacts
from audio_dataset_agent.processing.ffmpeg_normalizer import FFmpegAudioNormalizer
from audio_dataset_agent.repair.executor import CompositeRepairExecutor, FFmpegRepairExecutor
from audio_dataset_agent.reporting.agent import DatasetReportAgent, LLMDatasetReportAgent
from audio_dataset_agent.segmentation.materializer import FFmpegSampleMaterializer
from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent, LLMProcessingStrategyAgent
from audio_dataset_agent.tools.registry import default_tool_registry
from audio_dataset_agent.preprocessing.msst import MSSTModelSpec, MSSTPreprocessor
from audio_dataset_agent.transcription.faster_whisper_transcriber import FasterWhisperTranscriber
from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult
from audio_dataset_agent.web.summary import build_summary_from_export_dir, write_summary_file
from audio_dataset_agent.storage.sqlite import SQLiteRunStore


WorkflowName = Literal["basic", "repair"]
TargetPolicy = Literal["voice_clone"]
AsrMode = Literal["none"]
VadMode = Literal["fixed", "wave-energy", "whole-file"]
StrategyAgentMode = Literal["baseline", "qwen-local"]
CuratorAgentMode = Literal["baseline", "qwen-local"]
ReportAgentMode = Literal["deterministic", "qwen-local"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
MSSTMissingPolicy = Literal["error", "skip"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(
    current_stage: str,
    stage_message: str,
    percent: int,
    processed_samples: int = 0,
    total_samples: int = 0,
) -> dict[str, Any]:
    return {
        "current_stage": current_stage,
        "stage_message": stage_message,
        "processed_samples": processed_samples,
        "total_samples": total_samples,
        "percent": max(0, min(100, int(percent))),
    }


@dataclass
class RunRequest:
    # 前端一次运行会固化成 RunRequest；后台线程只读这个快照，不再依赖页面状态。
    input_dir: str
    target_policy: TargetPolicy = "voice_clone"
    workflow: WorkflowName = "basic"
    asr: AsrMode = "none"
    vad: VadMode = "wave-energy"
    strategy_agent: StrategyAgentMode = "baseline"
    curator_agent: CuratorAgentMode = "baseline"
    report_agent: ReportAgentMode = "deterministic"
    qwen_model_path: str = "local_tools/models/qwen/Qwen3-4B-Instruct-2507"
    qwen_max_new_tokens: int = 192
    llm_only_issue_samples: bool = True
    curator_confidence_threshold: float = 0.8
    speaker_consistency_check: bool = False
    demo_transcript: str = "demo transcript"
    demo_confidence: float = 0.95
    vad_min_duration_seconds: float = 0.6
    vad_merge_gap_seconds: float = 0.45
    metadata_path: str = ""
    metadata_audio_column: str = "audio_path"
    metadata_text_column: str = "text"
    metadata_language_column: str = "language"
    metadata_speaker_column: str = "speaker_id"
    use_sidecar_text: bool = True
    repair_chain: str = "single"
    rag_mode: str = "none"
    rag_documents: str = "knowledge_base/documents"
    rag_index: str = "local_tools/cache/rag/index.sqlite"
    rag_rebuild: bool = False
    msst_enabled: bool = False
    msst_missing_policy: MSSTMissingPolicy = "error"
    msst_model_name: str = "deverb_bs_roformer_8_256dim_8depth"
    msst_model_path: str = "local_tools/models/msst/deverb_bs_roformer_8_256dim_8depth/deverb_bs_roformer_8_256dim_8depth.ckpt"
    msst_config_path: str = "local_tools/models/msst/deverb_bs_roformer_8_256dim_8depth/deverb_bs_roformer_8_256dim_8depth.ckpt.yaml"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRequest":
        """从 SQLite 快照恢复请求，忽略未来版本新增的未知字段。"""
        field_names = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in field_names}
        return cls(**values)

    def to_run_config(self, ffmpeg_path: str) -> dict[str, Any]:
        """生成可写入批次产物的运行参数快照，不包含密钥。"""
        return {
            "target_policy": self.target_policy,
            "workflow": self.workflow,
            "asr": self.asr,
            "vad": self.vad,
            "strategy_agent": self.strategy_agent,
            "curator_agent": self.curator_agent,
            "report_agent": self.report_agent,
            "llm_only_issue_samples": self.llm_only_issue_samples,
            "curator_confidence_threshold": self.curator_confidence_threshold,
            "speaker_consistency_check": self.speaker_consistency_check,
            "qwen_max_new_tokens": self.qwen_max_new_tokens,
            "vad_min_duration_seconds": self.vad_min_duration_seconds,
            "vad_merge_gap_seconds": self.vad_merge_gap_seconds,
            "models": {
                "qwen": self.qwen_model_path,
                "faster_whisper": (
                    "local_tools/models/faster-whisper/small"
                    if self.asr == "faster-whisper"
                    else None
                ),
            },
            "tools": {"ffmpeg_path": ffmpeg_path},
            "metadata": {
                "path": self.metadata_path or None,
                "audio_column": self.metadata_audio_column,
                "text_column": self.metadata_text_column,
                "language_column": self.metadata_language_column,
                "speaker_column": self.metadata_speaker_column,
                "use_sidecar_text": self.use_sidecar_text,
            },
            "repair_chain": self.repair_chain,
            "rag": {
                "mode": self.rag_mode,
                "documents": self.rag_documents,
                "index": self.rag_index,
                "rebuild": self.rag_rebuild,
            },
            "msst": {
                "enabled": self.msst_enabled,
                "missing_policy": self.msst_missing_policy,
                "model_name": self.msst_model_name,
                "model_path": self.msst_model_path,
                "config_path": self.msst_config_path,
                "runtime": "pymss/MSST adapter",
            },
        }

    def to_metadata_config(self) -> MetadataImportConfig:
        return MetadataImportConfig(
            metadata_path=self.metadata_path or None,
            audio_column=self.metadata_audio_column,
            text_column=self.metadata_text_column,
            language_column=self.metadata_language_column,
            speaker_column=self.metadata_speaker_column,
            use_sidecar_text=self.use_sidecar_text,
        )


@dataclass
class AudioDatasetJob:
    # Job 保存 Web 视角的任务状态、导出目录、summary 和错误信息。
    job_id: str
    request: RunRequest
    status: JobStatus = "queued"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    work_dir: str | None = None
    export_dir: str | None = None
    batch_export_dir: str | None = None
    summary: dict[str, Any] | None = None
    error_message: str | None = None
    cancel_requested: bool = False
    task_runtime: dict[str, Any] | None = None
    checkpoint_path: str | None = None
    checkpoint_stage: str | None = None
    progress: dict[str, Any] = field(
        default_factory=lambda: _progress("queued", "任务已创建，等待执行", 0)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request": {
                "input_dir": self.request.input_dir,
                "target_policy": self.request.target_policy,
                "workflow": self.request.workflow,
                "asr": self.request.asr,
                "vad": self.request.vad,
                "strategy_agent": self.request.strategy_agent,
                "curator_agent": self.request.curator_agent,
                "report_agent": self.request.report_agent,
                "qwen_model_path": self.request.qwen_model_path,
                "qwen_max_new_tokens": self.request.qwen_max_new_tokens,
                "llm_only_issue_samples": self.request.llm_only_issue_samples,
                "curator_confidence_threshold": self.request.curator_confidence_threshold,
                "speaker_consistency_check": self.request.speaker_consistency_check,
                "vad_min_duration_seconds": self.request.vad_min_duration_seconds,
                "vad_merge_gap_seconds": self.request.vad_merge_gap_seconds,
                "metadata_path": self.request.metadata_path,
                "metadata_audio_column": self.request.metadata_audio_column,
                "metadata_text_column": self.request.metadata_text_column,
                "metadata_language_column": self.request.metadata_language_column,
                "metadata_speaker_column": self.request.metadata_speaker_column,
                "use_sidecar_text": self.request.use_sidecar_text,
                "repair_chain": self.request.repair_chain,
                "rag_mode": self.request.rag_mode,
                "rag_documents": self.request.rag_documents,
                "rag_index": self.request.rag_index,
                "rag_rebuild": self.request.rag_rebuild,
                "msst_enabled": self.request.msst_enabled,
                "msst_missing_policy": self.request.msst_missing_policy,
                "msst_model_name": self.request.msst_model_name,
                "msst_model_path": self.request.msst_model_path,
                "msst_config_path": self.request.msst_config_path,
            },
            "work_dir": self.work_dir,
            "export_dir": self.export_dir,
            "batch_export_dir": self.batch_export_dir,
            "summary": self.summary,
            "error_message": self.error_message,
            "cancel_requested": self.cancel_requested,
            "task_runtime": self.task_runtime,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_stage": self.checkpoint_stage,
            "progress": self.progress,
        }


PipelineRunner = Callable[..., Path]


def _safe_upload_parts(raw_name: str) -> list[str]:
    """校验浏览器提供的相对路径，阻止上传文件逃逸到临时目录之外。"""
    normalized = str(raw_name or "").replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("上传文件路径无效")
    return parts


class IssueOnlyAgentWrapper:
    """只把带问题标签的样本交给 LLM，干净样本走 baseline。"""

    def __init__(self, issue_agent, clean_fallback_agent) -> None:
        self.issue_agent = issue_agent
        self.clean_fallback_agent = clean_fallback_agent

    def decide(self, result, strategy=None):
        if getattr(result, "issue_tags", None):
            return _call_decider(self.issue_agent, result, strategy)
        return _call_decider(self.clean_fallback_agent, result, strategy)


def _call_decider(agent, result, strategy):
    """兼容只接收 AnalysisResult 的旧 Agent 实现。"""
    if strategy is None:
        return agent.decide(result)
    parameters = inspect.signature(agent.decide).parameters.values()
    supports_strategy = any(
        parameter.name == "strategy"
        or parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    )
    if supports_strategy:
        return agent.decide(result, strategy)
    return agent.decide(result)


class AudioDatasetJobManager:
    """管理 Web UI 触发的本地音频数据集任务。"""

    def __init__(
        self,
        runs_root: Path | str,
        runner: PipelineRunner | None = None,
        store: SQLiteRunStore | None = None,
        queue_mode: bool = False,
        ):
        self.runs_root = Path(runs_root)
        self.runner = runner or _run_pipeline_for_request
        self._jobs: dict[str, AudioDatasetJob] = {}
        self._lock = threading.Lock()
        self.store = store or SQLiteRunStore(self.runs_root / "harness.sqlite3")
        self.harness = HarnessRuntime(self.store)
        self.queue_mode = queue_mode
        self._restore_persisted_jobs()

    def _restore_persisted_jobs(self) -> None:
        """恢复最近一次服务进程退出前已登记的 Run 元数据。"""
        for snapshot in self.store.list_runs():
            try:
                request = RunRequest.from_dict(snapshot.get("request") or {})
                job = AudioDatasetJob(
                    job_id=snapshot["job_id"],
                    request=request,
                    status=snapshot["status"],
                    created_at=snapshot["created_at"],
                    updated_at=snapshot["updated_at"],
                    work_dir=snapshot.get("work_dir"),
                    export_dir=snapshot.get("export_dir"),
                    batch_export_dir=snapshot.get("batch_export_dir"),
                    summary=snapshot.get("summary"),
                    error_message=snapshot.get("error_message"),
                    cancel_requested=snapshot.get("cancel_requested", False),
                    progress=snapshot.get("progress") or _progress("restored", "已恢复任务记录", 0),
                )
                # 队列任务由独立 Worker 执行；服务重启后保留 queued/running，
                # 让租约过期机制负责重新领取，而不是丢失可恢复的 Run。
                if job.status in {"queued", "running"}:
                    job.status = "queued"
                    job.error_message = None
                    job.progress = _progress("queued", "服务已恢复，等待 Worker 重新领取", 0)
                self._jobs[job.job_id] = job
                self.store.upsert_run(job.to_dict())
            except (KeyError, TypeError, ValueError):
                continue

    def start_job(self, request: RunRequest) -> AudioDatasetJob:
        job = self._create_job(request)
        self.harness.create_run(job.job_id, job.to_dict()["request"])
        if self.queue_mode:
            task_id = self.store.create_task(job.job_id, "pipeline", {"job_id": job.job_id})
            self.store.append_event(
                job.job_id,
                event_type="task_enqueued",
                stage="queued",
                status="queued",
                payload={"task_id": task_id},
            )
            return job
        thread = threading.Thread(target=self._run_job, args=(job.job_id,), daemon=True)
        thread.start()
        return job

    def stage_uploaded_directory(
        self,
        files: list[tuple[str, bytes]],
    ) -> dict[str, str | int]:
        """将浏览器目录选择器上传的文件落到隔离目录，返回可运行的本地路径。"""
        if not files:
            raise ValueError("没有收到可处理的文件")

        upload_id = f"upload_{uuid.uuid4().hex[:12]}"
        upload_root = self.runs_root / "_uploads" / upload_id
        first_parts = _safe_upload_parts(files[0][0])
        directory_name = first_parts[0] if len(first_parts) > 1 else "上传目录"
        input_dir = upload_root / directory_name
        input_dir.mkdir(parents=True, exist_ok=True)

        for raw_name, content in files:
            parts = _safe_upload_parts(raw_name)
            relative_parts = parts[1:] if len(parts) > 1 else parts
            destination = input_dir.joinpath(*relative_parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

        return {
            "input_dir": str(input_dir.resolve()),
            "directory_name": directory_name,
            "file_count": len(files),
        }

    def run_sync(self, request: RunRequest) -> AudioDatasetJob:
        job = self._create_job(request)
        self.harness.create_run(job.job_id, job.to_dict()["request"])
        self._run_job(job.job_id)
        return self.get_job(job.job_id)

    def get_job(self, job_id: str) -> AudioDatasetJob:
        with self._lock:
            if job_id not in self._jobs:
                snapshot = self.store.get_run(job_id)
                if snapshot is None:
                    raise KeyError(job_id)
                request = RunRequest.from_dict(snapshot.get("request") or {})
                self._jobs[job_id] = AudioDatasetJob(
                    job_id=snapshot["job_id"], request=request, status=snapshot["status"],
                    created_at=snapshot["created_at"], updated_at=snapshot["updated_at"],
                    work_dir=snapshot.get("work_dir"), export_dir=snapshot.get("export_dir"),
                    batch_export_dir=snapshot.get("batch_export_dir"), summary=snapshot.get("summary"),
                    error_message=snapshot.get("error_message"), cancel_requested=snapshot.get("cancel_requested", False),
                    progress=snapshot.get("progress") or _progress("restored", "已恢复任务记录", 0),
                )
            job = self._jobs[job_id]
            # API 进程与独立 Worker 可能持有不同的内存对象；每次读取时
            # 从 SQLite 同步一次，确保前端看到的是 Worker 的最新状态。
            snapshot = self.store.get_run(job_id)
            if snapshot is not None and snapshot.get("updated_at", "") > job.updated_at:
                job.status = snapshot["status"]
                job.updated_at = snapshot["updated_at"]
                job.work_dir = snapshot.get("work_dir")
                job.export_dir = snapshot.get("export_dir")
                job.batch_export_dir = snapshot.get("batch_export_dir")
                job.summary = snapshot.get("summary")
                job.error_message = snapshot.get("error_message")
                job.cancel_requested = snapshot.get("cancel_requested", False)
                job.progress = snapshot.get("progress") or job.progress
            task = self.store.get_task_for_job(job_id)
            job.task_runtime = task
            checkpoint_path = self.runs_root / job_id / "checkpoint.json"
            checkpoint = self.harness.load_checkpoint(checkpoint_path)
            job.checkpoint_path = str(checkpoint_path) if checkpoint_path.exists() else None
            job.checkpoint_stage = checkpoint.get("stage") if checkpoint else None
            return job

    def list_jobs(self) -> list[AudioDatasetJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> AudioDatasetJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            previous_status = job.status
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.progress = _progress("cancelled", "任务已取消", 100)
            elif job.status == "running":
                job.progress = _progress(
                    "cancelling",
                    "已收到取消请求，正在等待当前阶段结束",
                    job.progress.get("percent", 0),
                )
            job.updated_at = _now_iso()
            snapshot = job.to_dict()
            current_status = job.status
            progress = job.progress
        self.store.upsert_run(snapshot)
        if current_status != previous_status:
            self.harness.transition(
                job_id,
                previous=previous_status,
                current=current_status,
                stage=str(progress.get("current_stage", "cancelled")),
                message=str(progress.get("stage_message", "任务已取消")),
            )
        else:
            self.store.append_event(
                job_id,
                event_type="cancel_requested",
                stage=str(progress.get("current_stage", "pipeline")),
                status=current_status,
                payload={"message": "已收到取消请求"},
            )
        return job

    def _create_job(self, request: RunRequest) -> AudioDatasetJob:
        input_dir = Path(request.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"音频目录不存在或不是文件夹：{input_dir}")
        if request.asr == "none" and request.target_policy != "voice_clone":
            raise ValueError("只有 voice_clone 流程允许关闭 ASR。")
        if request.qwen_max_new_tokens <= 0:
            raise ValueError("qwen_max_new_tokens 必须为正数")
        if request.repair_chain not in {"single", "auto"}:
            raise ValueError("不支持的修复链模式：只能选择 single 或 auto")
        if request.rag_mode not in {"none", "local"}:
            raise ValueError("不支持的 RAG 模式：只能选择 none 或 local")
        job_id = f"audio_job_{uuid.uuid4().hex[:12]}"
        job = AudioDatasetJob(job_id=job_id, request=request)
        with self._lock:
            self._jobs[job_id] = job
        self.store.upsert_run(job.to_dict())
        return job

    def _run_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job.cancel_requested or job.status == "cancelled":
            self._update_job(job_id, status="cancelled", progress=_progress("cancelled", "任务已取消", 100))
            return False

        work_dir = self.runs_root / job_id / "work"
        export_dir = self.runs_root / job_id / "exports"
        checkpoint_path = work_dir.parent / "checkpoint.json"
        job.checkpoint_path = str(checkpoint_path)
        self._update_job(
            job_id,
            status="running",
            work_dir=str(work_dir),
            export_dir=str(export_dir),
            progress=_progress("pipeline", "正在运行音频清洗与 Agent 决策流程", 10),
        )
        self.harness.checkpoint(
            job_id,
            work_dir.parent / "checkpoint.json",
            stage="pipeline",
            status="running",
            artifacts=[],
            state={"work_dir": str(work_dir), "export_dir": str(export_dir), "resume_policy": "rerun_current_pipeline"},
        )

        try:
            progress_callback = lambda event: self._handle_pipeline_event(job_id, event)
            try:
                batch_export_dir = self.runner(job.request, work_dir, export_dir, progress_callback)
            except TypeError as exc:
                if "positional argument" not in str(exc) and "given" not in str(exc):
                    raise
                batch_export_dir = self.runner(job.request, work_dir, export_dir)
            if self.get_job(job_id).cancel_requested:
                self._update_job(
                    job_id,
                    status="cancelled",
                    batch_export_dir=str(batch_export_dir),
                    progress=_progress("cancelled", "任务已取消，已保留当前导出产物", 100),
                )
                return

            self._update_job(job_id, progress=_progress("summary", "正在生成前端摘要", 92))
            write_summary_file(batch_export_dir)
            summary = build_summary_from_export_dir(batch_export_dir)
            total_samples = int(summary.get("batch", {}).get("total_samples") or 0)
            self._update_job(
                job_id,
                status="succeeded",
                batch_export_dir=str(batch_export_dir),
                summary=summary,
                progress=_progress("succeeded", "处理完成", 100, total_samples, total_samples),
            )
            self.harness.register_artifacts(
                job_id,
                [
                    {"name": "manifest", "path": str(Path(batch_export_dir) / "manifest.jsonl"), "kind": "manifest"},
                    {"name": "quality_report", "path": str(Path(batch_export_dir) / "dataset_quality_report.json"), "kind": "report"},
                ],
            )
            self.harness.checkpoint(
                job_id,
                work_dir.parent / "checkpoint.json",
                stage="export",
                status="succeeded",
                artifacts=self.store.list_artifacts(job_id),
                state={"work_dir": str(work_dir), "export_dir": str(export_dir), "batch_export_dir": str(batch_export_dir), "resume_policy": "already_complete"},
            )
            return True
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                error_message=str(exc),
                progress=_progress("failed", "处理失败", 100),
            )
            return False

    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            previous_status = job.status
            previous_progress = dict(job.progress)
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now_iso()
            snapshot = job.to_dict()
        self.store.upsert_run(snapshot)
        progress = changes.get("progress")
        if progress and (
            progress.get("current_stage") != previous_progress.get("current_stage")
            or progress.get("percent") != previous_progress.get("percent")
        ):
            self.store.append_event(
                job_id,
                event_type="stage_progress",
                stage=str(progress.get("current_stage", "unknown")),
                status=str(snapshot.get("status", previous_status)),
                payload={
                    "message": progress.get("stage_message", "阶段进度更新"),
                    "percent": progress.get("percent", 0),
                    "processed_samples": progress.get("processed_samples", 0),
                    "total_samples": progress.get("total_samples", 0),
                },
            )
        if "status" in changes and changes["status"] != previous_status:
            progress = changes.get("progress") or snapshot.get("progress") or {}
            self.harness.transition(
                job_id,
                previous=previous_status,
                current=changes["status"],
                stage=str(progress.get("current_stage", "unknown")),
                message=str(progress.get("stage_message", "状态更新")),
            )

    def _handle_pipeline_event(self, job_id: str, event: dict[str, Any]) -> None:
        stage = str(event.get("stage", "pipeline"))
        starts = {"import": 12, "preprocess": 18, "normalize": 28, "vad": 40, "materialize": 48, "asr": 54, "inspect": 64, "strategy": 72, "curate": 80, "repair_plan": 84, "repair_execute": 88, "repair_evaluate": 91, "final_select": 94, "export": 97}
        percent = starts.get(stage, 10)
        if event.get("event") == "stage_completed":
            percent = min(99, percent + 5)
        self._update_job(
            job_id,
            progress=_progress(stage, str(event.get("message", "处理阶段更新")), percent),
        )


def _run_pipeline_for_request(request: RunRequest, work_dir: Path, export_dir: Path, progress_callback=None) -> Path:
    # 这里是 Web job 进入真正音频处理 pipeline 的桥梁。
    work_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    # 集中实例化工具适配器，workflow 只接收抽象接口，便于后续替换工具实现。
    ffmpeg_path = resolve_ffmpeg_path("ffmpeg")
    run_config = request.to_run_config(ffmpeg_path)
    metadata_config = request.to_metadata_config()
    pipeline_input_dir = _prepare_msst_input(request, Path(request.input_dir), work_dir, progress_callback)
    if request.msst_enabled:
        run_config.setdefault("msst", {})["output_dir"] = str(work_dir / "preprocessed_msst")
        run_config["msst"]["status"] = "completed_or_skipped"
    vad_analyzer = _build_vad_analyzer(request)
    normalizer = FFmpegAudioNormalizer(ffmpeg_path=ffmpeg_path)
    sample_materializer = FFmpegSampleMaterializer(ffmpeg_path=ffmpeg_path)
    audio_metrics = WaveAudioMetricsAnalyzer()
    acoustic_consistency = (
        BatchAcousticConsistencyAnalyzer(audio_metrics)
        if request.target_policy == "voice_clone" and request.speaker_consistency_check
        else None
    )
    transcriber = _build_transcriber(request)
    run_asr = request.asr != "none"
    repair_executor = CompositeRepairExecutor(ffmpeg_executor=FFmpegRepairExecutor(ffmpeg_path=ffmpeg_path))
    tool_registry = default_tool_registry()
    agent_components = _build_agent_components(request, tool_registry)

    if request.workflow == "repair":
        batch = run_repair_enabled_pipeline(
            input_dir=pipeline_input_dir,
            output_dir=work_dir,
            export_dir=export_dir,
            vad_analyzer=vad_analyzer,
            transcriber=transcriber,
            audio_metrics_analyzer=audio_metrics,
            normalizer=normalizer,
            sample_materializer=sample_materializer,
            tool_registry=tool_registry,
            repair_executor=repair_executor,
            processing_strategy_agent=agent_components["processing_strategy_agent"],
            curator_policy=agent_components["curator_policy"],
            report_agent=agent_components["report_agent"],
            target_policy=request.target_policy,
            run_asr=run_asr,
            run_config=run_config,
            metadata_config=metadata_config,
            acoustic_consistency_analyzer=acoustic_consistency,
            repair_chain_mode=request.repair_chain,
            progress_callback=progress_callback,
        )
    else:
        batch = run_batch_pipeline(
            input_dir=pipeline_input_dir,
            output_dir=work_dir,
            export_dir=export_dir,
            vad_analyzer=vad_analyzer,
            transcriber=transcriber,
            audio_metrics_analyzer=audio_metrics,
            normalizer=normalizer,
            sample_materializer=sample_materializer,
            tool_registry=tool_registry,
            processing_strategy_agent=agent_components["processing_strategy_agent"],
            curator_policy=agent_components["curator_policy"],
            report_agent=agent_components["report_agent"],
            target_policy=request.target_policy,
            run_asr=run_asr,
            run_config=run_config,
            metadata_config=metadata_config,
            acoustic_consistency_analyzer=acoustic_consistency,
            repair_chain_mode=request.repair_chain,
            progress_callback=progress_callback,
        )
    if agent_components["knowledge_retriever"] is not None:
        agent_components["knowledge_retriever"].write_audit_files(
            Path(batch.export_paths["report"]).parent
        )
    batch = write_portfolio_artifacts(
        batch,
        build_local_portfolio_research(batch),
    )
    return Path(batch.export_paths["manifest"]).parent


def _prepare_msst_input(request: RunRequest, input_dir: Path, work_dir: Path, progress_callback=None) -> Path:
    """在标准化前执行可选 MSST 去混响，输出只进入本次 Run 工作区。"""
    if not request.msst_enabled:
        return input_dir
    model = MSSTModelSpec(
        name=request.msst_model_name,
        checkpoint_path=Path(request.msst_model_path),
        config_path=Path(request.msst_config_path),
    )
    preprocessor = MSSTPreprocessor(model, separator_factory=MSSTPreprocessor.pymss_separator_factory)
    output_dir = work_dir / "preprocessed_msst"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".wav", ".ogg", ".mp3", ".flac", ".m4a"})
    if progress_callback:
        progress_callback({"event": "stage_started", "stage": "preprocess", "message": "开始 MSST 去混响", "percent": 15})
    pairs = [(source, output_dir / f"{source.stem}.wav") for source in audio_files]
    parameters = {"device": "auto", "output_format": "wav", "use_tta": False, "cache_dir": str(work_dir / "msst_cache")}
    results = preprocessor.prepare_files(pairs, request.msst_missing_policy, parameters)
    for index, (source, result) in enumerate(zip(audio_files, results), start=1):
        if result.status == "skipped":
            destination = output_dir / source.name
            destination.write_bytes(source.read_bytes())
        if progress_callback:
            progress_callback({"event": "stage_progress", "stage": "preprocess", "message": f"MSST 处理 {index}/{len(audio_files)}", "percent": 15 + int(index / max(1, len(audio_files)) * 10)})
    if progress_callback:
        progress_callback({"event": "stage_completed", "stage": "preprocess", "message": "MSST 去混响完成", "percent": 25})
    return output_dir


def _build_agent_components(request: RunRequest, tool_registry):
    """按前端选项创建可选的本地 LLM Agent。"""
    shared_qwen_client = None
    knowledge_retriever = _build_knowledge_retriever(request)

    def qwen_client():
        nonlocal shared_qwen_client
        if shared_qwen_client is None:
            shared_qwen_client = TransformersQwenClient(
                model_path=request.qwen_model_path,
                max_new_tokens=request.qwen_max_new_tokens,
            )
        return shared_qwen_client

    processing_strategy_agent = None
    if request.strategy_agent == "qwen-local":
        baseline_strategy = BaselineProcessingStrategyAgent(tool_registry, target_policy=request.target_policy)
        processing_strategy_agent = LLMProcessingStrategyAgent(
            qwen_client(),
            tool_registry=tool_registry,
            fallback_agent=baseline_strategy,
            target_policy=request.target_policy,
            knowledge_retriever=knowledge_retriever,
        )
        if request.llm_only_issue_samples:
            processing_strategy_agent = IssueOnlyAgentWrapper(processing_strategy_agent, baseline_strategy)

    curator_policy = None
    if request.curator_agent == "qwen-local":
        baseline_curator = BaselineCurationPolicy(target_policy=request.target_policy)
        curator_policy = LLMCuratorAgent(
            qwen_client(),
            fallback_policy=baseline_curator,
            target_policy=request.target_policy,
        )
        if request.llm_only_issue_samples:
            curator_policy = IssueOnlyAgentWrapper(curator_policy, baseline_curator)

    report_agent = None
    base_report_agent = DatasetReportAgent(knowledge_retriever=knowledge_retriever)
    if request.report_agent == "qwen-local":
        report_agent = LLMDatasetReportAgent(
            qwen_client(),
            base_agent=base_report_agent,
        )
    elif knowledge_retriever is not None:
        report_agent = base_report_agent

    return {
        "processing_strategy_agent": processing_strategy_agent,
        "curator_policy": curator_policy,
        "report_agent": report_agent,
        "knowledge_retriever": knowledge_retriever,
    }


def _build_knowledge_retriever(request: RunRequest):
    if request.rag_mode == "none":
        return None
    try:
        retriever = LocalKnowledgeRetriever(request.rag_documents, request.rag_index)
        if request.rag_rebuild:
            retriever.rebuild()
        return retriever
    except Exception:
        return None


def _build_vad_analyzer(request: RunRequest):
    if request.vad == "whole-file":
        return WholeFileVADAnalyzer()
    if request.vad == "wave-energy":
        return WaveEnergyVADAnalyzer(
            min_duration_seconds=request.vad_min_duration_seconds,
            merge_gap_seconds=request.vad_merge_gap_seconds,
        )
    return FixedVADAnalyzer([SpeechSegment(0.0, 1.5)])


def _build_transcriber(request: RunRequest):
    if request.asr == "faster-whisper":
        return FasterWhisperTranscriber(
            model_size_or_path="local_tools/models/faster-whisper/small",
            device="cpu",
            compute_type="int8",
        )
    if request.asr == "fixed":
        return FixedTranscriber(TranscriptResult(request.demo_transcript, request.demo_confidence))
    return FixedTranscriber(TranscriptResult("", None))



