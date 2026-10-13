# 命令行入口：支持不用前端时直接运行同一套处理流程。
import argparse
import json
import shutil
import sys
from pathlib import Path

from audio_dataset_agent.analyzers.silero_vad import SileroVADAnalyzer
from audio_dataset_agent.analyzers.vad import (
    FixedVADAnalyzer,
    SpeechSegment,
    WaveEnergyVADAnalyzer,
    WholeFileVADAnalyzer,
)
from audio_dataset_agent.core.serialization import batch_to_dict
from audio_dataset_agent.curation.agent import LLMCuratorAgent
from audio_dataset_agent.curation.policy import BaselineCurationPolicy
from audio_dataset_agent.curation.qwen_client import TransformersQwenClient
from audio_dataset_agent.inspection.audio_metrics import (
    FFmpegAudioMetricsAnalyzer,
    WaveAudioMetricsAnalyzer,
)
from audio_dataset_agent.inspection.consistency import BatchAcousticConsistencyAnalyzer
from audio_dataset_agent.knowledge.retriever import LocalKnowledgeRetriever
from audio_dataset_agent.importer.metadata import MetadataImportConfig
from audio_dataset_agent.orchestrator.supervisor import (
    AudioDatasetSupervisor,
    SupervisorRunResult,
    supervisor_run_result_to_dict,
)
from audio_dataset_agent.orchestrator.workflow import (
    run_batch_pipeline,
    run_repair_enabled_pipeline,
)
from audio_dataset_agent.processing.ffmpeg_normalizer import FFmpegAudioNormalizer
from audio_dataset_agent.processing.normalizer import CopyAudioNormalizer
from audio_dataset_agent.portfolio.context import build_local_portfolio_research
from audio_dataset_agent.portfolio.renderer import write_portfolio_artifacts
from audio_dataset_agent.reporting.agent import LLMDatasetReportAgent
from audio_dataset_agent.reporting.agent import DatasetReportAgent
from audio_dataset_agent.repair.executor import (
    CompositeRepairExecutor,
    FFmpegRepairExecutor,
)
from audio_dataset_agent.segmentation.materializer import (
    CopySampleMaterializer,
    FFmpegSampleMaterializer,
)
from audio_dataset_agent.strategy.agent import (
    BaselineProcessingStrategyAgent,
    LLMProcessingStrategyAgent,
)
from audio_dataset_agent.tools.registry import (
    default_tool_registry,
    tool_registry_to_dict,
)
from audio_dataset_agent.transcription.faster_whisper_transcriber import (
    FasterWhisperTranscriber,
)
from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult


def _imageio_ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def resolve_ffmpeg_path(ffmpeg_path: str) -> str:
    if ffmpeg_path != "ffmpeg":
        return ffmpeg_path
    if shutil.which(ffmpeg_path):
        return ffmpeg_path
    local_path = _imageio_ffmpeg_path()
    return local_path or ffmpeg_path


class PlaceholderCuratorClient:
    """用于验证 Agent 决策路径的确定性 LLM 替身。"""

    def complete(self, prompt):
        return {
            "decision": "review",
            "reason": "占位 LLM Curator 仅用于验证调用链，真实决策前必须替换为实际模型",
            "confidence": 0.0,
            "recommended_action": "manual_decision_review",
        }


class PlaceholderStrategyClient:
    """用于验证 Strategy Agent 路径的确定性 LLM 替身。"""

    def complete(self, prompt):
        return {
            "action": "manual_review",
            "recommended_operation": "manual_strategy_review",
            "selected_tool_ids": [],
            "reason": "占位 Strategy 仅用于验证调用链，真实决策前必须替换为实际模型",
            "confidence": 0.0,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行音频数据集 Agent 处理流水线。"
    )
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--export-dir")
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("--demo-transcript", default="demo transcript")
    parser.add_argument("--demo-confidence", type=float, default=0.95)
    parser.add_argument("--demo-start", type=float, default=0.0)
    parser.add_argument("--demo-end", type=float, default=1.5)
    parser.add_argument(
        "--target-policy",
        choices=["asr", "tts", "voice_clone"],
        default="voice_clone",
    )
    parser.add_argument(
        "--orchestrator",
        choices=["workflow", "supervisor"],
        default="workflow",
    )
    parser.add_argument(
        "--vad",
        choices=["fixed", "wave-energy", "silero", "whole-file"],
        default="fixed",
    )
    parser.add_argument("--vad-sampling-rate", type=int, default=16000)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=int, default=100)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=30)
    parser.add_argument("--workflow", choices=["basic", "repair"], default="basic")
    parser.add_argument("--normalizer", choices=["copy", "ffmpeg"], default="copy")
    parser.add_argument(
        "--sample-materializer",
        choices=["copy", "ffmpeg"],
        default="copy",
    )
    parser.add_argument(
        "--audio-metrics",
        choices=["none", "wave", "ffmpeg"],
        default="none",
    )
    parser.add_argument("--asr", choices=["none", "fixed", "faster-whisper"], default="fixed")
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-compute-type", default="int8")
    parser.add_argument("--asr-language", default=None)
    parser.add_argument("--asr-beam-size", type=int, default=5)
    parser.add_argument("--asr-vad-filter", action="store_true")
    parser.add_argument(
        "--curator",
        choices=["baseline", "llm-placeholder", "qwen-local"],
        default="baseline",
    )
    parser.add_argument(
        "--strategy",
        choices=["baseline", "llm-placeholder", "qwen-local"],
        default="baseline",
    )
    parser.add_argument(
        "--report",
        choices=["deterministic", "qwen-local"],
        default="deterministic",
    )
    parser.add_argument(
        "--curator-model-path",
        default="local_tools/models/qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument("--curator-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--report-model-path",
        default="local_tools/models/qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument("--report-max-new-tokens", type=int, default=256)
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument(
        "--repair-chain",
        choices=["single", "auto"],
        default="single",
        help="single 只执行单步修复，auto 启用受控多步修复链。",
    )
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--metadata-audio-column", default="audio_path")
    parser.add_argument("--metadata-text-column", default="text")
    parser.add_argument("--metadata-language-column", default="language")
    parser.add_argument("--metadata-speaker-column", default="speaker_id")
    parser.add_argument(
        "--no-sidecar-text",
        action="store_false",
        dest="use_sidecar_text",
        help="不自动读取与音频同名的 TXT 文本。",
    )
    parser.add_argument("--rag", choices=["none", "local"], default="none")
    parser.add_argument("--rag-documents", default="knowledge_base/documents")
    parser.add_argument("--rag-index", default="local_tools/cache/rag/index.sqlite")
    parser.add_argument("--rag-rebuild", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ffmpeg_path = resolve_ffmpeg_path(args.ffmpeg_path)
    run_config = _build_run_config(args, ffmpeg_path)
    metadata_config = MetadataImportConfig(
        metadata_path=args.metadata_path,
        audio_column=args.metadata_audio_column,
        text_column=args.metadata_text_column,
        language_column=args.metadata_language_column,
        speaker_column=args.metadata_speaker_column,
        use_sidecar_text=args.use_sidecar_text,
    )
    knowledge_retriever = _build_knowledge_retriever(args)

    if args.list_tools:
        print(
            json.dumps(
                tool_registry_to_dict(default_tool_registry()),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not args.input_dir or not args.output_dir or not args.export_dir:
        print("错误：必须提供 --input-dir、--output-dir 和 --export-dir")
        return 2

    # CLI 在默认模式下使用可控的本地实现，便于先验证端到端流程。
    if args.normalizer == "ffmpeg":
        normalizer = FFmpegAudioNormalizer(
            ffmpeg_path=ffmpeg_path,
            sample_rate=args.sample_rate,
            channels=args.channels,
        )
    else:
        normalizer = CopyAudioNormalizer()

    if args.sample_materializer == "ffmpeg":
        sample_materializer = FFmpegSampleMaterializer(ffmpeg_path=ffmpeg_path)
    else:
        sample_materializer = CopySampleMaterializer()

    if args.audio_metrics == "ffmpeg":
        audio_metrics_analyzer = FFmpegAudioMetricsAnalyzer(ffmpeg_path=ffmpeg_path)
    elif args.audio_metrics == "wave":
        audio_metrics_analyzer = WaveAudioMetricsAnalyzer()
    else:
        audio_metrics_analyzer = None

    if args.vad == "silero":
        vad_analyzer = SileroVADAnalyzer(
            sampling_rate=args.vad_sampling_rate,
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_ms,
            min_silence_duration_ms=args.vad_min_silence_ms,
            speech_pad_ms=args.vad_speech_pad_ms,
        )
    elif args.vad == "wave-energy":
        vad_analyzer = WaveEnergyVADAnalyzer()
    elif args.vad == "whole-file":
        vad_analyzer = WholeFileVADAnalyzer()
    else:
        vad_analyzer = FixedVADAnalyzer(
            [SpeechSegment(start_time=args.demo_start, end_time=args.demo_end)]
        )

    run_asr = args.asr != "none"
    if args.asr == "none" and args.target_policy != "voice_clone":
        print("错误：只有 --target-policy voice_clone 时才允许 --asr none")
        return 2

    if args.asr == "faster-whisper":
        transcriber = FasterWhisperTranscriber(
            model_size_or_path=args.asr_model,
            device=args.asr_device,
            compute_type=args.asr_compute_type,
            beam_size=args.asr_beam_size,
            language=args.asr_language,
            vad_filter=args.asr_vad_filter,
        )
    elif args.asr == "none":
        transcriber = FixedTranscriber(
            TranscriptResult(
                text="",
                confidence=None,
            )
        )
    else:
        transcriber = FixedTranscriber(
            TranscriptResult(
                text=args.demo_transcript,
                confidence=args.demo_confidence,
            )
        )

    tool_registry = default_tool_registry()
    repair_executor = CompositeRepairExecutor(
        ffmpeg_executor=FFmpegRepairExecutor(ffmpeg_path=ffmpeg_path)
    )
    acoustic_consistency_analyzer = (
        BatchAcousticConsistencyAnalyzer(audio_metrics_analyzer)
        if args.target_policy == "voice_clone" and audio_metrics_analyzer is not None
        else None
    )

    curator_policy = None
    if args.curator == "llm-placeholder":
        curator_policy = LLMCuratorAgent(
            PlaceholderCuratorClient(),
            target_policy=args.target_policy,
        )
    elif args.curator == "qwen-local":
        curator_policy = LLMCuratorAgent(
            TransformersQwenClient(
                model_path=args.curator_model_path,
                max_new_tokens=args.curator_max_new_tokens,
            ),
            fallback_policy=BaselineCurationPolicy(
                target_policy=args.target_policy,
            ),
            target_policy=args.target_policy,
        )

    processing_strategy_agent = None
    if args.strategy == "llm-placeholder":
        processing_strategy_agent = LLMProcessingStrategyAgent(
            PlaceholderStrategyClient(),
            tool_registry=tool_registry,
            fallback_agent=BaselineProcessingStrategyAgent(
                tool_registry,
                target_policy=args.target_policy,
            ),
            target_policy=args.target_policy,
            knowledge_retriever=knowledge_retriever,
        )
    elif args.strategy == "qwen-local":
        processing_strategy_agent = LLMProcessingStrategyAgent(
            TransformersQwenClient(
                model_path=args.curator_model_path,
                max_new_tokens=args.curator_max_new_tokens,
            ),
            tool_registry=tool_registry,
            fallback_agent=BaselineProcessingStrategyAgent(
                tool_registry,
                target_policy=args.target_policy,
            ),
            target_policy=args.target_policy,
            knowledge_retriever=knowledge_retriever,
        )

    report_agent = None
    base_report_agent = DatasetReportAgent(knowledge_retriever=knowledge_retriever)
    if args.report == "qwen-local":
        report_agent = LLMDatasetReportAgent(
            TransformersQwenClient(
                model_path=args.report_model_path,
                max_new_tokens=args.report_max_new_tokens,
            ),
            base_agent=base_report_agent,
        )
    elif knowledge_retriever is not None:
        report_agent = base_report_agent

    if args.orchestrator == "supervisor":
        supervisor = AudioDatasetSupervisor(
            vad_analyzer=vad_analyzer,
            transcriber=transcriber,
            normalizer=normalizer,
            sample_materializer=sample_materializer,
            audio_metrics_analyzer=audio_metrics_analyzer,
            curator_policy=curator_policy,
            repair_executor=repair_executor,
            processing_strategy_agent=processing_strategy_agent,
            report_agent=report_agent,
            tool_registry=tool_registry,
            target_policy=args.target_policy,
            run_asr=run_asr,
            run_config=run_config,
            metadata_config=metadata_config,
            acoustic_consistency_analyzer=acoustic_consistency_analyzer,
            repair_chain_mode=args.repair_chain,
        )
        if args.workflow == "repair":
            if audio_metrics_analyzer is None:
                print("错误：--workflow repair 需要同时设置 --audio-metrics wave 或 ffmpeg")
                return 2
            result = supervisor.run_repair(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                export_dir=args.export_dir,
            )
        else:
            result = supervisor.run_basic(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                export_dir=args.export_dir,
            )
        _write_rag_audit(result.batch, knowledge_retriever)
        result = SupervisorRunResult(
            batch=write_portfolio_artifacts(
                result.batch,
                build_local_portfolio_research(result.batch),
            ),
            trace=result.trace,
        )
        print(json.dumps(supervisor_run_result_to_dict(result), ensure_ascii=False, indent=2))
        return 0

    if args.workflow == "repair":
        if audio_metrics_analyzer is None:
            print("错误：--workflow repair 需要同时设置 --audio-metrics wave 或 ffmpeg")
            return 2
        exported = run_repair_enabled_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            export_dir=args.export_dir,
            vad_analyzer=vad_analyzer,
            transcriber=transcriber,
            audio_metrics_analyzer=audio_metrics_analyzer,
            normalizer=normalizer,
            sample_materializer=sample_materializer,
            curator_policy=curator_policy,
            processing_strategy_agent=processing_strategy_agent,
            report_agent=report_agent,
            tool_registry=tool_registry,
            repair_executor=repair_executor,
            target_policy=args.target_policy,
            run_asr=run_asr,
            run_config=run_config,
            metadata_config=metadata_config,
            acoustic_consistency_analyzer=acoustic_consistency_analyzer,
            repair_chain_mode=args.repair_chain,
        )
    else:
        exported = run_batch_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            export_dir=args.export_dir,
            vad_analyzer=vad_analyzer,
            transcriber=transcriber,
            normalizer=normalizer,
            sample_materializer=sample_materializer,
            audio_metrics_analyzer=audio_metrics_analyzer,
            curator_policy=curator_policy,
            processing_strategy_agent=processing_strategy_agent,
            report_agent=report_agent,
            tool_registry=tool_registry,
            target_policy=args.target_policy,
            run_asr=run_asr,
            run_config=run_config,
            metadata_config=metadata_config,
            acoustic_consistency_analyzer=acoustic_consistency_analyzer,
            repair_chain_mode=args.repair_chain,
        )

    _write_rag_audit(exported, knowledge_retriever)
    exported = write_portfolio_artifacts(
        exported,
        build_local_portfolio_research(exported),
    )
    print(json.dumps(batch_to_dict(exported), ensure_ascii=False, indent=2))
    return 0


def _build_run_config(args, ffmpeg_path: str) -> dict[str, object]:
    """把 CLI 参数整理成不含密钥的可复现配置。"""
    return {
        "target_policy": args.target_policy,
        "workflow": args.workflow,
        "orchestrator": args.orchestrator,
        "normalizer": args.normalizer,
        "sample_materializer": args.sample_materializer,
        "audio_metrics": args.audio_metrics,
        "vad": args.vad,
        "asr": args.asr,
        "strategy": args.strategy,
        "curator": args.curator,
        "report": args.report,
        "sample_rate": args.sample_rate,
        "channels": args.channels,
        "repair_chain": args.repair_chain,
        "models": {
            "asr": args.asr_model if args.asr == "faster-whisper" else None,
            "curator": args.curator_model_path,
            "report": args.report_model_path,
        },
        "tools": {"ffmpeg_path": ffmpeg_path},
        "metadata": {
            "path": args.metadata_path,
            "audio_column": args.metadata_audio_column,
            "text_column": args.metadata_text_column,
            "language_column": args.metadata_language_column,
            "speaker_column": args.metadata_speaker_column,
            "use_sidecar_text": args.use_sidecar_text,
        },
        "rag": {
            "mode": args.rag,
            "documents": args.rag_documents,
            "index": args.rag_index,
            "rebuild": args.rag_rebuild,
        },
    }


def _build_knowledge_retriever(args):
    if args.rag == "none":
        return None
    try:
        retriever = LocalKnowledgeRetriever(args.rag_documents, args.rag_index)
        if args.rag_rebuild:
            retriever.rebuild()
        return retriever
    except Exception as exc:
        print(f"警告：本地 RAG 初始化失败，已降级为无 RAG 模式：{exc}", file=sys.stderr)
        return None


def _write_rag_audit(batch, knowledge_retriever) -> None:
    if knowledge_retriever is None:
        return
    report_path = batch.export_paths.get("report")
    if not report_path:
        return
    knowledge_retriever.write_audit_files(Path(report_path).parent)


if __name__ == "__main__":
    raise SystemExit(main())
