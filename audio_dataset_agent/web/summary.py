# 前端摘要构建层：把导出产物压缩成 dashboard 可消费的 summary.json。
import json
from pathlib import Path
from typing import Any

from audio_dataset_agent.repair.presets import get_repair_preset
from audio_dataset_agent.inspection.policy import InspectionPolicy


def build_summary_from_export_dir(export_dir: Path | str) -> dict[str, Any]:
    """从单个批次导出目录生成前端可消费的摘要 JSON。"""
    root = Path(export_dir)
    quality = _read_json(root / "dataset_quality_report.json", default={})
    # 人工复核产生的 manifest 是当前批次的最新事实源；没有人工覆盖时才使用原始 manifest。
    manifest_path = root / "manual_review_manifest.jsonl"
    has_manual_manifest = manifest_path.exists()
    if not has_manual_manifest:
        manifest_path = root / "manifest.jsonl"
    manifest = _read_jsonl(manifest_path)
    strategies = _read_json(
        root / "state" / "processing_strategies.json",
        default=[],
    )
    curator_decisions = _read_json(
        root / "state" / "curator_decisions.json",
        default=[],
    )
    graph_trace = _normalize_graph_trace(
        _read_json(root / "agent_graph_trace.json", default=[])
    )
    portfolio_research = _read_json(root / "portfolio_research.json", default={})
    rag_retrievals = _read_jsonl(root / "state" / "rag_retrievals.jsonl")
    run_config = _read_json(root / "run_config.json", default={})

    strategy_by_sample = {
        item.get("sample_id"): item
        for item in strategies
        if isinstance(item, dict) and item.get("sample_id")
    }
    curator_by_sample = {
        item.get("sample_id"): item
        for item in curator_decisions
        if isinstance(item, dict) and item.get("sample_id")
    }

    samples = []
    for record in manifest:
        strategy = strategy_by_sample.get(record.get("sample_id"), {})
        curator = curator_by_sample.get(record.get("sample_id"), {})
        samples.append(
            {
                "sample_id": record.get("sample_id", ""),
                "duration": float(record.get("duration") or 0.0),
                "transcript": record.get("text") or "",
                "transcript_source": record.get("transcript_source") or "none",
                "reference_text": record.get("reference_text") or "",
                "reference_text_source": record.get("reference_text_source") or "none",
                "language": record.get("language"),
                "speaker_id": record.get("speaker_id"),
                "metadata_status": record.get("metadata_status") or "missing",
                "text_alignment": record.get("text_alignment"),
                "speaker_consistency": record.get("speaker_consistency"),
                "asr_confidence": record.get("asr_confidence"),
                "snr": record.get("snr"),
                "quality_score": record.get("quality_score"),
                "issue_tags": record.get("issue_tags") or [],
                "strategy": strategy.get("recommended_operation")
                or strategy.get("action")
                or "unknown",
                "strategy_reason": strategy.get("reason") or "",
                "curator_decision": record.get("decision") or "unknown",
                "curator_reason": record.get("decision_reason") or "",
                "strategy_consistency": curator.get("strategy_consistency") or "not_applicable",
                "final_status": record.get("final_status") or "unknown",
                "final_audio_path": record.get("final_audio_path"),
                "manual_archive_group": record.get("manual_archive_group"),
                "manual_archive_path": record.get("manual_archive_path"),
                "source_audio_path": record.get("source_audio_path"),
                "work_audio_path": record.get("work_audio_path"),
                "sample_audio_path": record.get("sample_audio_path"),
                "repair_plan": record.get("repair_plan"),
                "repair_chain_id": record.get("repair_chain_id"),
                "repair_chain": record.get("repair_chain") or [],
                "repair_execution": record.get("repair_execution"),
                "repair_evaluation": record.get("repair_evaluation"),
            }
        )

    total_samples = int(quality.get("total_samples") or len(samples))
    usable_samples = int(quality.get("usable_samples") or _count_status(samples, "selected"))
    review_samples = int(quality.get("review_samples") or _count_exact(samples, "needs_review"))
    rejected_samples = int(quality.get("rejected_samples") or _count_exact(samples, "excluded"))
    if has_manual_manifest:
        total_samples = len(samples)
        usable_samples = sum(sample.get("final_status") in {"selected", "manual_keep"} for sample in samples)
        review_samples = sum(sample.get("final_status") in {"needs_review", "manual_review"} for sample in samples)
        rejected_samples = sum(sample.get("final_status") in {"excluded", "manual_reject"} for sample in samples)

    target_policy = quality.get("target_policy") or "voice_clone"
    repair_preset = get_repair_preset(target_policy)
    quality_policy = InspectionPolicy.from_profile(target_policy)

    return {
        "qualityScore": quality.get("qualityScore", quality.get("quality_score")),
        "batch": {
            "batch_id": quality.get("batch_id") or root.name,
            "target_policy": target_policy,
            "total_samples": total_samples,
            "usable_samples": usable_samples,
            "review_samples": review_samples,
            "rejected_samples": rejected_samples,
            "export_dir": str(root),
        },
        "funnel": [
            {"label": "\u539f\u59cb\u97f3\u9891", "value": _read_raw_asset_count(root)},
            {"label": "\u5019\u9009\u6837\u672c", "value": total_samples},
            {"label": "\u53ef\u7528\u6837\u672c", "value": usable_samples},
            {"label": "\u5f85\u590d\u6838", "value": review_samples},
            {"label": "\u5df2\u62d2\u7edd", "value": rejected_samples},
        ],
        "agents": _build_agent_cards(),
        "tools": _build_tool_cards(),
        "strategy_collaboration": quality.get("strategy_collaboration", {}),
        "quality_policy": quality_policy.to_dict(),
        "text_quality_summary": quality.get("text_quality_summary", {}),
        "rag": {
            "enabled": bool(rag_retrievals or quality.get("knowledge_references")),
            "retrieval_count": len(rag_retrievals),
            "references": quality.get("knowledge_references", []),
        },
        "run_config": run_config,
        "repair_preset": {
            "target_lufs": repair_preset.target_lufs,
            "silence_threshold_dbfs": repair_preset.silence_threshold_dbfs,
            "min_silence_duration_ms": repair_preset.min_silence_duration_ms,
            "noise_reduction_db": repair_preset.noise_reduction_db,
        },
        "samples": samples,
        "graph_trace": graph_trace,
        "portfolio_research": {
            "status": portfolio_research.get("status", "disabled"),
            "message": portfolio_research.get("message", ""),
            "problem_explanation": portfolio_research.get("problem_explanation", ""),
            "audio_processing_background": portfolio_research.get(
                "audio_processing_background",
                [],
            ),
            "tool_notes": portfolio_research.get("tool_notes", []),
            "external_references": portfolio_research.get("external_references", []),
            "local_references": portfolio_research.get("local_references", []),
            "limitations": portfolio_research.get("limitations", []),
        },
        "report_paths": {
            "batch_report": str(root / "batch_report.md"),
            "dataset_quality_report": str(root / "dataset_quality_report.md"),
            "portfolio_report_html": str(root / "portfolio_report.html"),
            "portfolio_report_markdown": str(root / "portfolio_report.md"),
            "run_config": str(root / "run_config.json"),
            "source_manifest": str(root / "source_manifest.json"),
            "tool_versions": str(root / "tool_versions.json"),
            "metadata_import_report": str(root / "metadata_import_report.json"),
        },
    }


def write_summary_file(export_dir: Path | str) -> Path:
    """Write summary.json for frontend/API consumption."""
    root = Path(export_dir)
    summary = build_summary_from_export_dir(root)
    output_path = root / "summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _normalize_graph_trace(value) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        nodes = value.get("nodes", [])
        if isinstance(nodes, list):
            return [item for item in nodes if isinstance(item, dict)]
    return []


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _read_raw_asset_count(root: Path) -> int:
    raw_assets = _read_json(root / "state" / "raw_assets.json", default=[])
    return len(raw_assets) if isinstance(raw_assets, list) else 0


def _count_status(samples: list[dict[str, Any]], prefix: str) -> int:
    return sum(1 for sample in samples if str(sample.get("final_status", "")).startswith(prefix))


def _count_exact(samples: list[dict[str, Any]], status: str) -> int:
    return sum(1 for sample in samples if sample.get("final_status") == status)


def _build_agent_cards() -> list[dict[str, str]]:
    return [
        {
            "name": "策略智能体",
            "role": "\u6839\u636e Inspector \u6807\u7b7e\u3001\u8bc1\u636e\u548c\u5de5\u5177\u6ce8\u518c\u8868\u9009\u62e9\u5904\u7406\u7b56\u7565\u3002",
            "input": "问题标签 / 证据 / 工具注册表 / 数据用途画像",
            "output": "处理策略建议",
            "decision_power": "\u9009\u62e9\u5904\u7406\u65b9\u5411\u548c\u5019\u9009\u5de5\u5177\uff0c\u4e0d\u76f4\u63a5\u4fee\u6539\u97f3\u9891\u6587\u4ef6\u3002",
        },
        {
            "name": "样本审核智能体",
            "role": "\u7ed3\u5408\u8d28\u91cf\u8bc1\u636e\u548c\u5904\u7406\u7b56\u7565\u5224\u65ad\u6837\u672c\u53bb\u7559\u3002",
            "input": "分析结果 / 处理策略建议 / 用途约束",
            "output": "保留 / 拒绝 / 修复 / 复核",
            "decision_power": "\u8d1f\u8d23\u6838\u5fc3\u6837\u672c\u7ea7\u51b3\u7b56\u3002",
        },
        {
            "name": "报告智能体",
            "role": "\u6c47\u603b\u8d28\u91cf\u7edf\u8ba1\u3001\u590d\u6838\u961f\u5217\u548c\u6570\u636e\u96c6\u5efa\u8bae\u3002",
            "input": "样本清单 / 最终选择 / 修复评估",
            "output": "数据集质量报告",
            "decision_power": "\u89e3\u91ca\u548c\u603b\u7ed3\u4e8b\u5b9e\uff0c\u4e0d\u8986\u76d6\u6837\u672c\u51b3\u7b56\u3002",
        },
        {
            "name": "本地作品集报告",
            "role": "基于本地事实、质量统计和知识库生成展示说明。",
            "input": "质量报告 / 本地知识检索结果",
            "output": "本地 Markdown / HTML 报告",
            "decision_power": "解释处理事实，不参与 keep/reject/repair/review。",
        },
    ]


def _build_tool_cards() -> list[dict[str, str]]:
    return [
        {
            "name": "Silero VAD / Wave VAD",
            "type": "VAD",
            "purpose": "\u68c0\u6d4b\u8bed\u97f3\u6d3b\u52a8\u533a\u57df\uff0c\u7528\u4e8e\u5207\u5206\u5019\u9009\u6837\u672c\u548c\u8fc7\u6ee4\u65e0\u4fe1\u606f\u7247\u6bb5\u3002",
        },
        {
            "name": "faster-whisper",
            "type": "ASR",
            "purpose": "\u751f\u6210\u5019\u9009\u6587\u672c\u548c ASR \u7f6e\u4fe1\u5ea6\uff0c\u670d\u52a1\u4e8e TTS/ASR \u6570\u636e\u8d28\u91cf\u5224\u65ad\u3002",
        },
        {
            "name": "DeepFilterNet / FFmpeg",
            "type": "修复",
            "purpose": "\u5bf9\u53ef\u4fee\u590d\u6837\u672c\u6267\u884c\u964d\u566a\u3001\u54cd\u5ea6\u6216\u9759\u97f3\u76f8\u5173\u5904\u7406\u3002",
        },
        {
            "name": "FFmpeg",
            "type": "预处理",
            "purpose": "\u7edf\u4e00\u683c\u5f0f\u3001\u91c7\u6837\u7387\u3001\u58f0\u9053\u5e76\u751f\u6210\u53ef\u5206\u6790\u7684\u5de5\u4f5c\u97f3\u9891\u3002",
        },
    ]
