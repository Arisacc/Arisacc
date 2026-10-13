"""导出服务：写出 manifest、状态文件、报告和复核队列。"""
import json
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from audio_dataset_agent.core.models import Batch, BatchPhase
from audio_dataset_agent.core.serialization import (
    analysis_result_to_dict,
    candidate_sample_to_dict,
    curator_decision_to_dict,
    final_selection_to_dict,
    processing_strategy_to_dict,
    raw_asset_to_dict,
    repair_evaluation_to_dict,
    repair_execution_to_dict,
    repair_plan_to_dict,
)
from audio_dataset_agent.exporters.manifest import build_manifest_records
from audio_dataset_agent.exporters.provenance import write_provenance_artifacts
from audio_dataset_agent.core.run_context import RunContext
from audio_dataset_agent.reporting.agent import DatasetReportAgent
from audio_dataset_agent.reporting.agent import DatasetQualityReport
from audio_dataset_agent.reporting.agent import write_dataset_quality_report


class ReportAgent(Protocol):
    def build(self, batch: Batch) -> DatasetQualityReport:
        """为一个已导出的 batch 生成数据集质量报告。"""


def export_curated_batch(
    batch: Batch,
    export_root: Path | str,
    report_agent: ReportAgent | None = None,
) -> Batch:
    """写出已策展 batch 的产物，并返回导出后的 batch 状态。"""
    # 导出阶段只消费现有状态，不重新分析或修改音频文件。
    # 每个 batch 单独一个导出目录，避免多次运行互相覆盖。
    batch_export_dir = RunContext.from_batch(batch).batch_export_dir(export_root)
    batch_export_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = batch_export_dir / "manifest.jsonl"
    report_path = batch_export_dir / "batch_report.md"
    state_dir = batch_export_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # manifest 是给下游训练/复核消费的主产物，每行对应一个最终样本记录。
    records = build_manifest_records(batch)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # state 文件保留每个中间阶段，方便解释 Agent 为什么做出某个判断。
    state_files = _write_state_files(batch, state_dir)
    report_path.write_text(_build_batch_report(batch, records), encoding="utf-8")
    active_report_agent = report_agent or DatasetReportAgent()
    quality_report = active_report_agent.build(batch)
    quality_report_paths = write_dataset_quality_report(
        quality_report,
        batch_export_dir,
    )
    review_queue_paths = _write_review_queue_files(
        quality_report.review_queue,
        batch_export_dir,
    )
    provenance_paths = write_provenance_artifacts(batch, batch_export_dir)
    metadata_report_path = batch_export_dir / "metadata_import_report.json"
    metadata_report_path.write_text(
        json.dumps(batch.metadata_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return replace(
        batch,
        phase=BatchPhase.EXPORTED,
        updated_at=datetime.now(timezone.utc),
        export_paths={
            "manifest": str(manifest_path),
            "report": str(report_path),
            "dataset_quality_json": quality_report_paths["json"],
            "dataset_quality_report": quality_report_paths["markdown"],
            "review_queue_json": review_queue_paths["json"],
            "review_queue_csv": review_queue_paths["csv"],
            "state_dir": str(state_dir),
            "state_files": state_files,
            **provenance_paths,
            "metadata_import_report": str(metadata_report_path),
        },
    )


def _write_state_files(batch: Batch, state_dir: Path) -> dict[str, str]:
    # 这些 JSON 是前端、作品集报告和调试排查的共同事实来源。
    state_payloads = {
        "raw_assets": [raw_asset_to_dict(asset) for asset in batch.raw_assets],
        "candidate_samples": [
            candidate_sample_to_dict(sample) for sample in batch.candidate_samples
        ],
        "analysis_results": [
            analysis_result_to_dict(result) for result in batch.analysis_results
        ],
        "curator_decisions": [
            curator_decision_to_dict(decision)
            for decision in batch.curator_decisions
        ],
        "processing_strategies": [
            processing_strategy_to_dict(strategy)
            for strategy in batch.processing_strategies
        ],
        "repair_plans": [repair_plan_to_dict(plan) for plan in batch.repair_plans],
        "repair_executions": [
            repair_execution_to_dict(execution)
            for execution in batch.repair_executions
        ],
        "repair_evaluations": [
            repair_evaluation_to_dict(evaluation)
            for evaluation in batch.repair_evaluations
        ],
        "final_selections": [
            final_selection_to_dict(selection)
            for selection in batch.final_selections
        ],
    }

    state_files = {}
    for name, payload in state_payloads.items():
        path = state_dir / f"{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state_files[name] = str(path)

    return state_files


def _write_review_queue_files(
    review_queue: list[dict],
    output_dir: Path,
) -> dict[str, str]:
    json_path = output_dir / "review_queue.json"
    csv_path = output_dir / "review_queue.csv"
    json_path.write_text(
        json.dumps(review_queue, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = ["sample_id", "recommended_action", "issue_tags", "confidence", "reason"]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in review_queue:
            writer.writerow(
                {
                    "sample_id": item.get("sample_id", ""),
                    "recommended_action": item.get("recommended_action", ""),
                    "issue_tags": ";".join(str(tag) for tag in item.get("issue_tags", [])),
                    "confidence": item.get("confidence", ""),
                    "reason": item.get("reason", ""),
                }
            )
    return {"json": str(json_path), "csv": str(csv_path)}


def _build_batch_report(batch: Batch, records: list[dict]) -> str:
    decision_counts = _count_field(records, "decision")
    final_selection_counts = _count_field(records, "final_status")
    issue_counts = _count_issues(records)

    report_lines = [
        f"# 批次处理报告：{batch.batch_id}",
        "",
        "## 概览",
        "",
        f"- 数据用途：{batch.target_policy}",
        f"- 原始音频数：{batch.raw_asset_count}",
        f"- 候选样本数：{batch.candidate_sample_count}",
        f"- 分析结果数：{batch.analysis_result_count}",
        f"- 策略建议数：{batch.processing_strategy_count}",
        f"- 样本审核决策数：{batch.curator_decision_count}",
        f"- 修复计划数：{batch.repair_plan_count}",
        f"- 修复执行数：{batch.repair_execution_count}",
        f"- 修复评估数：{batch.repair_evaluation_count}",
        f"- 最终选择数：{batch.final_selection_count}",
        "",
        "## 决策统计",
        "",
        *_format_count_lines(decision_counts),
        "",
        "## 最终选择统计",
        "",
        *_format_count_lines(final_selection_counts),
        "",
        "## 问题标签统计",
        "",
        *_format_count_lines(issue_counts),
        "",
        "## 样本明细",
        "",
        "| 样本 ID | 决策 | 最终状态 | 时长 | ASR 置信度 | 问题标签 | 文本 |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
        *[_format_sample_row(record) for record in records],
        "",
    ]
    return "\n".join(report_lines)


def _count_field(records: list[dict], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(field_name) or "unknown"
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _count_issues(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        issues = record.get("issue_tags") or []
        if not issues:
            counts["none"] = counts.get("none", 0) + 1
            continue
        for issue in issues:
            counts[str(issue)] = counts.get(str(issue), 0) + 1
    return counts


def _format_count_lines(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- 无：0"]
    return [f"- {key}: {counts[key]}" for key in sorted(counts)]


def _format_sample_row(record: dict) -> str:
    issues = ", ".join(record.get("issue_tags") or []) or "无"
    confidence = record.get("asr_confidence")
    confidence_text = f"{confidence:.3f}" if isinstance(confidence, float) else ""
    duration = record.get("duration")
    duration_text = f"{duration:.2f}" if isinstance(duration, float) else ""
    text = _escape_markdown_cell(str(record.get("text") or ""))
    return (
        f"| {record.get('sample_id')} "
        f"| {record.get('decision') or 'unknown'} "
        f"| {record.get('final_status') or 'unknown'} "
        f"| {duration_text} "
        f"| {confidence_text} "
        f"| {_escape_markdown_cell(issues)} "
        f"| {text} |"
    )


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
