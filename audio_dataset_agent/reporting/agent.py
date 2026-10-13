# 本地报告 Agent：生成数据质量统计、风险说明和建议。
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from audio_dataset_agent.core.models import Batch, DecisionType
from audio_dataset_agent.core.llm_json import parse_json_object
from audio_dataset_agent.profiles.policy import get_dataset_profile
from audio_dataset_agent.knowledge.context import build_knowledge_context


@dataclass(frozen=True)
class DatasetQualityReport:
    batch_id: str
    target_policy: str
    total_samples: int
    usable_samples: int
    review_samples: int
    rejected_samples: int
    quality_score: float | None = None
    issue_counts: dict[str, int] = field(default_factory=dict)
    decision_counts: dict[str, int] = field(default_factory=dict)
    final_status_counts: dict[str, int] = field(default_factory=dict)
    repair_summary: dict[str, int] = field(default_factory=dict)
    strategy_collaboration: dict[str, int] = field(default_factory=dict)
    text_quality_summary: dict[str, float | int] = field(default_factory=dict)
    knowledge_context: str = ""
    knowledge_references: list[dict[str, Any]] = field(default_factory=list)
    review_queue: list[dict[str, Any]] = field(default_factory=list)
    report_focus: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    agent_name: str = "dataset_report_agent"

    @property
    def usable_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.usable_samples / self.total_samples


class DatasetReportAgent:
    """将一个已处理 batch 汇总为数据集质量证据。"""

    agent_name = "dataset_report_agent"

    def __init__(self, knowledge_retriever=None) -> None:
        self.knowledge_retriever = knowledge_retriever

    def build(self, batch: Batch) -> DatasetQualityReport:
        profile = get_dataset_profile(batch.target_policy)
        analysis_by_sample = {
            result.sample_id: result for result in batch.analysis_results
        }
        decision_counts = _count_values(
            decision.decision.value for decision in batch.curator_decisions
        )
        final_status_counts = _count_values(
            selection.status for selection in batch.final_selections
        )
        issue_counts = self._count_issues(batch)
        repair_summary = self._summarize_repairs(batch)
        strategy_collaboration = _count_values(
            decision.strategy_consistency
            for decision in batch.curator_decisions
        )
        text_quality_summary = _summarize_text_quality(batch)
        quality_score = _summarize_quality_score(batch)
        knowledge_context = build_knowledge_context([], max_chars=4000)
        if self.knowledge_retriever is not None:
            query = " ".join(
                [
                    batch.target_policy,
                    " ".join(issue_counts.keys()),
                    "语音数据集质量工具处理复核规范",
                ]
            )
            knowledge_context = build_knowledge_context(
                self.knowledge_retriever.search(
                    query,
                    profile=batch.target_policy,
                    top_k=3,
                    agent_name="report_agent",
                ),
                max_chars=4000,
            )
        review_queue = self._build_review_queue(batch, analysis_by_sample)

        usable_samples = sum(
            final_status_counts.get(status, 0)
            for status in ("selected_original", "selected_repaired")
        )
        review_samples = (
            decision_counts.get(DecisionType.REVIEW.value, 0)
            + final_status_counts.get("needs_review", 0)
        )
        if decision_counts.get(DecisionType.REVIEW.value, 0) and final_status_counts.get(
            "needs_review", 0
        ):
            review_samples = max(
                decision_counts.get(DecisionType.REVIEW.value, 0),
                final_status_counts.get("needs_review", 0),
            )
        rejected_samples = (
            decision_counts.get(DecisionType.REJECT.value, 0)
            + final_status_counts.get("rejected", 0)
        )
        if decision_counts.get(DecisionType.REJECT.value, 0) and final_status_counts.get(
            "rejected", 0
        ):
            rejected_samples = max(
                decision_counts.get(DecisionType.REJECT.value, 0),
                final_status_counts.get("rejected", 0),
            )

        report_without_recommendations = DatasetQualityReport(
            batch_id=batch.batch_id,
            target_policy=batch.target_policy,
            total_samples=batch.candidate_sample_count,
            usable_samples=usable_samples,
            review_samples=review_samples,
            rejected_samples=rejected_samples,
            quality_score=quality_score,
            issue_counts=issue_counts,
            decision_counts=decision_counts,
            final_status_counts=final_status_counts,
            repair_summary=repair_summary,
            strategy_collaboration=strategy_collaboration,
            text_quality_summary=text_quality_summary,
            knowledge_context=knowledge_context.text,
            knowledge_references=knowledge_context.references,
            review_queue=review_queue,
            report_focus=list(profile.report_focus),
            risk_notes=list(profile.risk_notes),
            recommendations=[],
            agent_name=self.agent_name,
        )
        return DatasetQualityReport(
            batch_id=report_without_recommendations.batch_id,
            target_policy=report_without_recommendations.target_policy,
            total_samples=report_without_recommendations.total_samples,
            usable_samples=report_without_recommendations.usable_samples,
            review_samples=report_without_recommendations.review_samples,
            rejected_samples=report_without_recommendations.rejected_samples,
            quality_score=report_without_recommendations.quality_score,
            issue_counts=report_without_recommendations.issue_counts,
            decision_counts=report_without_recommendations.decision_counts,
            final_status_counts=report_without_recommendations.final_status_counts,
            repair_summary=report_without_recommendations.repair_summary,
            strategy_collaboration=report_without_recommendations.strategy_collaboration,
            text_quality_summary=report_without_recommendations.text_quality_summary,
            knowledge_context=report_without_recommendations.knowledge_context,
            knowledge_references=report_without_recommendations.knowledge_references,
            review_queue=report_without_recommendations.review_queue,
            report_focus=report_without_recommendations.report_focus,
            risk_notes=report_without_recommendations.risk_notes,
            recommendations=self._recommend(report_without_recommendations),
            agent_name=report_without_recommendations.agent_name,
        )

    def _count_issues(self, batch: Batch) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in batch.analysis_results:
            if not result.issue_tags:
                counts["none"] = counts.get("none", 0) + 1
                continue
            for tag in result.issue_tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def _summarize_repairs(self, batch: Batch) -> dict[str, int]:
        evaluated = len(batch.repair_evaluations)
        improved = sum(1 for evaluation in batch.repair_evaluations if evaluation.improved)
        not_improved = evaluated - improved
        adopted = sum(
            1
            for evaluation in batch.repair_evaluations
            if evaluation.recommendation == "adopt_repaired"
        )
        return {
            "evaluated": evaluated,
            "improved": improved,
            "not_improved": not_improved,
            "adopted": adopted,
        }

    def _build_review_queue(
        self,
        batch: Batch,
        analysis_by_sample: dict[str, object],
    ) -> list[dict[str, Any]]:
        queue = []
        queued_sample_ids = set()
        for decision in batch.curator_decisions:
            if decision.decision != DecisionType.REVIEW:
                continue
            analysis = analysis_by_sample.get(decision.sample_id)
            queue.append(
                {
                    "sample_id": decision.sample_id,
                    "recommended_action": decision.recommended_action,
                    "issue_tags": list(getattr(analysis, "issue_tags", [])),
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                }
            )
            queued_sample_ids.add(decision.sample_id)

        decisions_by_sample = {
            decision.sample_id: decision for decision in batch.curator_decisions
        }
        for selection in batch.final_selections:
            if selection.status != "needs_review":
                continue
            if selection.sample_id in queued_sample_ids:
                continue
            analysis = analysis_by_sample.get(selection.sample_id)
            decision = decisions_by_sample.get(selection.sample_id)
            queue.append(
                {
                    "sample_id": selection.sample_id,
                    "recommended_action": getattr(
                        decision,
                        "recommended_action",
                        "manual_decision_review",
                    ),
                    "issue_tags": list(getattr(analysis, "issue_tags", [])),
                    "reason": selection.reason,
                    "confidence": getattr(decision, "confidence", 0.0),
                }
            )
            queued_sample_ids.add(selection.sample_id)
        return queue

    def _recommend(self, report: DatasetQualityReport) -> list[str]:
        recommendations = []
        if report.review_samples:
            if report.target_policy == "voice_clone":
                recommendations.append(
                    f"声音克隆任务有 {report.review_samples} 条样本需要复核，优先检查噪声、削波和音色稳定性风险。"
                )
            else:
                recommendations.append(
                    f"优先复核 {report.review_samples} 条样本，尤其关注文本可靠性和音频内容是否匹配。"
                )
        if report.issue_counts.get("audio_noise_high", 0):
            if report.target_policy == "voice_clone":
                recommendations.append(
                    "声音克隆任务中出现噪声问题，降噪后需要复评是否影响说话人音色。"
                )
            else:
                recommendations.append(
                    "噪声问题已经出现，建议保留 FFmpeg 或 DeepFilterNet 处理后的复评记录。"
                )
        if report.repair_summary.get("evaluated", 0) and not report.repair_summary.get(
            "improved", 0
        ):
            recommendations.append(
                "当前修复没有带来可观测改善，建议降低自动修复置信度或转入人工复核。"
            )
        overridden = report.strategy_collaboration.get("overridden", 0)
        if overridden:
            recommendations.append(
                f"Curator 覆盖了 {overridden} 条 Strategy 建议，建议复核覆盖原因并评估策略提示词或规则是否需要调整。"
            )
        if not recommendations:
            recommendations.append(
                f"{_target_policy_label(report.target_policy)} 批次质量较稳定，可以进入导出或扩大样本验证。"
            )
        return recommendations


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str | dict[str, Any]:
        """返回包含报告建议的 JSON 字符串或类 JSON 字典。"""


class LLMDatasetReportAgent:
    """使用 LLM 根据报告事实生成面向人的建议。"""

    agent_name = "llm_dataset_report_agent"

    def __init__(
        self,
        llm_client: LLMClient,
        base_agent: DatasetReportAgent | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.base_agent = base_agent or DatasetReportAgent()

    def build(self, batch: Batch) -> DatasetQualityReport:
        base_report = self.base_agent.build(batch)
        prompt = self._build_prompt(base_report)
        try:
            payload = parse_json_object(self.llm_client.complete(prompt))
            recommendations = self._recommendations_from_payload(payload)
            return replace(
                base_report,
                recommendations=recommendations,
                agent_name=self.agent_name,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return replace(
                base_report,
                agent_name="llm_dataset_report_agent_guarded",
            )

    def _build_prompt(self, report: DatasetQualityReport) -> str:
        context = dataset_quality_report_to_dict(report)
        return "\n".join(
            [
                "你是音频数据集准备流水线中的报告 Agent。",
                "下面是数据集质量报告上下文:",
                json.dumps(context, ensure_ascii=False),
                "请为数据集处理人员写出简洁、可执行的中文建议。",
                "重点关注人工复核优先级、修复风险，以及当前批次是否适合导出。",
                "请只返回一个 JSON 对象，唯一顶层字段是 recommendations。",
                'Schema: {"recommendations": ["可执行建议"]}',
            ]
        )

    def _recommendations_from_payload(self, payload: dict[str, Any]) -> list[str]:
        recommendations = payload.get("recommendations")
        if not isinstance(recommendations, list):
            raise TypeError("recommendations must be a list")
        cleaned = [str(item).strip() for item in recommendations if str(item).strip()]
        if not cleaned:
            raise ValueError("recommendations must not be empty")
        return cleaned


def dataset_quality_report_to_dict(
    report: DatasetQualityReport,
) -> dict[str, Any]:
    return {
        "batch_id": report.batch_id,
        "target_policy": report.target_policy,
        "total_samples": report.total_samples,
        "usable_samples": report.usable_samples,
        "usable_rate": report.usable_rate,
        "review_samples": report.review_samples,
        "rejected_samples": report.rejected_samples,
        "qualityScore": report.quality_score,
        "issue_counts": dict(report.issue_counts),
        "decision_counts": dict(report.decision_counts),
        "final_status_counts": dict(report.final_status_counts),
        "repair_summary": dict(report.repair_summary),
        "strategy_collaboration": dict(report.strategy_collaboration),
        "text_quality_summary": dict(report.text_quality_summary),
        "knowledge_context": report.knowledge_context,
        "knowledge_references": list(report.knowledge_references),
        "review_queue": list(report.review_queue),
        "report_focus": list(report.report_focus),
        "risk_notes": list(report.risk_notes),
        "recommendations": list(report.recommendations),
        "agent_name": report.agent_name,
    }


def render_dataset_quality_markdown(report: DatasetQualityReport) -> str:
    lines = [
        "# 数据集质量报告",
        "",
        "## 概览",
        "",
        f"- 批次 ID: {report.batch_id}",
        f"- 数据目标: {_target_policy_label(report.target_policy)}",
        f"- 样本总数: {report.total_samples}",
        f"- 可用样本: {report.usable_samples}",
        f"- 可用率: {report.usable_rate:.2%}",
        f"- 待复核样本: {report.review_samples}",
        f"- 已拒绝样本: {report.rejected_samples}",
        f"- 平均质量分: {_format_quality_score(report.quality_score)}",
        "",
        "## 任务关注点",
        "",
        *[f"- {_profile_focus_label(item)}" for item in report.report_focus],
        "",
        "## 任务风险提示",
        "",
        *[f"- {_risk_note_label(item)}" for item in report.risk_notes],
        "",
        "## 问题标签统计",
        "",
        *_format_count_lines(report.issue_counts, labeler=_issue_label),
        "",
        "## 决策统计",
        "",
        *_format_count_lines(report.decision_counts, labeler=_decision_label),
        "",
        "## 修复评估概览",
        "",
        *_format_count_lines(report.repair_summary, labeler=_repair_summary_label),
        "",
        "## 文本质量概览",
        "",
        *[
            f"- {_text_quality_label(key)}: {value:.3f}"
            if isinstance(value, float)
            else f"- {_text_quality_label(key)}: {value}"
            for key, value in report.text_quality_summary.items()
        ],
        "",
        "## Strategy 与 Curator 协作统计",
        "",
        *_format_count_lines(
            report.strategy_collaboration,
            labeler=_strategy_consistency_label,
        ),
        "",
        "## 人工复核队列",
        "",
        "| 样本 ID | 建议动作 | 问题标签 | 置信度 | 原因 |",
        "| --- | --- | --- | ---: | --- |",
        *[_format_review_row(item) for item in report.review_queue],
        "",
        "## 建议",
        "",
        *[f"- {item}" for item in report.recommendations],
        "",
    ]
    return "\n".join(lines)


def write_dataset_quality_report(
    report: DatasetQualityReport,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_quality_report.json"
    markdown_path = output_dir / "dataset_quality_report.md"

    json_path.write_text(
        json.dumps(dataset_quality_report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_dataset_quality_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _count_values(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _format_count_lines(
    counts: dict[str, int],
    labeler=None,
) -> list[str]:
    if not counts:
        return ["- 无: 0"]
    labeler = labeler or (lambda value: value)
    return [f"- {labeler(key)}: {counts[key]}" for key in sorted(counts)]


def _format_review_row(item: dict[str, Any]) -> str:
    issues = ", ".join(_issue_label(tag) for tag in item.get("issue_tags") or []) or "无"
    confidence = item.get("confidence")
    confidence_text = f"{confidence:.3f}" if isinstance(confidence, float) else ""
    return (
        f"| {item.get('sample_id')} "
        f"| {_action_label(str(item.get('recommended_action') or ''))} "
        f"| {_escape_markdown_cell(issues)} "
        f"| {confidence_text} "
        f"| {_escape_markdown_cell(str(item.get('reason') or ''))} |"
    )


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _target_policy_label(value: str) -> str:
    labels = {
        "asr": "ASR 语音识别",
        "tts": "TTS 语音合成",
        "voice_clone": "声音克隆",
    }
    return labels.get(value, value)


def _profile_focus_label(value: str) -> str:
    labels = {
        "text_reliability": "文本可靠性",
        "recognizable_speech": "语音可识别性",
        "noise_distribution": "噪声分布",
        "text_audio_pairing": "文本与音频匹配",
        "audio_cleanliness": "音频干净度",
        "silence": "静音比例",
        "loudness": "响度稳定性",
        "speaker_timbre": "说话人音色稳定性",
        "noise": "噪声水平",
        "clipping": "削波/爆音风险",
    }
    return labels.get(value, value)


def _risk_note_label(value: str) -> str:
    labels = {
        "Low confidence transcripts should enter human review.": "低置信度转写应进入人工复核。",
        "Moderate noise can be acceptable when speech remains recognizable.": "当语音仍可识别时，ASR 数据可以容忍一定噪声。",
        "Text/audio mismatch can poison supervised TTS training.": "文本与音频不匹配会影响监督式 TTS 训练质量。",
        "Noise, clipping, and excessive silence should be fixed or reviewed.": "噪声、削波和过长静音应被修复或复核。",
        "Text is optional for pure voice cloning data preparation.": "纯声音克隆数据准备中，文本不是硬性要求。",
        "Noise reduction may affect speaker timbre and should be evaluated.": "降噪可能影响说话人音色，需要复评。",
    }
    return labels.get(value, value)


def _issue_label(value: str) -> str:
    labels = {
        "none": "无问题",
        "too_short": "时长过短",
        "missing_transcript": "缺少文本",
        "asr_low_confidence": "ASR 置信度低",
        "missing_reference_text": "缺少人工参考文本",
        "reference_alignment_required": "整段文本需要片段级对齐",
        "text_alignment_review": "文本一致性需要复核",
        "text_mismatch": "参考文本与 ASR 文本不匹配",
        "text_invalid": "文本规范化后无有效内容",
        "speaker_consistency_warning": "批次声学一致性提醒",
        "audio_clipping": "音频削波",
        "audio_too_quiet": "音量过低",
        "audio_high_silence_ratio": "静音比例过高",
        "audio_noise_high": "噪声偏高",
    }
    return labels.get(value, value)


def _decision_label(value: str) -> str:
    labels = {
        "keep": "保留",
        "reject": "拒绝",
        "repair": "修复",
        "review": "复核",
    }
    return labels.get(value, value)


def _repair_summary_label(value: str) -> str:
    labels = {
        "evaluated": "已评估",
        "improved": "有改善",
        "not_improved": "无改善",
        "adopted": "已采用修复版本",
    }
    return labels.get(value, value)


def _strategy_consistency_label(value: str) -> str:
    labels = {
        "accepted": "Curator 接受 Strategy 建议",
        "overridden": "Curator 覆盖 Strategy 建议",
        "not_applicable": "无可用 Strategy 建议",
    }
    return labels.get(value, value)


def _text_quality_label(value: str) -> str:
    return {
        "compared_samples": "参与文本比对样本",
        "average_cer": "平均 CER",
        "average_wer": "平均 WER",
        "pass_samples": "文本一致性通过",
        "review_samples": "文本一致性复核",
        "mismatch_samples": "文本不匹配",
    }.get(value, value)


def _summarize_text_quality(batch: Batch) -> dict[str, float | int]:
    alignments = []
    for result in batch.analysis_results:
        value = result.evidence.get("text_alignment")
        if isinstance(value, dict) and isinstance(value.get("cer"), (int, float)):
            alignments.append(value)
    if not alignments:
        return {
            "compared_samples": 0,
            "average_cer": 0.0,
            "average_wer": 0.0,
            "pass_samples": 0,
            "review_samples": 0,
            "mismatch_samples": 0,
        }
    mismatch = sum(
        1 for result in batch.analysis_results if "text_mismatch" in result.issue_tags
    )
    review = sum(
        1
        for result in batch.analysis_results
        if "text_alignment_review" in result.issue_tags
    )
    return {
        "compared_samples": len(alignments),
        "average_cer": round(sum(float(item["cer"]) for item in alignments) / len(alignments), 6),
        "average_wer": round(sum(float(item.get("wer", 0.0)) for item in alignments) / len(alignments), 6),
        "pass_samples": len(alignments) - mismatch - review,
        "review_samples": review,
        "mismatch_samples": mismatch,
    }


def _summarize_quality_score(batch: Batch) -> float | None:
    scores = [
        float(result.evidence["quality_score"])
        for result in batch.analysis_results
        if isinstance(result.evidence.get("quality_score"), (int, float))
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def _format_quality_score(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


def _action_label(value: str) -> str:
    labels = {
        "prepare_for_export": "准备导出",
        "manual_text_review": "人工文本复核",
        "manual_decision_review": "人工决策复核",
        "exclude_sample": "排除样本",
        "exclude_clipped_sample": "排除削波样本",
        "normalize_loudness": "响度标准化",
        "trim_silence": "裁剪静音",
        "denoise_light": "轻量降噪",
        "neural_denoise": "神经网络降噪",
    }
    return labels.get(value, value)
