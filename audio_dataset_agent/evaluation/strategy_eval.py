import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from audio_dataset_agent.core.models import AnalysisResult, ProcessingStrategy


class StrategyAgent(Protocol):
    def decide(self, result: AnalysisResult) -> ProcessingStrategy:
        """根据一条分析结果创建一个处理策略。"""


@dataclass(frozen=True)
class StrategyEvalCase:
    name: str
    analysis_result: AnalysisResult
    expected_action: str
    expected_operation: str
    expected_tool_ids: list[str]


@dataclass(frozen=True)
class StrategyEvalRecord:
    case_name: str
    expected_action: str
    actual_action: str
    expected_operation: str
    actual_operation: str
    expected_tool_ids: list[str]
    actual_selected_tool_ids: list[str]
    actual_reason: str
    actual_confidence: float
    agent_name: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class StrategyEvalSummary:
    records: list[StrategyEvalRecord]

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def pass_count(self) -> int:
        return sum(1 for record in self.records if record.passed)

    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.pass_count / self.total_count

    @property
    def fallback_count(self) -> int:
        return sum(1 for record in self.records if _is_fallback_record(record))

    @property
    def autonomous_pass_count(self) -> int:
        return sum(
            1
            for record in self.records
            if record.passed and not _is_fallback_record(record)
        )

    @property
    def passed(self) -> bool:
        return self.total_count > 0 and self.pass_count == self.total_count


def default_strategy_eval_cases() -> list[StrategyEvalCase]:
    return [
        StrategyEvalCase(
            name="clean_sample_no_processing",
            analysis_result=AnalysisResult(
                sample_id="strategy_clean",
                duration=2.4,
                transcript_length=24,
                asr_confidence=0.94,
                issue_tags=[],
                evidence={"duration": 2.4, "asr_confidence": 0.94},
            ),
            expected_action="keep_without_processing",
            expected_operation="prepare_for_export",
            expected_tool_ids=[],
        ),
        StrategyEvalCase(
            name="quiet_audio_loudness_repair",
            analysis_result=AnalysisResult(
                sample_id="strategy_quiet",
                duration=2.4,
                transcript_length=18,
                asr_confidence=0.92,
                issue_tags=["audio_too_quiet"],
                evidence={"audio_metrics": {"rms_dbfs": -55.0}},
            ),
            expected_action="repair_audio",
            expected_operation="normalize_loudness",
            expected_tool_ids=["repair.ffmpeg_loudness", "metrics.wave"],
        ),
        StrategyEvalCase(
            name="high_silence_trim_repair",
            analysis_result=AnalysisResult(
                sample_id="strategy_silence",
                duration=4.0,
                transcript_length=18,
                asr_confidence=0.92,
                issue_tags=["audio_high_silence_ratio"],
                evidence={"audio_metrics": {"silence_ratio": 0.72}},
            ),
            expected_action="repair_audio",
            expected_operation="trim_silence",
            expected_tool_ids=["repair.ffmpeg_silence_trim", "metrics.wave"],
        ),
        StrategyEvalCase(
            name="noisy_audio_denoise_repair",
            analysis_result=AnalysisResult(
                sample_id="strategy_noise",
                duration=3.0,
                transcript_length=18,
                asr_confidence=0.92,
                issue_tags=["audio_noise_high"],
                evidence={"audio_metrics": {"noise_floor_dbfs": -32.0}},
            ),
            expected_action="repair_audio",
            expected_operation="denoise_light",
            expected_tool_ids=["repair.ffmpeg_denoise", "metrics.wave"],
        ),
        StrategyEvalCase(
            name="low_asr_manual_text_review",
            analysis_result=AnalysisResult(
                sample_id="strategy_low_asr",
                duration=2.4,
                transcript_length=12,
                asr_confidence=0.42,
                issue_tags=["asr_low_confidence"],
                evidence={"asr_confidence": 0.42},
            ),
            expected_action="manual_review",
            expected_operation="manual_text_review",
            expected_tool_ids=["asr.faster_whisper"],
        ),
        StrategyEvalCase(
            name="clipped_audio_manual_review",
            analysis_result=AnalysisResult(
                sample_id="strategy_clipping",
                duration=2.4,
                transcript_length=12,
                asr_confidence=0.91,
                issue_tags=["audio_clipping"],
                evidence={"audio_metrics": {"clipping_ratio": 0.08}},
            ),
            expected_action="manual_review",
            expected_operation="review_clipped_audio",
            expected_tool_ids=["metrics.wave"],
        ),
    ]


def evaluate_strategy_agent(
    agent: StrategyAgent,
    cases: list[StrategyEvalCase],
) -> StrategyEvalSummary:
    records = []
    for case in cases:
        strategy = agent.decide(case.analysis_result)
        action_matches = strategy.action == case.expected_action
        operation_matches = strategy.recommended_operation == case.expected_operation
        tools_match = set(strategy.selected_tool_ids) == set(case.expected_tool_ids)
        passed = action_matches and operation_matches and tools_match
        if passed:
            reason = "passed"
        else:
            reason = (
                f"expected {case.expected_action}/{case.expected_operation}/"
                f"{case.expected_tool_ids}, got {strategy.action}/"
                f"{strategy.recommended_operation}/{strategy.selected_tool_ids}"
            )
        records.append(
            StrategyEvalRecord(
                case_name=case.name,
                expected_action=case.expected_action,
                actual_action=strategy.action,
                expected_operation=case.expected_operation,
                actual_operation=strategy.recommended_operation,
                expected_tool_ids=list(case.expected_tool_ids),
                actual_selected_tool_ids=list(strategy.selected_tool_ids),
                actual_reason=strategy.reason,
                actual_confidence=strategy.confidence,
                agent_name=strategy.agent_name,
                passed=passed,
                reason=reason,
            )
        )
    return StrategyEvalSummary(records=records)


def strategy_eval_summary_to_dict(summary: StrategyEvalSummary) -> dict[str, object]:
    return {
        "passed": summary.passed,
        "pass_count": summary.pass_count,
        "total_count": summary.total_count,
        "pass_rate": summary.pass_rate,
        "fallback_count": summary.fallback_count,
        "autonomous_pass_count": summary.autonomous_pass_count,
        "records": [
            {
                "case_name": record.case_name,
                "expected_action": record.expected_action,
                "actual_action": record.actual_action,
                "expected_operation": record.expected_operation,
                "actual_operation": record.actual_operation,
                "expected_tool_ids": list(record.expected_tool_ids),
                "actual_selected_tool_ids": list(record.actual_selected_tool_ids),
                "actual_reason": record.actual_reason,
                "actual_confidence": record.actual_confidence,
                "agent_name": record.agent_name,
                "passed": record.passed,
                "reason": record.reason,
            }
            for record in summary.records
        ],
    }


def render_strategy_eval_markdown(
    summary: StrategyEvalSummary,
    title: str = "Strategy Evaluation Report",
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Passed: {str(summary.passed).lower()}",
        f"- Pass count: {summary.pass_count}",
        f"- Total count: {summary.total_count}",
        f"- Pass rate: {summary.pass_rate:.2%}",
        f"- Fallback count: {summary.fallback_count}",
        f"- Autonomous pass count: {summary.autonomous_pass_count}",
        "",
        "## Cases",
        "",
        "| case | expected | actual | tools | passed | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in summary.records:
        expected = f"{record.expected_action} / {record.expected_operation}"
        actual = f"{record.actual_action} / {record.actual_operation}"
        tools = ", ".join(record.actual_selected_tool_ids) or "none"
        lines.append(
            "| "
            f"{record.case_name} | "
            f"{expected} | "
            f"{actual} | "
            f"{_escape_markdown_table(tools)} | "
            f"{str(record.passed).lower()} | "
            f"{_escape_markdown_table(record.reason)} |"
        )
    return "\n".join(lines) + "\n"


def write_strategy_eval_report(
    summary: StrategyEvalSummary,
    output_dir: Path,
    title: str = "Strategy Evaluation Report",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "strategy_eval_summary.json"
    markdown_path = output_dir / "strategy_eval_report.md"

    json_path.write_text(
        json.dumps(strategy_eval_summary_to_dict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_strategy_eval_markdown(summary, title=title),
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _is_fallback_record(record: StrategyEvalRecord) -> bool:
    return record.agent_name.endswith("_guarded") or "fallback:" in record.actual_reason
