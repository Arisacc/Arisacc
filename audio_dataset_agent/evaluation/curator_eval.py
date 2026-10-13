import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from audio_dataset_agent.core.models import AnalysisResult, CuratorDecision, DecisionType


class CuratorPolicy(Protocol):
    def decide(self, result: AnalysisResult) -> CuratorDecision:
        """根据一条分析结果创建一个 Curator 决策。"""


@dataclass(frozen=True)
class CuratorEvalCase:
    name: str
    analysis_result: AnalysisResult
    expected_decision: DecisionType
    expected_action: str


@dataclass(frozen=True)
class CuratorEvalRecord:
    case_name: str
    expected_decision: DecisionType
    actual_decision: DecisionType
    expected_action: str
    actual_action: str
    actual_reason: str
    actual_confidence: float
    policy_name: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class CuratorEvalSummary:
    records: list[CuratorEvalRecord]

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
    def passed(self) -> bool:
        return self.total_count > 0 and self.pass_count == self.total_count

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


def default_curator_eval_cases() -> list[CuratorEvalCase]:
    return [
        CuratorEvalCase(
            name="clean_sample_kept",
            analysis_result=AnalysisResult(
                sample_id="eval_clean",
                duration=2.4,
                transcript_length=24,
                asr_confidence=0.94,
                issue_tags=[],
                evidence={"duration": 2.4, "asr_confidence": 0.94},
            ),
            expected_decision=DecisionType.KEEP,
            expected_action="prepare_for_export",
        ),
        CuratorEvalCase(
            name="too_short_rejected",
            analysis_result=AnalysisResult(
                sample_id="eval_short",
                duration=0.4,
                transcript_length=8,
                asr_confidence=0.9,
                issue_tags=["too_short"],
                evidence={"duration": 0.4},
            ),
            expected_decision=DecisionType.REJECT,
            expected_action="exclude_sample",
        ),
        CuratorEvalCase(
            name="low_asr_confidence_reviewed",
            analysis_result=AnalysisResult(
                sample_id="eval_low_asr",
                duration=2.0,
                transcript_length=16,
                asr_confidence=0.42,
                issue_tags=["asr_low_confidence"],
                evidence={"asr_confidence": 0.42},
            ),
            expected_decision=DecisionType.REVIEW,
            expected_action="manual_text_review",
        ),
        CuratorEvalCase(
            name="high_silence_repaired",
            analysis_result=AnalysisResult(
                sample_id="eval_silence",
                duration=3.2,
                transcript_length=20,
                asr_confidence=0.91,
                issue_tags=["audio_high_silence_ratio"],
                evidence={"silence_ratio": 0.48},
            ),
            expected_decision=DecisionType.REPAIR,
            expected_action="trim_silence",
        ),
        CuratorEvalCase(
            name="noisy_audio_repaired",
            analysis_result=AnalysisResult(
                sample_id="eval_noise",
                duration=3.0,
                transcript_length=20,
                asr_confidence=0.91,
                issue_tags=["audio_noise_high"],
                evidence={"noise_floor_dbfs": -32.0},
            ),
            expected_decision=DecisionType.REPAIR,
            expected_action="denoise_light",
        ),
    ]


def evaluate_curator_policy(
    policy: CuratorPolicy,
    cases: list[CuratorEvalCase],
) -> CuratorEvalSummary:
    records = []
    for case in cases:
        decision = policy.decide(case.analysis_result)
        decision_matches = decision.decision == case.expected_decision
        action_matches = decision.recommended_action == case.expected_action
        passed = decision_matches and action_matches
        if passed:
            reason = "passed"
        else:
            reason = (
                f"expected {case.expected_decision.value}/"
                f"{case.expected_action}, got {decision.decision.value}/"
                f"{decision.recommended_action}"
            )
        records.append(
            CuratorEvalRecord(
                case_name=case.name,
                expected_decision=case.expected_decision,
                actual_decision=decision.decision,
                expected_action=case.expected_action,
                actual_action=decision.recommended_action,
                actual_reason=decision.reason,
                actual_confidence=decision.confidence,
                policy_name=decision.policy_name,
                passed=passed,
                reason=reason,
            )
        )
    return CuratorEvalSummary(records=records)


def curator_eval_summary_to_dict(summary: CuratorEvalSummary) -> dict[str, object]:
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
                "expected_decision": record.expected_decision.value,
                "actual_decision": record.actual_decision.value,
                "expected_action": record.expected_action,
                "actual_action": record.actual_action,
                "actual_reason": record.actual_reason,
                "actual_confidence": record.actual_confidence,
                "policy_name": record.policy_name,
                "passed": record.passed,
                "reason": record.reason,
            }
            for record in summary.records
        ],
    }


def render_curator_eval_markdown(
    summary: CuratorEvalSummary,
    title: str = "Curator Evaluation Report",
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
        "| case | expected | actual | passed | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in summary.records:
        expected = f"{record.expected_decision.value} / {record.expected_action}"
        actual = f"{record.actual_decision.value} / {record.actual_action}"
        lines.append(
            "| "
            f"{record.case_name} | "
            f"{expected} | "
            f"{actual} | "
            f"{str(record.passed).lower()} | "
            f"{_escape_markdown_table(record.reason)} |"
        )
    return "\n".join(lines) + "\n"


def write_curator_eval_report(
    summary: CuratorEvalSummary,
    output_dir: Path,
    title: str = "Curator Evaluation Report",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "curator_eval_summary.json"
    markdown_path = output_dir / "curator_eval_report.md"

    json_path.write_text(
        json.dumps(curator_eval_summary_to_dict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_curator_eval_markdown(summary, title=title),
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _is_fallback_record(record: CuratorEvalRecord) -> bool:
    return record.policy_name.endswith("_guarded") or "fallback:" in record.actual_reason
