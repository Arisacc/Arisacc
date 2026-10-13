"""Inspector 服务：把候选样本转换成 Agent 可读的证据包。

issue_tags 是短标签，方便 Strategy/Curator 快速判断；evidence 是完整证据，保存时长、文本来源、ASR 置信度和音频指标，用于报告、前端展示和 LLM prompt。
"""
from dataclasses import replace
from datetime import datetime, timezone

from audio_dataset_agent.core.models import (
    AnalysisResult,
    Batch,
    BatchPhase,
    CandidateSampleStatus,
)
from audio_dataset_agent.inspection.audio_metrics import AudioMetricsAnalyzer
from audio_dataset_agent.inspection.consistency import ConsistencyAnalyzer
from audio_dataset_agent.inspection.policy import InspectionPolicy
from audio_dataset_agent.inspection.quality import assess_audio_quality
from audio_dataset_agent.profiles.policy import get_dataset_profile
from audio_dataset_agent.text_quality.analyzer import evaluate_text_alignment


def inspect_candidate_samples(
    batch: Batch,
    policy: InspectionPolicy | None = None,
    audio_metrics_analyzer: AudioMetricsAnalyzer | None = None,
    acoustic_consistency_analyzer: ConsistencyAnalyzer | None = None,
) -> Batch:
    """为候选样本生成 issue_tags 和 evidence。"""
    # Inspector 将原始字段转换成 issue_tags，供 Curator 决策层消费。
    profile = get_dataset_profile(batch.target_policy)
    active_policy = policy or InspectionPolicy.from_profile(profile)
    analyzed_samples = []
    analysis_results = []

    for sample in batch.candidate_samples:
        transcript = sample.transcript or ""
        issue_tags: list[str] = []
        # evidence 记录标签产生的依据，便于报告、复核和 Agent 解释。
        evidence = {
            "duration": sample.duration,
            "transcript_length": len(transcript),
            "transcript_source": sample.transcript_source,
            "asr_confidence": sample.asr_confidence,
            "reference_text": sample.reference_text,
            "reference_text_source": sample.reference_text_source,
            "language": sample.language,
            "speaker_id": sample.speaker_id,
            "metadata_status": sample.metadata_status,
            "profile": profile.name,
            "quality_policy": active_policy.to_dict(),
        }
        profile_interpretation: dict[str, str] = {}

        # 第一类证据来自元数据：时长、文本是否存在、ASR 置信度。
        if sample.duration < active_policy.min_duration_seconds:
            issue_tags.append("too_short")
        if (
            active_policy.max_duration_seconds is not None
            and sample.duration > active_policy.max_duration_seconds
        ):
            issue_tags.append("too_long")

        if not transcript.strip():
            if profile.allow_missing_transcript:
                profile_interpretation["missing_transcript"] = "allowed"
            else:
                issue_tags.append("missing_transcript")

        reference_text = sample.reference_text or ""
        if profile.requires_reference_text and not reference_text.strip():
            issue_tags.append("missing_reference_text")
        elif sample.metadata_status == "requires_alignment":
            issue_tags.append("reference_alignment_required")
        elif reference_text.strip() and transcript.strip() and profile.text_cer_pass is not None:
            alignment = evaluate_text_alignment(reference_text, transcript)
            evidence["text_alignment"] = alignment.to_dict()
            if not alignment.normalized_reference or not alignment.normalized_candidate:
                issue_tags.append("text_invalid")
            elif alignment.cer <= profile.text_cer_pass:
                profile_interpretation["text_alignment"] = "matched"
            elif (
                profile.text_cer_review is not None
                and alignment.cer <= profile.text_cer_review
            ):
                issue_tags.append("text_alignment_review")
            else:
                issue_tags.append("text_mismatch")

        asr_generated_sources = {"asr", "faster_whisper"}
        if (
            sample.transcript_source in asr_generated_sources
            and sample.asr_confidence is not None
            and sample.asr_confidence < active_policy.min_asr_confidence
        ):
            if profile.allow_low_asr_confidence:
                profile_interpretation["asr_low_confidence"] = "warning"
            else:
                issue_tags.append("asr_low_confidence")

        if profile_interpretation:
            evidence["profile_interpretation"] = profile_interpretation

        # 第二类证据来自真实音频指标：响度、削波、静音比例、噪声底。
        if audio_metrics_analyzer is not None:
            audio_path = sample.sample_path or sample.source_path
            metrics = audio_metrics_analyzer.analyze(audio_path)
            quality = assess_audio_quality(sample.duration, metrics)
            evidence["audio_metrics"] = metrics.to_dict()
            evidence["snr_db"] = quality.snr_db
            evidence["quality_score"] = quality.quality_score
            evidence["quality_score_version"] = quality.version

            if metrics.rms_dbfs < active_policy.min_rms_dbfs:
                issue_tags.append("audio_too_quiet")

            if metrics.clipping_ratio > active_policy.max_clipping_ratio:
                issue_tags.append("audio_clipping")

            if metrics.silence_ratio > active_policy.max_silence_ratio:
                issue_tags.append("audio_high_silence_ratio")

            if (
                metrics.noise_floor_dbfs is not None
                and metrics.noise_floor_dbfs > active_policy.max_noise_floor_dbfs
            ):
                issue_tags.append("audio_noise_high")

        # AnalysisResult 是 Inspector 阶段的正式输出，也是后续两个 Agent 的主要输入。
        analysis_results.append(
            AnalysisResult(
                sample_id=sample.sample_id,
                duration=sample.duration,
                transcript_length=len(transcript),
                asr_confidence=sample.asr_confidence,
                issue_tags=issue_tags,
                evidence=evidence,
            )
        )
        analyzed_samples.append(
            replace(sample, status=CandidateSampleStatus.ANALYZED)
        )

    if acoustic_consistency_analyzer is not None and profile.name == "voice_clone":
        paths = [sample.sample_path or sample.source_path for sample in analyzed_samples]
        consistency = acoustic_consistency_analyzer.analyze(paths)
        consistency_payload = consistency.to_dict()
        for result_index, result in enumerate(analysis_results):
            evidence = dict(result.evidence)
            evidence["speaker_consistency"] = consistency_payload
            issue_tags = list(result.issue_tags)
            if consistency.warning:
                issue_tags.append("speaker_consistency_warning")
            analysis_results[result_index] = replace(
                result,
                issue_tags=issue_tags,
                evidence=evidence,
            )

    return replace(
        batch,
        phase=BatchPhase.ANALYZED,
        updated_at=datetime.now(timezone.utc),
        candidate_samples=analyzed_samples,
        analysis_results=analysis_results,
    )
