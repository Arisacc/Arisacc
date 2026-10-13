# 数据用途策略：定义 voice_clone/TTS/ASR 的质量阈值和判断重点。
from dataclasses import replace

from audio_dataset_agent.core.models import (
    AnalysisResult,
    CuratorDecision,
    DecisionType,
    ProcessingStrategy,
)
from audio_dataset_agent.profiles.policy import DatasetProfile, get_dataset_profile


class BaselineCurationPolicy:
    """根据数据目标调整决策的规则版 Curator 策略。"""

    policy_name = "baseline"

    def __init__(self, target_policy: str | DatasetProfile = "tts") -> None:
        self.profile = get_dataset_profile(target_policy)
        self.target_policy = self.profile.name

    def decide(
        self,
        result: AnalysisResult,
        strategy: ProcessingStrategy | None = None,
    ) -> CuratorDecision:
        issue_tags = set(result.issue_tags)

        if "too_short" in issue_tags:
            return self._finish(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REJECT,
                reason="拒绝：too_short，样本时长低于最小有效时长，缺乏修复价值。",
                confidence=0.95,
                recommended_action="exclude_sample",
                policy_name=self.policy_name,
            ), strategy)

        if "too_long" in issue_tags:
            return self._finish(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REVIEW,
                reason="复核：too_long，样本超过当前用途的最长时长，需要确认是否重新切分。",
                confidence=0.9,
                recommended_action="manual_decision_review",
                policy_name=self.policy_name,
            ), strategy)

        has_missing_text = "missing_transcript" in issue_tags
        has_low_asr_confidence = "asr_low_confidence" in issue_tags
        has_reference_text_issue = bool(
            issue_tags
            & {
                "missing_reference_text",
                "reference_alignment_required",
                "text_alignment_review",
                "text_mismatch",
                "text_invalid",
            }
        )
        if (
            has_missing_text
            and not self.profile.allow_missing_transcript
            or has_low_asr_confidence
            and not self.profile.allow_low_asr_confidence
            or has_reference_text_issue
        ):
            return self._finish(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REVIEW,
                reason="复核：参考文本或 ASR 候选文本的一致性不足，需要人工确认。",
                confidence=0.75,
                recommended_action="manual_text_review",
                policy_name=self.policy_name,
            ), strategy)

        if "audio_clipping" in issue_tags:
            return self._finish(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REJECT,
                reason="拒绝：audio_clipping，检测到音频削波，样本可能已发生失真。",
                confidence=0.9,
                recommended_action="exclude_clipped_sample",
                policy_name=self.policy_name,
            ), strategy)

        if "speaker_consistency_warning" in issue_tags:
            return self._finish(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REVIEW,
                reason="复核：批次声学特征差异较大，需要确认说话人或录音环境是否一致。",
                confidence=0.78,
                recommended_action="manual_decision_review",
                policy_name=self.policy_name,
            ), strategy)

        if "audio_too_quiet" in issue_tags:
            return self._finish(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REPAIR,
                reason=(
                    "修复：audio_too_quiet，音量过低，可尝试响度标准化。"
                ),
                confidence=0.8,
                recommended_action="normalize_loudness",
                policy_name=self.policy_name,
            ), strategy)

        if "audio_high_silence_ratio" in issue_tags:
            return self._finish(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REPAIR,
                reason=(
                    "修复：audio_high_silence_ratio，静音比例过高，可裁剪无信息静音。"
                ),
                confidence=0.8,
                recommended_action="trim_silence",
                policy_name=self.policy_name,
            ), strategy)

        if "audio_noise_high" in issue_tags:
            return self._decide_noisy_audio(result, strategy)

        return self._finish(CuratorDecision(
            sample_id=result.sample_id,
            decision=DecisionType.KEEP,
            reason="保留：未检测到阻断导出的质量问题。",
            confidence=0.8,
            recommended_action="prepare_for_export",
            policy_name=self.policy_name,
        ), strategy)

    def _decide_noisy_audio(
        self,
        result: AnalysisResult,
        strategy: ProcessingStrategy | None = None,
    ) -> CuratorDecision:
        if self.profile.name == "asr" and (
            result.asr_confidence is not None and result.asr_confidence >= 0.9
        ):
            return self._finish(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.KEEP,
                reason=(
                    "保留：ASR 文本置信度较高时，可容忍一定背景噪声。"
                ),
                confidence=0.72,
                recommended_action="prepare_for_export",
                policy_name=self.policy_name,
            ), strategy)

        if self.profile.preferred_denoise_operation == "neural_denoise":
            return self._finish(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REPAIR,
                reason=(
                    "修复：声音克隆更关注音色稳定性，优先采用神经网络降噪。"
                ),
                confidence=0.78,
                recommended_action=self.profile.preferred_denoise_operation,
                policy_name=self.policy_name,
            ), strategy)

        return self._finish(CuratorDecision(
            sample_id=result.sample_id,
            decision=DecisionType.REPAIR,
            reason="修复：audio_noise_high，背景噪声偏高，可尝试轻量降噪。",
            confidence=0.76,
            recommended_action=self.profile.preferred_denoise_operation,
            policy_name=self.policy_name,
        ), strategy)

    @staticmethod
    def _finish(
        decision: CuratorDecision,
        strategy: ProcessingStrategy | None,
    ) -> CuratorDecision:
        if strategy is None:
            return decision
        consistency = (
            "accepted"
            if decision.recommended_action == strategy.recommended_operation
            else "overridden"
        )
        return replace(decision, strategy_consistency=consistency)
