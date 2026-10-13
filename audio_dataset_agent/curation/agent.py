"""Curator Agent：根据用途画像、证据和策略决定样本去留。

输出 keep/reject/repair/review 四类决策
"""
import json
from typing import Any, Protocol

from audio_dataset_agent.core.llm_json import parse_json_object
from audio_dataset_agent.core.models import (
    AnalysisResult,
    CuratorDecision,
    DecisionType,
    ProcessingStrategy,
)
from audio_dataset_agent.profiles.policy import DatasetProfile, get_dataset_profile


class LLMClient(Protocol):
    """Curator Agent 使用的最小文本补全接口。"""

    def complete(self, prompt: str) -> str | dict[str, Any]:
        """返回包含 Curator 决策的 JSON 字符串或类 JSON 字典。"""


class LLMCuratorAgent:
    """把 Inspector 证据转换为结构化决策的 Curator Agent。"""

    policy_name = "llm_curator"
    allowed_recommended_actions = {
        "prepare_for_export",
        "exclude_sample",
        "exclude_clipped_sample",
        "trim_silence",
        "normalize_loudness",
        "denoise_light",
        "neural_denoise",
        "manual_text_review",
        "manual_decision_review",
    }

    def __init__(
        self,
        llm_client: LLMClient,
        fallback_policy=None,
        target_policy: str | DatasetProfile = "tts",
    ):
        self.llm_client = llm_client
        self.fallback_policy = fallback_policy
        self.profile = get_dataset_profile(target_policy)
        self.target_policy = self.profile.name

    # decide 是 Curator 的样本级入口：LLM 先给出判断，再经过安全规则兜底。
    def decide(
        self,
        result: AnalysisResult,
        strategy: ProcessingStrategy | None = None,
    ) -> CuratorDecision:
        prompt = self._build_prompt(result, strategy)
        try:
            payload = self._parse_response(self.llm_client.complete(prompt))
            decision = self._decision_from_payload(result.sample_id, payload)
            return self._guard_decision(result, decision, strategy)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if self.fallback_policy is not None:
                return self._fallback_decision(
                    result,
                    f"降级处理：LLM Curator 返回无效结果（{exc}）",
                    strategy,
                )
            return self._mark_strategy(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REVIEW,
                reason=f"人工复核：LLM Curator 返回无效结果（{exc}）",
                confidence=0.0,
                recommended_action="manual_decision_review",
                policy_name=self.policy_name,
            ), strategy)

    def _build_prompt(
        self,
        result: AnalysisResult,
        strategy: ProcessingStrategy | None = None,
    ) -> str:
        evidence_packet = {
            "sample_id": result.sample_id,
            "target_policy": self.target_policy,
            "dataset_profile": _profile_to_prompt_packet(self.profile),
            "duration": result.duration,
            "transcript_length": result.transcript_length,
            "asr_confidence": result.asr_confidence,
            "issue_tags": list(result.issue_tags),
            "evidence": dict(result.evidence),
        }
        return "\n".join(self._prompt_sections(evidence_packet, strategy))

    def _prompt_sections(
        self,
        evidence_packet: dict[str, Any],
        strategy: ProcessingStrategy | None = None,
    ) -> list[str]:
        strategy_packet = _strategy_to_prompt_packet(strategy)
        return [
            "你是语音数据集准备流水线中的 Curator Agent。",
            "请只返回一个完整 JSON 对象，不要输出 Markdown、注释或额外解释。",
            f"当前数据用途（target_policy）：{self.target_policy}",
            "你的任务是判断当前样本应该保留、拒绝、修复，还是进入人工复核。",
            "Strategy Agent 的内容只是候选建议，不是最终命令；你必须结合证据和硬规则审核它，必要时可以覆盖。",
            "Strategy Agent 建议（其中 action、recommended_operation、selected_tool_ids 为协议字段）："
            f"{json.dumps(strategy_packet, ensure_ascii=False)}",
            "ASR 用途优先关注语音是否可识别，以及候选文本是否可靠。",
            "TTS 用途优先关注音频干净、文本准确、响度稳定、静音比例合理。",
            "voice_clone 用途优先关注说话人音色稳定、低噪声和低混响。",
            "数据用途画像（DatasetProfile）："
            f"{json.dumps(_profile_to_prompt_packet(self.profile), ensure_ascii=False)}",
            "允许的 decision 协议值：keep、reject、repair、review",
            "允许的 recommended_action 协议值："
            f"{', '.join(sorted(self.allowed_recommended_actions))}",
            "请先执行硬性安全规则，再结合数据用途做判断。",
            "如果命中硬性安全规则，优先采用第一条匹配规则。",
            "不要编造证据中不存在的质量问题。",
            "问题标签到决策的示例（issue_tags、keep 等为协议字段）：",
            "- 如果 issue_tags 为空且文本证据可用，选择 keep。",
            "- too_short -> reject / exclude_sample。",
            "- too_long -> review / manual_decision_review。",
            "- 当当前用途需要可靠文本时，missing_transcript 或 asr_low_confidence -> review / manual_text_review。",
            "- missing_reference_text、reference_alignment_required、text_alignment_review、text_mismatch 或 text_invalid -> review / manual_text_review。",
            "- audio_clipping -> reject / exclude_clipped_sample，或 review / manual_decision_review。",
            "- speaker_consistency_warning -> review / manual_decision_review。",
            "- audio_too_quiet -> repair / normalize_loudness。",
            "- audio_high_silence_ratio -> repair / trim_silence。",
            "- TTS 中的 audio_noise_high -> repair / denoise_light。",
            "- voice_clone 或严重噪声下的 audio_noise_high -> repair / neural_denoise。",
            "- ASR 中如果 ASR 置信度高，audio_noise_high 可以 keep / prepare_for_export。",
            "- 如果当前 profile 允许缺失文本或低置信文本，不要只因为文本证据阻断样本。",
            "- 不要因为证据有限就直接 reject；不确定时选择 review。",
            "严格 JSON 格式（键名和值必须保持协议英文）：",
            '{"decision": "允许的决策值", "reason": "简短中文决策理由", '
            '"confidence": 0.0, "recommended_action": "允许的处理动作"}',
            f"样本证据包：\n{json.dumps(evidence_packet, ensure_ascii=False)}",
            "现在输出 JSON 对象。",
        ]

    def _parse_response(self, response: str | dict[str, Any]) -> dict[str, Any]:
        return parse_json_object(response)

    def _strip_markdown_json_fence(self, response: str) -> str:
        text = response.strip()
        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if len(lines) < 3:
            return text
        if not lines[-1].strip().startswith("```"):
            return text
        return "\n".join(lines[1:-1]).strip()

    def _decision_from_payload(
        self,
        sample_id: str,
        payload: dict[str, Any],
    ) -> CuratorDecision:
        decision_text = str(payload["decision"]).strip().lower()
        decision = DecisionType(decision_text)
        confidence = float(payload["confidence"])
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence 必须在 0 到 1 之间")

        reason = str(payload["reason"]).strip()
        recommended_action = str(payload["recommended_action"]).strip()
        if not reason:
            raise ValueError("reason 不能为空")
        if not recommended_action:
            raise ValueError("recommended_action 不能为空")
        if recommended_action not in self.allowed_recommended_actions:
            raise ValueError(f"不支持的 recommended_action：{recommended_action}")

        return CuratorDecision(
            sample_id=sample_id,
            decision=decision,
            reason=reason,
            confidence=confidence,
            recommended_action=recommended_action,
            policy_name=self.policy_name,
        )

    def _guard_decision(
        self,
        result: AnalysisResult,
        decision: CuratorDecision,
        strategy: ProcessingStrategy | None = None,
    ) -> CuratorDecision:
        # 安全校验负责给 Agent 放权后的边界：模型可以判断，但不能违反硬规则。
        violation = self._safety_violation(result, decision)
        if violation is None:
            return self._mark_strategy(decision, strategy)

        if self.fallback_policy is None:
            return self._mark_strategy(CuratorDecision(
                sample_id=result.sample_id,
                decision=DecisionType.REVIEW,
                reason=f"人工复核：LLM Curator 决策违反安全规则（{violation}）",
                confidence=0.0,
                recommended_action="manual_decision_review",
                policy_name=self.policy_name,
            ), strategy)

        return self._fallback_decision(
            result,
            f"降级处理：LLM Curator 决策违反安全规则（{violation}）",
            strategy,
        )

    def _safety_violation(
        self,
        result: AnalysisResult,
        decision: CuratorDecision,
    ) -> str | None:
        issue_tags = set(result.issue_tags)

        # 硬规则优先级高于模型理由，避免明显不合格样本被误保留。
        if "too_short" in issue_tags:
            if (
                decision.decision != DecisionType.REJECT
                or decision.recommended_action != "exclude_sample"
            ):
                return "too_short 样本必须拒绝"

        if "too_long" in issue_tags:
            if decision.decision != DecisionType.REVIEW:
                return "too_long 样本必须进入人工复核"

        if "missing_transcript" in issue_tags:
            if not self.profile.allow_missing_transcript and (
                decision.decision != DecisionType.REVIEW
                or decision.recommended_action != "manual_text_review"
            ):
                return "缺失文本必须进入人工文本复核"

        if "asr_low_confidence" in issue_tags:
            if not self.profile.allow_low_asr_confidence and (
                decision.decision != DecisionType.REVIEW
                or decision.recommended_action != "manual_text_review"
            ):
                return "ASR 低置信度必须进入人工文本复核"

        if issue_tags & {
            "missing_reference_text",
            "reference_alignment_required",
            "text_alignment_review",
            "text_mismatch",
            "text_invalid",
        }:
            if (
                decision.decision != DecisionType.REVIEW
                or decision.recommended_action != "manual_text_review"
            ):
                return "参考文本一致性问题必须进入人工文本复核"

        if "audio_clipping" in issue_tags:
            if decision.decision in {DecisionType.KEEP, DecisionType.REPAIR}:
                return "存在削波的音频不能自动保留或修复"

        if "speaker_consistency_warning" in issue_tags and decision.decision != DecisionType.REVIEW:
            return "声学一致性提醒必须进入人工复核"

        repair_required_tags = {
            "normalize_loudness": "audio_too_quiet",
            "trim_silence": "audio_high_silence_ratio",
            "denoise_light": "audio_noise_high",
            "neural_denoise": "audio_noise_high",
        }
        if decision.decision == DecisionType.REPAIR:
            required_tag = repair_required_tags.get(decision.recommended_action)
            if required_tag is not None and required_tag not in issue_tags:
                return (
                    f"{decision.recommended_action} 修复需要 "
                    f"{required_tag} 证据"
                )

        if decision.confidence < 0.5 and decision.decision != DecisionType.REVIEW:
            return "低置信度的自动审核决策必须进入人工复核"

        return None

    def _fallback_decision(
        self,
        result: AnalysisResult,
        reason_prefix: str,
        strategy: ProcessingStrategy | None = None,
    ) -> CuratorDecision:
        fallback = self.fallback_policy.decide(result, strategy)
        return self._mark_strategy(CuratorDecision(
            sample_id=fallback.sample_id,
            decision=fallback.decision,
            reason=f"{reason_prefix}; {fallback.reason}",
            confidence=fallback.confidence,
            recommended_action=fallback.recommended_action,
            policy_name="llm_curator_guarded",
        ), strategy)

    @staticmethod
    def _mark_strategy(
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
        from dataclasses import replace

        return replace(decision, strategy_consistency=consistency)


def _profile_to_prompt_packet(profile: DatasetProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "requires_asr": profile.requires_asr,
        "requires_text": profile.requires_text,
        "text_reliability_required": profile.text_reliability_required,
        "allow_missing_transcript": profile.allow_missing_transcript,
        "allow_low_asr_confidence": profile.allow_low_asr_confidence,
        "strict_audio_cleanliness": profile.strict_audio_cleanliness,
        "moderate_noise_tolerated": profile.moderate_noise_tolerated,
        "preferred_denoise_operation": profile.preferred_denoise_operation,
        "min_duration_seconds": profile.min_duration_seconds,
        "max_duration_seconds": profile.max_duration_seconds,
        "min_asr_confidence": profile.min_asr_confidence,
        "min_rms_dbfs": profile.min_rms_dbfs,
        "max_clipping_ratio": profile.max_clipping_ratio,
        "max_silence_ratio": profile.max_silence_ratio,
        "max_noise_floor_dbfs": profile.max_noise_floor_dbfs,
        "requires_reference_text": profile.requires_reference_text,
        "text_cer_pass": profile.text_cer_pass,
        "text_cer_review": profile.text_cer_review,
        "report_focus": list(profile.report_focus),
        "risk_notes": list(profile.risk_notes),
    }


def _strategy_to_prompt_packet(
    strategy: ProcessingStrategy | None,
) -> dict[str, Any]:
    if strategy is None:
        return {"available": False, "reason": "没有找到对应的 Strategy 建议"}
    return {
        "available": True,
        "sample_id": strategy.sample_id,
        "action": strategy.action,
        "recommended_operation": strategy.recommended_operation,
        "selected_tool_ids": list(strategy.selected_tool_ids),
        "reason": strategy.reason,
        "confidence": strategy.confidence,
        "agent_name": strategy.agent_name,
    }
