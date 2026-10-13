"""Strategy Agent：根据样本问题和工具能力选择处理策略。

它的职责是“选择下一步怎么处理”，不是执行工具。LLM 会读取 DatasetProfile、Inspector evidence 和工具注册表，然后输出结构化策略，再经过安全规则校验。
"""
import json
from typing import Any, Protocol

from audio_dataset_agent.core.llm_json import parse_json_object
from audio_dataset_agent.core.models import AnalysisResult, ProcessingStrategy
from audio_dataset_agent.profiles.policy import DatasetProfile, get_dataset_profile
from audio_dataset_agent.tools.registry import ToolRegistry, tool_registry_to_dict
from audio_dataset_agent.knowledge.context import build_knowledge_context


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str | dict[str, Any]:
        """返回包含处理策略的 JSON 字符串或类 JSON 字典。"""


class BaselineProcessingStrategyAgent:
    """将证据映射为目标相关工具选择的规则版策略 Agent。"""

    agent_name = "baseline_processing_strategy"

    def __init__(
        self,
        tool_registry: ToolRegistry,
        target_policy: str | DatasetProfile = "tts",
    ) -> None:
        self.tool_registry = tool_registry
        self.profile = get_dataset_profile(target_policy)
        self.target_policy = self.profile.name

    # decide 是样本级入口：输入一个 AnalysisResult，输出一个 ProcessingStrategy。
    def decide(self, result: AnalysisResult) -> ProcessingStrategy:
        issue_tags = set(result.issue_tags)

        # 过短样本没有修复价值，直接建议排除，避免后续工具浪费时间。
        if "too_short" in issue_tags:
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="exclude_sample",
                recommended_operation="exclude_sample",
                reason="样本时长过短，不具备训练使用价值",
                confidence=0.95,
                agent_name=self.agent_name,
            )

        if "too_long" in issue_tags:
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="manual_review",
                recommended_operation="manual_strategy_review",
                reason="样本超过当前用途的最长时长，需要人工确认是否重新切分",
                confidence=0.9,
                agent_name=self.agent_name,
            )

        if "audio_clipping" in issue_tags:
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="manual_review",
                recommended_operation="review_clipped_audio",
                selected_tool_ids=self._available(["metrics.wave"]),
                reason="削波可能代表失真，自动修复风险较高",
                confidence=0.85,
                agent_name=self.agent_name,
            )

        if "speaker_consistency_warning" in issue_tags:
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="manual_review",
                recommended_operation="manual_strategy_review",
                reason="批次声学特征差异较大，需要人工确认是否混入不同说话人或录音环境",
                confidence=0.78,
                agent_name=self.agent_name,
            )

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
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="manual_review",
                recommended_operation="manual_text_review",
                selected_tool_ids=self._available(["asr.faster_whisper"]),
                reason="参考文本或 ASR 候选文本的一致性不足，不能自动导出",
                confidence=0.8,
                agent_name=self.agent_name,
            )

        if "audio_too_quiet" in issue_tags:
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="repair_audio",
                recommended_operation="normalize_loudness",
                selected_tool_ids=self._available(
                    ["repair.ffmpeg_loudness", "metrics.wave"]
                ),
                reason="音量偏低，通常可通过响度标准化改善",
                confidence=0.82,
                agent_name=self.agent_name,
            )

        if "audio_high_silence_ratio" in issue_tags:
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="repair_audio",
                recommended_operation="trim_silence",
                selected_tool_ids=self._available(
                    ["repair.ffmpeg_silence_trim", "metrics.wave"]
                ),
                reason="静音比例偏高，可裁剪无信息静音",
                confidence=0.82,
                agent_name=self.agent_name,
            )

        # 噪声样本的处理方式和数据用途强相关，因此单独交给 _decide_noisy_audio。
        if "audio_noise_high" in issue_tags:
            return self._decide_noisy_audio(result)

        return ProcessingStrategy(
            sample_id=result.sample_id,
            action="keep_without_processing",
            recommended_operation="prepare_for_export",
            selected_tool_ids=[],
            reason="未发现需要修复或人工复核的问题",
            confidence=0.8,
            agent_name=self.agent_name,
        )

    def _decide_noisy_audio(self, result: AnalysisResult) -> ProcessingStrategy:
        if self.profile.name == "asr" and (
            result.asr_confidence is not None and result.asr_confidence >= 0.9
        ):
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="keep_without_processing",
                recommended_operation="prepare_for_export",
                selected_tool_ids=[],
                reason=(
                    "ASR 用途在文本置信度较高时，可保留中等噪声的语音"
                ),
                confidence=0.72,
                agent_name=self.agent_name,
            )

        if self.profile.preferred_denoise_operation == "neural_denoise":
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="repair_audio",
                recommended_operation=self.profile.preferred_denoise_operation,
                selected_tool_ids=self._available(
                    ["repair.deepfilternet_denoise", "metrics.ffmpeg_astats"]
                ),
                reason=(
                    "voice_clone 用途需要稳定的说话人音色；噪声样本优先使用神经网络降噪"
                ),
                confidence=0.78,
                agent_name=self.agent_name,
            )

        return ProcessingStrategy(
            sample_id=result.sample_id,
            action="repair_audio",
            recommended_operation=self.profile.preferred_denoise_operation,
            selected_tool_ids=self._available(
                ["repair.ffmpeg_denoise", "metrics.wave"]
            ),
            reason="背景噪声通常可通过轻量降噪减弱",
            confidence=0.76,
            agent_name=self.agent_name,
        )

    def _available(self, tool_ids: list[str]) -> list[str]:
        return [tool_id for tool_id in tool_ids if self.tool_registry.get(tool_id)]


class LLMProcessingStrategyAgent:
    """受硬性安全规则保护的 LLM 策略 Agent。"""

    agent_name = "llm_processing_strategy"

    allowed_actions = {
        "keep_without_processing",
        "repair_audio",
        "manual_review",
        "exclude_sample",
    }
    allowed_operations = {
        "prepare_for_export",
        "normalize_loudness",
        "trim_silence",
        "denoise_light",
        "neural_denoise",
        "manual_text_review",
        "review_clipped_audio",
        "exclude_sample",
        "manual_strategy_review",
    }

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        fallback_agent: BaselineProcessingStrategyAgent | None = None,
        target_policy: str | DatasetProfile = "tts",
        knowledge_retriever=None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.fallback_agent = fallback_agent
        self.profile = get_dataset_profile(target_policy)
        self.target_policy = self.profile.name
        self.knowledge_retriever = knowledge_retriever

    def decide(self, result: AnalysisResult) -> ProcessingStrategy:
        prompt = self._build_prompt(result)
        try:
            payload = self._parse_response(self.llm_client.complete(prompt))
            strategy = self._strategy_from_payload(result.sample_id, payload)
            return self._guard_strategy(result, strategy)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if self.fallback_agent is not None:
                return self._fallback_strategy(
                    result,
                    f"降级处理：LLM Strategy 返回无效结果（{exc}）",
                )
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="manual_review",
                recommended_operation="manual_strategy_review",
                selected_tool_ids=[],
                reason=f"LLM Strategy 返回无效结果（{exc}）",
                confidence=0.0,
                agent_name=self.agent_name,
            )

    # LLM 版本会把 profile、证据和工具注册表一起放进 prompt，让模型在受限集合内选择。
    def _build_prompt(self, result: AnalysisResult) -> str:
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
        tool_packet = _strategy_tool_registry_to_dict(self.tool_registry)
        knowledge = build_knowledge_context([], max_chars=3000)
        if self.knowledge_retriever is not None:
            knowledge = build_knowledge_context(
                self.knowledge_retriever.search(
                    f"{self.target_policy} {' '.join(result.issue_tags)} 音频工具处理建议",
                    profile=self.target_policy,
                    top_k=3,
                    agent_name="strategy_agent",
                    sample_id=result.sample_id,
                ),
                max_chars=3000,
            )
        return "\n".join(
            self._strategy_prompt_sections(evidence_packet, tool_packet, knowledge)
        )

    def _parse_response(self, response: str | dict[str, Any]) -> dict[str, Any]:
        return parse_json_object(response)

    def _strategy_prompt_sections(
        self,
        evidence_packet: dict[str, Any],
        tool_packet: dict[str, Any],
        knowledge,
    ) -> list[str]:
        return [
            "你是音频数据集流水线中的处理策略 Agent。",
            "请只返回一个完整 JSON 对象，不要输出 Markdown、注释或额外解释。",
            f"当前数据用途（target_policy）：{self.target_policy}",
            "数据用途画像（DatasetProfile）："
            f"{json.dumps(_profile_to_prompt_packet(self.profile), ensure_ascii=False)}",
            "你的任务是选择下一步处理策略和候选工具，只做决策，不执行工具。",
            f"允许的 action 协议值：{', '.join(sorted(self.allowed_actions))}",
            "允许的 recommended_operation 协议值："
            f"{', '.join(sorted(self.allowed_operations))}",
            "硬性安全规则:",
            "- too_short: 不要尝试修复，只能 exclude 或 manual_review。",
            "- too_long: 必须 manual_review，不能直接保留或自动修复。",
            "- missing_transcript: 当当前用途需要可靠文本时，必须选择 manual_text_review。",
            "- asr_low_confidence: 当当前用途需要可靠文本时，必须选择 manual_text_review。",
            "- missing_reference_text、reference_alignment_required、text_alignment_review、text_mismatch、text_invalid: 必须选择 manual_text_review。",
            "- audio_clipping: 必须人工复核，不要自动降噪、响度修复或直接保留。",
            "- review_clipped_audio 只有在存在 audio_clipping 证据时才能选择。",
            "- selected_tool_ids 只能使用工具注册表里实际存在的 tool_id。",
            "- 如果证据不足，请选择 manual_review。",
            "问题标签到策略的示例:",
            "- 无 issue_tags -> keep_without_processing / prepare_for_export / 不选工具。",
            "- audio_too_quiet -> repair_audio / normalize_loudness / repair.ffmpeg_loudness, metrics.wave。",
            "- audio_high_silence_ratio -> repair_audio / trim_silence / repair.ffmpeg_silence_trim, metrics.wave。",
            "- audio_noise_high -> repair_audio / denoise_light / repair.ffmpeg_denoise, metrics.wave。",
            "- voice_clone 或严重噪声下的 audio_noise_high -> repair_audio / neural_denoise / repair.deepfilternet_denoise, metrics.ffmpeg_astats。",
            "- asr_low_confidence -> manual_review / manual_text_review / asr.faster_whisper。",
            "- 需要文本但 missing_transcript -> manual_review / manual_text_review / asr.faster_whisper。",
            "- audio_clipping -> manual_review / review_clipped_audio / metrics.wave。",
            "- speaker_consistency_warning -> manual_review / manual_strategy_review。",
            "- too_long -> manual_review / manual_strategy_review / 不选工具。",
            "- text_mismatch -> manual_review / manual_text_review / 不选工具。",
            "按目标用途判断:",
            "- ASR 关注语音是否可识别以及文本是否可靠；当 ASR 置信度高时，可容忍中等噪声。",
            "- TTS 关注音频干净、文本准确、响度稳定和静音比例合理。",
            "- voice_clone 关注说话人音色稳定；噪声可能需要 repair.deepfilternet_denoise。",
            "- 轻微噪声可以选择 repair.ffmpeg_denoise。",
            "- 强噪声或 high noise_floor_dbfs 可以选择 repair.deepfilternet_denoise。",
            "严格 JSON 格式（键名和值必须保持协议英文）：",
            '{"action": "允许的处理动作", "recommended_operation": "允许的处理操作", '
            '"selected_tool_ids": ["tool.id"], "reason": "简短中文决策理由", '
            '"confidence": 0.0}',
            f"样本证据包：\n{json.dumps(evidence_packet, ensure_ascii=False)}",
            f"工具注册表：\n{json.dumps(tool_packet, ensure_ascii=False)}",
            f"本地工具知识（仅用于解释，不得绕过工具注册表）：\n{knowledge.text or '无命中'}",
            f"本地知识来源：\n{json.dumps(knowledge.references, ensure_ascii=False)}",
            "现在输出 JSON 对象。",
        ]

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

    def _strategy_from_payload(
        self,
        sample_id: str,
        payload: dict[str, Any],
    ) -> ProcessingStrategy:
        action = str(payload["action"]).strip()
        if action not in self.allowed_actions:
            raise ValueError(f"不支持的 action：{action}")

        recommended_operation = str(payload["recommended_operation"]).strip()
        if not recommended_operation:
            raise ValueError("recommended_operation 不能为空")
        if recommended_operation not in self.allowed_operations:
            raise ValueError(
                f"不支持的 recommended_operation：{recommended_operation}"
            )

        selected_tool_ids = payload.get("selected_tool_ids", [])
        if not isinstance(selected_tool_ids, list):
            raise TypeError("selected_tool_ids 必须是列表")
        selected_tool_ids = [str(tool_id).strip() for tool_id in selected_tool_ids]
        for tool_id in selected_tool_ids:
            if self.tool_registry.get(tool_id) is None:
                raise ValueError(f"选择的工具未注册：{tool_id}")
        selected_tool_ids = self._normalize_selected_tool_ids(
            recommended_operation,
            selected_tool_ids,
        )

        confidence = float(payload["confidence"])
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence 必须在 0 到 1 之间")

        reason = str(payload["reason"]).strip()
        if not reason:
            raise ValueError("reason 不能为空")

        return ProcessingStrategy(
            sample_id=sample_id,
            action=action,
            recommended_operation=recommended_operation,
            selected_tool_ids=selected_tool_ids,
            reason=reason,
            confidence=confidence,
            agent_name=self.agent_name,
        )

    def _normalize_selected_tool_ids(
        self,
        recommended_operation: str,
        selected_tool_ids: list[str],
    ) -> list[str]:
        normalized = []
        for tool_id in selected_tool_ids:
            if tool_id not in normalized:
                normalized.append(tool_id)

        repair_evaluation_tools = {
            "normalize_loudness": "metrics.wave",
            "trim_silence": "metrics.wave",
            "denoise_light": "metrics.wave",
            "neural_denoise": "metrics.ffmpeg_astats",
        }
        evaluation_tool = repair_evaluation_tools.get(recommended_operation)
        if (
            evaluation_tool is not None
            and evaluation_tool not in normalized
            and self.tool_registry.get(evaluation_tool) is not None
        ):
            normalized.append(evaluation_tool)
        return normalized

    def _guard_strategy(
        self,
        result: AnalysisResult,
        strategy: ProcessingStrategy,
    ) -> ProcessingStrategy:
        violation = self._safety_violation(result, strategy)
        if violation is None:
            return strategy

        reason_prefix = (
            "降级处理：LLM Strategy 违反安全规则"
            f"({violation})"
        )
        return self._fallback_strategy(result, reason_prefix)

    def _safety_violation(
        self,
        result: AnalysisResult,
        strategy: ProcessingStrategy,
    ) -> str | None:
        issue_tags = set(result.issue_tags)

        if "too_short" in issue_tags:
            safe_operations = {"exclude_sample", "manual_strategy_review"}
            if (
                strategy.action == "repair_audio"
                or strategy.recommended_operation not in safe_operations
            ):
                return "too_short 样本必须排除或进入人工复核"

        if "too_long" in issue_tags:
            if strategy.action != "manual_review":
                return "too_long 样本必须进入人工复核"

        if "missing_transcript" in issue_tags:
            if self.profile.allow_missing_transcript:
                return None
            if (
                strategy.action != "manual_review"
                or strategy.recommended_operation != "manual_text_review"
            ):
                return "缺失文本必须进入人工文本复核"

        if "asr_low_confidence" in issue_tags:
            if self.profile.allow_low_asr_confidence:
                return None
            if (
                strategy.action != "manual_review"
                or strategy.recommended_operation != "manual_text_review"
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
                strategy.action != "manual_review"
                or strategy.recommended_operation != "manual_text_review"
            ):
                return "参考文本一致性问题必须进入人工文本复核"

        if "audio_clipping" in issue_tags:
            if strategy.action in {"repair_audio", "keep_without_processing"}:
                return "存在削波的音频不能自动修复或保留"

        if "speaker_consistency_warning" in issue_tags and strategy.action != "manual_review":
            return "声学一致性提醒必须进入人工复核"

        if (
            strategy.recommended_operation == "review_clipped_audio"
            and "audio_clipping" not in issue_tags
        ):
            return "削波复核需要 audio_clipping 证据"

        if strategy.confidence < 0.5 and strategy.action != "manual_review":
            return "低置信度自动决策必须进入人工复核"

        return None

    def _fallback_strategy(
        self,
        result: AnalysisResult,
        reason_prefix: str,
    ) -> ProcessingStrategy:
        if self.fallback_agent is None:
            return ProcessingStrategy(
                sample_id=result.sample_id,
                action="manual_review",
                recommended_operation="manual_strategy_review",
                selected_tool_ids=[],
                reason=reason_prefix,
                confidence=0.0,
                agent_name="llm_processing_strategy_guarded",
            )

        fallback = self.fallback_agent.decide(result)
        return ProcessingStrategy(
            sample_id=fallback.sample_id,
            action=fallback.action,
            recommended_operation=fallback.recommended_operation,
            selected_tool_ids=list(fallback.selected_tool_ids),
            reason=f"{reason_prefix}; {fallback.reason}",
            confidence=fallback.confidence,
            agent_name="llm_processing_strategy_guarded",
        )


def processing_strategy_to_dict(strategy: ProcessingStrategy) -> dict[str, Any]:
    return {
        "sample_id": strategy.sample_id,
        "action": strategy.action,
        "recommended_operation": strategy.recommended_operation,
        "selected_tool_ids": list(strategy.selected_tool_ids),
        "reason": strategy.reason,
        "confidence": strategy.confidence,
        "agent_name": strategy.agent_name,
    }


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


def _strategy_tool_registry_to_dict(registry: ToolRegistry) -> dict[str, Any]:
    strategy_tool_ids = {
        "asr.faster_whisper",
        "metrics.wave",
        "metrics.ffmpeg_astats",
        "repair.ffmpeg_loudness",
        "repair.ffmpeg_silence_trim",
        "repair.ffmpeg_denoise",
        "repair.deepfilternet_denoise",
    }
    strategy_tools = [
        capability
        for capability in registry.capabilities
        if capability.tool_id in strategy_tool_ids
    ]
    return tool_registry_to_dict(ToolRegistry(strategy_tools))
