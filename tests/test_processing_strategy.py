import json
import unittest


class TestBaselineProcessingStrategyAgent(unittest.TestCase):
    def test_strategy_applies_target_policy_to_noisy_audio(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        result = AnalysisResult(
            sample_id="sample_noise",
            duration=3.0,
            transcript_length=16,
            asr_confidence=0.94,
            issue_tags=["audio_noise_high"],
            evidence={"audio_metrics": {"noise_floor_dbfs": -30.0}},
        )
        registry = default_tool_registry()

        asr_strategy = BaselineProcessingStrategyAgent(
            registry,
            target_policy="asr",
        ).decide(result)
        tts_strategy = BaselineProcessingStrategyAgent(
            registry,
            target_policy="tts",
        ).decide(result)
        clone_strategy = BaselineProcessingStrategyAgent(
            registry,
            target_policy="voice_clone",
        ).decide(result)

        self.assertEqual(asr_strategy.action, "keep_without_processing")
        self.assertEqual(asr_strategy.recommended_operation, "prepare_for_export")
        self.assertEqual(tts_strategy.action, "repair_audio")
        self.assertEqual(tts_strategy.recommended_operation, "denoise_light")
        self.assertEqual(clone_strategy.action, "repair_audio")
        self.assertEqual(clone_strategy.recommended_operation, "neural_denoise")
        self.assertIn("repair.deepfilternet_denoise", clone_strategy.selected_tool_ids)

    def test_strategy_selects_loudness_repair_for_quiet_audio(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        decision = BaselineProcessingStrategyAgent(default_tool_registry()).decide(
            AnalysisResult(
                sample_id="sample_0001",
                duration=2.0,
                transcript_length=11,
                asr_confidence=0.95,
                issue_tags=["audio_too_quiet"],
                evidence={},
            )
        )

        self.assertEqual(decision.action, "repair_audio")
        self.assertEqual(decision.recommended_operation, "normalize_loudness")
        self.assertIn("repair.ffmpeg_loudness", decision.selected_tool_ids)
        self.assertIn("metrics.wave", decision.selected_tool_ids)

    def test_strategy_selects_silence_trimming_for_high_silence(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        decision = BaselineProcessingStrategyAgent(default_tool_registry()).decide(
            AnalysisResult(
                sample_id="sample_0001",
                duration=4.0,
                transcript_length=11,
                asr_confidence=0.95,
                issue_tags=["audio_high_silence_ratio"],
                evidence={},
            )
        )

        self.assertEqual(decision.action, "repair_audio")
        self.assertEqual(decision.recommended_operation, "trim_silence")
        self.assertIn("repair.ffmpeg_silence_trim", decision.selected_tool_ids)

    def test_strategy_selects_light_denoise_for_noisy_audio(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        decision = BaselineProcessingStrategyAgent(default_tool_registry()).decide(
            AnalysisResult(
                sample_id="sample_0001",
                duration=3.0,
                transcript_length=11,
                asr_confidence=0.95,
                issue_tags=["audio_noise_high"],
                evidence={"audio_metrics": {"noise_floor_dbfs": -32.0}},
            )
        )

        self.assertEqual(decision.action, "repair_audio")
        self.assertEqual(decision.recommended_operation, "denoise_light")
        self.assertEqual(
            decision.selected_tool_ids,
            ["repair.ffmpeg_denoise", "metrics.wave"],
        )

    def test_strategy_routes_low_confidence_asr_to_manual_review(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        decision = BaselineProcessingStrategyAgent(default_tool_registry()).decide(
            AnalysisResult(
                sample_id="sample_0001",
                duration=3.0,
                transcript_length=8,
                asr_confidence=0.42,
                issue_tags=["asr_low_confidence"],
                evidence={},
            )
        )

        self.assertEqual(decision.action, "manual_review")
        self.assertEqual(decision.recommended_operation, "manual_text_review")
        self.assertIn("asr.faster_whisper", decision.selected_tool_ids)

    def test_strategy_uses_profile_to_allow_missing_text_for_voice_clone(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        decision = BaselineProcessingStrategyAgent(
            default_tool_registry(),
            target_policy="voice_clone",
        ).decide(
            AnalysisResult(
                sample_id="sample_no_text",
                duration=3.0,
                transcript_length=0,
                asr_confidence=None,
                issue_tags=["missing_transcript"],
                evidence={},
            )
        )

        self.assertEqual(decision.action, "keep_without_processing")
        self.assertEqual(decision.recommended_operation, "prepare_for_export")

    def test_strategy_keeps_clean_sample_without_processing(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import (
            BaselineProcessingStrategyAgent,
            processing_strategy_to_dict,
        )
        from audio_dataset_agent.tools.registry import default_tool_registry

        decision = BaselineProcessingStrategyAgent(default_tool_registry()).decide(
            AnalysisResult(
                sample_id="sample_0001",
                duration=3.0,
                transcript_length=11,
                asr_confidence=0.95,
                issue_tags=[],
                evidence={},
            )
        )
        payload = processing_strategy_to_dict(decision)
        encoded = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(decision.action, "keep_without_processing")
        self.assertEqual(decision.selected_tool_ids, [])
        self.assertIn('"sample_id": "sample_0001"', encoded)


class TestProcessingStrategyService(unittest.TestCase):
    def test_select_processing_strategies_attaches_decisions_to_batch(self):
        from pathlib import Path

        from audio_dataset_agent.core.models import AnalysisResult, Batch
        from audio_dataset_agent.strategy.service import select_processing_strategies
        from audio_dataset_agent.tools.registry import default_tool_registry

        batch = Batch(
            batch_id="batch_0001",
            input_dir=Path("input"),
            output_dir=Path("out"),
            target_policy="voice_clone",
            analysis_results=[
                AnalysisResult(
                    sample_id="sample_0001",
                    duration=2.0,
                    transcript_length=11,
                    asr_confidence=0.95,
                    issue_tags=["audio_too_quiet"],
                    evidence={},
                )
            ],
        )

        decided = select_processing_strategies(
            batch,
            tool_registry=default_tool_registry(),
        )

        self.assertEqual(len(decided.processing_strategies), 1)
        self.assertEqual(
            decided.processing_strategies[0].recommended_operation,
            "normalize_loudness",
        )

    def test_select_processing_strategies_uses_batch_target_policy(self):
        from pathlib import Path

        from audio_dataset_agent.core.models import AnalysisResult, Batch
        from audio_dataset_agent.strategy.service import select_processing_strategies
        from audio_dataset_agent.tools.registry import default_tool_registry

        batch = Batch(
            batch_id="batch_0002",
            input_dir=Path("input"),
            output_dir=Path("out"),
            target_policy="voice_clone",
            analysis_results=[
                AnalysisResult(
                    sample_id="sample_noise",
                    duration=3.0,
                    transcript_length=16,
                    asr_confidence=0.94,
                    issue_tags=["audio_noise_high"],
                    evidence={"audio_metrics": {"noise_floor_dbfs": -30.0}},
                )
            ],
        )

        decided = select_processing_strategies(
            batch,
            tool_registry=default_tool_registry(),
        )

        self.assertEqual(
            decided.processing_strategies[0].recommended_operation,
            "neural_denoise",
        )


class TestLLMProcessingStrategyAgent(unittest.TestCase):
    def test_llm_strategy_uses_structured_json_decision(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import LLMProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        class FakeLLMClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "action": "repair_audio",
                    "recommended_operation": "trim_silence",
                    "selected_tool_ids": ["repair.ffmpeg_silence_trim", "metrics.wave"],
                    "reason": "silence ratio is high",
                    "confidence": 0.83,
                }

        client = FakeLLMClient()
        decision = LLMProcessingStrategyAgent(
            client,
            tool_registry=default_tool_registry(),
        ).decide(
            AnalysisResult(
                sample_id="sample_0002",
                duration=4.0,
                transcript_length=16,
                asr_confidence=0.95,
                issue_tags=["audio_high_silence_ratio"],
                evidence={"audio_metrics": {"silence_ratio": 0.72}},
            )
        )

        self.assertIn("工具注册表", client.prompt)
        self.assertIn("sample_0002", client.prompt)
        self.assertEqual(decision.action, "repair_audio")
        self.assertEqual(decision.recommended_operation, "trim_silence")
        self.assertEqual(decision.selected_tool_ids, ["repair.ffmpeg_silence_trim", "metrics.wave"])
        self.assertEqual(decision.agent_name, "llm_processing_strategy")

    def test_llm_strategy_falls_back_when_response_is_invalid(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import (
            BaselineProcessingStrategyAgent,
            LLMProcessingStrategyAgent,
        )
        from audio_dataset_agent.tools.registry import default_tool_registry

        class BadClient:
            def complete(self, prompt):
                return "not json"

        registry = default_tool_registry()
        decision = LLMProcessingStrategyAgent(
            BadClient(),
            tool_registry=registry,
            fallback_agent=BaselineProcessingStrategyAgent(registry),
        ).decide(
            AnalysisResult(
                sample_id="sample_0003",
                duration=2.0,
                transcript_length=11,
                asr_confidence=0.95,
                issue_tags=["audio_too_quiet"],
            )
        )

        self.assertEqual(decision.action, "repair_audio")
        self.assertEqual(decision.recommended_operation, "normalize_loudness")
        self.assertEqual(decision.agent_name, "llm_processing_strategy_guarded")
        self.assertIn("降级处理", decision.reason)

    def test_llm_strategy_guard_falls_back_on_policy_mismatch(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import (
            BaselineProcessingStrategyAgent,
            LLMProcessingStrategyAgent,
        )
        from audio_dataset_agent.tools.registry import default_tool_registry

        class OverStrictClient:
            def complete(self, prompt):
                return {
                    "action": "exclude_sample",
                    "recommended_operation": "exclude_sample",
                    "selected_tool_ids": [],
                    "reason": "too risky",
                    "confidence": 0.9,
                }

        registry = default_tool_registry()
        decision = LLMProcessingStrategyAgent(
            OverStrictClient(),
            tool_registry=registry,
            fallback_agent=BaselineProcessingStrategyAgent(registry),
        ).decide(
            AnalysisResult(
                sample_id="sample_0004",
                duration=3.0,
                transcript_length=11,
                asr_confidence=0.42,
                issue_tags=["asr_low_confidence"],
            )
        )

        self.assertEqual(decision.action, "manual_review")
        self.assertEqual(decision.recommended_operation, "manual_text_review")
        self.assertEqual(decision.agent_name, "llm_processing_strategy_guarded")

    def test_llm_strategy_accepts_autonomous_deepfilternet_choice(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import (
            BaselineProcessingStrategyAgent,
            LLMProcessingStrategyAgent,
        )
        from audio_dataset_agent.tools.registry import default_tool_registry

        class NeuralDenoiseClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "action": "repair_audio",
                    "recommended_operation": "neural_denoise",
                    "selected_tool_ids": [
                        "repair.deepfilternet_denoise",
                        "metrics.ffmpeg_astats",
                    ],
                    "reason": "噪声底为 -25.0 dBFS，轻量降噪可能不足",
                    "confidence": 0.78,
                }

        registry = default_tool_registry()
        client = NeuralDenoiseClient()
        decision = LLMProcessingStrategyAgent(
            client,
            tool_registry=registry,
            fallback_agent=BaselineProcessingStrategyAgent(registry),
        ).decide(
            AnalysisResult(
                sample_id="sample_0006",
                duration=4.0,
                transcript_length=18,
                asr_confidence=0.94,
                issue_tags=["audio_noise_high"],
                evidence={"audio_metrics": {"noise_floor_dbfs": -25.0}},
            )
        )

        self.assertEqual(decision.action, "repair_audio")
        self.assertEqual(decision.recommended_operation, "neural_denoise")
        self.assertEqual(
            decision.selected_tool_ids,
            ["repair.deepfilternet_denoise", "metrics.ffmpeg_astats"],
        )
        self.assertEqual(decision.agent_name, "llm_processing_strategy")

    def test_llm_strategy_adds_evaluation_tool_for_repair_choice(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import LLMProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        class RepairOnlyClient:
            def complete(self, prompt):
                return {
                    "action": "repair_audio",
                    "recommended_operation": "denoise_light",
                    "selected_tool_ids": ["repair.ffmpeg_denoise"],
                    "reason": "noise is high",
                    "confidence": 0.84,
                }

        decision = LLMProcessingStrategyAgent(
            RepairOnlyClient(),
            tool_registry=default_tool_registry(),
        ).decide(
            AnalysisResult(
                sample_id="sample_repair_eval_tool",
                duration=3.0,
                transcript_length=12,
                asr_confidence=0.91,
                issue_tags=["audio_noise_high"],
            )
        )

        self.assertEqual(
            decision.selected_tool_ids,
            ["repair.ffmpeg_denoise", "metrics.wave"],
        )

    def test_llm_strategy_guard_rejects_unsafe_keep_with_missing_transcript(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import (
            BaselineProcessingStrategyAgent,
            LLMProcessingStrategyAgent,
        )
        from audio_dataset_agent.tools.registry import default_tool_registry

        class UnsafeKeepClient:
            def complete(self, prompt):
                return {
                    "action": "keep_without_processing",
                    "recommended_operation": "prepare_for_export",
                    "selected_tool_ids": [],
                    "reason": "音频听起来干净",
                    "confidence": 0.91,
                }

        registry = default_tool_registry()
        decision = LLMProcessingStrategyAgent(
            UnsafeKeepClient(),
            tool_registry=registry,
            fallback_agent=BaselineProcessingStrategyAgent(registry),
        ).decide(
            AnalysisResult(
                sample_id="sample_0007",
                duration=3.0,
                transcript_length=0,
                asr_confidence=None,
                issue_tags=["missing_transcript"],
            )
        )

        self.assertEqual(decision.action, "manual_review")
        self.assertEqual(decision.recommended_operation, "manual_text_review")
        self.assertEqual(decision.agent_name, "llm_processing_strategy_guarded")

    def test_llm_strategy_guard_rejects_unsafe_repair_with_clipping(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import (
            BaselineProcessingStrategyAgent,
            LLMProcessingStrategyAgent,
        )
        from audio_dataset_agent.tools.registry import default_tool_registry

        class UnsafeRepairClient:
            def complete(self, prompt):
                return {
                    "action": "repair_audio",
                    "recommended_operation": "denoise_light",
                    "selected_tool_ids": ["repair.ffmpeg_denoise"],
                    "reason": "先降噪处理",
                    "confidence": 0.72,
                }

        registry = default_tool_registry()
        decision = LLMProcessingStrategyAgent(
            UnsafeRepairClient(),
            tool_registry=registry,
            fallback_agent=BaselineProcessingStrategyAgent(registry),
        ).decide(
            AnalysisResult(
                sample_id="sample_0008",
                duration=2.5,
                transcript_length=12,
                asr_confidence=0.92,
                issue_tags=["audio_clipping", "audio_noise_high"],
                evidence={"audio_metrics": {"clipping_ratio": 0.08}},
            )
        )

        self.assertEqual(decision.action, "manual_review")
        self.assertEqual(decision.recommended_operation, "review_clipped_audio")
        self.assertEqual(decision.agent_name, "llm_processing_strategy_guarded")

    def test_llm_strategy_guard_requires_clipping_evidence_for_clipping_review(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import (
            BaselineProcessingStrategyAgent,
            LLMProcessingStrategyAgent,
        )
        from audio_dataset_agent.tools.registry import default_tool_registry

        class UnsupportedClippingReviewClient:
            def complete(self, prompt):
                return {
                    "action": "manual_review",
                    "recommended_operation": "review_clipped_audio",
                    "selected_tool_ids": ["metrics.wave"],
                    "reason": "疑似存在削波风险，需要复核",
                    "confidence": 0.7,
                }

        registry = default_tool_registry()
        decision = LLMProcessingStrategyAgent(
            UnsupportedClippingReviewClient(),
            tool_registry=registry,
            fallback_agent=BaselineProcessingStrategyAgent(registry),
        ).decide(
            AnalysisResult(
                sample_id="sample_0009",
                duration=2.5,
                transcript_length=12,
                asr_confidence=0.92,
                issue_tags=[],
                evidence={"audio_metrics": {"clipping_ratio": 0.0}},
            )
        )

        self.assertEqual(decision.action, "keep_without_processing")
        self.assertEqual(decision.recommended_operation, "prepare_for_export")
        self.assertEqual(decision.agent_name, "llm_processing_strategy_guarded")

    def test_llm_strategy_prompt_describes_safety_rules_and_deepfilternet(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import LLMProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        class PromptCaptureClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "action": "repair_audio",
                    "recommended_operation": "denoise_light",
                    "selected_tool_ids": ["repair.ffmpeg_denoise"],
                    "reason": "噪声较高",
                    "confidence": 0.7,
                }

        client = PromptCaptureClient()
        LLMProcessingStrategyAgent(
            client,
            tool_registry=default_tool_registry(),
        ).decide(
            AnalysisResult(
                sample_id="sample_0009",
                duration=3.0,
                transcript_length=12,
                asr_confidence=0.91,
                issue_tags=["audio_noise_high"],
            )
        )

        self.assertIn("处理策略 Agent", client.prompt)
        self.assertIn("硬性安全规则", client.prompt)
        self.assertIn("repair.deepfilternet_denoise", client.prompt)

    def test_llm_strategy_prompt_includes_target_policy(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import LLMProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        class PromptCaptureClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "action": "repair_audio",
                    "recommended_operation": "neural_denoise",
                    "selected_tool_ids": ["repair.deepfilternet_denoise"],
                    "reason": "voice cloning needs stable speaker timbre",
                    "confidence": 0.72,
                }

        client = PromptCaptureClient()
        LLMProcessingStrategyAgent(
            client,
            tool_registry=default_tool_registry(),
            target_policy="voice_clone",
        ).decide(
            AnalysisResult(
                sample_id="sample_policy",
                duration=3.0,
                transcript_length=12,
                asr_confidence=0.92,
                issue_tags=["audio_noise_high"],
            )
        )

        self.assertIn("target_policy", client.prompt)
        self.assertIn("voice_clone", client.prompt)

    def test_llm_strategy_prompt_includes_strict_schema_and_issue_tool_map(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import LLMProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        class PromptCaptureClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "action": "repair_audio",
                    "recommended_operation": "denoise_light",
                    "selected_tool_ids": ["repair.ffmpeg_denoise", "metrics.wave"],
                    "reason": "noise is high",
                    "confidence": 0.7,
                }

        client = PromptCaptureClient()
        LLMProcessingStrategyAgent(
            client,
            tool_registry=default_tool_registry(),
        ).decide(
            AnalysisResult(
                sample_id="sample_prompt_schema",
                duration=3.0,
                transcript_length=12,
                asr_confidence=0.91,
                issue_tags=["audio_noise_high"],
            )
        )

        self.assertIn("允许的 action 协议值：", client.prompt)
        self.assertIn("允许的 recommended_operation 协议值：", client.prompt)
        self.assertIn('"max_silence_ratio": 0.35', client.prompt)
        self.assertIn('"max_duration_seconds": 15.0', client.prompt)
        self.assertIn("问题标签到策略的示例:", client.prompt)
        self.assertIn(
            "audio_noise_high -> repair_audio / denoise_light / repair.ffmpeg_denoise, metrics.wave",
            client.prompt,
        )
        self.assertIn(
            "asr_low_confidence -> manual_review / manual_text_review / asr.faster_whisper",
            client.prompt,
        )
        self.assertIn('"selected_tool_ids": ["tool.id"]', client.prompt)

    def test_llm_strategy_accepts_markdown_fenced_json(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import LLMProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        class FencedClient:
            def complete(self, prompt):
                return """```json
                {
                  "action": "keep_without_processing",
                  "recommended_operation": "prepare_for_export",
                  "selected_tool_ids": [],
                  "reason": "clean sample",
                  "confidence": 0.76
                }
                ```"""

        decision = LLMProcessingStrategyAgent(
            FencedClient(),
            tool_registry=default_tool_registry(),
        ).decide(
            AnalysisResult(
                sample_id="sample_0005",
                duration=3.0,
                transcript_length=11,
                asr_confidence=0.95,
                issue_tags=[],
            )
        )

        self.assertEqual(decision.action, "keep_without_processing")
        self.assertEqual(decision.recommended_operation, "prepare_for_export")

    def test_llm_strategy_accepts_json_wrapped_in_explanation_text(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.strategy.agent import LLMProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        class WrappedClient:
            def complete(self, prompt):
                return """
                先说明判断：静音比例过高，适合裁剪。
                {
                  "action": "repair_audio",
                  "recommended_operation": "trim_silence",
                  "selected_tool_ids": ["repair.ffmpeg_silence_trim", "metrics.wave"],
                  "reason": "silence ratio is high",
                  "confidence": 0.82
                }
                上面 JSON 是工具选择结果。
                """

        decision = LLMProcessingStrategyAgent(
            WrappedClient(),
            tool_registry=default_tool_registry(),
        ).decide(
            AnalysisResult(
                sample_id="sample_wrapped_strategy",
                duration=4.0,
                transcript_length=11,
                asr_confidence=0.95,
                issue_tags=["audio_high_silence_ratio"],
            )
        )

        self.assertEqual(decision.action, "repair_audio")
        self.assertEqual(decision.recommended_operation, "trim_silence")
        self.assertEqual(
            decision.selected_tool_ids,
            ["repair.ffmpeg_silence_trim", "metrics.wave"],
        )


class TestProcessingStrategyCli(unittest.TestCase):
    def test_cli_accepts_strategy_argument(self):
        from audio_dataset_agent.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--input-dir",
                "input",
                "--output-dir",
                "out",
                "--export-dir",
                "exports",
                "--orchestrator",
                "supervisor",
                "--strategy",
                "qwen-local",
            ]
        )

        self.assertEqual(args.strategy, "qwen-local")


if __name__ == "__main__":
    unittest.main()
