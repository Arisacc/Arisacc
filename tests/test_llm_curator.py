import unittest


class TestLLMCuratorAgent(unittest.TestCase):
    def test_curator_prompt_includes_strategy_recommendation(self):
        from audio_dataset_agent.core.models import AnalysisResult, ProcessingStrategy
        from audio_dataset_agent.curation.agent import LLMCuratorAgent

        class PromptClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "decision": "repair",
                    "reason": "noise requires repair",
                    "confidence": 0.82,
                    "recommended_action": "denoise_light",
                }

        client = PromptClient()
        strategy = ProcessingStrategy(
            sample_id="sample_strategy",
            action="repair_audio",
            recommended_operation="denoise_light",
            selected_tool_ids=["repair.ffmpeg_denoise"],
            reason="background noise is high",
            confidence=0.8,
            agent_name="llm_processing_strategy",
        )
        decision = LLMCuratorAgent(client).decide(
            AnalysisResult(
                sample_id="sample_strategy",
                duration=3.0,
                transcript_length=12,
                asr_confidence=0.92,
                issue_tags=["audio_noise_high"],
            ),
            strategy=strategy,
        )

        self.assertIn("Strategy Agent", client.prompt)
        self.assertIn("denoise_light", client.prompt)
        self.assertEqual(decision.strategy_consistency, "accepted")

    def test_llm_curator_uses_structured_json_decision(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.agent import LLMCuratorAgent

        class FakeLLMClient:
            def complete(self, prompt):
                self.prompt = prompt
                return """
                {
                  "decision": "repair",
                  "reason": "speech is usable but silence should be trimmed",
                  "confidence": 0.82,
                  "recommended_action": "trim_silence"
                }
                """

        client = FakeLLMClient()
        result = AnalysisResult(
            sample_id="sample_001",
            duration=3.2,
            transcript_length=18,
            asr_confidence=0.91,
            issue_tags=["audio_high_silence_ratio"],
            evidence={"silence_ratio": 0.48},
        )

        decision = LLMCuratorAgent(client).decide(result)

        self.assertIn("sample_001", client.prompt)
        self.assertIn("如果 issue_tags 为空", client.prompt)
        self.assertIn("不要因为证据有限就直接 reject", client.prompt)
        self.assertIn("不要编造证据中不存在的质量问题", client.prompt)
        self.assertIn("优先采用第一条匹配规则", client.prompt)
        self.assertEqual(decision.sample_id, "sample_001")
        self.assertEqual(decision.decision, DecisionType.REPAIR)
        self.assertEqual(decision.reason, "speech is usable but silence should be trimmed")
        self.assertEqual(decision.confidence, 0.82)
        self.assertEqual(decision.recommended_action, "trim_silence")
        self.assertEqual(decision.policy_name, "llm_curator")

    def test_llm_curator_prompt_includes_target_policy(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.curation.agent import LLMCuratorAgent

        class PromptClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "decision": "repair",
                    "reason": "noise is risky for voice cloning",
                    "confidence": 0.78,
                    "recommended_action": "neural_denoise",
                }

        client = PromptClient()
        LLMCuratorAgent(client, target_policy="voice_clone").decide(
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
        self.assertIn("ASR", client.prompt)
        self.assertIn("TTS", client.prompt)
        self.assertIn('"min_duration_seconds": 1.5', client.prompt)
        self.assertIn('"max_noise_floor_dbfs": -40.0', client.prompt)

    def test_llm_curator_prompt_includes_strict_schema_and_decision_examples(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.curation.agent import LLMCuratorAgent

        class PromptClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "decision": "repair",
                    "reason": "noise should be repaired",
                    "confidence": 0.78,
                    "recommended_action": "denoise_light",
                }

        client = PromptClient()
        LLMCuratorAgent(client, target_policy="tts").decide(
            AnalysisResult(
                sample_id="sample_prompt",
                duration=3.0,
                transcript_length=12,
                asr_confidence=0.92,
                issue_tags=["audio_noise_high"],
            )
        )

        self.assertIn("允许的 decision 协议值：", client.prompt)
        self.assertIn("允许的 recommended_action 协议值：", client.prompt)
        self.assertIn("问题标签到决策的示例（issue_tags、keep 等为协议字段）：", client.prompt)
        self.assertIn(
            "TTS 中的 audio_noise_high -> repair / denoise_light",
            client.prompt,
        )
        self.assertIn('"recommended_action": "允许的处理动作"', client.prompt)

    def test_llm_curator_returns_review_when_response_is_invalid(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.agent import LLMCuratorAgent

        class BadLLMClient:
            def complete(self, prompt):
                return "not json"

        result = AnalysisResult(
            sample_id="sample_002",
            duration=2.4,
            transcript_length=10,
            asr_confidence=0.65,
            issue_tags=["asr_low_confidence"],
            evidence={"asr_confidence": 0.65},
        )

        decision = LLMCuratorAgent(BadLLMClient()).decide(result)

        self.assertEqual(decision.sample_id, "sample_002")
        self.assertEqual(decision.decision, DecisionType.REVIEW)
        self.assertIn("LLM Curator 返回无效结果", decision.reason)
        self.assertEqual(decision.confidence, 0.0)
        self.assertEqual(decision.recommended_action, "manual_decision_review")
        self.assertEqual(decision.policy_name, "llm_curator")

    def test_llm_curator_can_fallback_to_policy_when_response_is_invalid(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.agent import LLMCuratorAgent
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        class BadLLMClient:
            def complete(self, prompt):
                return "not json"

        result = AnalysisResult(
            sample_id="sample_005",
            duration=2.4,
            transcript_length=10,
            asr_confidence=0.95,
            issue_tags=[],
        )

        decision = LLMCuratorAgent(
            BadLLMClient(),
            fallback_policy=BaselineCurationPolicy(),
        ).decide(result)

        self.assertEqual(decision.decision, DecisionType.KEEP)
        self.assertEqual(decision.recommended_action, "prepare_for_export")
        self.assertEqual(decision.policy_name, "llm_curator_guarded")
        self.assertIn("降级处理", decision.reason)

    def test_llm_curator_falls_back_when_decision_disagrees_with_policy_guard(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.agent import LLMCuratorAgent
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        class OverStrictClient:
            def complete(self, prompt):
                return {
                    "decision": "reject",
                    "reason": "too risky",
                    "confidence": 0.9,
                    "recommended_action": "exclude_sample",
                }

        result = AnalysisResult(
            sample_id="sample_006",
            duration=2.4,
            transcript_length=10,
            asr_confidence=0.42,
            issue_tags=["asr_low_confidence"],
        )

        decision = LLMCuratorAgent(
            OverStrictClient(),
            fallback_policy=BaselineCurationPolicy(),
        ).decide(result)

        self.assertEqual(decision.decision, DecisionType.REVIEW)
        self.assertEqual(decision.recommended_action, "manual_text_review")
        self.assertEqual(decision.policy_name, "llm_curator_guarded")

    def test_llm_curator_keeps_safe_autonomous_decision_that_differs_from_baseline(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.agent import LLMCuratorAgent
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        class SafeNeuralDenoiseClient:
            def complete(self, prompt):
                return {
                    "decision": "repair",
                    "reason": "noise floor is high enough to justify neural denoise",
                    "confidence": 0.82,
                    "recommended_action": "neural_denoise",
                }

        decision = LLMCuratorAgent(
            SafeNeuralDenoiseClient(),
            fallback_policy=BaselineCurationPolicy(target_policy="tts"),
            target_policy="tts",
        ).decide(
            AnalysisResult(
                sample_id="sample_noise_autonomous",
                duration=3.0,
                transcript_length=12,
                asr_confidence=0.92,
                issue_tags=["audio_noise_high"],
                evidence={"audio_metrics": {"noise_floor_dbfs": -25.0}},
            )
        )

        self.assertEqual(decision.decision, DecisionType.REPAIR)
        self.assertEqual(decision.recommended_action, "neural_denoise")
        self.assertEqual(decision.policy_name, "llm_curator")

    def test_llm_curator_falls_back_when_repair_action_lacks_matching_issue_tag(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.agent import LLMCuratorAgent
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        class UnsupportedRepairClient:
            def complete(self, prompt):
                return {
                    "decision": "repair",
                    "reason": "noise floor is high",
                    "confidence": 0.95,
                    "recommended_action": "denoise_light",
                }

        decision = LLMCuratorAgent(
            UnsupportedRepairClient(),
            fallback_policy=BaselineCurationPolicy(target_policy="tts"),
            target_policy="tts",
        ).decide(
            AnalysisResult(
                sample_id="sample_clean",
                duration=3.0,
                transcript_length=20,
                asr_confidence=0.92,
                issue_tags=[],
                evidence={
                    "audio_metrics": {
                        "noise_floor_dbfs": -55.0,
                        "silence_ratio": 0.12,
                    }
                },
            )
        )

        self.assertEqual(decision.decision, DecisionType.KEEP)
        self.assertEqual(decision.recommended_action, "prepare_for_export")
        self.assertEqual(decision.policy_name, "llm_curator_guarded")

    def test_llm_curator_accepts_markdown_fenced_json(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.agent import LLMCuratorAgent

        class FencedClient:
            def complete(self, prompt):
                return """```json
                {
                  "decision": "keep",
                  "reason": "sample has no blocking issues",
                  "confidence": 0.74,
                  "recommended_action": "prepare_for_export"
                }
                ```"""

        result = AnalysisResult(
            sample_id="sample_004",
            duration=2.4,
            transcript_length=16,
            asr_confidence=0.92,
        )

        decision = LLMCuratorAgent(FencedClient()).decide(result)

        self.assertEqual(decision.decision, DecisionType.KEEP)
        self.assertEqual(decision.recommended_action, "prepare_for_export")

    def test_llm_curator_accepts_json_wrapped_in_explanation_text(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.agent import LLMCuratorAgent

        class WrappedClient:
            def complete(self, prompt):
                return """
                根据音频证据，样本需要先做响度处理。
                {
                  "decision": "repair",
                  "reason": "audio is too quiet but otherwise usable",
                  "confidence": 0.79,
                  "recommended_action": "normalize_loudness"
                }
                这个 JSON 是最终决策。
                """

        decision = LLMCuratorAgent(WrappedClient()).decide(
            AnalysisResult(
                sample_id="sample_wrapped",
                duration=2.4,
                transcript_length=16,
                asr_confidence=0.92,
                issue_tags=["audio_too_quiet"],
            )
        )

        self.assertEqual(decision.decision, DecisionType.REPAIR)
        self.assertEqual(decision.recommended_action, "normalize_loudness")


class TestLLMCuratorServiceIntegration(unittest.TestCase):
    def test_curator_receives_strategy_and_records_override(self):
        from audio_dataset_agent.core.models import (
            AnalysisResult,
            Batch,
            DecisionType,
            ProcessingStrategy,
        )
        from audio_dataset_agent.curation.agent import LLMCuratorAgent
        from audio_dataset_agent.curation.service import curate_candidate_samples

        class OverrideClient:
            def complete(self, prompt):
                return {
                    "decision": "review",
                    "reason": "the repair suggestion needs human confirmation",
                    "confidence": 0.8,
                    "recommended_action": "manual_decision_review",
                }

        batch = Batch(
            batch_id="batch_strategy_handoff",
            input_dir="input",
            output_dir="out",
            target_policy="voice_clone",
            analysis_results=[
                AnalysisResult(
                    sample_id="sample_strategy_handoff",
                    duration=3.0,
                    transcript_length=12,
                    asr_confidence=0.92,
                    issue_tags=["audio_noise_high"],
                )
            ],
            processing_strategies=[
                ProcessingStrategy(
                    sample_id="sample_strategy_handoff",
                    action="repair_audio",
                    recommended_operation="neural_denoise",
                    selected_tool_ids=["repair.deepfilternet_denoise"],
                    reason="voice clone needs cleaner timbre",
                    confidence=0.8,
                )
            ],
        )

        curated = curate_candidate_samples(
            batch,
            policy=LLMCuratorAgent(OverrideClient(), target_policy="voice_clone"),
        )

        self.assertEqual(curated.curator_decisions[0].decision, DecisionType.REVIEW)
        self.assertEqual(curated.curator_decisions[0].strategy_consistency, "overridden")

    def test_curate_candidate_samples_uses_batch_target_policy_by_default(self):
        from audio_dataset_agent.core.models import AnalysisResult, Batch, DecisionType
        from audio_dataset_agent.curation.service import curate_candidate_samples

        batch = Batch(
            batch_id="batch_target",
            input_dir="input",
            output_dir="out",
            target_policy="asr",
            analysis_results=[
                AnalysisResult(
                    sample_id="sample_noise",
                    duration=3.0,
                    transcript_length=12,
                    asr_confidence=0.94,
                    issue_tags=["audio_noise_high"],
                )
            ],
        )

        curated = curate_candidate_samples(batch)

        self.assertEqual(curated.curator_decisions[0].decision, DecisionType.KEEP)
        self.assertEqual(
            curated.curator_decisions[0].recommended_action,
            "prepare_for_export",
        )

    def test_curate_candidate_samples_accepts_llm_curator_agent(self):
        from audio_dataset_agent.core.models import AnalysisResult, Batch, DecisionType
        from audio_dataset_agent.curation.agent import LLMCuratorAgent
        from audio_dataset_agent.curation.service import curate_candidate_samples

        class KeepClient:
            def complete(self, prompt):
                return {
                    "decision": "keep",
                    "reason": "sample passes the policy",
                    "confidence": 0.77,
                    "recommended_action": "prepare_for_export",
                }

        batch = Batch(
            batch_id="batch_001",
            input_dir="input",
            output_dir="out",
            target_policy="voice_clone",
            analysis_results=[
                AnalysisResult(
                    sample_id="sample_003",
                    duration=2.1,
                    transcript_length=12,
                    asr_confidence=0.9,
                )
            ],
        )

        curated = curate_candidate_samples(batch, policy=LLMCuratorAgent(KeepClient()))

        self.assertEqual(len(curated.curator_decisions), 1)
        self.assertEqual(curated.curator_decisions[0].decision, DecisionType.KEEP)
        self.assertEqual(curated.curator_decisions[0].policy_name, "llm_curator")


class TestTransformersQwenClient(unittest.TestCase):
    def test_transformers_qwen_client_generates_text_from_prompt(self):
        import torch

        from audio_dataset_agent.curation.qwen_client import TransformersQwenClient

        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                return "rendered chat"

            def __call__(self, texts, return_tensors):
                self.texts = texts
                return {"input_ids": torch.tensor([[1, 2]])}

            def decode(self, token_ids, skip_special_tokens):
                return '{"decision":"keep"}'

        class FakeModel:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[1, 2, 3, 4]])

        tokenizer = FakeTokenizer()
        model = FakeModel()
        client = TransformersQwenClient(
            model_path="local-model",
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=24,
        )

        text = client.complete("curate this sample")

        self.assertEqual(text, '{"decision":"keep"}')
        self.assertEqual(tokenizer.messages[0]["role"], "user")
        self.assertEqual(tokenizer.messages[0]["content"], "curate this sample")
        self.assertEqual(model.kwargs["max_new_tokens"], 24)

    def test_transformers_qwen_client_accepts_device_map_and_torch_dtype(self):
        from audio_dataset_agent.curation.qwen_client import TransformersQwenClient

        client = TransformersQwenClient(
            model_path="local-model",
            device_map="auto",
            torch_dtype="auto",
        )

        self.assertEqual(client.device_map, "auto")
        self.assertEqual(client.torch_dtype, "auto")

    def test_transformers_qwen_client_moves_inputs_to_model_device(self):
        from audio_dataset_agent.curation.qwen_client import TransformersQwenClient

        class FakeTensor:
            shape = (1, 2)

            def __init__(self):
                self.moved_to = None

            def to(self, device):
                self.moved_to = device
                return self

        class FakeTokenizer:
            def __init__(self):
                self.input_ids = FakeTensor()

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "rendered chat"

            def __call__(self, texts, return_tensors):
                return {"input_ids": self.input_ids}

            def decode(self, token_ids, skip_special_tokens):
                return "OK"

        class FakeOutputRow:
            def __getitem__(self, item):
                return [3, 4]

        class FakeOutputs:
            def __getitem__(self, item):
                return FakeOutputRow()

        class FakeModel:
            device = "cuda:0"

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return FakeOutputs()

        tokenizer = FakeTokenizer()
        model = FakeModel()
        client = TransformersQwenClient(
            model_path="local-model",
            tokenizer=tokenizer,
            model=model,
        )

        text = client.complete("curate this sample")

        self.assertEqual(text, "OK")
        self.assertEqual(tokenizer.input_ids.moved_to, "cuda:0")
