import unittest
from pathlib import Path


class TestDatasetProfiles(unittest.TestCase):
    def test_profiles_define_target_specific_text_requirements(self):
        from audio_dataset_agent.profiles.policy import get_dataset_profile

        asr = get_dataset_profile("asr")
        tts = get_dataset_profile("tts")
        voice_clone = get_dataset_profile("voice_clone")

        self.assertTrue(asr.requires_asr)
        self.assertTrue(asr.text_reliability_required)
        self.assertTrue(tts.requires_text)
        self.assertTrue(tts.text_reliability_required)
        self.assertFalse(voice_clone.requires_asr)
        self.assertFalse(voice_clone.text_reliability_required)
        self.assertTrue(voice_clone.allow_missing_transcript)
        self.assertEqual(voice_clone.preferred_denoise_operation, "neural_denoise")

    def test_unknown_profile_is_rejected(self):
        from audio_dataset_agent.profiles.policy import get_dataset_profile

        with self.assertRaises(ValueError):
            get_dataset_profile("unknown")

    def test_profiles_define_distinct_audio_quality_thresholds(self):
        from audio_dataset_agent.profiles.policy import get_dataset_profile

        tts = get_dataset_profile("tts")
        asr = get_dataset_profile("asr")
        voice_clone = get_dataset_profile("voice_clone")

        self.assertGreater(tts.min_duration_seconds, asr.min_duration_seconds)
        self.assertLess(tts.max_silence_ratio, asr.max_silence_ratio)
        self.assertLess(tts.max_noise_floor_dbfs, asr.max_noise_floor_dbfs)
        self.assertGreater(voice_clone.min_duration_seconds, asr.min_duration_seconds)
        self.assertIsNotNone(tts.max_duration_seconds)

    def test_inspector_uses_profile_thresholds_when_policy_is_not_explicit(self):
        from audio_dataset_agent.core.models import Batch, CandidateSample
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.inspection.service import inspect_candidate_samples
        from audio_dataset_agent.profiles.policy import get_dataset_profile

        class FixedMetrics:
            def analyze(self, audio_path):
                return AudioMetrics(
                    rms_dbfs=-30.0,
                    peak_dbfs=-3.0,
                    clipping_ratio=0.0,
                    silence_ratio=0.5,
                    noise_floor_dbfs=-35.0,
                )

        def inspect(target_policy):
            sample = CandidateSample(
                sample_id=f"sample_{target_policy}",
                batch_id="batch_profile",
                asset_id="asset_0001",
                source_path=Path("sample.wav"),
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
                transcript="测试文本",
                transcript_source="manual",
                asr_confidence=0.95,
            )
            batch = Batch(
                batch_id="batch_profile",
                input_dir=Path("input"),
                output_dir=Path("output"),
                target_policy=target_policy,
                candidate_samples=[sample],
            )
            return inspect_candidate_samples(
                batch,
                audio_metrics_analyzer=FixedMetrics(),
            ).analysis_results[0]

        tts_result = inspect("tts")
        asr_result = inspect("asr")

        self.assertIn("audio_high_silence_ratio", tts_result.issue_tags)
        self.assertIn("audio_noise_high", tts_result.issue_tags)
        self.assertNotIn("audio_high_silence_ratio", asr_result.issue_tags)
        self.assertNotIn("audio_noise_high", asr_result.issue_tags)
        self.assertEqual(
            tts_result.evidence["quality_policy"]["max_silence_ratio"],
            get_dataset_profile("tts").max_silence_ratio,
        )

    def test_inspector_marks_profile_specific_too_long_samples(self):
        from audio_dataset_agent.core.models import Batch, CandidateSample
        from audio_dataset_agent.inspection.service import inspect_candidate_samples

        sample = CandidateSample(
            sample_id="sample_long",
            batch_id="batch_long",
            asset_id="asset_0001",
            source_path=Path("long.wav"),
            start_time=0.0,
            end_time=20.0,
            duration=20.0,
            transcript="这是一条过长的 TTS 样本",
            transcript_source="manual",
            asr_confidence=0.95,
        )
        batch = Batch(
            batch_id="batch_long",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="tts",
            candidate_samples=[sample],
        )

        result = inspect_candidate_samples(batch).analysis_results[0]

        self.assertIn("too_long", result.issue_tags)

    def test_too_long_sample_is_routed_to_manual_review(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        result = AnalysisResult(
            sample_id="sample_long",
            duration=20.0,
            transcript_length=12,
            asr_confidence=0.95,
            issue_tags=["too_long"],
            evidence={},
        )

        strategy = BaselineProcessingStrategyAgent(
            default_tool_registry(),
            target_policy="tts",
        ).decide(result)
        decision = BaselineCurationPolicy(target_policy="tts").decide(
            result,
            strategy,
        )

        self.assertEqual(strategy.action, "manual_review")
        self.assertEqual(strategy.recommended_operation, "manual_strategy_review")
        self.assertEqual(decision.decision, DecisionType.REVIEW)
        self.assertEqual(decision.recommended_action, "manual_decision_review")


if __name__ == "__main__":
    unittest.main()
