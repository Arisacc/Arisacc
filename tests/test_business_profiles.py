import unittest
from pathlib import Path


class TestTextQuality(unittest.TestCase):
    def test_chinese_text_normalization_removes_formatting_noise(self):
        from audio_dataset_agent.text_quality.analyzer import normalize_text

        self.assertEqual(normalize_text(" 你好，ＷＯＲＬＤ！\n"), "你好world")

    def test_alignment_calculates_cer_and_wer(self):
        from audio_dataset_agent.text_quality.analyzer import evaluate_text_alignment

        result = evaluate_text_alignment("今天天气很好", "今天天气好")

        self.assertAlmostEqual(result.cer, 1 / 6)
        self.assertGreaterEqual(result.wer, result.cer)
        self.assertEqual(result.normalized_reference, "今天天气很好")
        self.assertEqual(result.normalized_candidate, "今天天气好")


class TestProfileTextInspection(unittest.TestCase):
    def _inspect(self, target_policy, reference_text, candidate_text, metadata_status="matched"):
        from audio_dataset_agent.core.models import Batch, CandidateSample
        from audio_dataset_agent.inspection.service import inspect_candidate_samples

        batch = Batch(
            batch_id="batch_text",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy=target_policy,
            candidate_samples=[
                CandidateSample(
                    sample_id="sample_text",
                    batch_id="batch_text",
                    asset_id="asset_text",
                    source_path=Path("sample.wav"),
                    start_time=0.0,
                    end_time=3.0,
                    duration=3.0,
                    transcript=candidate_text,
                    transcript_source="faster_whisper",
                    asr_confidence=0.95,
                    reference_text=reference_text,
                    reference_text_source="csv" if reference_text else None,
                    metadata_status=metadata_status,
                )
            ],
        )
        return inspect_candidate_samples(batch).analysis_results[0]

    def test_tts_requires_reference_text(self):
        result = self._inspect("tts", None, "ASR 候选文本", metadata_status="missing")

        self.assertIn("missing_reference_text", result.issue_tags)

    def test_tts_marks_close_text_for_review(self):
        result = self._inspect("tts", "今天天气很好", "今天天气好")

        self.assertIn("text_alignment_review", result.issue_tags)
        self.assertAlmostEqual(result.evidence["text_alignment"]["cer"], 1 / 6)

    def test_tts_marks_large_text_difference_as_mismatch(self):
        result = self._inspect("tts", "欢迎使用语音系统", "今天下雨了")

        self.assertIn("text_mismatch", result.issue_tags)

    def test_asr_compares_reference_when_available_but_does_not_require_it(self):
        with_reference = self._inspect("asr", "请查询订单状态", "查询天气情况")
        without_reference = self._inspect("asr", None, "自动转写文本", metadata_status="missing")

        self.assertIn("text_mismatch", with_reference.issue_tags)
        self.assertNotIn("missing_reference_text", without_reference.issue_tags)

    def test_voice_clone_ignores_reference_text_requirement(self):
        result = self._inspect("voice_clone", None, None, metadata_status="missing")

        self.assertNotIn("missing_reference_text", result.issue_tags)
        self.assertNotIn("missing_transcript", result.issue_tags)

    def test_unaligned_segment_reference_is_sent_to_review(self):
        result = self._inspect(
            "tts",
            "整段音频参考文本",
            "第一段",
            metadata_status="requires_alignment",
        )

        self.assertIn("reference_alignment_required", result.issue_tags)


class TestTextIssueDecisionFlow(unittest.TestCase):
    def test_strategy_and_curator_route_text_mismatch_to_text_review(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        result = AnalysisResult(
            sample_id="sample_mismatch",
            duration=3.0,
            transcript_length=8,
            asr_confidence=0.95,
            issue_tags=["text_mismatch"],
            evidence={"text_alignment": {"cer": 0.8}},
        )
        strategy = BaselineProcessingStrategyAgent(
            default_tool_registry(), target_policy="tts"
        ).decide(result)
        decision = BaselineCurationPolicy(target_policy="tts").decide(result, strategy)

        self.assertEqual(strategy.recommended_operation, "manual_text_review")
        self.assertEqual(decision.decision, DecisionType.REVIEW)
        self.assertEqual(decision.recommended_action, "manual_text_review")


if __name__ == "__main__":
    unittest.main()
