import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestCuratorEvaluation(unittest.TestCase):
    def test_baseline_curator_applies_target_policy_to_noisy_audio(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        result = AnalysisResult(
            sample_id="sample_noise",
            duration=3.0,
            transcript_length=16,
            asr_confidence=0.94,
            issue_tags=["audio_noise_high"],
            evidence={"audio_metrics": {"noise_floor_dbfs": -30.0}},
        )

        asr_decision = BaselineCurationPolicy(target_policy="asr").decide(result)
        tts_decision = BaselineCurationPolicy(target_policy="tts").decide(result)
        clone_decision = BaselineCurationPolicy(target_policy="voice_clone").decide(result)

        self.assertEqual(asr_decision.decision, DecisionType.KEEP)
        self.assertEqual(asr_decision.recommended_action, "prepare_for_export")
        self.assertEqual(tts_decision.decision, DecisionType.REPAIR)
        self.assertEqual(tts_decision.recommended_action, "denoise_light")
        self.assertEqual(clone_decision.decision, DecisionType.REPAIR)
        self.assertEqual(clone_decision.recommended_action, "neural_denoise")

    def test_baseline_curator_uses_profile_to_allow_missing_text_for_voice_clone(self):
        from audio_dataset_agent.core.models import AnalysisResult, DecisionType
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        result = AnalysisResult(
            sample_id="sample_no_text",
            duration=3.0,
            transcript_length=0,
            asr_confidence=None,
            issue_tags=["missing_transcript"],
            evidence={},
        )

        decision = BaselineCurationPolicy(target_policy="voice_clone").decide(result)

        self.assertEqual(decision.decision, DecisionType.KEEP)
        self.assertEqual(decision.recommended_action, "prepare_for_export")

    def test_baseline_curator_passes_default_evaluation_cases(self):
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy
        from audio_dataset_agent.evaluation.curator_eval import (
            default_curator_eval_cases,
            evaluate_curator_policy,
        )

        summary = evaluate_curator_policy(
            BaselineCurationPolicy(),
            default_curator_eval_cases(),
        )

        self.assertTrue(summary.passed)
        self.assertEqual(summary.pass_count, summary.total_count)
        self.assertGreaterEqual(summary.total_count, 4)

    def test_evaluator_reports_failed_decision_and_action(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.evaluation.curator_eval import (
            default_curator_eval_cases,
            evaluate_curator_policy,
        )

        class AlwaysKeepPolicy:
            def decide(self, result):
                return CuratorDecision(
                    sample_id=result.sample_id,
                    decision=DecisionType.KEEP,
                    reason="always keep",
                    confidence=0.5,
                    recommended_action="prepare_for_export",
                    policy_name="always_keep",
                )

        summary = evaluate_curator_policy(
            AlwaysKeepPolicy(),
            default_curator_eval_cases(),
        )

        self.assertFalse(summary.passed)
        failed = [record for record in summary.records if not record.passed]
        self.assertGreater(len(failed), 0)
        self.assertIn("expected", failed[0].reason)

    def test_curator_evaluation_summary_can_be_serialized(self):
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy
        from audio_dataset_agent.evaluation.curator_eval import (
            curator_eval_summary_to_dict,
            default_curator_eval_cases,
            evaluate_curator_policy,
        )

        summary = evaluate_curator_policy(
            BaselineCurationPolicy(),
            default_curator_eval_cases(),
        )
        payload = curator_eval_summary_to_dict(summary)

        self.assertEqual(payload["pass_count"], summary.pass_count)
        self.assertEqual(payload["total_count"], summary.total_count)
        self.assertIn("records", payload)
        self.assertIn("actual_reason", payload["records"][0])
        self.assertIn("fallback_count", payload)
        self.assertIn("autonomous_pass_count", payload)

    def test_curator_evaluation_counts_guarded_fallback_passes(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.evaluation.curator_eval import (
            CuratorEvalCase,
            curator_eval_summary_to_dict,
            default_curator_eval_cases,
            evaluate_curator_policy,
        )

        class GuardedPolicy:
            def decide(self, result):
                return CuratorDecision(
                    sample_id=result.sample_id,
                    decision=DecisionType.REPAIR,
                    reason="fallback: invalid llm response; silence",
                    confidence=0.8,
                    recommended_action="trim_silence",
                    policy_name="llm_curator_guarded",
                )

        silence_case = default_curator_eval_cases()[3]
        case = CuratorEvalCase(
            name="silence_repair",
            analysis_result=silence_case.analysis_result,
            expected_decision=DecisionType.REPAIR,
            expected_action="trim_silence",
        )
        summary = evaluate_curator_policy(GuardedPolicy(), [case])
        payload = curator_eval_summary_to_dict(summary)

        self.assertTrue(summary.passed)
        self.assertEqual(summary.fallback_count, 1)
        self.assertEqual(summary.autonomous_pass_count, 0)
        self.assertEqual(payload["fallback_count"], 1)
        self.assertEqual(payload["autonomous_pass_count"], 0)

    def test_curator_evaluation_summary_can_render_markdown_report(self):
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy
        from audio_dataset_agent.evaluation.curator_eval import (
            default_curator_eval_cases,
            evaluate_curator_policy,
            render_curator_eval_markdown,
        )

        summary = evaluate_curator_policy(
            BaselineCurationPolicy(),
            default_curator_eval_cases(),
        )

        markdown = render_curator_eval_markdown(summary, title="Baseline Curator Eval")

        self.assertIn("# Baseline Curator Eval", markdown)
        self.assertIn("- Pass rate: 100.00%", markdown)
        self.assertIn("| case | expected | actual | passed |", markdown)
        self.assertIn("clean_sample_kept", markdown)

    def test_curator_evaluation_report_writer_creates_json_and_markdown(self):
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy
        from audio_dataset_agent.evaluation.curator_eval import (
            default_curator_eval_cases,
            evaluate_curator_policy,
            write_curator_eval_report,
        )

        summary = evaluate_curator_policy(
            BaselineCurationPolicy(),
            default_curator_eval_cases(),
        )

        with TemporaryDirectory() as tmp:
            paths = write_curator_eval_report(
                summary,
                output_dir=Path(tmp),
                title="Baseline Curator Eval",
            )

            json_path = Path(paths["json"])
            markdown_path = Path(paths["markdown"])

            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertIn("Baseline Curator Eval", markdown_path.read_text())
