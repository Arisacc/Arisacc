import json
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory


class TestStrategyEvaluation(unittest.TestCase):
    def test_default_strategy_eval_cases_cover_repair_review_and_keep(self):
        from audio_dataset_agent.evaluation.strategy_eval import (
            default_strategy_eval_cases,
        )

        cases = default_strategy_eval_cases()
        expected_operations = {
            case.expected_operation for case in cases
        }

        self.assertIn("prepare_for_export", expected_operations)
        self.assertIn("normalize_loudness", expected_operations)
        self.assertIn("trim_silence", expected_operations)
        self.assertIn("denoise_light", expected_operations)
        self.assertIn("manual_text_review", expected_operations)
        self.assertIn("review_clipped_audio", expected_operations)

        denoise_cases = [
            case for case in cases
            if case.expected_operation == "denoise_light"
        ]
        self.assertEqual(len(denoise_cases), 1)
        self.assertEqual(
            denoise_cases[0].expected_tool_ids,
            ["repair.ffmpeg_denoise", "metrics.wave"],
        )

    def test_evaluate_strategy_agent_reports_pass_rate(self):
        from audio_dataset_agent.evaluation.strategy_eval import (
            default_strategy_eval_cases,
            evaluate_strategy_agent,
        )
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        registry = default_tool_registry()
        summary = evaluate_strategy_agent(
            BaselineProcessingStrategyAgent(registry),
            default_strategy_eval_cases(),
        )

        self.assertTrue(summary.passed)
        self.assertEqual(summary.pass_count, summary.total_count)
        self.assertEqual(summary.pass_rate, 1.0)
        self.assertEqual(summary.fallback_count, 0)
        self.assertEqual(summary.autonomous_pass_count, summary.total_count)

    def test_strategy_eval_summary_serializes_to_json_ready_dict(self):
        from audio_dataset_agent.evaluation.strategy_eval import (
            default_strategy_eval_cases,
            evaluate_strategy_agent,
            strategy_eval_summary_to_dict,
        )
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        registry = default_tool_registry()
        summary = evaluate_strategy_agent(
            BaselineProcessingStrategyAgent(registry),
            default_strategy_eval_cases(),
        )
        payload = strategy_eval_summary_to_dict(summary)
        encoded = json.dumps(payload, ensure_ascii=False)

        self.assertIn('"passed": true', encoded)
        self.assertIn("records", payload)
        self.assertIn("actual_selected_tool_ids", payload["records"][0])
        self.assertIn("fallback_count", payload)
        self.assertIn("autonomous_pass_count", payload)

    def test_strategy_eval_summary_counts_guarded_fallback_passes(self):
        from audio_dataset_agent.core.models import ProcessingStrategy
        from audio_dataset_agent.evaluation.strategy_eval import (
            StrategyEvalCase,
            default_strategy_eval_cases,
            evaluate_strategy_agent,
            strategy_eval_summary_to_dict,
        )

        class GuardedAgent:
            def decide(self, result):
                return ProcessingStrategy(
                    sample_id=result.sample_id,
                    action="repair_audio",
                    recommended_operation="normalize_loudness",
                    selected_tool_ids=["repair.ffmpeg_loudness", "metrics.wave"],
                    reason="fallback: invalid llm response; quiet audio",
                    confidence=0.82,
                    agent_name="llm_processing_strategy_guarded",
                )

        quiet_case = default_strategy_eval_cases()[1]
        case = StrategyEvalCase(
            name="quiet_audio",
            analysis_result=quiet_case.analysis_result,
            expected_action="repair_audio",
            expected_operation="normalize_loudness",
            expected_tool_ids=["repair.ffmpeg_loudness", "metrics.wave"],
        )
        summary = evaluate_strategy_agent(GuardedAgent(), [case])
        payload = strategy_eval_summary_to_dict(summary)

        self.assertTrue(summary.passed)
        self.assertEqual(summary.fallback_count, 1)
        self.assertEqual(summary.autonomous_pass_count, 0)
        self.assertEqual(payload["fallback_count"], 1)
        self.assertEqual(payload["autonomous_pass_count"], 0)

    def test_render_and_write_strategy_eval_report(self):
        from audio_dataset_agent.evaluation.strategy_eval import (
            default_strategy_eval_cases,
            evaluate_strategy_agent,
            render_strategy_eval_markdown,
            write_strategy_eval_report,
        )
        from audio_dataset_agent.strategy.agent import BaselineProcessingStrategyAgent
        from audio_dataset_agent.tools.registry import default_tool_registry

        registry = default_tool_registry()
        summary = evaluate_strategy_agent(
            BaselineProcessingStrategyAgent(registry),
            default_strategy_eval_cases(),
        )
        markdown = render_strategy_eval_markdown(summary, title="Strategy Eval")

        with TemporaryDirectory() as tmp:
            paths = write_strategy_eval_report(
                summary,
                Path(tmp),
                title="Strategy Eval",
            )
            json_path = Path(paths["json"])
            markdown_path = Path(paths["markdown"])
            json_payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown_text = markdown_path.read_text(encoding="utf-8")

        self.assertIn("# Strategy Eval", markdown)
        self.assertTrue(json_payload["passed"])
        self.assertIn("Strategy Eval", markdown_text)


class TestStrategyEvaluationScript(unittest.TestCase):
    def test_strategy_evaluation_script_writes_reports(self):
        from local_tools.evaluate_strategy import main

        with TemporaryDirectory() as tmp:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--strategy",
                        "baseline",
                        "--output-dir",
                        tmp,
                    ]
                )
            json_path = Path(tmp) / "strategy_eval_summary.json"
            markdown_path = Path(tmp) / "strategy_eval_report.md"
            markdown_exists = markdown_path.exists()
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["passed"])
        self.assertTrue(markdown_exists)


if __name__ == "__main__":
    unittest.main()
