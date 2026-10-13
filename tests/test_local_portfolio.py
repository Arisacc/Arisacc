import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestLocalPortfolio(unittest.TestCase):
    def test_local_research_uses_project_facts_without_remote_service(self):
        from audio_dataset_agent.core.models import Batch
        from audio_dataset_agent.portfolio.context import build_local_portfolio_research

        batch = Batch(
            batch_id="batch_local_portfolio",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="voice_clone",
        )

        research = build_local_portfolio_research(batch)

        self.assertEqual(research["status"], "succeeded")
        self.assertIn("声音克隆", research["problem_explanation"])
        self.assertTrue(research["tool_notes"])
        self.assertEqual(research["external_references"], [])

    def test_renderer_writes_local_portfolio_artifacts(self):
        from audio_dataset_agent.core.models import Batch
        from audio_dataset_agent.portfolio.context import build_local_portfolio_research
        from audio_dataset_agent.portfolio.renderer import write_portfolio_artifacts

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_path = root / "dataset_quality_report.json"
            quality_path.write_text(
                json.dumps(
                    {
                        "batch_id": "batch_local_portfolio",
                        "target_policy": "tts",
                        "total_samples": 2,
                        "usable_samples": 1,
                        "review_samples": 1,
                        "rejected_samples": 0,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            batch = Batch(
                batch_id="batch_local_portfolio",
                input_dir=root / "input",
                output_dir=root / "output",
                target_policy="tts",
                export_paths={"dataset_quality_json": str(quality_path)},
            )

            exported = write_portfolio_artifacts(
                batch,
                build_local_portfolio_research(batch),
            )

            self.assertTrue(Path(exported.export_paths["portfolio_report_html"]).exists())
            self.assertTrue(Path(exported.export_paths["portfolio_report_markdown"]).exists())
            content = Path(exported.export_paths["portfolio_report_markdown"]).read_text(
                encoding="utf-8"
            )
            self.assertIn("作品集展示报告", content)
            self.assertIn("语音合成（TTS）", content)


if __name__ == "__main__":
    unittest.main()
