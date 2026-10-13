import unittest
import subprocess
import sys


class TestQwenDownloadScript(unittest.TestCase):
    def test_qwen_download_script_defaults_to_qwen3_4b_instruct_model(self):
        from pathlib import Path

        from local_tools.download_qwen_model import build_parser

        args = build_parser().parse_args([])

        self.assertEqual(args.repo_id, "Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(
            args.local_dir,
            Path("local_tools/models/qwen/Qwen3-4B-Instruct-2507"),
        )
        self.assertEqual(args.endpoint, "https://hf-mirror.com")

    def test_qwen_download_script_accepts_modelscope_provider(self):
        from local_tools.download_qwen_model import build_parser

        args = build_parser().parse_args(["--provider", "modelscope"])

        self.assertEqual(args.provider, "modelscope")


class TestCuratorEvaluationScript(unittest.TestCase):
    def test_curator_evaluation_script_accepts_qwen_local_arguments(self):
        from local_tools.evaluate_curator import build_parser

        args = build_parser().parse_args(
            [
                "--curator",
                "qwen-local",
                "--model-path",
                "local_tools/models/qwen/Qwen2.5-0.5B-Instruct",
                "--max-new-tokens",
                "96",
                "--output-dir",
                "local_tools/cache/eval_report",
            ]
        )

        self.assertEqual(args.curator, "qwen-local")
        self.assertEqual(
            args.model_path,
            "local_tools/models/qwen/Qwen2.5-0.5B-Instruct",
        )
        self.assertEqual(args.max_new_tokens, 96)
        self.assertEqual(args.output_dir, "local_tools/cache/eval_report")

    def test_curator_evaluation_script_runs_from_project_root(self):
        result = subprocess.run(
            [
                sys.executable,
                "local_tools/evaluate_curator.py",
                "--curator",
                "baseline",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"passed": true', result.stdout)

    def test_curator_evaluation_script_writes_report_files(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "local_tools/evaluate_curator.py",
                    "--curator",
                    "baseline",
                    "--output-dir",
                    tmp,
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(tmp) / "curator_eval_summary.json").exists())
            self.assertTrue((Path(tmp) / "curator_eval_report.md").exists())
