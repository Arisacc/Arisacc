import json
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory


class TestAudioDatasetSupervisor(unittest.TestCase):
    def test_supervisor_runs_basic_workflow_and_records_trace(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import BatchPhase
        from audio_dataset_agent.orchestrator.supervisor import AudioDatasetSupervisor
        from audio_dataset_agent.transcription.transcriber import (
            FixedTranscriber,
            TranscriptResult,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            export_dir = root / "exports"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            result = AudioDatasetSupervisor(
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.0, 1.5)]),
                transcriber=FixedTranscriber(TranscriptResult("hello", 0.95)),
            ).run_basic(
                input_dir=input_dir,
                output_dir=output_dir,
                export_dir=export_dir,
            )

        self.assertEqual(result.batch.phase, BatchPhase.EXPORTED)
        self.assertEqual(result.batch.final_selection_count, 1)
        self.assertEqual(result.trace[0].agent_name, "Data Intake")
        self.assertEqual(result.trace[-1].agent_name, "Archivist")
        self.assertEqual(
            [step.phase for step in result.trace],
            [
                "import",
                "normalize",
                "segment",
                "materialize",
                "transcribe",
                "inspect",
                "strategy",
                "curate",
                "repair_plan",
                "final_select",
                "export",
            ],
        )

    def test_supervisor_trace_serializes_to_json_ready_dict(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.orchestrator.supervisor import (
            AudioDatasetSupervisor,
            supervisor_run_result_to_dict,
        )
        from audio_dataset_agent.transcription.transcriber import (
            FixedTranscriber,
            TranscriptResult,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            export_dir = root / "exports"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            result = AudioDatasetSupervisor(
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.0, 1.5)]),
                transcriber=FixedTranscriber(TranscriptResult("hello", 0.95)),
            ).run_basic(
                input_dir=input_dir,
                output_dir=output_dir,
                export_dir=export_dir,
            )
            payload = supervisor_run_result_to_dict(result)
            encoded = json.dumps(payload, ensure_ascii=False)

        self.assertIn('"phase": "exported"', encoded)
        self.assertEqual(payload["trace"][0]["input_summary"]["input_dir"], str(input_dir))
        self.assertEqual(payload["trace"][0]["output_summary"]["raw_assets"], 1)
        self.assertEqual(payload["trace"][-1]["output_summary"]["final_selections"], 1)

    def test_supervisor_writes_trace_artifact_next_to_export(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.orchestrator.supervisor import AudioDatasetSupervisor
        from audio_dataset_agent.transcription.transcriber import (
            FixedTranscriber,
            TranscriptResult,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            export_dir = root / "exports"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            result = AudioDatasetSupervisor(
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.0, 1.5)]),
                transcriber=FixedTranscriber(TranscriptResult("hello", 0.95)),
            ).run_basic(
                input_dir=input_dir,
                output_dir=output_dir,
                export_dir=export_dir,
            )
            trace_path = Path(result.batch.export_paths["supervisor_trace"])
            trace_exists = trace_path.exists()
            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertTrue(trace_exists)
        self.assertEqual(trace_payload[0]["agent_name"], "Data Intake")
        self.assertEqual(trace_payload[-1]["agent_name"], "Archivist")


class TestSupervisorCli(unittest.TestCase):
    def test_cli_accepts_supervisor_orchestrator_argument(self):
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
            ]
        )

        self.assertEqual(args.orchestrator, "supervisor")

    def test_cli_supervisor_output_includes_trace(self):
        from audio_dataset_agent.cli import main

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            export_dir = root / "exports"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                        "--export-dir",
                        str(export_dir),
                        "--orchestrator",
                        "supervisor",
                        "--demo-transcript",
                        "hello",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertIn("batch", payload)
        self.assertIn("trace", payload)
        self.assertEqual(payload["trace"][0]["agent_name"], "Data Intake")
        self.assertEqual(payload["trace"][-1]["agent_name"], "Archivist")


if __name__ == "__main__":
    unittest.main()
