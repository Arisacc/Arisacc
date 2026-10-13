import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestAgentGraphTrace(unittest.TestCase):
    def test_workflow_exports_agent_graph_trace(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline
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

            exported = run_batch_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                export_dir=export_dir,
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.0, 1.5)]),
                transcriber=FixedTranscriber(TranscriptResult("hello", 0.95)),
            )
            trace_path = Path(exported.export_paths["agent_graph_trace"])
            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(trace_payload["graph_name"], "音频数据集 Agent 工作流")
        self.assertEqual(trace_payload["nodes"][0]["node_name"], "import")
        self.assertEqual(trace_payload["nodes"][-1]["node_name"], "export")
        self.assertTrue(trace_payload["nodes"][0]["duration_ms"] >= 0)

    def test_repair_workflow_trace_includes_repair_nodes(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import RepairExecution
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.orchestrator.workflow import run_repair_enabled_pipeline
        from audio_dataset_agent.transcription.transcriber import (
            FixedTranscriber,
            TranscriptResult,
        )

        class SequenceAudioMetricsAnalyzer:
            def __init__(self):
                self.metrics = [
                    AudioMetrics(-55.0, -12.0, 0.0, 0.1),
                    AudioMetrics(-55.0, -12.0, 0.0, 0.1),
                    AudioMetrics(-24.0, -3.0, 0.0, 0.1),
                ]

            def analyze(self, audio_path):
                return self.metrics.pop(0)

        class FakeRepairExecutor:
            def execute(self, plan, sample, repairs_dir):
                repairs_dir.mkdir(parents=True, exist_ok=True)
                output_path = repairs_dir / f"{sample.sample_id}_{plan.operation}.wav"
                output_path.write_bytes(b"repaired")
                return RepairExecution(
                    sample_id=sample.sample_id,
                    operation=plan.operation,
                    input_path=sample.sample_path or sample.source_path,
                    output_path=output_path,
                    status="succeeded",
                )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            export_dir = root / "exports"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            exported = run_repair_enabled_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                export_dir=export_dir,
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.0, 2.0)]),
                transcriber=FixedTranscriber(TranscriptResult("hello", 0.95)),
                audio_metrics_analyzer=SequenceAudioMetricsAnalyzer(),
                repair_executor=FakeRepairExecutor(),
            )
            trace_payload = json.loads(
                Path(exported.export_paths["agent_graph_trace"]).read_text(
                    encoding="utf-8"
                )
            )

        node_names = [node["node_name"] for node in trace_payload["nodes"]]
        self.assertIn("repair_execute", node_names)
        self.assertIn("repair_evaluate", node_names)


if __name__ == "__main__":
    unittest.main()
