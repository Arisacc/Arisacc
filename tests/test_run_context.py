import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestRunContext(unittest.TestCase):
    def test_run_context_owns_all_batch_working_paths(self):
        from audio_dataset_agent.core.models import Batch
        from audio_dataset_agent.core.run_context import RunContext

        batch = Batch(
            batch_id="batch_paths",
            input_dir=Path("input"),
            output_dir=Path("runs/job/work"),
            target_policy="voice_clone",
            run_config={"vad": "silero"},
        )

        context = RunContext.from_batch(batch)

        self.assertEqual(context.normalized_dir, Path("runs/job/work/work/batch_paths"))
        self.assertEqual(context.samples_dir, Path("runs/job/work/samples/batch_paths"))
        self.assertEqual(context.repairs_dir, Path("runs/job/work/repairs/batch_paths"))
        self.assertEqual(
            context.batch_export_dir(Path("runs/job/exports")),
            Path("runs/job/exports/batch_paths"),
        )
        self.assertEqual(context.run_config["vad"], "silero")

    def test_batch_serialization_includes_run_config(self):
        from audio_dataset_agent.core.models import Batch
        from audio_dataset_agent.core.serialization import batch_to_dict

        batch = Batch(
            batch_id="batch_config",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="asr",
            run_config={"asr": "faster-whisper", "vad": "silero"},
        )

        payload = batch_to_dict(batch)

        self.assertEqual(payload["run_config"]["asr"], "faster-whisper")
        self.assertEqual(payload["run_config"]["vad"], "silero")

    def test_workflow_preserves_runtime_configuration_in_export(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            exported = run_batch_pipeline(
                input_dir=input_dir,
                output_dir=root / "work",
                export_dir=root / "exports",
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.0, 1.5)]),
                transcriber=FixedTranscriber(TranscriptResult("测试", 0.95)),
                target_policy="voice_clone",
                run_config={"workflow": "basic", "vad": "fixed", "asr": "fixed"},
            )

            persisted = json.loads(
                Path(exported.export_paths["run_config"]).read_text(encoding="utf-8")
            )
            manifest = json.loads(
                Path(exported.export_paths["manifest"]).read_text(encoding="utf-8").strip()
            )

        self.assertEqual(exported.run_config["vad"], "fixed")
        self.assertEqual(persisted["parameters"]["workflow"], "basic")
        self.assertEqual(manifest["source_audio_path"], str(input_dir / "voice.wav"))
        self.assertEqual(manifest["work_audio_path"], str(exported.raw_assets[0].work_path))
        self.assertEqual(manifest["sample_audio_path"], str(exported.candidate_samples[0].sample_path))
        self.assertEqual(manifest["final_audio_path"], str(exported.final_selections[0].selected_audio_path))


class TestProvenanceArtifacts(unittest.TestCase):
    def test_export_writes_run_source_and_tool_provenance(self):
        from audio_dataset_agent.core.models import Batch, RawAsset
        from audio_dataset_agent.exporters.service import export_curated_batch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "work"
            export_root = root / "exports"
            input_dir.mkdir()
            source_path = input_dir / "voice.wav"
            source_bytes = b"source-audio"
            source_path.write_bytes(source_bytes)
            batch = Batch(
                batch_id="batch_provenance",
                input_dir=input_dir,
                output_dir=output_dir,
                target_policy="voice_clone",
                raw_assets=[
                    RawAsset(
                        asset_id="asset_0001",
                        batch_id="batch_provenance",
                        source_path=source_path,
                        file_name=source_path.name,
                        file_size=len(source_bytes),
                        extension=".wav",
                    )
                ],
                run_config={
                    "workflow": "repair",
                    "vad": "whole-file",
                    "asr": "none",
                    "tools": {"ffmpeg_path": "ffmpeg"},
                },
            )

            exported = export_curated_batch(batch, export_root)
            batch_dir = export_root / batch.batch_id
            run_config = json.loads((batch_dir / "run_config.json").read_text(encoding="utf-8"))
            source_manifest = json.loads((batch_dir / "source_manifest.json").read_text(encoding="utf-8"))
            tool_versions = json.loads((batch_dir / "tool_versions.json").read_text(encoding="utf-8"))

        self.assertEqual(run_config["batch_id"], "batch_provenance")
        self.assertEqual(run_config["target_policy"], "voice_clone")
        self.assertEqual(run_config["parameters"]["vad"], "whole-file")
        self.assertEqual(source_manifest["asset_count"], 1)
        self.assertEqual(
            source_manifest["assets"][0]["sha256"],
            hashlib.sha256(source_bytes).hexdigest(),
        )
        self.assertEqual(source_manifest["assets"][0]["relative_path"], "voice.wav")
        self.assertIn("python", tool_versions["runtime"])
        self.assertIn("ffmpeg", tool_versions["configured_tools"])
        self.assertIn("run_config", exported.export_paths)
        self.assertIn("source_manifest", exported.export_paths)
        self.assertIn("tool_versions", exported.export_paths)


if __name__ == "__main__":
    unittest.main()
