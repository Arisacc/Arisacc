import json
import struct
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory


class TestWorkflow(unittest.TestCase):
    def test_run_batch_pipeline_exports_manifest(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import BatchPhase
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

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

            manifest_path = Path(exported.export_paths["manifest"])
            manifest_record = json.loads(
                manifest_path.read_text(encoding="utf-8").strip()
            )

        self.assertEqual(exported.phase, BatchPhase.EXPORTED)
        self.assertEqual(manifest_record["decision"], "keep")
        self.assertEqual(manifest_record["text"], "hello")
        self.assertEqual(len(exported.final_selections), 1)
        self.assertEqual(manifest_record["final_status"], "selected_original")
        self.assertEqual(
            manifest_record["final_audio_path"],
            str(exported.candidate_samples[0].sample_path),
        )

    def test_run_batch_pipeline_accepts_custom_normalizer(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        class FakeNormalizer:
            def normalize(self, asset, work_dir):
                work_dir.mkdir(parents=True, exist_ok=True)
                destination = work_dir / f"{asset.asset_id}.custom.wav"
                destination.write_bytes(b"custom")
                return destination

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            export_dir = root / "exports"
            input_dir.mkdir()
            (input_dir / "voice.mp3").write_bytes(b"original")

            exported = run_batch_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                export_dir=export_dir,
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.0, 1.5)]),
                transcriber=FixedTranscriber(TranscriptResult("hello", 0.95)),
                normalizer=FakeNormalizer(),
            )

            manifest_path = Path(exported.export_paths["manifest"])
            manifest_record = json.loads(
                manifest_path.read_text(encoding="utf-8").strip()
            )

        self.assertTrue(exported.candidate_samples[0].source_path.name.endswith(".custom.wav"))
        self.assertEqual(
            manifest_record["audio_path"],
            str(exported.candidate_samples[0].sample_path),
        )

    def test_run_batch_pipeline_exports_materialized_sample_path(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline
        from audio_dataset_agent.segmentation.materializer import CopySampleMaterializer
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

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
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.25, 1.75)]),
                transcriber=FixedTranscriber(TranscriptResult("hello", 0.95)),
                sample_materializer=CopySampleMaterializer(),
            )

            sample = exported.candidate_samples[0]
            manifest_path = Path(exported.export_paths["manifest"])
            manifest_record = json.loads(
                manifest_path.read_text(encoding="utf-8").strip()
            )

        self.assertIsNotNone(sample.sample_path)
        self.assertTrue(sample.sample_path.name.endswith(".wav"))
        self.assertEqual(manifest_record["audio_path"], str(sample.sample_path))

    def test_run_batch_pipeline_accepts_custom_curator_policy(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        class FakeCuratorPolicy:
            def decide(self, result):
                return CuratorDecision(
                    sample_id=result.sample_id,
                    decision=DecisionType.REVIEW,
                    reason="review: fake agent requested human check",
                    confidence=0.66,
                    recommended_action="manual_decision_review",
                    policy_name="fake_agent",
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
                curator_policy=FakeCuratorPolicy(),
            )

        self.assertEqual(exported.curator_decisions[0].decision, DecisionType.REVIEW)
        self.assertEqual(exported.curator_decisions[0].policy_name, "fake_agent")

    def test_run_batch_pipeline_accepts_custom_processing_strategy_agent(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import ProcessingStrategy
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline
        from audio_dataset_agent.transcription.transcriber import (
            FixedTranscriber,
            TranscriptResult,
        )

        class FakeProcessingStrategyAgent:
            def decide(self, result):
                return ProcessingStrategy(
                    sample_id=result.sample_id,
                    action="manual_review",
                    recommended_operation="manual_strategy_review",
                    reason="fake strategy touched workflow",
                    confidence=0.5,
                    agent_name="fake_strategy_agent",
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
                processing_strategy_agent=FakeProcessingStrategyAgent(),
            )

        self.assertEqual(len(exported.processing_strategies), 1)
        self.assertEqual(
            exported.processing_strategies[0].agent_name,
            "fake_strategy_agent",
        )

    def test_voice_clone_pipeline_can_skip_asr(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline

        class FailingTranscriber:
            def transcribe(self, sample):
                raise AssertionError("ASR should be skipped for this profile")

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
                transcriber=FailingTranscriber(),
                target_policy="voice_clone",
                run_asr=False,
            )

        self.assertIsNone(exported.candidate_samples[0].transcript)
        self.assertEqual(exported.curator_decisions[0].decision.value, "keep")


class TestRepairEnabledWorkflow(unittest.TestCase):
    def test_repair_enabled_pipeline_selects_improved_repaired_output(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import RepairExecution
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.orchestrator.workflow import run_repair_enabled_pipeline
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

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

            manifest_path = Path(exported.export_paths["manifest"])
            manifest_record = json.loads(
                manifest_path.read_text(encoding="utf-8").strip()
            )

        self.assertEqual(len(exported.repair_plans), 1)
        self.assertEqual(len(exported.repair_executions), 1)
        self.assertEqual(len(exported.repair_evaluations), 1)
        self.assertEqual(manifest_record["decision"], "repair")
        self.assertEqual(manifest_record["final_status"], "selected_repaired")
        self.assertEqual(
            manifest_record["final_audio_path"],
            str(exported.repair_executions[0].output_path),
        )


class TestSampleMaterializer(unittest.TestCase):
    def test_ffmpeg_sample_materializer_invokes_expected_clip_command(self):
        from audio_dataset_agent.core.models import CandidateSample
        from audio_dataset_agent.segmentation.materializer import FFmpegSampleMaterializer

        commands = []

        def fake_runner(command):
            commands.append(command)
            Path(command[-1]).write_bytes(b"clip")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "work.wav"
            source.write_bytes(b"normalized")
            sample = CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=source,
                start_time=1.25,
                end_time=3.75,
                duration=2.5,
            )

            materializer = FFmpegSampleMaterializer(
                ffmpeg_path="ffmpeg",
                runner=fake_runner,
            )
            output = materializer.materialize(sample, root / "samples")

        self.assertEqual(output.name, "sample_0001.wav")
        self.assertEqual(commands[0][0], "ffmpeg")
        self.assertIn("-ss", commands[0])
        self.assertIn("1.25", commands[0])
        self.assertIn("-t", commands[0])
        self.assertIn("2.5", commands[0])
        self.assertEqual(commands[0][-1], str(output))


class TestWaveEnergyVAD(unittest.TestCase):
    def test_wave_energy_vad_detects_high_energy_region(self):
        from audio_dataset_agent.analyzers.vad import WaveEnergyVADAnalyzer

        sample_rate = 16000
        silence = [0] * int(sample_rate * 0.2)
        speech = [12000] * int(sample_rate * 0.3)
        samples = silence + speech + silence

        with TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "voice.wav"
            self._write_pcm_wav(wav_path, samples, sample_rate=sample_rate)

            segments = WaveEnergyVADAnalyzer(
                frame_ms=50,
                energy_threshold_dbfs=-35.0,
                min_duration_seconds=0.1,
            ).detect(wav_path)

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0].start_time, 0.2, places=2)
        self.assertAlmostEqual(segments[0].end_time, 0.5, places=2)

    def _write_pcm_wav(self, path, samples, sample_rate):
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(
                b"".join(struct.pack("<h", sample) for sample in samples)
            )


class TestCli(unittest.TestCase):
    def test_cli_accepts_sample_materializer_argument(self):
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
                "--sample-materializer",
                "ffmpeg",
            ]
        )

        self.assertEqual(args.sample_materializer, "ffmpeg")

    def test_cli_accepts_audio_metrics_argument(self):
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
                "--audio-metrics",
                "wave",
            ]
        )

        self.assertEqual(args.audio_metrics, "wave")

    def test_cli_accepts_ffmpeg_audio_metrics_argument(self):
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
                "--audio-metrics",
                "ffmpeg",
            ]
        )

        self.assertEqual(args.audio_metrics, "ffmpeg")

    def test_cli_resolves_default_ffmpeg_to_bundled_imageio_binary(self):
        import audio_dataset_agent.cli as cli

        original_which = cli.shutil.which
        original_imageio_path = cli._imageio_ffmpeg_path
        cli.shutil.which = lambda executable: None
        cli._imageio_ffmpeg_path = lambda: "local-ffmpeg.exe"
        try:
            resolved = cli.resolve_ffmpeg_path("ffmpeg")
        finally:
            cli.shutil.which = original_which
            cli._imageio_ffmpeg_path = original_imageio_path

        self.assertEqual(resolved, "local-ffmpeg.exe")

    def test_cli_keeps_explicit_ffmpeg_path(self):
        from audio_dataset_agent.cli import resolve_ffmpeg_path

        self.assertEqual(resolve_ffmpeg_path("custom-ffmpeg.exe"), "custom-ffmpeg.exe")

    def test_cli_accepts_workflow_argument(self):
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
                "--workflow",
                "repair",
            ]
        )

        self.assertEqual(args.workflow, "repair")

    def test_cli_accepts_target_policy_argument(self):
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
                "--target-policy",
                "tts",
            ]
        )

        self.assertEqual(args.target_policy, "tts")

    def test_cli_accepts_vad_argument(self):
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
                "--vad",
                "wave-energy",
            ]
        )

        self.assertEqual(args.vad, "wave-energy")

    def test_cli_accepts_whole_file_vad_argument(self):
        from audio_dataset_agent.cli import build_parser

        args = build_parser().parse_args(["--vad", "whole-file"])

        self.assertEqual(args.vad, "whole-file")

    def test_cli_accepts_local_rag_options(self):
        from audio_dataset_agent.cli import build_parser

        args = build_parser().parse_args([
            "--rag", "local",
            "--rag-documents", "knowledge_base/documents",
            "--rag-index", "local_tools/cache/rag/index.sqlite",
        ])

        self.assertEqual(args.rag, "local")
        self.assertEqual(args.rag_documents, "knowledge_base/documents")
        self.assertEqual(args.rag_index, "local_tools/cache/rag/index.sqlite")

    def test_cli_accepts_rag_rebuild_option(self):
        from audio_dataset_agent.cli import build_parser

        args = build_parser().parse_args(["--rag", "local", "--rag-rebuild"])

        self.assertTrue(args.rag_rebuild)

    def test_cli_run_config_includes_rag_rebuild(self):
        from audio_dataset_agent.cli import _build_run_config, build_parser

        args = build_parser().parse_args(["--rag", "local", "--rag-rebuild"])

        config = _build_run_config(args, "ffmpeg")

        self.assertTrue(config["rag"]["rebuild"])

    def test_cli_accepts_silero_vad_arguments(self):
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
                "--vad",
                "silero",
                "--vad-sampling-rate",
                "16000",
                "--vad-threshold",
                "0.6",
                "--vad-min-speech-ms",
                "200",
                "--vad-min-silence-ms",
                "120",
                "--vad-speech-pad-ms",
                "40",
            ]
        )

        self.assertEqual(args.vad, "silero")
        self.assertEqual(args.vad_sampling_rate, 16000)
        self.assertEqual(args.vad_threshold, 0.6)
        self.assertEqual(args.vad_min_speech_ms, 200)
        self.assertEqual(args.vad_min_silence_ms, 120)
        self.assertEqual(args.vad_speech_pad_ms, 40)

    def test_cli_accepts_faster_whisper_asr_arguments(self):
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
                "--asr",
                "faster-whisper",
                "--asr-model",
                "base",
                "--asr-device",
                "cpu",
                "--asr-compute-type",
                "int8",
                "--asr-language",
                "zh",
                "--asr-beam-size",
                "3",
                "--asr-vad-filter",
            ]
        )

        self.assertEqual(args.asr, "faster-whisper")
        self.assertEqual(args.asr_model, "base")
        self.assertEqual(args.asr_device, "cpu")
        self.assertEqual(args.asr_compute_type, "int8")
        self.assertEqual(args.asr_language, "zh")
        self.assertEqual(args.asr_beam_size, 3)
        self.assertTrue(args.asr_vad_filter)

    def test_cli_accepts_no_asr_argument(self):
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
                "--target-policy",
                "voice_clone",
                "--asr",
                "none",
            ]
        )

        self.assertEqual(args.asr, "none")

    def test_cli_accepts_curator_argument(self):
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
                "--curator",
                "llm-placeholder",
            ]
        )

        self.assertEqual(args.curator, "llm-placeholder")

    def test_cli_accepts_local_qwen_curator_arguments(self):
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
                "--curator",
                "qwen-local",
                "--curator-model-path",
                "local_tools/models/qwen/Qwen2.5-0.5B-Instruct",
                "--curator-max-new-tokens",
                "96",
            ]
        )

        self.assertEqual(args.curator, "qwen-local")
        self.assertEqual(
            args.curator_model_path,
            "local_tools/models/qwen/Qwen2.5-0.5B-Instruct",
        )
        self.assertEqual(args.curator_max_new_tokens, 96)

    def test_cli_defaults_to_qwen3_4b_model_path(self):
        from audio_dataset_agent.cli import build_parser

        args = build_parser().parse_args(
            [
                "--input-dir",
                "input",
                "--output-dir",
                "out",
                "--export-dir",
                "exports",
            ]
        )

        self.assertEqual(
            args.curator_model_path,
            "local_tools/models/qwen/Qwen3-4B-Instruct-2507",
        )

    def test_cli_accepts_local_qwen_report_arguments(self):
        from audio_dataset_agent.cli import build_parser

        args = build_parser().parse_args(
            [
                "--input-dir",
                "input",
                "--output-dir",
                "out",
                "--export-dir",
                "exports",
                "--report",
                "qwen-local",
                "--report-model-path",
                "local_tools/models/qwen/Qwen3-4B-Instruct-2507",
                "--report-max-new-tokens",
                "192",
            ]
        )

        self.assertEqual(args.report, "qwen-local")
        self.assertEqual(
            args.report_model_path,
            "local_tools/models/qwen/Qwen3-4B-Instruct-2507",
        )
        self.assertEqual(args.report_max_new_tokens, 192)

    def test_cli_repair_workflow_requires_audio_metrics(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.cli import main

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            export_dir = root / "exports"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            exit_code = main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--export-dir",
                    str(export_dir),
                    "--workflow",
                    "repair",
                    "--audio-metrics",
                    "none",
                ]
            )

        self.assertNotEqual(exit_code, 0)

    def test_cli_passes_target_policy_into_workflow_report(self):
        from contextlib import redirect_stdout
        import io
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

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
                        "--target-policy",
                        "asr",
                        "--demo-transcript",
                        "hello",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            quality_path = Path(payload["export_paths"]["dataset_quality_json"])
            quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["target_policy"], "asr")
        self.assertEqual(quality_payload["target_policy"], "asr")

    def test_cli_repair_workflow_passes_ffmpeg_path_to_repair_executor(self):
        from contextlib import redirect_stdout
        import io
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import audio_dataset_agent.cli as cli

        captured = {}
        original_wave_metrics = cli.WaveAudioMetricsAnalyzer
        original_composite = cli.CompositeRepairExecutor

        class FakeMetricsAnalyzer:
            def analyze(self, audio_path):
                from audio_dataset_agent.inspection.audio_metrics import AudioMetrics

                return AudioMetrics(-55.0, -20.0, 0.0, 0.0)

        class FakeCompositeRepairExecutor:
            def __init__(self, ffmpeg_executor=None, deepfilternet_executor=None):
                captured["ffmpeg_path"] = ffmpeg_executor.ffmpeg_path

            def execute(self, plan, sample, repairs_dir):
                from audio_dataset_agent.core.models import RepairExecution

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

        cli.WaveAudioMetricsAnalyzer = FakeMetricsAnalyzer
        cli.CompositeRepairExecutor = FakeCompositeRepairExecutor
        try:
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_dir = root / "input"
                output_dir = root / "out"
                export_dir = root / "exports"
                input_dir.mkdir()
                (input_dir / "voice.wav").write_bytes(b"original")

                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = cli.main(
                        [
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                            "--export-dir",
                            str(export_dir),
                            "--workflow",
                            "repair",
                            "--audio-metrics",
                            "wave",
                            "--ffmpeg-path",
                            "custom-ffmpeg",
                            "--demo-transcript",
                            "hello",
                        ]
                    )
                json.loads(stdout.getvalue())
        finally:
            cli.WaveAudioMetricsAnalyzer = original_wave_metrics
            cli.CompositeRepairExecutor = original_composite

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["ffmpeg_path"], "custom-ffmpeg")


class TestAudioMetricsInspector(unittest.TestCase):
    def _write_pcm_wav(self, path, samples):
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(
                b"".join(struct.pack("<h", sample) for sample in samples)
            )

    def test_wave_audio_metrics_analyzer_reports_loudness_clipping_and_silence(self):
        from audio_dataset_agent.inspection.audio_metrics import WaveAudioMetricsAnalyzer

        with TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "sample.wav"
            self._write_pcm_wav(wav_path, [0, 0, 32767, -32768])

            metrics = WaveAudioMetricsAnalyzer().analyze(wav_path)

        self.assertLess(metrics.rms_dbfs, 0)
        self.assertAlmostEqual(metrics.peak_dbfs, 0.0, places=3)
        self.assertEqual(metrics.clipping_ratio, 0.5)
        self.assertEqual(metrics.silence_ratio, 0.5)
        self.assertIn("noise_floor_dbfs", metrics.to_dict())
        self.assertIn("dynamic_range_db", metrics.to_dict())
        self.assertIn("dc_offset", metrics.to_dict())

    def test_ffmpeg_audio_metrics_analyzer_parses_astats_and_silencedetect_output(self):
        from audio_dataset_agent.inspection.audio_metrics import FFmpegAudioMetricsAnalyzer

        commands = []

        def fake_runner(command):
            commands.append(command)
            if any("astats" in part for part in command):
                return """
                [Parsed_astats_0] DC offset: 0.001
                [Parsed_astats_0] Min level: -1200.000000
                [Parsed_astats_0] Max level: 1400.000000
                [Parsed_astats_0] Peak level dB: -1.200000
                [Parsed_astats_0] RMS level dB: -31.500000
                [Parsed_astats_0] Noise floor dB: -32.000000
                [Parsed_astats_0] Dynamic range: 28.000000
                """
            return """
            [silencedetect @ 000001] silence_start: 0
            [silencedetect @ 000001] silence_end: 0.4 | silence_duration: 0.4
            [silencedetect @ 000001] silence_start: 1.5
            [silencedetect @ 000001] silence_end: 2.0 | silence_duration: 0.5
            """

        metrics = FFmpegAudioMetricsAnalyzer(runner=fake_runner).analyze(
            Path("sample.wav")
        )

        self.assertEqual(len(commands), 2)
        self.assertEqual(metrics.rms_dbfs, -31.5)
        self.assertEqual(metrics.peak_dbfs, -1.2)
        self.assertEqual(metrics.noise_floor_dbfs, -32.0)
        self.assertEqual(metrics.dynamic_range_db, 28.0)
        self.assertAlmostEqual(metrics.silence_ratio, 0.45)

    def test_inspector_adds_audio_metric_issue_tags_and_evidence(self):
        from audio_dataset_agent.core.models import Batch, CandidateSample
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.inspection.policy import InspectionPolicy
        from audio_dataset_agent.inspection.service import inspect_candidate_samples

        class FixedAudioMetricsAnalyzer:
            def analyze(self, audio_path):
                return AudioMetrics(
                    rms_dbfs=-55.0,
                    peak_dbfs=0.0,
                    clipping_ratio=0.03,
                    silence_ratio=0.75,
                    noise_floor_dbfs=-32.0,
                )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_path = root / "sample.wav"
            sample_path.write_bytes(b"fake")
            batch = Batch(
                batch_id="batch_0001",
                input_dir=root / "input",
                output_dir=root / "out",
                target_policy="voice_clone",
                candidate_samples=[
                    CandidateSample(
                        sample_id="sample_0001",
                        batch_id="batch_0001",
                        asset_id="asset_0001",
                        source_path=sample_path,
                        start_time=0.0,
                        end_time=2.0,
                        duration=2.0,
                        sample_path=sample_path,
                        transcript="hello world",
                        transcript_source="asr",
                        asr_confidence=0.95,
                    )
                ],
            )

            analyzed = inspect_candidate_samples(
                batch,
                policy=InspectionPolicy(
                    min_rms_dbfs=-45.0,
                    max_clipping_ratio=0.01,
                    max_silence_ratio=0.6,
                    max_noise_floor_dbfs=-35.0,
                ),
                audio_metrics_analyzer=FixedAudioMetricsAnalyzer(),
            )

        result = analyzed.analysis_results[0]
        self.assertIn("audio_too_quiet", result.issue_tags)
        self.assertIn("audio_clipping", result.issue_tags)
        self.assertIn("audio_high_silence_ratio", result.issue_tags)
        self.assertIn("audio_noise_high", result.issue_tags)
        self.assertEqual(result.evidence["audio_metrics"]["rms_dbfs"], -55.0)
        self.assertEqual(result.evidence["audio_metrics"]["clipping_ratio"], 0.03)
        self.assertEqual(result.evidence["audio_metrics"]["noise_floor_dbfs"], -32.0)


class TestAudioAwareCuration(unittest.TestCase):
    def _analysis_result(self, issue_tags):
        from audio_dataset_agent.core.models import AnalysisResult

        return AnalysisResult(
            sample_id="sample_0001",
            duration=2.0,
            transcript_length=11,
            asr_confidence=0.95,
            issue_tags=issue_tags,
            evidence={},
        )

    def test_curator_rejects_clipped_audio(self):
        from audio_dataset_agent.core.models import DecisionType
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        decision = BaselineCurationPolicy().decide(
            self._analysis_result(["audio_clipping"])
        )

        self.assertEqual(decision.decision, DecisionType.REJECT)
        self.assertEqual(decision.recommended_action, "exclude_clipped_sample")
        self.assertIn("audio_clipping", decision.reason)

    def test_curator_repairs_quiet_audio(self):
        from audio_dataset_agent.core.models import DecisionType
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        decision = BaselineCurationPolicy().decide(
            self._analysis_result(["audio_too_quiet"])
        )

        self.assertEqual(decision.decision, DecisionType.REPAIR)
        self.assertEqual(decision.recommended_action, "normalize_loudness")
        self.assertIn("audio_too_quiet", decision.reason)

    def test_curator_repairs_high_silence_audio(self):
        from audio_dataset_agent.core.models import DecisionType
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        decision = BaselineCurationPolicy().decide(
            self._analysis_result(["audio_high_silence_ratio"])
        )

        self.assertEqual(decision.decision, DecisionType.REPAIR)
        self.assertEqual(decision.recommended_action, "trim_silence")
        self.assertIn("audio_high_silence_ratio", decision.reason)

    def test_curator_repairs_noisy_audio_with_light_denoise(self):
        from audio_dataset_agent.core.models import DecisionType
        from audio_dataset_agent.curation.policy import BaselineCurationPolicy

        decision = BaselineCurationPolicy().decide(
            self._analysis_result(["audio_noise_high"])
        )

        self.assertEqual(decision.decision, DecisionType.REPAIR)
        self.assertEqual(decision.recommended_action, "denoise_light")
        self.assertIn("audio_noise_high", decision.reason)


class TestRepairPlanner(unittest.TestCase):
    def test_repair_planner_uses_asr_preset_for_loudness(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.repair.planner import BaselineRepairPlanner

        decision = CuratorDecision(
            sample_id="sample_asr",
            decision=DecisionType.REPAIR,
            reason="repair: audio_too_quiet",
            confidence=0.8,
            recommended_action="normalize_loudness",
        )

        plan = BaselineRepairPlanner(target_policy="asr").plan(decision)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.parameters["target_lufs"], -20.0)

    def test_repair_planner_uses_less_aggressive_asr_denoise_preset(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.repair.planner import BaselineRepairPlanner

        decision = CuratorDecision(
            sample_id="sample_asr_noise",
            decision=DecisionType.REPAIR,
            reason="repair: audio_noise_high",
            confidence=0.8,
            recommended_action="denoise_light",
        )

        plan = BaselineRepairPlanner(target_policy="asr").plan(decision)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.parameters["noise_reduction_db"], 8.0)

    def test_repair_executor_rejects_out_of_range_loudness_parameter(self):
        from pathlib import Path

        from audio_dataset_agent.core.models import RepairPlan
        from audio_dataset_agent.repair.executor import FFmpegRepairExecutor

        plan = RepairPlan(
            sample_id="sample_invalid",
            operation="loudness_normalization",
            parameters={"target_lufs": -50.0},
        )

        with self.assertRaisesRegex(ValueError, "目标响度"):
            FFmpegRepairExecutor().build_command(
                plan,
                Path("input.wav"),
                Path("output.wav"),
            )

    def test_repair_planner_maps_loudness_action_to_plan(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.repair.planner import BaselineRepairPlanner

        decision = CuratorDecision(
            sample_id="sample_0001",
            decision=DecisionType.REPAIR,
            reason="repair: audio_too_quiet",
            confidence=0.8,
            recommended_action="normalize_loudness",
        )

        plan = BaselineRepairPlanner().plan(decision)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "loudness_normalization")
        self.assertEqual(plan.parameters["target_lufs"], -23.0)

    def test_repair_planner_maps_silence_action_to_plan(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.repair.planner import BaselineRepairPlanner

        decision = CuratorDecision(
            sample_id="sample_0001",
            decision=DecisionType.REPAIR,
            reason="repair: audio_high_silence_ratio",
            confidence=0.8,
            recommended_action="trim_silence",
        )

        plan = BaselineRepairPlanner().plan(decision)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "silence_trimming")
        self.assertEqual(plan.parameters["silence_threshold_dbfs"], -50.0)

    def test_repair_planner_maps_denoise_action_to_plan(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.repair.planner import BaselineRepairPlanner

        decision = CuratorDecision(
            sample_id="sample_0001",
            decision=DecisionType.REPAIR,
            reason="repair: audio_noise_high",
            confidence=0.8,
            recommended_action="denoise_light",
        )

        plan = BaselineRepairPlanner().plan(decision)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "noise_reduction")
        self.assertEqual(plan.parameters["noise_reduction_db"], 12.0)

    def test_repair_planner_maps_neural_denoise_action_to_plan(self):
        from audio_dataset_agent.core.models import CuratorDecision, DecisionType
        from audio_dataset_agent.repair.planner import BaselineRepairPlanner

        decision = CuratorDecision(
            sample_id="sample_0001",
            decision=DecisionType.REPAIR,
            reason="repair: stronger denoise requested",
            confidence=0.72,
            recommended_action="neural_denoise",
        )

        plan = BaselineRepairPlanner().plan(decision)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "neural_noise_reduction")
        self.assertTrue(plan.parameters["post_filter"])

    def test_workflow_manifest_includes_repair_plan(self):
        from audio_dataset_agent.core.models import DecisionType
        from audio_dataset_agent.exporters.manifest import build_manifest_records
        from audio_dataset_agent.repair.service import plan_repairs

        analysis = self._analysis_result(["audio_too_quiet"])
        batch = self._batch_with_analysis(analysis)
        planned = plan_repairs(batch)

        record = build_manifest_records(planned)[0]

        self.assertEqual(planned.repair_plans[0].sample_id, "sample_0001")
        self.assertEqual(record["decision"], DecisionType.REPAIR.value)
        self.assertEqual(record["repair_plan"]["operation"], "loudness_normalization")

    def _analysis_result(self, issue_tags):
        from audio_dataset_agent.core.models import AnalysisResult

        return AnalysisResult(
            sample_id="sample_0001",
            duration=2.0,
            transcript_length=11,
            asr_confidence=0.95,
            issue_tags=issue_tags,
            evidence={},
        )

    def _batch_with_analysis(self, analysis):
        from pathlib import Path

        from audio_dataset_agent.core.models import Batch, CandidateSample
        from audio_dataset_agent.curation.service import curate_candidate_samples

        batch = Batch(
            batch_id="batch_0001",
            input_dir=Path("input"),
            output_dir=Path("out"),
            target_policy="voice_clone",
            candidate_samples=[
                CandidateSample(
                    sample_id="sample_0001",
                    batch_id="batch_0001",
                    asset_id="asset_0001",
                    source_path=Path("sample.wav"),
                    start_time=0.0,
                    end_time=2.0,
                    duration=2.0,
                    transcript="hello world",
                    transcript_source="asr",
                    asr_confidence=0.95,
                )
            ],
            analysis_results=[analysis],
        )
        return curate_candidate_samples(batch)


class TestRepairExecutor(unittest.TestCase):
    def test_ffmpeg_repair_executor_runs_loudness_command(self):
        from audio_dataset_agent.core.models import CandidateSample, RepairPlan
        from audio_dataset_agent.repair.executor import FFmpegRepairExecutor

        commands = []

        def fake_runner(command):
            commands.append(command)
            Path(command[-1]).write_bytes(b"repaired")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.wav"
            source.write_bytes(b"audio")
            sample = CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=source,
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
                sample_path=source,
            )
            plan = RepairPlan(
                sample_id="sample_0001",
                operation="loudness_normalization",
                parameters={"target_lufs": -23.0},
            )

            execution = FFmpegRepairExecutor(runner=fake_runner).execute(
                plan,
                sample,
                root / "repairs",
            )

        self.assertEqual(execution.status, "succeeded")
        self.assertEqual(execution.output_path.name, "sample_0001_loudness_normalization.wav")
        self.assertIn("loudnorm=I=-23.0", commands[0])
        self.assertIn("-c:a", commands[0])
        self.assertIn("pcm_s16le", commands[0])

    def test_ffmpeg_repair_executor_runs_silence_trimming_command(self):
        from audio_dataset_agent.core.models import CandidateSample, RepairPlan
        from audio_dataset_agent.repair.executor import FFmpegRepairExecutor

        commands = []

        def fake_runner(command):
            commands.append(command)
            Path(command[-1]).write_bytes(b"repaired")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.wav"
            source.write_bytes(b"audio")
            sample = CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=source,
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
                sample_path=source,
            )
            plan = RepairPlan(
                sample_id="sample_0001",
                operation="silence_trimming",
                parameters={
                    "silence_threshold_dbfs": -50.0,
                    "min_silence_duration_ms": 200,
                },
            )

            execution = FFmpegRepairExecutor(runner=fake_runner).execute(
                plan,
                sample,
                root / "repairs",
            )

        self.assertEqual(execution.status, "succeeded")
        command_text = " ".join(commands[0])
        self.assertIn("silenceremove=", command_text)
        self.assertIn("start_threshold=-50.0dB", command_text)

    def test_ffmpeg_repair_executor_runs_noise_reduction_command(self):
        from audio_dataset_agent.core.models import CandidateSample, RepairPlan
        from audio_dataset_agent.repair.executor import FFmpegRepairExecutor

        commands = []

        def fake_runner(command):
            commands.append(command)
            Path(command[-1]).write_bytes(b"repaired")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.wav"
            source.write_bytes(b"audio")
            sample = CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=source,
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
                sample_path=source,
            )
            plan = RepairPlan(
                sample_id="sample_0001",
                operation="noise_reduction",
                parameters={"noise_reduction_db": 12.0},
            )

            execution = FFmpegRepairExecutor(runner=fake_runner).execute(
                plan,
                sample,
                root / "repairs",
            )

        self.assertEqual(execution.status, "succeeded")
        self.assertEqual(execution.output_path.name, "sample_0001_noise_reduction.wav")
        command_text = " ".join(commands[0])
        self.assertIn("afftdn=", command_text)
        self.assertIn("nr=12.0", command_text)

    def test_deepfilternet_repair_executor_runs_neural_denoise_command(self):
        from audio_dataset_agent.core.models import CandidateSample, RepairPlan
        from audio_dataset_agent.repair.executor import DeepFilterNetRepairExecutor

        commands = []

        def fake_runner(command):
            commands.append(command)
            if "--output-dir" in command:
                out_dir = Path(command[command.index("--output-dir") + 1])
            else:
                out_dir = Path(command[command.index("-o") + 1])
            input_path = Path(command[-1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / input_path.name).write_bytes(b"enhanced")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.wav"
            source.write_bytes(b"audio")
            sample = CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=source,
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
                sample_path=source,
            )
            plan = RepairPlan(
                sample_id="sample_0001",
                operation="neural_noise_reduction",
                parameters={"post_filter": True},
            )

            execution = DeepFilterNetRepairExecutor(
                deepfilter_path="deep-filter",
                runner=fake_runner,
            ).execute(
                plan,
                sample,
                root / "repairs",
            )

        self.assertEqual(execution.status, "succeeded")
        self.assertEqual(
            execution.output_path.name,
            "sample_0001_neural_noise_reduction.wav",
        )
        command_text = " ".join(commands[0])
        self.assertIn("deep-filter", command_text)
        self.assertIn("-o", command_text)
        self.assertIn("--pf", command_text)

    def test_composite_repair_executor_routes_neural_and_ffmpeg_repairs(self):
        from audio_dataset_agent.core.models import CandidateSample, RepairPlan
        from audio_dataset_agent.repair.executor import (
            CompositeRepairExecutor,
            DeepFilterNetRepairExecutor,
            FFmpegRepairExecutor,
        )

        routed = []

        class FakeExecutor:
            def __init__(self, name):
                self.name = name

            def execute(self, plan, sample, repairs_dir):
                routed.append((self.name, plan.operation))
                return None

        sample = CandidateSample(
            sample_id="sample_0001",
            batch_id="batch_0001",
            asset_id="asset_0001",
            source_path=Path("sample.wav"),
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
        )
        executor = CompositeRepairExecutor(
            ffmpeg_executor=FakeExecutor("ffmpeg"),
            deepfilternet_executor=FakeExecutor("deepfilter"),
        )

        executor.execute(
            RepairPlan(sample_id="sample_0001", operation="noise_reduction"),
            sample,
            Path("repairs"),
        )
        executor.execute(
            RepairPlan(sample_id="sample_0001", operation="neural_noise_reduction"),
            sample,
            Path("repairs"),
        )

        self.assertEqual(
            routed,
            [
                ("ffmpeg", "noise_reduction"),
                ("deepfilter", "neural_noise_reduction"),
            ],
        )

    def test_execute_repairs_defaults_to_composite_executor(self):
        import audio_dataset_agent.repair.service as repair_service

        from audio_dataset_agent.core.models import Batch, CandidateSample, RepairPlan
        from audio_dataset_agent.repair.executor import CompositeRepairExecutor

        original = repair_service.CompositeRepairExecutor
        constructed = []

        class FakeCompositeExecutor(CompositeRepairExecutor):
            def __init__(self):
                constructed.append(True)

            def execute(self, plan, sample, repairs_dir):
                return None

        repair_service.CompositeRepairExecutor = FakeCompositeExecutor
        try:
            repair_service.execute_repairs(
                Batch(
                    batch_id="batch_0001",
                    input_dir=Path("input"),
                    output_dir=Path("out"),
                    target_policy="voice_clone",
                    candidate_samples=[
                        CandidateSample(
                            sample_id="sample_0001",
                            batch_id="batch_0001",
                            asset_id="asset_0001",
                            source_path=Path("sample.wav"),
                            start_time=0.0,
                            end_time=2.0,
                            duration=2.0,
                        )
                    ],
                    repair_plans=[
                        RepairPlan(
                            sample_id="sample_0001",
                            operation="neural_noise_reduction",
                        )
                    ],
                )
            )
        finally:
            repair_service.CompositeRepairExecutor = original

        self.assertTrue(constructed)

    def test_manifest_includes_repair_execution(self):
        from audio_dataset_agent.core.models import (
            Batch,
            CandidateSample,
            RepairExecution,
            RepairPlan,
        )
        from audio_dataset_agent.exporters.manifest import build_manifest_records

        sample = CandidateSample(
            sample_id="sample_0001",
            batch_id="batch_0001",
            asset_id="asset_0001",
            source_path=Path("sample.wav"),
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            sample_path=Path("sample.wav"),
        )
        batch = Batch(
            batch_id="batch_0001",
            input_dir=Path("input"),
            output_dir=Path("out"),
            target_policy="voice_clone",
            candidate_samples=[sample],
            repair_plans=[
                RepairPlan(
                    sample_id="sample_0001",
                    operation="loudness_normalization",
                )
            ],
            repair_executions=[
                RepairExecution(
                    sample_id="sample_0001",
                    operation="loudness_normalization",
                    input_path=Path("sample.wav"),
                    output_path=Path("repaired.wav"),
                    status="succeeded",
                )
            ],
        )

        record = build_manifest_records(batch)[0]

        self.assertEqual(record["repair_execution"]["status"], "succeeded")
        self.assertEqual(record["repair_execution"]["output_path"], "repaired.wav")


class TestRepairEvaluation(unittest.TestCase):
    def test_evaluator_marks_loudness_repair_improved(self):
        from audio_dataset_agent.core.models import RepairExecution
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.repair.evaluator import BaselineRepairEvaluator

        class SequenceAudioMetricsAnalyzer:
            def __init__(self):
                self.metrics = [
                    AudioMetrics(-55.0, -12.0, 0.0, 0.1),
                    AudioMetrics(-24.0, -3.0, 0.0, 0.1),
                ]

            def analyze(self, audio_path):
                return self.metrics.pop(0)

        execution = RepairExecution(
            sample_id="sample_0001",
            operation="loudness_normalization",
            input_path=Path("before.wav"),
            output_path=Path("after.wav"),
            status="succeeded",
        )

        evaluation = BaselineRepairEvaluator().evaluate(
            execution,
            SequenceAudioMetricsAnalyzer(),
        )

        self.assertTrue(evaluation.improved)
        self.assertEqual(evaluation.recommendation, "adopt_repaired")
        self.assertEqual(evaluation.before_metrics["rms_dbfs"], -55.0)
        self.assertEqual(evaluation.after_metrics["rms_dbfs"], -24.0)

    def test_evaluator_marks_silence_trimming_improved(self):
        from audio_dataset_agent.core.models import RepairExecution
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.repair.evaluator import BaselineRepairEvaluator

        class SequenceAudioMetricsAnalyzer:
            def __init__(self):
                self.metrics = [
                    AudioMetrics(-30.0, -3.0, 0.0, 0.8),
                    AudioMetrics(-30.0, -3.0, 0.0, 0.2),
                ]

            def analyze(self, audio_path):
                return self.metrics.pop(0)

        execution = RepairExecution(
            sample_id="sample_0001",
            operation="silence_trimming",
            input_path=Path("before.wav"),
            output_path=Path("after.wav"),
            status="succeeded",
        )

        evaluation = BaselineRepairEvaluator().evaluate(
            execution,
            SequenceAudioMetricsAnalyzer(),
        )

        self.assertTrue(evaluation.improved)
        self.assertEqual(evaluation.recommendation, "adopt_repaired")
        self.assertEqual(evaluation.after_metrics["silence_ratio"], 0.2)

    def test_evaluator_marks_noise_reduction_improved_when_noise_floor_drops(self):
        from audio_dataset_agent.core.models import RepairExecution
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.repair.evaluator import BaselineRepairEvaluator

        class SequenceAudioMetricsAnalyzer:
            def __init__(self):
                self.metrics = [
                    AudioMetrics(
                        -32.0,
                        -5.0,
                        0.0,
                        0.1,
                        noise_floor_dbfs=-25.0,
                    ),
                    AudioMetrics(
                        -34.0,
                        -5.0,
                        0.0,
                        0.1,
                        noise_floor_dbfs=-39.0,
                    ),
                ]

            def analyze(self, audio_path):
                return self.metrics.pop(0)

        execution = RepairExecution(
            sample_id="sample_0001",
            operation="neural_noise_reduction",
            input_path=Path("before.wav"),
            output_path=Path("after.wav"),
            status="succeeded",
        )

        evaluation = BaselineRepairEvaluator().evaluate(
            execution,
            SequenceAudioMetricsAnalyzer(),
        )

        self.assertTrue(evaluation.improved)
        self.assertEqual(evaluation.recommendation, "adopt_repaired")
        self.assertIn("噪声底下降", evaluation.reason)

    def test_evaluator_explains_minor_noise_floor_drop_is_not_adopted(self):
        from audio_dataset_agent.core.models import RepairExecution
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.repair.evaluator import BaselineRepairEvaluator

        class SequenceAudioMetricsAnalyzer:
            def __init__(self):
                self.metrics = [
                    AudioMetrics(
                        -17.0,
                        -10.0,
                        0.0,
                        0.02,
                        noise_floor_dbfs=-31.0,
                    ),
                    AudioMetrics(
                        -17.2,
                        -10.0,
                        0.0,
                        0.03,
                        noise_floor_dbfs=-32.0,
                    ),
                ]

            def analyze(self, audio_path):
                return self.metrics.pop(0)

        execution = RepairExecution(
            sample_id="sample_0001",
            operation="noise_reduction",
            input_path=Path("before.wav"),
            output_path=Path("after.wav"),
            status="succeeded",
        )

        evaluation = BaselineRepairEvaluator().evaluate(
            execution,
            SequenceAudioMetricsAnalyzer(),
        )

        self.assertFalse(evaluation.improved)
        self.assertEqual(evaluation.recommendation, "keep_original")
        self.assertIn("未达到采用阈值", evaluation.reason)

    def test_evaluator_rejects_denoise_that_turns_audio_into_silence(self):
        from audio_dataset_agent.core.models import RepairExecution
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.repair.evaluator import BaselineRepairEvaluator

        class SequenceAudioMetricsAnalyzer:
            def __init__(self):
                self.metrics = [
                    AudioMetrics(
                        -15.0,
                        -8.0,
                        0.0,
                        0.02,
                        noise_floor_dbfs=-32.0,
                    ),
                    AudioMetrics(
                        -90.0,
                        -68.0,
                        0.0,
                        1.0,
                        noise_floor_dbfs=-90.0,
                    ),
                ]

            def analyze(self, audio_path):
                return self.metrics.pop(0)

        execution = RepairExecution(
            sample_id="sample_0001",
            operation="neural_noise_reduction",
            input_path=Path("before.wav"),
            output_path=Path("after.wav"),
            status="succeeded",
        )

        evaluation = BaselineRepairEvaluator().evaluate(
            execution,
            SequenceAudioMetricsAnalyzer(),
        )

        self.assertFalse(evaluation.improved)
        self.assertEqual(evaluation.recommendation, "keep_original")
        self.assertIn("降噪过度", evaluation.reason)

    def test_manifest_includes_repair_evaluation(self):
        from audio_dataset_agent.core.models import (
            Batch,
            CandidateSample,
            RepairEvaluation,
        )
        from audio_dataset_agent.exporters.manifest import build_manifest_records

        batch = Batch(
            batch_id="batch_0001",
            input_dir=Path("input"),
            output_dir=Path("out"),
            target_policy="voice_clone",
            candidate_samples=[
                CandidateSample(
                    sample_id="sample_0001",
                    batch_id="batch_0001",
                    asset_id="asset_0001",
                    source_path=Path("sample.wav"),
                    start_time=0.0,
                    end_time=2.0,
                    duration=2.0,
                )
            ],
            repair_evaluations=[
                RepairEvaluation(
                    sample_id="sample_0001",
                    operation="loudness_normalization",
                    before_metrics={"rms_dbfs": -55.0},
                    after_metrics={"rms_dbfs": -24.0},
                    improved=True,
                    recommendation="adopt_repaired",
                )
            ],
        )

        record = build_manifest_records(batch)[0]

        self.assertTrue(record["repair_evaluation"]["improved"])
        self.assertEqual(
            record["repair_evaluation"]["recommendation"],
            "adopt_repaired",
        )


class TestFinalSelector(unittest.TestCase):
    def test_final_selector_keeps_original_sample_for_keep_decision(self):
        from audio_dataset_agent.core.models import (
            CandidateSample,
            CuratorDecision,
            DecisionType,
        )
        from audio_dataset_agent.selection.service import select_final_samples

        batch = self._batch(
            sample=CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=Path("source.wav"),
                sample_path=Path("sample.wav"),
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
            ),
            decision=CuratorDecision(
                sample_id="sample_0001",
                decision=DecisionType.KEEP,
                reason="keep",
                confidence=0.9,
                recommended_action="prepare_for_export",
            ),
        )

        selected = select_final_samples(batch)

        self.assertEqual(selected.final_selections[0].status, "selected_original")
        self.assertEqual(selected.final_selections[0].selected_audio_path, Path("sample.wav"))

    def test_final_selector_uses_repaired_sample_when_evaluation_improved(self):
        from audio_dataset_agent.core.models import (
            CandidateSample,
            CuratorDecision,
            DecisionType,
            RepairEvaluation,
            RepairExecution,
        )
        from audio_dataset_agent.selection.service import select_final_samples

        batch = self._batch(
            sample=CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=Path("source.wav"),
                sample_path=Path("sample.wav"),
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
            ),
            decision=CuratorDecision(
                sample_id="sample_0001",
                decision=DecisionType.REPAIR,
                reason="repair",
                confidence=0.8,
                recommended_action="normalize_loudness",
            ),
            repair_execution=RepairExecution(
                sample_id="sample_0001",
                operation="loudness_normalization",
                input_path=Path("sample.wav"),
                output_path=Path("repaired.wav"),
                status="succeeded",
            ),
            repair_evaluation=RepairEvaluation(
                sample_id="sample_0001",
                operation="loudness_normalization",
                before_metrics={},
                after_metrics={},
                improved=True,
                recommendation="adopt_repaired",
            ),
        )

        selected = select_final_samples(batch)

        self.assertEqual(selected.final_selections[0].status, "selected_repaired")
        self.assertEqual(selected.final_selections[0].selected_audio_path, Path("repaired.wav"))

    def test_final_selector_excludes_rejected_sample(self):
        from audio_dataset_agent.core.models import (
            CandidateSample,
            CuratorDecision,
            DecisionType,
        )
        from audio_dataset_agent.selection.service import select_final_samples

        batch = self._batch(
            sample=CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=Path("source.wav"),
                sample_path=Path("sample.wav"),
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
            ),
            decision=CuratorDecision(
                sample_id="sample_0001",
                decision=DecisionType.REJECT,
                reason="reject",
                confidence=0.9,
                recommended_action="exclude_sample",
            ),
        )

        selected = select_final_samples(batch)

        self.assertEqual(selected.final_selections[0].status, "excluded")
        self.assertIsNone(selected.final_selections[0].selected_audio_path)

    def test_manifest_includes_final_selection(self):
        from audio_dataset_agent.exporters.manifest import build_manifest_records
        from audio_dataset_agent.selection.service import select_final_samples

        selected = select_final_samples(self._keep_batch())
        record = build_manifest_records(selected)[0]

        self.assertEqual(record["final_status"], "selected_original")
        self.assertEqual(record["final_audio_path"], "sample.wav")
        self.assertEqual(
            record["final_selection"]["reason"],
            "selected original sample from keep decision",
        )

    def _keep_batch(self):
        from audio_dataset_agent.core.models import (
            CandidateSample,
            CuratorDecision,
            DecisionType,
        )

        return self._batch(
            sample=CandidateSample(
                sample_id="sample_0001",
                batch_id="batch_0001",
                asset_id="asset_0001",
                source_path=Path("source.wav"),
                sample_path=Path("sample.wav"),
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
            ),
            decision=CuratorDecision(
                sample_id="sample_0001",
                decision=DecisionType.KEEP,
                reason="keep",
                confidence=0.9,
                recommended_action="prepare_for_export",
            ),
        )

    def _batch(
        self,
        sample,
        decision,
        repair_execution=None,
        repair_evaluation=None,
    ):
        from audio_dataset_agent.core.models import Batch

        return Batch(
            batch_id="batch_0001",
            input_dir=Path("input"),
            output_dir=Path("out"),
            target_policy="voice_clone",
            candidate_samples=[sample],
            curator_decisions=[decision],
            repair_executions=[repair_execution] if repair_execution else [],
            repair_evaluations=[repair_evaluation] if repair_evaluation else [],
        )


if __name__ == "__main__":
    unittest.main()
