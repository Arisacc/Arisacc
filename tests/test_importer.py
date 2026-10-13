import unittest
import json


class TestAudioScanner(unittest.TestCase):
    def test_scanner_returns_supported_audio_files_only(self):
        from audio_dataset_agent.importer.scanner import scan_audio_files

        with self.subTest("filters and sorts audio files"):
            from tempfile import TemporaryDirectory
            from pathlib import Path

            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "voice.wav").write_bytes(b"fake")
                (root / "clip.MP3").write_bytes(b"fake")
                (root / "notes.txt").write_text("ignore", encoding="utf-8")

                files = scan_audio_files(root)

        self.assertEqual([path.name for path in files], ["clip.MP3", "voice.wav"])


class TestBatchImport(unittest.TestCase):
    def test_batch_import_creates_raw_assets_with_stable_ids(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.core.models import BatchPhase
        from audio_dataset_agent.importer.service import create_import_batch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"fake")

            batch = create_import_batch(
                input_dir,
                output_dir=output_dir,
                target_policy="tts",
            )

        self.assertEqual(batch.phase, BatchPhase.IMPORTED)
        self.assertEqual(len(batch.raw_assets), 1)
        self.assertTrue(batch.raw_assets[0].asset_id.startswith(batch.batch_id))
        self.assertEqual(batch.raw_assets[0].source_path.name, "voice.wav")


class TestSerialization(unittest.TestCase):
    def test_batch_can_be_serialized_to_json_ready_dict(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.core.serialization import batch_to_dict
        from audio_dataset_agent.importer.service import create_import_batch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"fake")

            batch = create_import_batch(
                input_dir,
                output_dir=output_dir,
                target_policy="tts",
            )
            payload = batch_to_dict(batch)

        self.assertEqual(payload["batch_id"], batch.batch_id)
        self.assertEqual(payload["phase"], "imported")
        self.assertEqual(payload["raw_asset_count"], 1)
        self.assertTrue(payload["raw_assets"][0]["source_path"].endswith("voice.wav"))


class TestNormalization(unittest.TestCase):
    def test_normalize_batch_creates_working_copy_without_modifying_source(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.core.models import BatchPhase
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.processing.service import normalize_batch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            source = input_dir / "voice.wav"
            source.write_bytes(b"original")
            batch = create_import_batch(input_dir, output_dir=output_dir)

            normalized = normalize_batch(batch)

            self.assertEqual(normalized.phase, BatchPhase.NORMALIZED)
            self.assertEqual(source.read_bytes(), b"original")
            self.assertIsNotNone(normalized.raw_assets[0].work_path)
            self.assertTrue(normalized.raw_assets[0].work_path.exists())
            self.assertEqual(normalized.raw_assets[0].work_path.read_bytes(), b"original")

    def test_serialized_normalized_batch_includes_work_path(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.core.serialization import batch_to_dict
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.processing.service import normalize_batch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(
                input_dir,
                output_dir=output_dir,
                target_policy="tts",
            )
            normalized = normalize_batch(batch)
            payload = batch_to_dict(normalized)

        self.assertEqual(payload["phase"], "normalized")
        self.assertIsNotNone(payload["raw_assets"][0]["work_path"])
        self.assertTrue(payload["raw_assets"][0]["work_path"].endswith(".wav"))


class TestCandidateGeneration(unittest.TestCase):
    def test_generate_candidate_samples_from_vad_segments(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import BatchPhase
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(input_dir, output_dir=output_dir)
            normalized = normalize_batch(batch)
            analyzer = FixedVADAnalyzer([SpeechSegment(start_time=0.0, end_time=2.5)])

            segmented = generate_candidate_samples(normalized, analyzer)

        self.assertEqual(segmented.phase, BatchPhase.SEGMENTED)
        self.assertEqual(len(segmented.candidate_samples), 1)
        sample = segmented.candidate_samples[0]
        self.assertTrue(sample.sample_id.startswith(segmented.batch_id))
        self.assertEqual(sample.asset_id, segmented.raw_assets[0].asset_id)
        self.assertEqual(sample.start_time, 0.0)
        self.assertEqual(sample.end_time, 2.5)
        self.assertEqual(sample.duration, 2.5)
        self.assertEqual(sample.source_path, segmented.raw_assets[0].work_path)

    def test_serialized_segmented_batch_includes_candidate_samples(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.serialization import batch_to_dict
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(input_dir, output_dir=output_dir)
            normalized = normalize_batch(batch)
            segmented = generate_candidate_samples(
                normalized,
                FixedVADAnalyzer([SpeechSegment(start_time=1.0, end_time=3.0)]),
            )
            payload = batch_to_dict(segmented)

        self.assertEqual(payload["phase"], "segmented")
        self.assertEqual(payload["candidate_sample_count"], 1)
        self.assertEqual(payload["candidate_samples"][0]["start_time"], 1.0)
        self.assertEqual(payload["candidate_samples"][0]["duration"], 2.0)


class TestTranscription(unittest.TestCase):
    def test_transcribe_candidate_samples_adds_text_source_and_confidence(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import BatchPhase, CandidateSampleStatus
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples
        from audio_dataset_agent.transcription.service import transcribe_candidate_samples
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(input_dir, output_dir=output_dir)
            normalized = normalize_batch(batch)
            segmented = generate_candidate_samples(
                normalized,
                FixedVADAnalyzer([SpeechSegment(start_time=0.0, end_time=2.5)]),
            )

            transcribed = transcribe_candidate_samples(
                segmented,
                FixedTranscriber(TranscriptResult(text="hello world", confidence=0.91)),
            )

        self.assertEqual(transcribed.phase, BatchPhase.TRANSCRIBED)
        sample = transcribed.candidate_samples[0]
        self.assertEqual(sample.status, CandidateSampleStatus.TRANSCRIBED)
        self.assertEqual(sample.transcript, "hello world")
        self.assertEqual(sample.transcript_source, "asr")
        self.assertEqual(sample.asr_confidence, 0.91)

    def test_serialized_transcribed_batch_includes_transcript_fields(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.serialization import batch_to_dict
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples
        from audio_dataset_agent.transcription.service import transcribe_candidate_samples
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(input_dir, output_dir=output_dir)
            normalized = normalize_batch(batch)
            segmented = generate_candidate_samples(
                normalized,
                FixedVADAnalyzer([SpeechSegment(start_time=0.0, end_time=2.5)]),
            )
            transcribed = transcribe_candidate_samples(
                segmented,
                FixedTranscriber(TranscriptResult(text="hello world", confidence=0.91)),
            )
            payload = batch_to_dict(transcribed)

        sample_payload = payload["candidate_samples"][0]
        self.assertEqual(payload["phase"], "transcribed")
        self.assertEqual(sample_payload["transcript"], "hello world")
        self.assertEqual(sample_payload["transcript_source"], "asr")
        self.assertEqual(sample_payload["asr_confidence"], 0.91)


class TestInspection(unittest.TestCase):
    def test_inspector_flags_short_and_low_confidence_samples(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import BatchPhase, CandidateSampleStatus
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.inspection.service import inspect_candidate_samples
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples
        from audio_dataset_agent.selection.service import select_final_samples
        from audio_dataset_agent.transcription.service import transcribe_candidate_samples
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(
                input_dir,
                output_dir=output_dir,
                target_policy="tts",
            )
            normalized = normalize_batch(batch)
            segmented = generate_candidate_samples(
                normalized,
                FixedVADAnalyzer([SpeechSegment(start_time=0.0, end_time=0.5)]),
            )
            transcribed = transcribe_candidate_samples(
                segmented,
                FixedTranscriber(TranscriptResult(text="hi", confidence=0.4)),
            )

            analyzed = inspect_candidate_samples(transcribed)

        self.assertEqual(analyzed.phase, BatchPhase.ANALYZED)
        self.assertEqual(analyzed.candidate_samples[0].status, CandidateSampleStatus.ANALYZED)
        self.assertEqual(len(analyzed.analysis_results), 1)
        result = analyzed.analysis_results[0]
        self.assertIn("too_short", result.issue_tags)
        self.assertIn("asr_low_confidence", result.issue_tags)
        self.assertEqual(result.evidence["duration"], 0.5)
        self.assertEqual(result.evidence["asr_confidence"], 0.4)

    def test_inspector_downgrades_text_tags_for_voice_clone_profile(self):
        from pathlib import Path

        from audio_dataset_agent.core.models import Batch, CandidateSample
        from audio_dataset_agent.inspection.service import inspect_candidate_samples

        batch = Batch(
            batch_id="batch_voice_clone",
            input_dir=Path("input"),
            output_dir=Path("out"),
            target_policy="voice_clone",
            candidate_samples=[
                CandidateSample(
                    sample_id="sample_no_text",
                    batch_id="batch_voice_clone",
                    asset_id="asset_0001",
                    source_path=Path("voice.wav"),
                    start_time=0.0,
                    end_time=2.0,
                    duration=2.0,
                    transcript=None,
                    transcript_source=None,
                    asr_confidence=None,
                )
            ],
        )

        analyzed = inspect_candidate_samples(batch)
        result = analyzed.analysis_results[0]

        self.assertNotIn("missing_transcript", result.issue_tags)
        self.assertEqual(
            result.evidence["profile_interpretation"]["missing_transcript"],
            "allowed",
        )

    def test_inspector_flags_low_confidence_faster_whisper_transcript(self):
        from pathlib import Path

        from audio_dataset_agent.core.models import Batch, CandidateSample
        from audio_dataset_agent.inspection.service import inspect_candidate_samples

        batch = Batch(
            batch_id="batch_tts",
            input_dir=Path("input"),
            output_dir=Path("out"),
            target_policy="tts",
            candidate_samples=[
                CandidateSample(
                    sample_id="sample_whisper",
                    batch_id="batch_tts",
                    asset_id="asset_0001",
                    source_path=Path("voice.wav"),
                    start_time=0.0,
                    end_time=3.0,
                    duration=3.0,
                    transcript="recognized text",
                    transcript_source="faster_whisper",
                    asr_confidence=0.75,
                )
            ],
        )

        analyzed = inspect_candidate_samples(batch)

        self.assertIn("asr_low_confidence", analyzed.analysis_results[0].issue_tags)

    def test_serialized_analyzed_batch_includes_analysis_results(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.serialization import batch_to_dict
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.inspection.service import inspect_candidate_samples
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples
        from audio_dataset_agent.selection.service import select_final_samples
        from audio_dataset_agent.transcription.service import transcribe_candidate_samples
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(input_dir, output_dir=output_dir)
            normalized = normalize_batch(batch)
            segmented = generate_candidate_samples(
                normalized,
                FixedVADAnalyzer([SpeechSegment(start_time=0.0, end_time=0.5)]),
            )
            transcribed = transcribe_candidate_samples(
                segmented,
                FixedTranscriber(TranscriptResult(text="hi", confidence=0.4)),
            )
            analyzed = inspect_candidate_samples(transcribed)
            payload = batch_to_dict(analyzed)

        self.assertEqual(payload["phase"], "analyzed")
        self.assertEqual(payload["analysis_result_count"], 1)
        self.assertEqual(payload["analysis_results"][0]["sample_id"], analyzed.analysis_results[0].sample_id)
        self.assertIn("too_short", payload["analysis_results"][0]["issue_tags"])
        self.assertEqual(payload["analysis_results"][0]["evidence"]["transcript_length"], 2)


class TestCuration(unittest.TestCase):
    def test_curator_rejects_too_short_samples(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import BatchPhase, DecisionType
        from audio_dataset_agent.curation.service import curate_candidate_samples
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.inspection.service import inspect_candidate_samples
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples
        from audio_dataset_agent.selection.service import select_final_samples
        from audio_dataset_agent.transcription.service import transcribe_candidate_samples
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(input_dir, output_dir=output_dir)
            normalized = normalize_batch(batch)
            segmented = generate_candidate_samples(
                normalized,
                FixedVADAnalyzer([SpeechSegment(start_time=0.0, end_time=0.5)]),
            )
            transcribed = transcribe_candidate_samples(
                segmented,
                FixedTranscriber(TranscriptResult(text="hi", confidence=0.95)),
            )
            analyzed = inspect_candidate_samples(transcribed)

            curated = curate_candidate_samples(analyzed)

        self.assertEqual(curated.phase, BatchPhase.CURATED)
        self.assertEqual(len(curated.curator_decisions), 1)
        decision = curated.curator_decisions[0]
        self.assertEqual(decision.decision, DecisionType.REJECT)
        self.assertEqual(decision.sample_id, curated.analysis_results[0].sample_id)
        self.assertIn("too_short", decision.reason)

    def test_serialized_curated_batch_includes_curator_decisions(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.serialization import batch_to_dict
        from audio_dataset_agent.curation.service import curate_candidate_samples
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.inspection.service import inspect_candidate_samples
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples
        from audio_dataset_agent.selection.service import select_final_samples
        from audio_dataset_agent.transcription.service import transcribe_candidate_samples
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(input_dir, output_dir=output_dir)
            normalized = normalize_batch(batch)
            segmented = generate_candidate_samples(
                normalized,
                FixedVADAnalyzer([SpeechSegment(start_time=0.0, end_time=0.5)]),
            )
            transcribed = transcribe_candidate_samples(
                segmented,
                FixedTranscriber(TranscriptResult(text="hi", confidence=0.95)),
            )
            analyzed = inspect_candidate_samples(transcribed)
            curated = curate_candidate_samples(analyzed)
            payload = batch_to_dict(curated)

        self.assertEqual(payload["phase"], "curated")
        self.assertEqual(payload["curator_decision_count"], 1)
        self.assertEqual(payload["curator_decisions"][0]["decision"], "reject")
        self.assertEqual(payload["curator_decisions"][0]["recommended_action"], "exclude_sample")


class TestExport(unittest.TestCase):
    def test_export_curated_batch_writes_manifest_and_report(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import BatchPhase
        from audio_dataset_agent.curation.service import curate_candidate_samples
        from audio_dataset_agent.exporters.service import export_curated_batch
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.inspection.service import inspect_candidate_samples
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples
        from audio_dataset_agent.transcription.service import transcribe_candidate_samples
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            export_root = root / "exports"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(input_dir, output_dir=output_dir)
            normalized = normalize_batch(batch)
            segmented = generate_candidate_samples(
                normalized,
                FixedVADAnalyzer([SpeechSegment(start_time=0.0, end_time=0.5)]),
            )
            transcribed = transcribe_candidate_samples(
                segmented,
                FixedTranscriber(TranscriptResult(text="hi", confidence=0.95)),
            )
            analyzed = inspect_candidate_samples(transcribed)
            curated = curate_candidate_samples(analyzed)
            from audio_dataset_agent.selection.service import select_final_samples

            selected = select_final_samples(curated)

            exported = export_curated_batch(selected, export_root)

            manifest_path = Path(exported.export_paths["manifest"])
            report_path = Path(exported.export_paths["report"])
            quality_json_path = Path(exported.export_paths["dataset_quality_json"])
            quality_report_path = Path(exported.export_paths["dataset_quality_report"])
            manifest_exists = manifest_path.exists()
            report_exists = report_path.exists()
            quality_json_exists = quality_json_path.exists()
            quality_report_exists = quality_report_path.exists()
            manifest_line = manifest_path.read_text(encoding="utf-8").strip()
            manifest_record = json.loads(manifest_line)
            report_text = report_path.read_text(encoding="utf-8")
            quality_payload = json.loads(
                quality_json_path.read_text(encoding="utf-8")
            )
            quality_report_text = quality_report_path.read_text(encoding="utf-8")
            state_paths = exported.export_paths["state_files"]
            candidate_state = json.loads(
                Path(state_paths["candidate_samples"]).read_text(encoding="utf-8")
            )
            decisions_state = json.loads(
                Path(state_paths["curator_decisions"]).read_text(encoding="utf-8")
            )

        self.assertEqual(exported.phase, BatchPhase.EXPORTED)
        self.assertTrue(manifest_exists)
        self.assertTrue(report_exists)
        self.assertTrue(quality_json_exists)
        self.assertTrue(quality_report_exists)
        self.assertEqual(manifest_record["decision"], "reject")
        self.assertEqual(manifest_record["text"], "hi")
        self.assertEqual(manifest_record["duration"], 0.5)
        self.assertIn("## 概览", report_text)
        self.assertIn("## 决策统计", report_text)
        self.assertIn("## 样本明细", report_text)
        self.assertIn("- excluded: 1", report_text)
        self.assertIn("| 样本 ID | 决策 | 最终状态 | 时长 | ASR 置信度 | 问题标签 | 文本 |", report_text)
        self.assertEqual(quality_payload["agent_name"], "dataset_report_agent")
        self.assertEqual(quality_payload["rejected_samples"], 1)
        self.assertIn("数据集质量报告", quality_report_text)
        self.assertIn("## 人工复核队列", quality_report_text)
        self.assertEqual(candidate_state[0]["transcript"], "hi")
        self.assertEqual(decisions_state[0]["decision"], "reject")

    def test_serialized_exported_batch_includes_export_paths(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.serialization import batch_to_dict
        from audio_dataset_agent.curation.service import curate_candidate_samples
        from audio_dataset_agent.exporters.service import export_curated_batch
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.inspection.service import inspect_candidate_samples
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples
        from audio_dataset_agent.selection.service import select_final_samples
        from audio_dataset_agent.transcription.service import transcribe_candidate_samples
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            export_root = root / "exports"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"original")

            batch = create_import_batch(input_dir, output_dir=output_dir)
            normalized = normalize_batch(batch)
            segmented = generate_candidate_samples(
                normalized,
                FixedVADAnalyzer([SpeechSegment(start_time=0.0, end_time=0.5)]),
            )
            transcribed = transcribe_candidate_samples(
                segmented,
                FixedTranscriber(TranscriptResult(text="hi", confidence=0.95)),
            )
            analyzed = inspect_candidate_samples(transcribed)
            curated = curate_candidate_samples(analyzed)
            selected = select_final_samples(curated)
            exported = export_curated_batch(selected, export_root)
            payload = batch_to_dict(exported)

        self.assertEqual(payload["phase"], "exported")
        self.assertIn("manifest", payload["export_paths"])
        self.assertIn("report", payload["export_paths"])
        self.assertIn("dataset_quality_json", payload["export_paths"])
        self.assertIn("dataset_quality_report", payload["export_paths"])
        self.assertIn("state_dir", payload["export_paths"])
        self.assertIn("state_files", payload["export_paths"])


class TestFFmpegNormalizer(unittest.TestCase):
    def test_ffmpeg_normalizer_invokes_expected_command(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.core.models import RawAsset
        from audio_dataset_agent.processing.ffmpeg_normalizer import FFmpegAudioNormalizer

        commands = []

        def fake_runner(command):
            commands.append(command)
            Path(command[-1]).write_bytes(b"normalized")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.mp3"
            source.write_bytes(b"fake")
            asset = RawAsset(
                asset_id="asset_0001",
                batch_id="batch_0001",
                source_path=source,
                file_name="input.mp3",
                file_size=source.stat().st_size,
                extension=".mp3",
            )

            normalizer = FFmpegAudioNormalizer(
                ffmpeg_path="ffmpeg",
                sample_rate=16000,
                channels=1,
                runner=fake_runner,
            )
            output = normalizer.normalize(asset, root / "work")

        self.assertEqual(output.name, "asset_0001.wav")
        self.assertEqual(commands[0][0], "ffmpeg")
        self.assertIn("-ac", commands[0])
        self.assertIn("1", commands[0])
        self.assertIn("-ar", commands[0])
        self.assertIn("16000", commands[0])


if __name__ == "__main__":
    unittest.main()
