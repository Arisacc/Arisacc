import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class TestRecursiveAudioScan(unittest.TestCase):
    def test_scanner_recursively_finds_audio_in_stable_relative_order(self):
        from audio_dataset_agent.importer.scanner import scan_audio_files

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b").mkdir()
            (root / "a").mkdir()
            (root / "b" / "second.wav").write_bytes(b"b")
            (root / "a" / "first.mp3").write_bytes(b"a")

            files = scan_audio_files(root)

        self.assertEqual(
            [path.relative_to(root).as_posix() for path in files],
            ["a/first.mp3", "b/second.wav"],
        )


class TestMetadataImport(unittest.TestCase):
    def test_configured_relative_metadata_path_can_resolve_from_project_root(self):
        from audio_dataset_agent.importer.metadata import _resolve_metadata_path

        with TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            input_dir = project_root / "datasets" / "prepared"
            input_dir.mkdir(parents=True)
            metadata_path = project_root / "metadata.csv"
            metadata_path.write_text("audio_path,text\n", encoding="utf-8")

            with patch(
                "audio_dataset_agent.importer.metadata.Path.cwd",
                return_value=project_root,
            ):
                resolved = _resolve_metadata_path(input_dir, "metadata.csv")

        self.assertEqual(resolved, metadata_path)

    def test_sidecar_text_is_attached_and_propagated_without_replacing_asr(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples
        from audio_dataset_agent.transcription.service import transcribe_candidate_samples
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"audio")
            (input_dir / "voice.txt").write_text("人工参考文本\n", encoding="utf-8")

            imported = create_import_batch(input_dir, root / "work", target_policy="tts")
            segmented = generate_candidate_samples(
                normalize_batch(imported),
                FixedVADAnalyzer([SpeechSegment(0.0, 2.0)]),
            )
            transcribed = transcribe_candidate_samples(
                segmented,
                FixedTranscriber(TranscriptResult("ASR 候选文本", 0.91)),
            )

        asset = imported.raw_assets[0]
        sample = transcribed.candidate_samples[0]
        self.assertEqual(asset.reference_text, "人工参考文本")
        self.assertEqual(asset.reference_text_source, "sidecar_txt")
        self.assertEqual(asset.metadata_status, "matched")
        self.assertEqual(sample.reference_text, "人工参考文本")
        self.assertEqual(sample.transcript, "ASR 候选文本")
        self.assertEqual(sample.transcript_source, "asr")
        self.assertEqual(imported.metadata_summary["matched_assets"], 1)

    def test_csv_supports_custom_columns_language_speaker_and_orphans(self):
        from audio_dataset_agent.importer.metadata import MetadataImportConfig
        from audio_dataset_agent.importer.service import create_import_batch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            nested = input_dir / "speaker_a"
            nested.mkdir(parents=True)
            (nested / "001.wav").write_bytes(b"audio")
            metadata_path = input_dir / "labels.csv"
            metadata_path.write_text(
                "file,sentence,locale,speaker\n"
                "speaker_a/001.wav,你好世界,zh-CN,spk_a\n"
                "missing.wav,孤立文本,zh-CN,spk_x\n",
                encoding="utf-8-sig",
            )
            config = MetadataImportConfig(
                metadata_path=metadata_path,
                audio_column="file",
                text_column="sentence",
                language_column="locale",
                speaker_column="speaker",
            )

            batch = create_import_batch(
                input_dir,
                root / "work",
                target_policy="tts",
                metadata_config=config,
            )

        asset = batch.raw_assets[0]
        self.assertEqual(asset.reference_text, "你好世界")
        self.assertEqual(asset.reference_text_source, "csv")
        self.assertEqual(asset.language, "zh-CN")
        self.assertEqual(asset.speaker_id, "spk_a")
        self.assertEqual(batch.metadata_summary["orphan_records"], 1)
        self.assertEqual(batch.metadata_summary["metadata_path"], str(metadata_path))

    def test_jsonl_is_auto_detected_and_serialized_to_manifest(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.serialization import batch_to_dict
        from audio_dataset_agent.exporters.manifest import build_manifest_records
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"audio")
            (input_dir / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "audio_path": "voice.wav",
                        "text": "参考文本",
                        "language": "zh",
                        "speaker_id": "speaker_01",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            imported = create_import_batch(input_dir, root / "work", target_policy="tts")
            segmented = generate_candidate_samples(
                normalize_batch(imported),
                FixedVADAnalyzer([SpeechSegment(0.0, 2.0)]),
            )
            payload = batch_to_dict(segmented)
            record = build_manifest_records(segmented)[0]

        self.assertEqual(imported.metadata_summary["source_type"], "jsonl")
        self.assertEqual(payload["metadata_summary"]["matched_assets"], 1)
        self.assertEqual(payload["candidate_samples"][0]["reference_text"], "参考文本")
        self.assertEqual(record["reference_text"], "参考文本")
        self.assertEqual(record["reference_text_source"], "jsonl")
        self.assertEqual(record["language"], "zh")
        self.assertEqual(record["speaker_id"], "speaker_01")

    def test_duplicate_basenames_are_not_silently_matched(self):
        from audio_dataset_agent.importer.metadata import MetadataImportConfig
        from audio_dataset_agent.importer.service import create_import_batch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            (input_dir / "a").mkdir(parents=True)
            (input_dir / "b").mkdir()
            (input_dir / "a" / "voice.wav").write_bytes(b"a")
            (input_dir / "b" / "voice.wav").write_bytes(b"b")
            metadata_path = input_dir / "labels.csv"
            metadata_path.write_text(
                "audio_path,text\nvoice.wav,无法确定属于哪条音频\n",
                encoding="utf-8",
            )

            batch = create_import_batch(
                input_dir,
                root / "work",
                metadata_config=MetadataImportConfig(metadata_path=metadata_path),
            )

        self.assertEqual(batch.metadata_summary["ambiguous_records"], 1)
        self.assertTrue(all(asset.metadata_status == "missing" for asset in batch.raw_assets))
        self.assertTrue(all(asset.reference_text is None for asset in batch.raw_assets))

    def test_workflow_accepts_metadata_config_and_exports_reference_text(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.importer.metadata import MetadataImportConfig
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"audio")
            labels = input_dir / "labels.csv"
            labels.write_text("file,sentence\nvoice.wav,人工文本\n", encoding="utf-8")

            exported = run_batch_pipeline(
                input_dir=input_dir,
                output_dir=root / "work",
                export_dir=root / "exports",
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.0, 2.0)]),
                transcriber=FixedTranscriber(TranscriptResult("ASR 文本", 0.95)),
                target_policy="tts",
                metadata_config=MetadataImportConfig(
                    metadata_path=labels,
                    audio_column="file",
                    text_column="sentence",
                ),
            )
            record = json.loads(
                Path(exported.export_paths["manifest"]).read_text(encoding="utf-8").strip()
            )
            import_report = json.loads(
                Path(exported.export_paths["metadata_import_report"]).read_text(encoding="utf-8")
            )
            source_manifest = json.loads(
                Path(exported.export_paths["source_manifest"]).read_text(encoding="utf-8")
            )

        self.assertEqual(record["reference_text"], "人工文本")
        self.assertEqual(record["text"], "ASR 文本")
        self.assertEqual(import_report["matched_assets"], 1)
        self.assertEqual(source_manifest["assets"][0]["reference_text_source"], "csv")
        self.assertEqual(source_manifest["assets"][0]["metadata_status"], "matched")

    def test_multiple_segments_mark_reference_text_as_requiring_alignment(self):
        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.importer.service import create_import_batch
        from audio_dataset_agent.processing.service import normalize_batch
        from audio_dataset_agent.segmentation.service import generate_candidate_samples

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "long.wav").write_bytes(b"audio")
            (input_dir / "long.txt").write_text("整段音频对应的参考文本", encoding="utf-8")

            batch = create_import_batch(input_dir, root / "work", target_policy="tts")
            segmented = generate_candidate_samples(
                normalize_batch(batch),
                FixedVADAnalyzer(
                    [SpeechSegment(0.0, 2.0), SpeechSegment(2.1, 4.0)]
                ),
            )

        self.assertEqual(len(segmented.candidate_samples), 2)
        self.assertTrue(
            all(
                sample.metadata_status == "requires_alignment"
                for sample in segmented.candidate_samples
            )
        )
        self.assertTrue(
            all(
                sample.reference_text == "整段音频对应的参考文本"
                for sample in segmented.candidate_samples
            )
        )


if __name__ == "__main__":
    unittest.main()
