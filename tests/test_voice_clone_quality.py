import unittest
from pathlib import Path


class TestAcousticConsistency(unittest.TestCase):
    def test_consistency_analyzer_reports_feature_spread(self):
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.inspection.consistency import (
            AcousticConsistencyPolicy,
            BatchAcousticConsistencyAnalyzer,
        )

        metrics = {
            Path("a.wav"): AudioMetrics(-20.0, -3.0, 0.0, 0.1, -55.0, 30.0),
            Path("b.wav"): AudioMetrics(-35.0, -5.0, 0.0, 0.7, -30.0, 60.0),
        }

        class FixedAnalyzer:
            def analyze(self, audio_path):
                return metrics[audio_path]

        result = BatchAcousticConsistencyAnalyzer(
            FixedAnalyzer(),
            AcousticConsistencyPolicy(
                max_rms_range_db=6.0,
                max_silence_range=0.2,
                max_noise_floor_range_db=8.0,
                max_dynamic_range_db=15.0,
            ),
        ).analyze([Path("a.wav"), Path("b.wav")])

        self.assertEqual(result.sample_count, 2)
        self.assertTrue(result.warning)
        self.assertIn("rms_range_db", result.to_dict())
        self.assertIn("silence_range", result.to_dict())

    def test_single_sample_does_not_trigger_consistency_warning(self):
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.inspection.consistency import BatchAcousticConsistencyAnalyzer

        class FixedAnalyzer:
            def analyze(self, audio_path):
                return AudioMetrics(-20.0, -3.0, 0.0, 0.1, -55.0, 30.0)

        result = BatchAcousticConsistencyAnalyzer(FixedAnalyzer()).analyze([Path("a.wav")])

        self.assertFalse(result.warning)
        self.assertEqual(result.compared_sample_count, 1)

    def test_voice_clone_inspector_attaches_consistency_evidence(self):
        from audio_dataset_agent.core.models import Batch, CandidateSample
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.inspection.consistency import BatchAcousticConsistencyAnalyzer
        from audio_dataset_agent.inspection.service import inspect_candidate_samples

        class FixedAnalyzer:
            def analyze(self, audio_path):
                return AudioMetrics(-20.0, -3.0, 0.0, 0.1, -55.0, 30.0)

        class FixedConsistencyAnalyzer:
            def analyze(self, audio_paths):
                return BatchAcousticConsistencyAnalyzer(FixedAnalyzer()).analyze(audio_paths)

        samples = [
            CandidateSample(
                sample_id=f"sample_{index}",
                batch_id="batch_voice",
                asset_id=f"asset_{index}",
                source_path=Path(f"{index}.wav"),
                sample_path=Path(f"{index}.wav"),
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
            )
            for index in range(2)
        ]
        batch = Batch(
            batch_id="batch_voice",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="voice_clone",
            candidate_samples=samples,
        )

        analyzed = inspect_candidate_samples(
            batch,
            audio_metrics_analyzer=FixedAnalyzer(),
            acoustic_consistency_analyzer=FixedConsistencyAnalyzer(),
        )

        evidence = analyzed.analysis_results[0].evidence
        self.assertEqual(evidence["speaker_consistency"]["sample_count"], 2)
        self.assertNotIn("speaker_consistency_warning", analyzed.analysis_results[0].issue_tags)

    def test_workflow_can_receive_consistency_analyzer(self):
        from tempfile import TemporaryDirectory

        from audio_dataset_agent.analyzers.vad import FixedVADAnalyzer, SpeechSegment
        from audio_dataset_agent.core.models import BatchPhase
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.inspection.consistency import BatchAcousticConsistencyAnalyzer
        from audio_dataset_agent.orchestrator.workflow import run_batch_pipeline
        from audio_dataset_agent.transcription.transcriber import FixedTranscriber, TranscriptResult

        class FixedAnalyzer:
            def analyze(self, audio_path):
                return AudioMetrics(-20.0, -3.0, 0.0, 0.1, -55.0, 30.0)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "voice.wav").write_bytes(b"audio")
            exported = run_batch_pipeline(
                input_dir=input_dir,
                output_dir=root / "work",
                export_dir=root / "exports",
                vad_analyzer=FixedVADAnalyzer([SpeechSegment(0.0, 2.0)]),
                transcriber=FixedTranscriber(TranscriptResult("", None)),
                audio_metrics_analyzer=FixedAnalyzer(),
                acoustic_consistency_analyzer=BatchAcousticConsistencyAnalyzer(FixedAnalyzer()),
                target_policy="voice_clone",
                run_asr=False,
            )

        self.assertEqual(exported.phase, BatchPhase.EXPORTED)
        self.assertEqual(
            exported.analysis_results[0].evidence["speaker_consistency"]["sample_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
