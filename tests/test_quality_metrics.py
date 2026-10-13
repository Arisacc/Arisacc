import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestQualityMetrics(unittest.TestCase):
    def test_estimate_snr_and_quality_score_from_audio_metrics(self):
        from audio_dataset_agent.inspection.audio_metrics import AudioMetrics
        from audio_dataset_agent.inspection.quality import assess_audio_quality

        assessment = assess_audio_quality(
            duration=2.0,
            metrics=AudioMetrics(
                rms_dbfs=-20.0,
                peak_dbfs=-3.0,
                clipping_ratio=0.001,
                silence_ratio=0.1,
                noise_floor_dbfs=-45.0,
            ),
        )

        self.assertEqual(assessment.snr_db, 25.0)
        self.assertGreater(assessment.quality_score, 0.8)
        self.assertLessEqual(assessment.quality_score, 1.0)

    def test_manifest_contains_snr_and_quality_score(self):
        from audio_dataset_agent.core.models import AnalysisResult, Batch, CandidateSample
        from audio_dataset_agent.exporters.manifest import build_manifest_records

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_path = root / "sample.wav"
            sample_path.write_bytes(b"audio")
            batch = Batch(
                batch_id="batch_quality",
                input_dir=root,
                output_dir=root / "out",
                target_policy="voice_clone",
                candidate_samples=[
                    CandidateSample(
                        sample_id="sample_1",
                        batch_id="batch_quality",
                        asset_id="asset_1",
                        source_path=sample_path,
                        sample_path=sample_path,
                        start_time=0.0,
                        end_time=2.0,
                        duration=2.0,
                    )
                ],
                analysis_results=[
                    AnalysisResult(
                        sample_id="sample_1",
                        duration=2.0,
                        transcript_length=0,
                        asr_confidence=None,
                        evidence={
                            "snr_db": 25.0,
                            "quality_score": 0.91,
                            "audio_metrics": {"rms_dbfs": -20.0},
                        },
                    )
                ],
            )

            record = build_manifest_records(batch)[0]

        self.assertEqual(record["snr"], 25.0)
        self.assertEqual(record["quality_score"], 0.91)
        self.assertEqual(record["audio_metrics"]["rms_dbfs"], -20.0)

    def test_report_exposes_average_quality_score(self):
        from audio_dataset_agent.core.models import AnalysisResult, Batch
        from audio_dataset_agent.reporting.agent import DatasetReportAgent

        batch = Batch(
            batch_id="batch_report_quality",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="voice_clone",
            analysis_results=[
                AnalysisResult(
                    sample_id="sample_1",
                    duration=2.0,
                    transcript_length=0,
                    asr_confidence=None,
                    evidence={"quality_score": 0.82},
                ),
                AnalysisResult(
                    sample_id="sample_2",
                    duration=2.0,
                    transcript_length=0,
                    asr_confidence=None,
                    evidence={"quality_score": 0.94},
                ),
            ],
        )

        report = DatasetReportAgent().build(batch)

        self.assertAlmostEqual(report.quality_score, 0.88)


if __name__ == "__main__":
    unittest.main()
