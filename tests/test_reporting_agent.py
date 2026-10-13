import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestDatasetReportAgent(unittest.TestCase):
    def test_report_agent_summarizes_batch_quality_and_repair_effect(self):
        from audio_dataset_agent.reporting.agent import DatasetReportAgent

        batch = _build_report_batch()

        report = DatasetReportAgent().build(batch)

        self.assertEqual(report.batch_id, "batch_report")
        self.assertEqual(report.total_samples, 4)
        self.assertEqual(report.usable_samples, 2)
        self.assertEqual(report.review_samples, 1)
        self.assertEqual(report.rejected_samples, 1)
        self.assertEqual(report.issue_counts["audio_noise_high"], 1)
        self.assertEqual(report.issue_counts["asr_low_confidence"], 1)
        self.assertEqual(report.decision_counts["repair"], 1)
        self.assertEqual(report.final_status_counts["selected_repaired"], 1)
        self.assertEqual(report.repair_summary["evaluated"], 1)
        self.assertEqual(report.repair_summary["improved"], 1)
        self.assertEqual(report.strategy_collaboration["accepted"], 1)
        self.assertEqual(report.strategy_collaboration["overridden"], 1)
        self.assertAlmostEqual(report.usable_rate, 0.5)
        self.assertEqual(report.agent_name, "dataset_report_agent")
        self.assertIn("audio_cleanliness", report.report_focus)

    def test_report_includes_text_quality_summary(self):
        from audio_dataset_agent.core.models import AnalysisResult, Batch
        from audio_dataset_agent.reporting.agent import DatasetReportAgent

        batch = Batch(
            batch_id="batch_text_report",
            input_dir=Path("input"),
            output_dir=Path("output"),
            target_policy="tts",
            analysis_results=[
                AnalysisResult(
                    sample_id="a", duration=2.0, transcript_length=4,
                    asr_confidence=0.9, evidence={"text_alignment": {"cer": 0.1, "wer": 0.2}},
                ),
                AnalysisResult(
                    sample_id="b", duration=2.0, transcript_length=4,
                    asr_confidence=0.9, issue_tags=["text_mismatch"],
                    evidence={"text_alignment": {"cer": 0.8, "wer": 0.8}},
                ),
            ],
        )

        report = DatasetReportAgent().build(batch)

        self.assertEqual(report.text_quality_summary["compared_samples"], 2)
        self.assertEqual(report.text_quality_summary["mismatch_samples"], 1)
        self.assertAlmostEqual(report.text_quality_summary["average_cer"], 0.45)

    def test_report_agent_builds_human_review_queue_with_reasons(self):
        from audio_dataset_agent.reporting.agent import DatasetReportAgent

        report = DatasetReportAgent().build(_build_report_batch())

        self.assertEqual(len(report.review_queue), 1)
        review_item = report.review_queue[0]
        self.assertEqual(review_item["sample_id"], "sample_review")
        self.assertEqual(review_item["recommended_action"], "manual_text_review")
        self.assertIn("asr_low_confidence", review_item["issue_tags"])
        self.assertIn("低置信", review_item["reason"])

    def test_report_agent_adds_final_selection_review_items_to_queue(self):
        from dataclasses import replace

        from audio_dataset_agent.reporting.agent import DatasetReportAgent

        batch = _build_report_batch()
        batch.final_selections = [
            replace(
                selection,
                status="needs_review",
                selected_audio_path=None,
                reason="repair did not produce an adopted output",
            )
            if selection.sample_id == "sample_repair"
            else selection
            for selection in batch.final_selections
        ]

        report = DatasetReportAgent().build(batch)
        review_ids = {item["sample_id"] for item in report.review_queue}

        self.assertIn("sample_review", review_ids)
        self.assertIn("sample_repair", review_ids)

    def test_report_agent_recommends_next_actions_from_batch_pattern(self):
        from audio_dataset_agent.reporting.agent import DatasetReportAgent

        report = DatasetReportAgent().build(_build_report_batch())

        joined = "\n".join(report.recommendations)
        self.assertIn("优先复核 1 条样本", joined)
        self.assertIn("噪声问题", joined)

    def test_render_and_write_dataset_quality_report(self):
        from audio_dataset_agent.reporting.agent import (
            DatasetReportAgent,
            dataset_quality_report_to_dict,
            render_dataset_quality_markdown,
            write_dataset_quality_report,
        )

        report = DatasetReportAgent().build(_build_report_batch())
        payload = dataset_quality_report_to_dict(report)
        markdown = render_dataset_quality_markdown(report)

        with TemporaryDirectory() as tmp:
            paths = write_dataset_quality_report(report, Path(tmp))
            json_payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            markdown_text = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(payload["batch_id"], "batch_report")
        self.assertIn("# 数据集质量报告", markdown)
        self.assertIn("## 任务关注点", markdown)
        self.assertIn("## 人工复核队列", markdown)
        self.assertEqual(json_payload["usable_samples"], 2)
        self.assertIn("report_focus", json_payload)
        self.assertIn("数据集质量报告", markdown_text)

    def test_report_agent_uses_voice_clone_profile_focus(self):
        from audio_dataset_agent.reporting.agent import DatasetReportAgent

        batch = _build_report_batch()
        batch.target_policy = "voice_clone"

        report = DatasetReportAgent().build(batch)

        self.assertIn("speaker_timbre", report.report_focus)
        self.assertTrue(
            any("声音克隆" in item or "音色" in item for item in report.recommendations)
        )


    def test_llm_report_agent_generates_recommendations_from_report_context(self):
        from audio_dataset_agent.reporting.agent import LLMDatasetReportAgent

        class RecommendationClient:
            def complete(self, prompt):
                self.prompt = prompt
                return {
                    "recommendations": [
                        "优先复核低 ASR 置信度样本，确认文本和音频是否匹配。",
                        "保留降噪前后指标，避免修复引入音色变化。",
                    ]
                }

        client = RecommendationClient()
        report = LLMDatasetReportAgent(client).build(_build_report_batch())

        self.assertIn("数据集质量报告上下文", client.prompt)
        self.assertIn("请为数据集处理人员写出简洁、可执行的中文建议", client.prompt)
        self.assertIn("review_queue", client.prompt)
        self.assertEqual(report.agent_name, "llm_dataset_report_agent")
        self.assertEqual(
            report.recommendations,
            [
                "优先复核低 ASR 置信度样本，确认文本和音频是否匹配。",
                "保留降噪前后指标，避免修复引入音色变化。",
            ],
        )

    def test_llm_report_agent_falls_back_to_deterministic_recommendations(self):
        from audio_dataset_agent.reporting.agent import DatasetReportAgent
        from audio_dataset_agent.reporting.agent import LLMDatasetReportAgent

        class BadReportClient:
            def complete(self, prompt):
                return "not json"

        batch = _build_report_batch()
        baseline_report = DatasetReportAgent().build(batch)
        report = LLMDatasetReportAgent(BadReportClient()).build(batch)

        self.assertEqual(report.agent_name, "llm_dataset_report_agent_guarded")
        self.assertEqual(report.recommendations, baseline_report.recommendations)

    def test_export_curated_batch_accepts_custom_report_agent(self):
        from dataclasses import replace

        from audio_dataset_agent.exporters.service import export_curated_batch
        from audio_dataset_agent.reporting.agent import DatasetReportAgent

        class FakeReportAgent:
            def build(self, batch):
                return replace(
                    DatasetReportAgent().build(batch),
                    agent_name="fake_report_agent",
                    recommendations=["fake report recommendation"],
                )

        with TemporaryDirectory() as tmp:
            exported = export_curated_batch(
                _build_report_batch(),
                Path(tmp),
                report_agent=FakeReportAgent(),
            )
            quality_path = Path(exported.export_paths["dataset_quality_json"])
            payload = json.loads(quality_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["agent_name"], "fake_report_agent")
        self.assertEqual(payload["recommendations"], ["fake report recommendation"])

    def test_export_curated_batch_writes_review_queue_files(self):
        from audio_dataset_agent.exporters.service import export_curated_batch

        with TemporaryDirectory() as tmp:
            exported = export_curated_batch(_build_report_batch(), Path(tmp))
            review_json_path = Path(exported.export_paths["review_queue_json"])
            review_csv_path = Path(exported.export_paths["review_queue_csv"])
            review_payload = json.loads(review_json_path.read_text(encoding="utf-8"))
            review_csv = review_csv_path.read_text(encoding="utf-8")

        self.assertEqual(len(review_payload), 1)
        self.assertEqual(review_payload[0]["sample_id"], "sample_review")
        self.assertIn("sample_id,recommended_action,issue_tags,confidence,reason", review_csv)
        self.assertIn("sample_review", review_csv)


def _build_report_batch():
    from audio_dataset_agent.core.models import (
        AnalysisResult,
        Batch,
        CandidateSample,
        CandidateSampleStatus,
        CuratorDecision,
        DecisionType,
        FinalSelection,
        ProcessingStrategy,
        RepairEvaluation,
    )

    batch = Batch(
        batch_id="batch_report",
        input_dir=Path("input"),
        output_dir=Path("out"),
        target_policy="tts",
    )
    batch.candidate_samples = [
        CandidateSample(
            sample_id="sample_keep",
            batch_id=batch.batch_id,
            asset_id="asset_1",
            source_path=Path("input/keep.wav"),
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            sample_path=Path("out/keep.wav"),
            transcript="干净样本",
            transcript_source="asr",
            asr_confidence=0.94,
            status=CandidateSampleStatus.ANALYZED,
        ),
        CandidateSample(
            sample_id="sample_repair",
            batch_id=batch.batch_id,
            asset_id="asset_2",
            source_path=Path("input/noise.wav"),
            start_time=0.0,
            end_time=3.0,
            duration=3.0,
            sample_path=Path("out/noise.wav"),
            transcript="有噪声样本",
            transcript_source="asr",
            asr_confidence=0.91,
            status=CandidateSampleStatus.ANALYZED,
        ),
        CandidateSample(
            sample_id="sample_review",
            batch_id=batch.batch_id,
            asset_id="asset_3",
            source_path=Path("input/review.wav"),
            start_time=0.0,
            end_time=2.8,
            duration=2.8,
            sample_path=Path("out/review.wav"),
            transcript="低置信样本",
            transcript_source="asr",
            asr_confidence=0.42,
            status=CandidateSampleStatus.ANALYZED,
        ),
        CandidateSample(
            sample_id="sample_reject",
            batch_id=batch.batch_id,
            asset_id="asset_4",
            source_path=Path("input/reject.wav"),
            start_time=0.0,
            end_time=0.4,
            duration=0.4,
            sample_path=Path("out/reject.wav"),
            transcript="短",
            transcript_source="asr",
            asr_confidence=0.9,
            status=CandidateSampleStatus.ANALYZED,
        ),
    ]
    batch.analysis_results = [
        AnalysisResult(
            sample_id="sample_keep",
            duration=2.0,
            transcript_length=4,
            asr_confidence=0.94,
            issue_tags=[],
        ),
        AnalysisResult(
            sample_id="sample_repair",
            duration=3.0,
            transcript_length=5,
            asr_confidence=0.91,
            issue_tags=["audio_noise_high"],
            evidence={"audio_metrics": {"noise_floor_dbfs": -25.0}},
        ),
        AnalysisResult(
            sample_id="sample_review",
            duration=2.8,
            transcript_length=5,
            asr_confidence=0.42,
            issue_tags=["asr_low_confidence"],
        ),
        AnalysisResult(
            sample_id="sample_reject",
            duration=0.4,
            transcript_length=1,
            asr_confidence=0.9,
            issue_tags=["too_short"],
        ),
    ]
    batch.processing_strategies = [
        ProcessingStrategy(
            sample_id="sample_keep",
            action="keep_without_processing",
            recommended_operation="prepare_for_export",
            reason="clean",
            confidence=0.8,
        ),
        ProcessingStrategy(
            sample_id="sample_repair",
            action="repair_audio",
            recommended_operation="neural_denoise",
            selected_tool_ids=["repair.deepfilternet_denoise"],
            reason="noise floor is high",
            confidence=0.78,
            agent_name="llm_processing_strategy",
        ),
    ]
    batch.curator_decisions = [
        CuratorDecision(
            sample_id="sample_keep",
            decision=DecisionType.KEEP,
            reason="clean sample",
            confidence=0.9,
            recommended_action="prepare_for_export",
            strategy_consistency="accepted",
        ),
        CuratorDecision(
            sample_id="sample_repair",
            decision=DecisionType.REPAIR,
            reason="noise should be repaired",
            confidence=0.82,
            recommended_action="neural_denoise",
            strategy_consistency="overridden",
        ),
        CuratorDecision(
            sample_id="sample_review",
            decision=DecisionType.REVIEW,
            reason="ASR 低置信，需要人工复核文本",
            confidence=0.8,
            recommended_action="manual_text_review",
        ),
        CuratorDecision(
            sample_id="sample_reject",
            decision=DecisionType.REJECT,
            reason="too short",
            confidence=0.95,
            recommended_action="exclude_sample",
        ),
    ]
    batch.repair_evaluations = [
        RepairEvaluation(
            sample_id="sample_repair",
            operation="neural_noise_reduction",
            before_metrics={"noise_floor_dbfs": -25.0},
            after_metrics={"noise_floor_dbfs": -41.0},
            improved=True,
            recommendation="adopt_repaired",
            reason="noise floor decreased",
        )
    ]
    batch.final_selections = [
        FinalSelection(
            sample_id="sample_keep",
            status="selected_original",
            selected_audio_path=Path("out/keep.wav"),
            reason="clean",
        ),
        FinalSelection(
            sample_id="sample_repair",
            status="selected_repaired",
            selected_audio_path=Path("out/noise_repaired.wav"),
            reason="repair improved quality",
        ),
        FinalSelection(
            sample_id="sample_review",
            status="needs_review",
            selected_audio_path=None,
            reason="manual text review",
        ),
        FinalSelection(
            sample_id="sample_reject",
            status="rejected",
            selected_audio_path=None,
            reason="too short",
        ),
    ]
    return batch


if __name__ == "__main__":
    unittest.main()
