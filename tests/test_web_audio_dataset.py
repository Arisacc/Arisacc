import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestAudioPortfolioSummary(unittest.TestCase):
    def test_build_summary_from_export_dir_reads_core_artifacts(self):
        from audio_dataset_agent.web.summary import build_summary_from_export_dir

        with TemporaryDirectory() as tmp:
            export_dir = Path(tmp) / "batch_demo"
            state_dir = export_dir / "state"
            state_dir.mkdir(parents=True)
            (export_dir / "dataset_quality_report.json").write_text(
                json.dumps(
                    {
                        "batch_id": "batch_demo",
                        "target_policy": "voice_clone",
                        "total_samples": 2,
                        "usable_samples": 1,
                        "review_samples": 1,
                        "rejected_samples": 0,
                        "qualityScore": 0.88,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (export_dir / "manifest.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "sample_keep",
                                "duration": 1.5,
                                "transcript_source": "faster_whisper",
                                "asr_confidence": 0.91,
                                "snr": 23.5,
                                "quality_score": 0.91,
                                "issue_tags": [],
                                "decision": "keep",
                                "final_status": "selected_original",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "sample_id": "sample_review",
                                "duration": 2.1,
                                "transcript_source": "none",
                                "asr_confidence": None,
                                "issue_tags": ["asr_missing"],
                                "decision": "review",
                                "final_status": "needs_review",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            (state_dir / "processing_strategies.json").write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "sample_keep",
                            "recommended_operation": "keep_without_processing",
                        },
                        {
                            "sample_id": "sample_review",
                            "recommended_operation": "manual_text_review",
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (state_dir / "curator_decisions.json").write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "sample_keep",
                            "decision": "keep",
                            "strategy_consistency": "accepted",
                        },
                        {
                            "sample_id": "sample_review",
                            "decision": "review",
                            "strategy_consistency": "overridden",
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (export_dir / "agent_graph_trace.json").write_text(
                json.dumps(
                    {
                        "graph_name": "demo graph",
                        "nodes": [
                            {
                                "node_name": "strategy",
                                "agent_name": "Strategy Agent",
                                "uses_agent": True,
                                "uses_tool": False,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = build_summary_from_export_dir(export_dir)

        self.assertEqual(summary["batch"]["batch_id"], "batch_demo")
        self.assertEqual(summary["batch"]["target_policy"], "voice_clone")
        self.assertTrue(summary["funnel"][0]["label"])
        self.assertEqual(summary["samples"][0]["strategy"], "keep_without_processing")
        self.assertEqual(summary["samples"][1]["curator_decision"], "review")
        self.assertEqual(summary["samples"][0]["strategy_consistency"], "accepted")
        self.assertEqual(summary["samples"][1]["strategy_consistency"], "overridden")
        self.assertEqual(summary["qualityScore"], 0.88)
        self.assertEqual(summary["samples"][0]["snr"], 23.5)
        self.assertEqual(summary["samples"][0]["quality_score"], 0.91)
        self.assertEqual(summary["graph_trace"][0]["node_name"], "strategy")
        self.assertEqual(summary["repair_preset"]["target_lufs"], -23.0)
        self.assertEqual(summary["repair_preset"]["noise_reduction_db"], 10.0)
        self.assertEqual(summary["quality_policy"]["min_duration_seconds"], 1.5)
        self.assertEqual(summary["quality_policy"]["max_silence_ratio"], 0.45)


class TestAudioDatasetJobManager(unittest.TestCase):
    def test_stage_uploaded_directory_preserves_relative_files(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJobManager

        with TemporaryDirectory() as tmp:
            manager = AudioDatasetJobManager(Path(tmp) / "runs")
            staged = manager.stage_uploaded_directory(
                [
                    ("selected/sample_01.wav", b"audio"),
                    ("selected/labels/sample_01.txt", b"text"),
                ]
            )

            input_dir = Path(staged["input_dir"])
            self.assertEqual(staged["file_count"], 2)
            self.assertEqual(staged["directory_name"], "selected")
            self.assertEqual((input_dir / "sample_01.wav").read_bytes(), b"audio")
            self.assertEqual(
                (input_dir / "labels" / "sample_01.txt").read_bytes(),
                b"text",
            )

    def test_run_request_builds_reproducible_pipeline_config(self):
        from audio_dataset_agent.web.jobs import RunRequest

        request = RunRequest(
            input_dir="input",
            target_policy="tts",
            workflow="repair",
            asr="faster-whisper",
            vad="wave-energy",
            strategy_agent="qwen-local",
            curator_agent="qwen-local",
            report_agent="qwen-local",
            qwen_model_path="models/qwen",
            metadata_path="input/metadata.csv",
            metadata_audio_column="file",
            metadata_text_column="sentence",
        )

        config = request.to_run_config("tools/ffmpeg.exe")

        self.assertEqual(config["target_policy"], "tts")
        self.assertEqual(config["workflow"], "repair")
        self.assertEqual(config["asr"], "faster-whisper")
        self.assertEqual(config["tools"]["ffmpeg_path"], "tools/ffmpeg.exe")
        self.assertEqual(config["models"]["qwen"], "models/qwen")
        self.assertEqual(config["metadata"]["path"], "input/metadata.csv")
        self.assertEqual(config["metadata"]["audio_column"], "file")
        self.assertEqual(config["metadata"]["text_column"], "sentence")

    def test_batch_acoustic_consistency_is_disabled_by_default(self):
        from audio_dataset_agent.web.jobs import RunRequest

        request = RunRequest(input_dir="input")
        config = request.to_run_config("ffmpeg")

        self.assertFalse(config["speaker_consistency_check"])
        self.assertEqual(request.curator_confidence_threshold, 0.8)

    def test_run_request_rejects_unknown_repair_chain_mode(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            manager = AudioDatasetJobManager(root / "runs")
            with self.assertRaisesRegex(ValueError, "修复链模式"):
                manager.run_sync(RunRequest(input_dir=str(input_dir), repair_chain="unsafe"))

    def test_run_request_serializes_local_rag_options(self):
        from audio_dataset_agent.web.jobs import RunRequest

        request = RunRequest(
            input_dir="input",
            rag_mode="local",
            rag_documents="knowledge_base/documents",
            rag_index="local_tools/cache/rag/index.sqlite",
        )

        config = request.to_run_config("ffmpeg")

        self.assertEqual(config["rag"]["mode"], "local")
        self.assertEqual(config["rag"]["documents"], "knowledge_base/documents")
        self.assertEqual(config["rag"]["index"], "local_tools/cache/rag/index.sqlite")

    def test_run_request_serializes_rag_rebuild_option(self):
        from audio_dataset_agent.web.jobs import RunRequest

        config = RunRequest(input_dir="input", rag_mode="local", rag_rebuild=True).to_run_config("ffmpeg")

        self.assertTrue(config["rag"]["rebuild"])

    def test_job_progress_is_serialized_during_run(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest

        def fake_runner(request, work_dir, export_dir):
            batch_dir = export_dir / "batch_fake"
            batch_dir.mkdir(parents=True)
            (batch_dir / "dataset_quality_report.json").write_text(
                json.dumps(
                    {
                        "batch_id": "batch_fake",
                        "target_policy": request.target_policy,
                        "total_samples": 0,
                        "usable_samples": 0,
                        "review_samples": 0,
                        "rejected_samples": 0,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (batch_dir / "manifest.jsonl").write_text("", encoding="utf-8")
            return batch_dir

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            manager = AudioDatasetJobManager(root / "runs", runner=fake_runner)
            job = manager.run_sync(
                RunRequest(input_dir=str(input_dir), target_policy="voice_clone", asr="none")
            )

        payload = job.to_dict()
        self.assertEqual(payload["progress"]["current_stage"], "succeeded")
        self.assertEqual(payload["progress"]["stage_message"], "处理完成")
        self.assertEqual(payload["progress"]["percent"], 100)

    def test_cancel_job_marks_queued_job_as_cancelled(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            manager = AudioDatasetJobManager(root / "runs")
            job = manager._create_job(
                RunRequest(input_dir=str(input_dir), target_policy="voice_clone", asr="none")
            )

            cancelled = manager.cancel_job(job.job_id)

        self.assertEqual(cancelled.status, "cancelled")
        self.assertTrue(cancelled.cancel_requested)
        self.assertEqual(cancelled.progress["current_stage"], "cancelled")

    def test_apply_manual_review_override_writes_override_and_corrected_manifest(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJob, RunRequest
        from audio_dataset_agent.web.local_server import apply_manual_review_override

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "exports" / "batch_demo"
            batch_dir.mkdir(parents=True)
            manifest_path = batch_dir / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample_review",
                        "text": "",
                        "decision": "review",
                        "decision_reason": "needs review",
                        "final_status": "needs_review",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            job = AudioDatasetJob(
                job_id="job_demo",
                request=RunRequest(input_dir=str(root)),
                export_dir=str(root / "exports"),
                batch_export_dir=str(batch_dir),
                summary={"samples": [{"sample_id": "sample_review"}]},
            )

            result = apply_manual_review_override(
                job,
                {
                    "sample_id": "sample_review",
                    "decision": "keep",
                    "note": "人工确认可用",
                    "transcript": "人工修订文本",
                },
            )
            self.assertEqual(result["sample_id"], "sample_review")
            self.assertTrue((batch_dir / "manual_review_overrides.json").exists())
            corrected = [
                json.loads(line)
                for line in (batch_dir / "manual_review_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(corrected[0]["decision"], "keep")
            self.assertEqual(corrected[0]["text"], "人工修订文本")
            self.assertEqual(corrected[0]["reference_text"], "人工修订文本")
            self.assertEqual(corrected[0]["reference_text_source"], "manual_review")
            self.assertEqual(corrected[0]["metadata_status"], "matched")
            self.assertEqual(corrected[0]["manual_review"]["note"], "人工确认可用")

    def test_issue_only_llm_wrapper_skips_clean_samples(self):
        from audio_dataset_agent.core.models import AnalysisResult
        from audio_dataset_agent.web.jobs import IssueOnlyAgentWrapper

        class FakeLLMAgent:
            def __init__(self):
                self.calls = 0

            def decide(self, result):
                self.calls += 1
                return "llm"

        class FakeBaselineAgent:
            def __init__(self):
                self.calls = 0

            def decide(self, result):
                self.calls += 1
                return "baseline"

        llm = FakeLLMAgent()
        baseline = FakeBaselineAgent()
        wrapper = IssueOnlyAgentWrapper(llm, baseline)

        clean = AnalysisResult("clean", 2.0, 0, None, [], {})
        noisy = AnalysisResult("noisy", 2.0, 0, None, ["audio_noise_high"], {})

        self.assertEqual(wrapper.decide(clean), "baseline")
        self.assertEqual(wrapper.decide(noisy), "llm")
        self.assertEqual(llm.calls, 1)
        self.assertEqual(baseline.calls, 1)

    def test_run_job_records_success_and_summary(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest

        def fake_runner(request, work_dir, export_dir):
            batch_dir = export_dir / "batch_fake"
            batch_dir.mkdir(parents=True)
            (batch_dir / "dataset_quality_report.json").write_text(
                json.dumps(
                    {
                        "batch_id": "batch_fake",
                        "target_policy": request.target_policy,
                        "total_samples": 0,
                        "usable_samples": 0,
                        "review_samples": 0,
                        "rejected_samples": 0,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (batch_dir / "manifest.jsonl").write_text("", encoding="utf-8")
            return batch_dir

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            manager = AudioDatasetJobManager(root / "runs", runner=fake_runner)

            job = manager.run_sync(
                RunRequest(
                    input_dir=str(input_dir),
                    target_policy="voice_clone",
                    workflow="basic",
                    asr="none",
                )
            )
            loaded = manager.get_job(job.job_id)

        self.assertEqual(job.status, "succeeded")
        self.assertEqual(loaded.status, "succeeded")
        self.assertEqual(job.summary["batch"]["batch_id"], "batch_fake")
        self.assertIsNotNone(job.export_dir)


    def test_run_request_serializes_local_llm_agent_options(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest

        def fake_runner(request, work_dir, export_dir):
            batch_dir = export_dir / "batch_fake"
            batch_dir.mkdir(parents=True)
            (batch_dir / "dataset_quality_report.json").write_text(
                json.dumps(
                    {
                        "batch_id": "batch_fake",
                        "target_policy": request.target_policy,
                        "total_samples": 0,
                        "usable_samples": 0,
                        "review_samples": 0,
                        "rejected_samples": 0,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (batch_dir / "manifest.jsonl").write_text("", encoding="utf-8")
            return batch_dir

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            manager = AudioDatasetJobManager(root / "runs", runner=fake_runner)

            job = manager.run_sync(
                RunRequest(
                    input_dir=str(input_dir),
                    target_policy="voice_clone",
                    workflow="basic",
                    asr="none",
                    strategy_agent="qwen-local",
                    curator_agent="qwen-local",
                    report_agent="qwen-local",
                    qwen_model_path="local_tools/models/qwen/Qwen3-4B-Instruct-2507",
                    qwen_max_new_tokens=192,
                )
            )

        request_payload = job.to_dict()["request"]
        self.assertEqual(request_payload["strategy_agent"], "qwen-local")
        self.assertEqual(request_payload["curator_agent"], "qwen-local")
        self.assertEqual(request_payload["report_agent"], "qwen-local")
        self.assertEqual(
            request_payload["qwen_model_path"],
            "local_tools/models/qwen/Qwen3-4B-Instruct-2507",
        )
        self.assertEqual(request_payload["qwen_max_new_tokens"], 192)

    def test_run_job_rejects_missing_input_dir(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest

        with TemporaryDirectory() as tmp:
            manager = AudioDatasetJobManager(Path(tmp) / "runs")
            with self.assertRaises(ValueError) as raised:
                manager.run_sync(
                    RunRequest(
                        input_dir=str(Path(tmp) / "missing"),
                        target_policy="voice_clone",
                    )
                )

        self.assertIn("missing", str(raised.exception))




class TestLocalServerJsonSerialization(unittest.TestCase):
    def test_send_json_replaces_non_finite_numbers_with_null(self):
        from audio_dataset_agent.web.local_server import dumps_strict_json

        payload = {
            "status": "success",
            "data": {
                "normal": 1.25,
                "bad_values": [float("inf"), float("-inf"), float("nan")],
            },
        }

        encoded = dumps_strict_json(payload)
        decoded = json.loads(encoded)

        self.assertNotIn("Infinity", encoded)
        self.assertNotIn("NaN", encoded)
        self.assertEqual(decoded["data"]["normal"], 1.25)
        self.assertEqual(decoded["data"]["bad_values"], [None, None, None])

    def test_frontend_route_falls_back_to_index_without_escaping_root(self):
        from audio_dataset_agent.web.local_server import resolve_frontend_path

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "dist"
            (root / "assets").mkdir(parents=True)
            (root / "index.html").write_text("index", encoding="utf-8")
            (root / "assets" / "app.js").write_text("app", encoding="utf-8")

            self.assertEqual(resolve_frontend_path(root, "/pipeline"), (root / "index.html").resolve())
            self.assertEqual(resolve_frontend_path(root, "/assets/app.js"), (root / "assets" / "app.js").resolve())

            outside = Path(tmp) / "outside.js"
            outside.write_text("outside", encoding="utf-8")
            self.assertEqual(resolve_frontend_path(root, "/../outside.js"), (root / "index.html").resolve())

class TestWebAssetResolution(unittest.TestCase):
    def test_resolve_sample_audio_path_allows_only_job_artifacts(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJob, RunRequest
        from audio_dataset_agent.web.local_server import resolve_sample_audio_path

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_dir = root / "work"
            sample_path = work_dir / "samples" / "sample_keep.wav"
            sample_path.parent.mkdir(parents=True)
            sample_path.write_bytes(b"RIFFdemo")
            outside_path = root / "outside.wav"
            outside_path.write_bytes(b"RIFFbad")

            job = AudioDatasetJob(
                job_id="job_demo",
                request=RunRequest(input_dir=str(root)),
                work_dir=str(work_dir),
                export_dir=str(root / "exports"),
                batch_export_dir=str(root / "exports" / "batch_demo"),
                summary={
                    "samples": [
                        {
                            "sample_id": "sample_keep",
                            "final_audio_path": str(sample_path),
                        },
                        {
                            "sample_id": "sample_outside",
                            "final_audio_path": str(outside_path),
                        },
                    ]
                },
            )

            self.assertEqual(resolve_sample_audio_path(job, "sample_keep"), sample_path.resolve())
            with self.assertRaises(ValueError):
                resolve_sample_audio_path(job, "sample_outside")
            with self.assertRaises(ValueError):
                resolve_sample_audio_path(job, "missing")

    def test_resolve_report_path_uses_known_report_names(self):
        from audio_dataset_agent.web.jobs import AudioDatasetJob, RunRequest
        from audio_dataset_agent.web.local_server import resolve_report_path

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "exports" / "batch_demo"
            batch_dir.mkdir(parents=True)
            report_path = batch_dir / "portfolio_report.html"
            report_path.write_text("<html>ok</html>", encoding="utf-8")
            outside_path = root / "outside.html"
            outside_path.write_text("bad", encoding="utf-8")

            job = AudioDatasetJob(
                job_id="job_demo",
                request=RunRequest(input_dir=str(root)),
                work_dir=str(root / "work"),
                export_dir=str(root / "exports"),
                batch_export_dir=str(batch_dir),
                summary={
                    "report_paths": {
                        "portfolio_report_html": str(report_path),
                        "bad_report": str(outside_path),
                    }
                },
            )

            self.assertEqual(resolve_report_path(job, "portfolio_report_html"), report_path.resolve())
            with self.assertRaises(ValueError):
                resolve_report_path(job, "bad_report")
            with self.assertRaises(ValueError):
                resolve_report_path(job, "unknown")


if __name__ == "__main__":
    unittest.main()

