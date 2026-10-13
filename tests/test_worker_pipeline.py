import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_dataset_agent.storage.sqlite import SQLiteRunStore
from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest
from audio_dataset_agent.web.worker import LocalWorker


class TestWorkerPipeline(unittest.TestCase):
    def test_worker_completes_queued_run_and_records_lifecycle(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "audio"
            input_dir.mkdir()
            export_dir = root / "export"

            def runner(request, work_dir, requested_export_dir):
                requested_export_dir.mkdir(parents=True, exist_ok=True)
                (requested_export_dir / "manifest.jsonl").write_text(
                    '{"sample_id":"sample-1"}\n', encoding="utf-8"
                )
                (requested_export_dir / "dataset_quality_report.json").write_text(
                    '{"qualityScore":0.8,"samples":[]}', encoding="utf-8"
                )
                return requested_export_dir

            store = SQLiteRunStore(root / "harness.sqlite3")
            manager = AudioDatasetJobManager(root / "runs", runner=runner, store=store, queue_mode=True)
            job = manager.start_job(RunRequest(input_dir=str(input_dir), target_policy="voice_clone"))

            worker = LocalWorker(root / "runs", worker_id="test-worker", store=store, runner=runner)
            self.assertTrue(worker.run_once())

            completed = manager.get_job(job.job_id)
            self.assertEqual(completed.status, "succeeded")
            task = store.get_task_for_job(job.job_id)
            self.assertEqual(task["status"], "succeeded")
            event_types = [event["event_type"] for event in store.list_events(job.job_id)]
            self.assertIn("task_claimed", event_types)
            self.assertIn("task_succeeded", event_types)
            self.assertTrue((Path(completed.batch_export_dir) / "manifest.jsonl").exists())

    def test_worker_requeues_failed_task_and_records_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "audio"
            input_dir.mkdir()

            def runner(request, work_dir, export_dir):
                raise RuntimeError("测试执行失败")

            store = SQLiteRunStore(root / "harness.sqlite3")
            manager = AudioDatasetJobManager(root / "runs", runner=runner, store=store, queue_mode=True)
            job = manager.start_job(RunRequest(input_dir=str(input_dir), target_policy="voice_clone"))
            worker = LocalWorker(root / "runs", worker_id="test-worker", store=store, runner=runner)

            self.assertTrue(worker.run_once())
            task = store.get_task_for_job(job.job_id)
            self.assertEqual(task["status"], "queued")
            self.assertEqual(task["attempts"], 1)
            self.assertEqual(manager.get_job(job.job_id).status, "queued")
            self.assertTrue(any(event["event_type"] == "task_failed" for event in store.list_events(job.job_id)))


if __name__ == "__main__":
    unittest.main()
