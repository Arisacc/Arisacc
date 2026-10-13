import json
import math
import struct
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_dataset_agent.storage.sqlite import SQLiteRunStore
from audio_dataset_agent.web.jobs import AudioDatasetJobManager, RunRequest
from audio_dataset_agent.web.worker import LocalWorker


def write_wav(path: Path, seconds: float = 1.2, sample_rate: int = 16000) -> None:
    frames = bytearray()
    for index in range(int(seconds * sample_rate)):
        value = int(0.22 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(frames)


class TestDataFlowContract(unittest.TestCase):
    def test_voice_clone_queue_flow_exports_traceable_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            write_wav(source / "speaker_001.wav")
            store = SQLiteRunStore(root / "harness.sqlite3")
            manager = AudioDatasetJobManager(root / "runs", store=store, queue_mode=True)
            job = manager.start_job(
                RunRequest(
                    input_dir=str(source),
                    target_policy="voice_clone",
                    workflow="basic",
                    asr="none",
                    vad="whole-file",
                )
            )

            worker = LocalWorker(root / "runs", worker_id="e2e-worker", store=store)
            self.assertTrue(worker.run_once())

            completed = manager.get_job(job.job_id)
            self.assertEqual(completed.status, "succeeded")
            export_dir = Path(completed.batch_export_dir)
            manifest = export_dir / "manifest.jsonl"
            report = export_dir / "dataset_quality_report.json"
            checkpoint = root / "runs" / job.job_id / "checkpoint.json"
            self.assertTrue(manifest.exists())
            self.assertTrue(report.exists())
            self.assertTrue(checkpoint.exists())

            records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(len(records), 1)
            self.assertTrue(all(record.get("sample_id") for record in records))
            self.assertTrue(all(record.get("audio_path") or record.get("final_audio_path") for record in records), records)
            report_payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(report_payload["total_samples"], len(records))
            self.assertIn("qualityScore", report_payload)
            events = store.list_events(job.job_id)
            self.assertGreater(len(events), 3)
            self.assertTrue(any(event["event_type"] == "stage_progress" for event in events))


if __name__ == "__main__":
    unittest.main()
