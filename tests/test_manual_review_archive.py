import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from audio_dataset_agent.web.local_server import apply_manual_review_override


class TestManualReviewArchive(unittest.TestCase):
    def test_manual_keep_copies_audio_into_keep_group_and_updates_manifest(self):
        with TemporaryDirectory() as tmp:
            batch_dir = Path(tmp) / "batch"
            audio_dir = batch_dir / "samples"
            audio_dir.mkdir(parents=True)
            audio_path = audio_dir / "sample_0001.wav"
            audio_path.write_bytes(b"wav-content")
            record = {
                "sample_id": "sample_0001",
                "audio_path": str(audio_path),
                "final_audio_path": str(audio_path),
                "final_status": "needs_review",
                "decision": "repair",
            }
            (batch_dir / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            job = SimpleNamespace(batch_export_dir=str(batch_dir), summary=None)

            result = apply_manual_review_override(job, {
                "sample_id": "sample_0001",
                "decision": "keep",
                "transcript": "",
                "note": "人工试听后保留",
            })

            archived = batch_dir / "manual_groups" / "keep" / "sample_0001.wav"
            self.assertTrue(archived.exists())
            self.assertEqual(archived.read_bytes(), b"wav-content")
            corrected = [json.loads(line) for line in Path(result["manual_manifest_path"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(corrected[0]["final_status"], "manual_keep")
            self.assertEqual(Path(corrected[0]["final_audio_path"]), archived)
            self.assertEqual(result["archive_group"], "keep")
            queue = json.loads((batch_dir / "review_queue.json").read_text(encoding="utf-8"))
            self.assertEqual(queue, [])

    def test_changing_manual_decision_moves_archive_copy_without_touching_source(self):
        with TemporaryDirectory() as tmp:
            batch_dir = Path(tmp) / "batch"
            source_dir = batch_dir / "samples"
            source_dir.mkdir(parents=True)
            source = source_dir / "sample_0002.wav"
            source.write_bytes(b"original")
            (batch_dir / "manifest.jsonl").write_text(json.dumps({
                "sample_id": "sample_0002", "source_audio_path": str(source),
                "audio_path": str(source), "final_status": "needs_review",
            }) + "\n", encoding="utf-8")
            job = SimpleNamespace(batch_export_dir=str(batch_dir), summary=None)
            apply_manual_review_override(job, {"sample_id": "sample_0002", "decision": "keep"})

            apply_manual_review_override(job, {"sample_id": "sample_0002", "decision": "reject"})

            self.assertFalse((batch_dir / "manual_groups" / "keep" / source.name).exists())
            self.assertTrue((batch_dir / "manual_groups" / "reject" / source.name).exists())
            self.assertEqual(source.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
