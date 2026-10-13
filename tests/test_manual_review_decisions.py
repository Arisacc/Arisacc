import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from audio_dataset_agent.web.local_server import apply_manual_review_override


class TestManualReviewDecisions(unittest.TestCase):
    def test_manual_review_accepts_only_keep_or_reject(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.jsonl").write_text('{"sample_id":"s1"}\n', encoding="utf-8")
            job = SimpleNamespace(batch_export_dir=str(root), summary=None)
            with self.assertRaises(ValueError):
                apply_manual_review_override(job, {"sample_id": "s1", "decision": "repair"})


if __name__ == "__main__":
    unittest.main()
