import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_dataset_agent.web.summary import build_summary_from_export_dir


class TestManualReviewSummary(unittest.TestCase):
    def test_summary_prefers_manual_manifest_after_review(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "state").mkdir()
            base = {"sample_id": "sample_1", "duration": 1.0, "final_status": "needs_review", "decision": "review", "audio_path": "old.wav"}
            corrected = {**base, "final_status": "manual_keep", "decision": "keep", "final_audio_path": "manual_groups/keep/sample_1.wav", "manual_archive_group": "keep"}
            (root / "manifest.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")
            (root / "manual_review_manifest.jsonl").write_text(json.dumps(corrected) + "\n", encoding="utf-8")
            (root / "dataset_quality_report.json").write_text(json.dumps({"target_policy": "voice_clone"}), encoding="utf-8")
            (root / "state" / "raw_assets.json").write_text("[]", encoding="utf-8")
            (root / "state" / "processing_strategies.json").write_text("[]", encoding="utf-8")
            (root / "state" / "curator_decisions.json").write_text("[]", encoding="utf-8")
            summary = build_summary_from_export_dir(root)
            self.assertEqual(summary["samples"][0]["final_status"], "manual_keep")
            self.assertEqual(summary["samples"][0]["manual_archive_group"], "keep")
            self.assertEqual(summary["batch"]["usable_samples"], 1)
            self.assertEqual(summary["batch"]["review_samples"], 0)


if __name__ == "__main__":
    unittest.main()
