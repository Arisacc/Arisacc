import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_dataset_agent.web.package_export import create_export_package


class TestExportPackage(unittest.TestCase):
    def test_package_contains_manifest_reports_and_audio_groups(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            (root / "manual_groups" / "keep").mkdir(parents=True)
            (root / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "dataset_quality_report.json").write_text("{}", encoding="utf-8")
            (root / "manual_groups" / "keep" / "sample.wav").write_bytes(b"wav")
            (root / "manual_groups" / "reject").mkdir(parents=True)
            (root / "manual_groups" / "reject" / "rejected.wav").write_bytes(b"wav")
            package = create_export_package(root)
            with zipfile.ZipFile(package) as archive:
                names = set(archive.namelist())
            self.assertIn("manual_groups/keep/sample.wav", names)
            self.assertNotIn("manual_groups/reject/rejected.wav", names)
            self.assertEqual(names, {"manual_groups/keep/sample.wav"})


if __name__ == "__main__":
    unittest.main()
