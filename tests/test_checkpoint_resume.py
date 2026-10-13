import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_dataset_agent.harness.runtime import HarnessRuntime
from audio_dataset_agent.storage.sqlite import SQLiteRunStore


class TestCheckpointResume(unittest.TestCase):
    def test_checkpoint_keeps_structured_state_and_reads_after_restart(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = HarnessRuntime(SQLiteRunStore(root / "runs.sqlite3"))
            path = root / "run" / "checkpoint.json"
            runtime.checkpoint("run-1", path, "normalize", "running", state={"next_stage": "vad", "sample_count": 4})
            loaded = HarnessRuntime.load_checkpoint(path)
            self.assertEqual(loaded["state"]["next_stage"], "vad")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["state"]["sample_count"], 4)


if __name__ == "__main__":
    unittest.main()
