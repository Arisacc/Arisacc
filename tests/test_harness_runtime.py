import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_dataset_agent.harness.runtime import HarnessRuntime
from audio_dataset_agent.harness.state_machine import validate_transition
from audio_dataset_agent.storage.sqlite import SQLiteRunStore


class TestHarnessRuntime(unittest.TestCase):
    def test_sqlite_events_artifacts_and_checkpoint_are_persisted(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteRunStore(root / "harness.sqlite3")
            runtime = HarnessRuntime(store)
            runtime.create_run("run-1", {"target_policy": "voice_clone"})
            runtime.transition("run-1", "queued", "running", "pipeline", "开始处理")
            artifact_path = root / "manifest.jsonl"
            artifact_path.write_text("{}\n", encoding="utf-8")
            runtime.register_artifacts(
                "run-1",
                [{"name": "manifest", "path": str(artifact_path), "kind": "manifest"}],
            )
            checkpoint = runtime.checkpoint(
                "run-1",
                root / "checkpoint.json",
                "export",
                "running",
                store.list_artifacts("run-1"),
            )

            self.assertTrue(checkpoint.exists())
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(payload["stage"], "export")
            self.assertEqual(len(store.list_events("run-1")), 3)
            self.assertEqual(store.list_artifacts("run-1")[0]["name"], "manifest")

            reopened = SQLiteRunStore(root / "harness.sqlite3")
            self.assertEqual(len(reopened.list_events("run-1")), 3)

    def test_state_machine_rejects_terminal_to_running_transition(self):
        with self.assertRaises(ValueError):
            validate_transition("succeeded", "running")


if __name__ == "__main__":
    unittest.main()
