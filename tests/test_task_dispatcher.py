import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_dataset_agent.storage.sqlite import SQLiteRunStore


class TestTaskDispatcher(unittest.TestCase):
    def test_task_is_claimed_once_and_expired_lease_can_be_reclaimed(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteRunStore(Path(tmp) / "run.sqlite3")
            store.create_task("run-1", "pipeline", {"stage": "import"})

            first = store.claim_task("worker-a", lease_seconds=30)
            second = store.claim_task("worker-b", lease_seconds=30)

            self.assertIsNotNone(first)
            self.assertIsNone(second)
            store.expire_task_lease(first["task_id"])
            reclaimed = store.claim_task("worker-b", lease_seconds=30)
            self.assertEqual(reclaimed["task_id"], first["task_id"])

    def test_completed_task_is_not_claimed_again(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteRunStore(Path(tmp) / "run.sqlite3")
            task_id = store.create_task("run-1", "pipeline", {})
            task = store.claim_task("worker-a", lease_seconds=30)
            store.complete_task(task_id)
            self.assertIsNone(store.claim_task("worker-b", lease_seconds=30))

    def test_task_lease_owner_controls_completion_and_renewal(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteRunStore(Path(tmp) / "run.sqlite3")
            task_id = store.create_task("run-1", "pipeline", {})
            task = store.claim_task("worker-a", lease_seconds=1)
            self.assertFalse(store.renew_task_lease(task_id, "worker-b", lease_seconds=30))
            self.assertTrue(store.renew_task_lease(task_id, "worker-a", lease_seconds=30))
            self.assertFalse(store.complete_task(task_id, worker_id="worker-b"))
            self.assertTrue(store.complete_task(task_id, worker_id="worker-a"))

    def test_failed_task_retries_until_attempt_limit(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteRunStore(Path(tmp) / "run.sqlite3")
            task_id = store.create_task("run-1", "pipeline", {})
            for attempt in range(1, 4):
                claimed = store.claim_task(f"worker-{attempt}", lease_seconds=30)
                task = store.fail_task(task_id, "transient failure", max_attempts=3, worker_id=claimed["lease_owner"])
                self.assertEqual(task["attempts"], attempt)
                self.assertEqual(task["status"], "queued" if attempt < 3 else "failed")


if __name__ == "__main__":
    unittest.main()
