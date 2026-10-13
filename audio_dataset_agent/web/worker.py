"""本地持久化任务 Worker。"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from pathlib import Path

from audio_dataset_agent.storage.sqlite import SQLiteRunStore
from audio_dataset_agent.web.jobs import AudioDatasetJobManager


class LocalWorker:
    """从 SQLite 领取任务并执行已有 JobManager pipeline。"""

    def __init__(self, runs_root: Path | str, worker_id: str | None = None, store=None, runner=None):
        self.runs_root = Path(runs_root)
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.store = store or SQLiteRunStore(self.runs_root / "harness.sqlite3")
        self.manager = AudioDatasetJobManager(self.runs_root, store=self.store, runner=runner, queue_mode=True)

    def _lease_heartbeat(self, task_id: str, stop: threading.Event) -> None:
        while not stop.wait(30):
            if not self.store.renew_task_lease(task_id, self.worker_id, lease_seconds=120):
                return

    def run_once(self) -> bool:
        task = self.store.claim_task(self.worker_id, lease_seconds=120)
        if task is None:
            return False
        self.store.append_event(
            task["job_id"], "task_claimed", task["stage"], "running",
            {"task_id": task["task_id"], "worker_id": self.worker_id, "attempts": task["attempts"]},
        )
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._lease_heartbeat,
            args=(task["task_id"], heartbeat_stop),
            daemon=True,
        )
        heartbeat.start()
        try:
            job = self.manager.get_job(task["job_id"])
            self.store.append_event(task["job_id"], "worker_started", task["stage"], "running", {"worker_id": self.worker_id})
            succeeded = self.manager._run_job(job.job_id)
            if succeeded:
                self.store.complete_task(task["task_id"], worker_id=self.worker_id)
                self.store.append_event(task["job_id"], "task_succeeded", task["stage"], "succeeded", {"task_id": task["task_id"]})
            else:
                failed = self.store.fail_task(task["task_id"], "pipeline 返回失败", worker_id=self.worker_id)
                if failed and failed["status"] == "queued":
                    self.manager._update_job(
                        task["job_id"],
                        status="queued",
                        error_message="本次执行失败，等待自动重试",
                        progress={"current_stage": "queued", "stage_message": "等待任务重试", "processed_samples": 0, "total_samples": 0, "percent": 0},
                    )
                self.store.append_event(task["job_id"], "task_failed", task["stage"], failed["status"] if failed else "failed", {"task_id": task["task_id"], "retrying": bool(failed and failed["status"] == "queued")})
        except Exception as exc:
            failed = self.store.fail_task(task["task_id"], str(exc), worker_id=self.worker_id)
            if failed and failed["status"] == "queued":
                self.manager._update_job(
                    task["job_id"],
                    status="queued",
                    error_message=str(exc),
                    progress={"current_stage": "queued", "stage_message": "等待任务重试", "processed_samples": 0, "total_samples": 0, "percent": 0},
                )
            self.store.append_event(task["job_id"], "worker_failed", task["stage"], failed["status"] if failed else "failed", {"task_id": task["task_id"], "error": str(exc), "retrying": bool(failed and failed["status"] == "queued")})
        finally:
            heartbeat_stop.set()
        return True

    def run_forever(self, poll_seconds: float = 1.0) -> None:
        while True:
            if not self.run_once():
                time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动音频数据处理 Worker。")
    parser.add_argument("--runs-root", default="local_tools/cache/audio_web_runs")
    parser.add_argument("--worker-id")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    worker = LocalWorker(args.runs_root, args.worker_id)
    if args.once:
        worker.run_once()
    else:
        worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
