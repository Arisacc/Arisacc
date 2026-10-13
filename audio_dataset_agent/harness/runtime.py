"""Harness Runtime：把任务生命周期、事件和检查点统一起来。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audio_dataset_agent.harness.state_machine import validate_transition
from audio_dataset_agent.storage.sqlite import SQLiteRunStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HarnessRuntime:
    """为一次 Run 提供持久化事件、状态迁移和 checkpoint。"""

    def __init__(self, store: SQLiteRunStore):
        self.store = store

    def create_run(self, job_id: str, request: dict[str, Any], status: str = "queued") -> None:
        self.store.append_event(
            job_id,
            event_type="run_created",
            stage="created",
            status=status,
            payload={"request": request},
        )

    def transition(
        self,
        job_id: str,
        previous: str,
        current: str,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        validate_transition(previous, current)
        event_payload = {"message": message, **(payload or {})}
        self.store.append_event(
            job_id,
            event_type="status_changed",
            stage=stage,
            status=current,
            payload=event_payload,
        )

    def checkpoint(
        self,
        job_id: str,
        checkpoint_path: Path,
        stage: str,
        status: str,
        artifacts: list[dict[str, Any]] | None = None,
        state: dict[str, Any] | None = None,
    ) -> Path:
        """用原子替换写入 checkpoint，避免进程中断时留下半个 JSON。"""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        previous_state = self.load_checkpoint(checkpoint_path)
        payload = {
            "job_id": job_id,
            "stage": stage,
            "status": status,
            "artifacts": artifacts or [],
            "updated_at": _now_iso(),
            "state": state if state is not None else (previous_state or {}).get("state"),
        }
        temporary_path = checkpoint_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(checkpoint_path)
        self.store.append_event(
            job_id,
            event_type="checkpoint_saved",
            stage=stage,
            status=status,
            payload={"checkpoint_path": str(checkpoint_path), "artifact_count": len(payload["artifacts"])},
        )
        return checkpoint_path

    @staticmethod
    def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
        """读取完整 checkpoint；损坏的快照视为不存在，由调用方从头运行。"""
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def register_artifacts(self, job_id: str, artifacts: list[dict[str, Any]]) -> None:
        for artifact in artifacts:
            self.store.register_artifact(
                job_id,
                name=str(artifact.get("name", "artifact")),
                path=str(artifact.get("path", "")),
                kind=str(artifact.get("kind", "file")),
            )
