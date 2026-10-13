"""基于 SQLite 的 Run、事件和产物索引。

SQLite 只保存任务元数据和路径，不复制音频内容；音频和报告仍位于当前 Run 的工作目录。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRunStore:
    """线程安全的 SQLite 仓储，后续可替换为 SQLAlchemy/PostgreSQL 实现。"""

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        """提交事务并显式关闭连接，避免 Windows 锁住 SQLite 文件。"""
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    work_dir TEXT,
                    export_dir TEXT,
                    batch_export_dir TEXT,
                    summary_json TEXT,
                    error_message TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    progress_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_run_events_job_id
                    ON run_events(job_id, event_id);
                CREATE TABLE IF NOT EXISTS run_tasks (
                    task_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_run_tasks_claim
                    ON run_tasks(status, lease_expires_at, created_at);
                CREATE TABLE IF NOT EXISTS run_artifacts (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_run_artifacts_job_id
                    ON run_artifacts(job_id, artifact_id);
                """
            )

    def create_task(self, job_id: str, stage: str, payload: dict[str, Any]) -> str:
        task_id = f"task_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        now = _now_iso()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO run_tasks (
                    task_id, job_id, stage, status, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?)
                """,
                (task_id, job_id, stage, json.dumps(payload, ensure_ascii=False), now, now),
            )
        return task_id

    def claim_task(self, worker_id: str, lease_seconds: int = 60) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        lease_expires = now.timestamp() + lease_seconds
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM run_tasks
                WHERE status = 'queued'
                   OR (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                ORDER BY created_at
                LIMIT 1
                """,
                (now.isoformat(),),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE run_tasks
                SET status='running', attempts=attempts+1, lease_owner=?,
                    lease_expires_at=?, updated_at=?
                WHERE task_id=?
                """,
                (
                    worker_id,
                    datetime.fromtimestamp(lease_expires, timezone.utc).isoformat(),
                    now.isoformat(),
                    row["task_id"],
                ),
            )
            return {
                "task_id": row["task_id"],
                "job_id": row["job_id"],
                "stage": row["stage"],
                "payload": json.loads(row["payload_json"]),
                "attempts": row["attempts"] + 1,
                "lease_owner": worker_id,
            }

    def expire_task_lease(self, task_id: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE run_tasks SET lease_expires_at=?, updated_at=? WHERE task_id=?",
                (datetime.fromtimestamp(0, timezone.utc).isoformat(), _now_iso(), task_id),
            )

    def complete_task(self, task_id: str, worker_id: str | None = None) -> bool:
        with self._lock, self._connection() as connection:
            where = "task_id=?"
            values: list[Any] = [task_id]
            if worker_id is not None:
                where += " AND status='running' AND lease_owner=?"
                values.append(worker_id)
            cursor = connection.execute(
                "UPDATE run_tasks SET status='succeeded', lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE " + where,
                [_now_iso(), *values],
            )
            return cursor.rowcount == 1

    def renew_task_lease(self, task_id: str, worker_id: str, lease_seconds: int = 60) -> bool:
        """Worker 继续执行长任务时延长租约，避免被其他 Worker 抢占。"""
        expires_at = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + lease_seconds,
            timezone.utc,
        ).isoformat()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE run_tasks
                SET lease_expires_at=?, updated_at=?
                WHERE task_id=? AND status='running' AND lease_owner=?
                """,
                (expires_at, _now_iso(), task_id, worker_id),
            )
            return cursor.rowcount == 1

    def fail_task(
        self,
        task_id: str,
        error: str,
        retry: bool = True,
        max_attempts: int = 3,
        worker_id: str | None = None,
    ) -> dict[str, Any] | None:
        """记录任务失败；未超过重试次数时回到 queued，否则进入 failed。"""
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT attempts, lease_owner, status FROM run_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            if worker_id is not None and (row["status"] != "running" or row["lease_owner"] != worker_id):
                return None
            next_status = "queued" if retry and row["attempts"] < max_attempts else "failed"
            connection.execute(
                """
                UPDATE run_tasks
                SET status=?, lease_owner=NULL, lease_expires_at=NULL,
                    payload_json=json_set(payload_json, '$.last_error', ?), updated_at=?
                WHERE task_id=?
                """,
                (next_status, error, _now_iso(), task_id),
            )
            updated = connection.execute(
                "SELECT * FROM run_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return self._task_row_to_dict(updated) if updated else None

    def get_task_for_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM run_tasks WHERE job_id=? ORDER BY created_at DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        return self._task_row_to_dict(row) if row else None

    def _task_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"],
            "job_id": row["job_id"],
            "stage": row["stage"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            "attempts": row["attempts"],
            "lease_owner": row["lease_owner"],
            "lease_expires_at": row["lease_expires_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_events_after(self, job_id: str, event_id: int = 0) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM run_events WHERE job_id=? AND event_id>? ORDER BY event_id",
                (job_id, event_id),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "job_id": row["job_id"],
                "event_type": row["event_type"],
                "stage": row["stage"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def upsert_run(self, snapshot: dict[str, Any]) -> None:
        now = snapshot.get("updated_at") or _now_iso()
        request = snapshot.get("request") or {}
        progress = snapshot.get("progress") or {}
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    job_id, status, created_at, updated_at, request_json,
                    work_dir, export_dir, batch_export_dir, summary_json,
                    error_message, cancel_requested, progress_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    work_dir=excluded.work_dir,
                    export_dir=excluded.export_dir,
                    batch_export_dir=excluded.batch_export_dir,
                    summary_json=excluded.summary_json,
                    error_message=excluded.error_message,
                    cancel_requested=excluded.cancel_requested,
                    progress_json=excluded.progress_json
                """,
                (
                    snapshot["job_id"],
                    snapshot["status"],
                    snapshot.get("created_at") or now,
                    now,
                    json.dumps(request, ensure_ascii=False),
                    snapshot.get("work_dir"),
                    snapshot.get("export_dir"),
                    snapshot.get("batch_export_dir"),
                    json.dumps(snapshot.get("summary"), ensure_ascii=False) if snapshot.get("summary") is not None else None,
                    snapshot.get("error_message"),
                    int(bool(snapshot.get("cancel_requested"))),
                    json.dumps(progress, ensure_ascii=False),
                ),
            )

    def append_event(
        self,
        job_id: str,
        event_type: str,
        stage: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO run_events (job_id, event_type, stage, status, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    event_type,
                    stage,
                    status,
                    json.dumps(payload or {}, ensure_ascii=False),
                    _now_iso(),
                ),
            )
            return int(cursor.lastrowid)

    def register_artifact(self, job_id: str, name: str, path: str, kind: str = "file") -> int:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO run_artifacts (job_id, name, path, kind, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, name, path, kind, _now_iso()),
            )
            return int(cursor.lastrowid)

    def get_run(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM runs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_events(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM run_events WHERE job_id = ? ORDER BY event_id",
                (job_id,),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "job_id": row["job_id"],
                "event_type": row["event_type"],
                "stage": row["stage"],
                "status": row["status"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM run_artifacts WHERE job_id = ? ORDER BY artifact_id",
                (job_id,),
            ).fetchall()
        return [
            {
                "artifact_id": row["artifact_id"],
                "job_id": row["job_id"],
                "name": row["name"],
                "path": row["path"],
                "kind": row["kind"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "request": json.loads(row["request_json"]),
            "work_dir": row["work_dir"],
            "export_dir": row["export_dir"],
            "batch_export_dir": row["batch_export_dir"],
            "summary": json.loads(row["summary_json"]) if row["summary_json"] else None,
            "error_message": row["error_message"],
            "cancel_requested": bool(row["cancel_requested"]),
            "progress": json.loads(row["progress_json"]),
        }
