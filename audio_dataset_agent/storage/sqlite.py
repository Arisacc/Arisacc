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
