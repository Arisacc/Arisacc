"""Run 状态机和状态迁移约束。"""

from __future__ import annotations

from typing import Literal


RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"queued", "running", "cancelled", "failed"},
    "running": {"running", "succeeded", "failed", "cancelled"},
    "succeeded": {"succeeded"},
    # 任务失败后只有 Harness 的重试路径可以回到 queued，普通调用不能复活终态。
    "failed": {"failed", "queued"},
    "cancelled": {"cancelled"},
}


def validate_transition(previous: str, current: str) -> None:
    """阻止 Run 跳过必要阶段或从终态重新运行。"""
    allowed = _ALLOWED_TRANSITIONS.get(previous, set())
    if current not in allowed:
        raise ValueError(f"非法 Run 状态迁移：{previous} -> {current}")
