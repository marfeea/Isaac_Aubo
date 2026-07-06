from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskOutputPaths:
    """同一任务的日志与 checkpoint 输出位置。"""

    log_dir: Path
    checkpoint_dir: Path

    @classmethod
    def for_task(cls, task_name: str, root: Path | str = ".") -> "TaskOutputPaths":
        if not task_name or any(char in task_name for char in ("/", "\\")):
            raise ValueError("task_name 必须是非空的单级目录名。")
        root_path = Path(root)
        return cls(
            log_dir=root_path / "logs" / task_name / "sb3_aubo",
            checkpoint_dir=root_path / "checkpoints" / task_name / "sb3_aubo",
        )

    def ensure_exists(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def checkpoint_stem(task_name: str, run_label: str | None = None) -> str:
    """生成不会越出任务 checkpoint 目录的文件名。"""
    if run_label is None:
        return f"ppo_{task_name}_final"
    if not run_label or not all(char.isalnum() or char in ("-", "_") for char in run_label):
        raise ValueError("run_label 只能包含字母、数字、连字符和下划线。")
    return f"ppo_{task_name}_{run_label}"
