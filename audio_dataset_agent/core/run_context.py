"""一次批处理运行的目录和配置上下文。"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from audio_dataset_agent.core.models import Batch


@dataclass(frozen=True)
class RunContext:
    """集中生成工作目录，避免各 service 自行拼接路径。"""

    batch_id: str
    input_dir: Path
    output_dir: Path
    target_policy: str
    run_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_batch(cls, batch: "Batch") -> "RunContext":
        return cls(
            batch_id=batch.batch_id,
            input_dir=batch.input_dir,
            output_dir=batch.output_dir,
            target_policy=batch.target_policy,
            run_config=dict(batch.run_config),
        )

    @property
    def normalized_dir(self) -> Path:
        return self.output_dir / "work" / self.batch_id

    @property
    def samples_dir(self) -> Path:
        return self.output_dir / "samples" / self.batch_id

    @property
    def repairs_dir(self) -> Path:
        return self.output_dir / "repairs" / self.batch_id

    def batch_export_dir(self, export_root: Path | str) -> Path:
        return Path(export_root) / self.batch_id
