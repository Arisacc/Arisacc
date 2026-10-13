# 样本落盘工具：把候选时间段裁剪成实际 wav 文件。
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Protocol

from audio_dataset_agent.core.models import CandidateSample


CommandRunner = Callable[[list[str]], None]


def run_subprocess(command: list[str]) -> None:
    """执行 FFmpeg 命令；失败时让调用方看到明确异常。"""
    subprocess.run(command, check=True, capture_output=True, text=True)


class SampleMaterializer(Protocol):
    # SampleMaterializer 负责把候选片段元数据变成可被 ASR 读取的真实音频文件。
    def materialize(self, sample: CandidateSample, samples_dir: Path) -> Path:
        """为候选样本创建实际音频文件。"""


class CopySampleMaterializer:
    """复制标准化工作音频的开发用样本落盘器。"""

    def materialize(self, sample: CandidateSample, samples_dir: Path) -> Path:
        samples_dir.mkdir(parents=True, exist_ok=True)
        destination = samples_dir / f"{sample.sample_id}.wav"
        shutil.copy2(sample.source_path, destination)
        return destination


class FFmpegSampleMaterializer:
    """使用 FFmpeg 从标准化工作音频中裁剪候选样本。"""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        runner: CommandRunner = run_subprocess,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.runner = runner

    def build_command(self, sample: CandidateSample, destination_path: Path) -> list[str]:
        """构建裁剪命令：按起点和时长从工作音频中导出样本。"""
        return [
            self.ffmpeg_path,
            "-y",
            "-ss",
            str(sample.start_time),
            "-i",
            str(sample.source_path),
            "-t",
            str(sample.duration),
            "-vn",
            str(destination_path),
        ]

    def materialize(self, sample: CandidateSample, samples_dir: Path) -> Path:
        samples_dir.mkdir(parents=True, exist_ok=True)
        destination = samples_dir / f"{sample.sample_id}.wav"
        self.runner(self.build_command(sample, destination))
        return destination
