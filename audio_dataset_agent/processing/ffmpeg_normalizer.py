# FFmpeg 标准化适配器：负责采样率、声道和格式统一。
import subprocess
from pathlib import Path
from typing import Callable

from audio_dataset_agent.core.models import RawAsset


CommandRunner = Callable[[list[str]], None]


def run_subprocess(command: list[str]) -> None:
    """执行 FFmpeg 命令，失败时抛出 CalledProcessError。"""
    subprocess.run(command, check=True, capture_output=True, text=True)


class FFmpegAudioNormalizer:
    """使用 FFmpeg 将原始音频标准化为工作 WAV 文件。"""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        sample_rate: int = 16000,
        channels: int = 1,
        runner: CommandRunner = run_subprocess,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.runner = runner

    def build_command(self, source_path: Path, destination_path: Path) -> list[str]:
        """构建标准化命令：覆盖输出、禁用视频、指定声道和采样率。"""
        return [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(source_path),
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-vn",
            str(destination_path),
        ]

    def normalize(self, asset: RawAsset, work_dir: Path) -> Path:
        """生成统一格式的 WAV 工作副本。"""
        work_dir.mkdir(parents=True, exist_ok=True)
        destination = work_dir / f"{asset.asset_id}.wav"
        self.runner(self.build_command(asset.source_path, destination))
        return destination
