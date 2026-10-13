from pathlib import Path
from shutil import copy2
from typing import Protocol

from audio_dataset_agent.core.models import RawAsset


class AudioNormalizer(Protocol):
    # 真实 FFmpeg 转码器后续只要实现这个接口即可替换 CopyAudioNormalizer。
    def normalize(self, asset: RawAsset, work_dir: Path) -> Path:
        """为原始资产创建标准化工作文件。"""


class CopyAudioNormalizer:
    """在保留原始音频字节不变的前提下准备工作副本。"""

    def normalize(self, asset: RawAsset, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        destination = work_dir / f"{asset.asset_id}{asset.extension}"
        copy2(asset.source_path, destination)
        return destination
