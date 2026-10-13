# 文件扫描层：识别目录中的音频文件并收集基础文件信息。
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".wav",
    ".webm",
}


def scan_audio_files(input_dir: Path | str) -> list[Path]:
    """按稳定顺序返回输入目录下支持的音频文件。"""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    # 递归扫描子目录，并按相对路径排序，保证不同机器上的批处理顺序稳定。
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
