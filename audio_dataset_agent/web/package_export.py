"""把当前批次的音频和追溯文件打包为可下载 ZIP。"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def create_export_package(batch_dir: Path | str) -> Path:
    root = Path(batch_dir).resolve()
    if not root.is_dir():
        raise ValueError("批次导出目录不存在")
    package = root / "voice_clone_keep_audio.zip"
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path == package:
            continue
        relative = path.relative_to(root)
        if relative.parts[:2] == ("manual_groups", "keep"):
            files.append((path, relative))
    with ZipFile(package, "w", compression=ZIP_DEFLATED) as archive:
        for path, relative in files:
            archive.write(path, relative.as_posix())
    return package
