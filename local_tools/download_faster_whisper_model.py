"""下载 faster-whisper-small 到项目固定模型目录。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_REPO_ID = "Systran/faster-whisper-small"
DEFAULT_LOCAL_DIR = Path("local_tools/models/faster-whisper/small")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载 faster-whisper-small 本地模型")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from huggingface_hub import snapshot_download

    args.local_dir.mkdir(parents=True, exist_ok=True)
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    snapshot_download(repo_id=args.repo_id, local_dir=str(args.local_dir), local_dir_use_symlinks=False)
    print(f"已下载 {args.repo_id} 到 {args.local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
