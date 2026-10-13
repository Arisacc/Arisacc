import argparse
import os
from pathlib import Path

DEFAULT_REPO_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_LOCAL_DIR = Path("local_tools/models/qwen/Qwen3-4B-Instruct-2507")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a local Qwen model for the Audio Dataset Agent."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument(
        "--provider",
        choices=["huggingface", "modelscope"],
        default="huggingface",
    )
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="Optional Hugging Face mirror endpoint. Use an empty string for default HF.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.local_dir.mkdir(parents=True, exist_ok=True)

    if args.provider == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(
            model_id=args.repo_id,
            local_dir=str(args.local_dir),
        )
        print(f"downloaded {args.repo_id} to {args.local_dir} with ModelScope")
        return 0

    from huggingface_hub import snapshot_download

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"downloaded {args.repo_id} to {args.local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
