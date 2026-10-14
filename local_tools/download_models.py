"""交互式下载项目可选模型到 local_tools/models。"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelOption:
    key: str
    label: str
    size: str
    local_dir: Path
    description: str


MODEL_OPTIONS = {
    "qwen": ModelOption("qwen", "Qwen3-4B-Instruct", "约 8GB", Path("local_tools/models/qwen/Qwen3-4B-Instruct-2507"), "Strategy/Curator/Report 本地模型"),
    "faster-whisper": ModelOption("faster-whisper", "faster-whisper-small", "约 486MB", Path("local_tools/models/faster-whisper/small"), "可选 ASR 候选文本生成"),
    "msst": ModelOption("msst", "MSST BS-Roformer 去混响", "约 163MB", Path("local_tools/models/msst/deverb_bs_roformer_8_256dim_8depth"), "voice_clone 前置去混响"),
}

MSST_FILES = {
    "deverb_bs_roformer_8_256dim_8depth.ckpt": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/single_stem_models/deverb_bs_roformer_8_256dim_8depth.ckpt?download=true",
    "deverb_bs_roformer_8_256dim_8depth.ckpt.yaml": "https://raw.githubusercontent.com/SUC-DriverOld/MSST-WebUI/main/configs_backup/single_stem_models/deverb_bs_roformer_8_256dim_8depth.ckpt.yaml",
}


def resolve_option(key: str) -> ModelOption:
    if key not in MODEL_OPTIONS:
        raise ValueError(f"不支持的模型选项：{key}")
    return MODEL_OPTIONS[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载声音数据处理工作台的可选模型")
    parser.add_argument("--model", choices=list(MODEL_OPTIONS), help="非交互式选择模型")
    parser.add_argument("--provider", choices=["huggingface", "modelscope"], default="huggingface")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    return parser


def _download_msst(option: ModelOption) -> None:
    option.local_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MSST_FILES.items():
        destination = option.local_dir / name
        if destination.exists() and destination.stat().st_size > 0:
            print(f"已存在，跳过：{destination}")
            continue
        print(f"下载：{name}")
        urllib.request.urlretrieve(url, destination)


def _download_snapshot(key: str, option: ModelOption, provider: str, endpoint: str) -> None:
    if key == "qwen":
        from local_tools.download_qwen_model import main as download_qwen
        args = ["--local-dir", str(option.local_dir), "--provider", provider]
        if endpoint:
            args.extend(["--endpoint", endpoint])
        download_qwen(args)
        return
    from local_tools.download_faster_whisper_model import main as download_whisper
    args = ["--local-dir", str(option.local_dir)]
    if endpoint:
        args.extend(["--endpoint", endpoint])
    download_whisper(args)


def _select_interactively() -> str:
    print("可下载模型：")
    for index, option in enumerate(MODEL_OPTIONS.values(), start=1):
        print(f"  {index}. {option.label} | {option.size} | {option.description}")
    value = input("请选择模型编号（直接回车取消）：").strip()
    if not value:
        return ""
    keys = list(MODEL_OPTIONS)
    try:
        return keys[int(value) - 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("模型编号无效") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    key = args.model or _select_interactively()
    if not key:
        print("已取消下载")
        return 0
    option = resolve_option(key)
    print(f"模型：{option.label}\n预计占用：{option.size}\n保存目录：{option.local_dir.resolve()}\n用途：{option.description}")
    if not args.yes and input("确认下载？[y/N] ").strip().lower() not in {"y", "yes"}:
        print("已取消下载")
        return 0
    if key == "msst":
        _download_msst(option)
    else:
        _download_snapshot(key, option, args.provider, args.endpoint)
    print(f"下载完成：{option.local_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
