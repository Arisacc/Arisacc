from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# 直接运行本脚本时，Python 默认只把 local_tools 加入模块搜索路径。
# 显式加入项目根目录，保证从仓库根目录执行命令时可以导入主包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_dataset_agent.curation.agent import LLMCuratorAgent
from audio_dataset_agent.curation.policy import BaselineCurationPolicy
from audio_dataset_agent.curation.qwen_client import TransformersQwenClient
from audio_dataset_agent.evaluation.curator_eval import (
    curator_eval_summary_to_dict,
    default_curator_eval_cases,
    evaluate_curator_policy,
    write_curator_eval_report,
)


DEFAULT_MODEL_PATH = "local_tools/models/qwen/Qwen3-4B-Instruct-2507"


def build_parser() -> argparse.ArgumentParser:
    """定义 Curator 评估脚本的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="评估 Curator Agent 在固定质量案例上的决策准确性。",
    )
    parser.add_argument(
        "--curator",
        choices=["baseline", "qwen-local"],
        default="baseline",
        help="选择规则基线或本地 Qwen Curator Agent。",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="本地 Qwen 模型目录，仅在 qwen-local 模式下使用。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="本地 Qwen 单次决策允许生成的最大 token 数。",
    )
    parser.add_argument(
        "--target-policy",
        choices=["tts", "asr", "voice_clone"],
        default="tts",
        help="评估时使用的数据集用途画像。",
    )
    parser.add_argument(
        "--output-dir",
        help="可选的报告输出目录；不填写时只向终端输出 JSON 摘要。",
    )
    return parser


def _build_policy(args: argparse.Namespace):
    baseline = BaselineCurationPolicy(target_policy=args.target_policy)
    if args.curator == "baseline":
        return baseline
    return LLMCuratorAgent(
        TransformersQwenClient(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
        ),
        fallback_policy=baseline,
        target_policy=args.target_policy,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_curator_policy(
        _build_policy(args),
        default_curator_eval_cases(),
    )
    if args.output_dir:
        write_curator_eval_report(
            summary,
            Path(args.output_dir),
            title="Curator Agent 评估报告",
        )
    print(
        json.dumps(
            curator_eval_summary_to_dict(summary),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
