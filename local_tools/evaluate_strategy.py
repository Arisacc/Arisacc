from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# 兼容 `python local_tools/evaluate_strategy.py` 这种直接脚本调用方式。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_dataset_agent.curation.qwen_client import TransformersQwenClient
from audio_dataset_agent.evaluation.strategy_eval import (
    default_strategy_eval_cases,
    evaluate_strategy_agent,
    strategy_eval_summary_to_dict,
    write_strategy_eval_report,
)
from audio_dataset_agent.strategy.agent import (
    BaselineProcessingStrategyAgent,
    LLMProcessingStrategyAgent,
)
from audio_dataset_agent.tools.registry import default_tool_registry


DEFAULT_MODEL_PATH = "local_tools/models/qwen/Qwen3-4B-Instruct-2507"


def build_parser() -> argparse.ArgumentParser:
    """定义 Strategy Agent 评估脚本的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="评估 Strategy Agent 的处理策略与工具选择。",
    )
    parser.add_argument(
        "--strategy",
        choices=["baseline", "qwen-local"],
        default="baseline",
        help="选择规则基线或本地 Qwen Strategy Agent。",
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


def _build_agent(args: argparse.Namespace):
    registry = default_tool_registry()
    baseline = BaselineProcessingStrategyAgent(
        registry,
        target_policy=args.target_policy,
    )
    if args.strategy == "baseline":
        return baseline
    return LLMProcessingStrategyAgent(
        TransformersQwenClient(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
        ),
        tool_registry=registry,
        fallback_agent=baseline,
        target_policy=args.target_policy,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_strategy_agent(
        _build_agent(args),
        default_strategy_eval_cases(),
    )
    if args.output_dir:
        write_strategy_eval_report(
            summary,
            Path(args.output_dir),
            title="Strategy Agent 评估报告",
        )
    print(
        json.dumps(
            strategy_eval_summary_to_dict(summary),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
