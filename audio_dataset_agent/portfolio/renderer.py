# 作品集渲染器：把事实报告和研究内容渲染成 Markdown/HTML。
import html
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from audio_dataset_agent.core.models import Batch


def write_portfolio_artifacts(
    batch: Batch,
    research: dict[str, Any],
) -> Batch:
    """写出作品集研究 JSON，并生成 Markdown/HTML 展示报告。"""
    export_dir = _batch_export_dir(batch)
    export_dir.mkdir(parents=True, exist_ok=True)

    research_path = export_dir / "portfolio_research.json"
    markdown_path = export_dir / "portfolio_report.md"
    html_path = export_dir / "portfolio_report.html"

    quality_report = _read_quality_report(batch)
    research_path.write_text(
        json.dumps(research, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = render_portfolio_markdown(quality_report, research)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(render_portfolio_html(markdown), encoding="utf-8")

    export_paths = dict(batch.export_paths)
    export_paths.update(
        {
            "portfolio_research_json": str(research_path),
            "portfolio_report_markdown": str(markdown_path),
            "portfolio_report_html": str(html_path),
        }
    )
    return replace(batch, export_paths=export_paths)


def render_portfolio_markdown(
    quality_report: dict[str, Any],
    research: dict[str, Any],
) -> str:
    lines = [
        "# 作品集展示报告",
        "",
        "## 项目定位",
        "",
        _text(research.get("problem_explanation")),
        "",
        "## 本地处理结果概览",
        "",
        f"- 批次 ID: {_text(quality_report.get('batch_id'))}",
        f"- 数据目标: {_text(quality_report.get('target_policy'))}",
        f"- 样本总数: {_text(quality_report.get('total_samples'))}",
        f"- 可用样本: {_text(quality_report.get('usable_samples'))}",
        f"- 复核样本: {_text(quality_report.get('review_samples'))}",
        f"- 拒绝样本: {_text(quality_report.get('rejected_samples'))}",
        "",
        "## 样本处理漏斗",
        "",
        *_render_sample_funnel(quality_report),
        "",
        "## Agent 分工",
        "",
        "- Strategy Agent: 根据 Inspector 的 issue_tags 和工具注册表选择处理策略，只输出策略建议和候选工具，不直接改写音频。",
        "- Curator Agent: 基于目标 profile 和证据判断 keep / reject / repair / review，是样本去留的核心决策者。",
        "- Report Agent: 汇总质量统计、复核队列和风险建议，不修改样本选择结果。",
        "- 本地报告与知识库：基于本地事实、质量统计和本地知识检索生成展示材料。",
        "",
        "## 音频处理背景",
        "",
    ]
    for item in _list(research.get("audio_processing_background")):
        lines.extend(
            [
                f"### {_text(item.get('topic'))}",
                "",
                _text(item.get("summary")),
                "",
                f"项目价值: {_text(item.get('why_it_matters_for_this_project'))}",
                "",
            ]
        )

    lines.extend(["## 工具链说明", ""])
    for item in _list(research.get("tool_notes")):
        lines.extend(
            [
                f"### {_text(item.get('tool_name'))}",
                "",
                f"- 用途: {_text(item.get('purpose'))}",
                f"- 本项目使用方式: {_text(item.get('how_this_project_uses_it'))}",
                f"- 展示价值: {_text(item.get('portfolio_value'))}",
                "",
            ]
        )

    lines.extend(["## 本地知识来源", ""])
    for item in _list(research.get("local_references")):
        source = _text(item.get("source"))
        section = _text(item.get("section"))
        chunk_id = _text(item.get("chunk_id"))
        lines.append(f"- {source} / {section} / {chunk_id}")

    lines.extend(["", "## 外部参考", ""])
    for item in _list(research.get("external_references")):
        title = _text(item.get("title"))
        url = _text(item.get("url"))
        used_for = _text(item.get("used_for"))
        lines.append(f"- [{title}]({url}): {used_for}")

    lines.extend(["", "## 局限与边界", ""])
    for item in _list(research.get("limitations")):
        lines.append(f"- {_text(item)}")

    lines.append("")
    return "\n".join(lines)


def _render_sample_funnel(quality_report: dict[str, Any]) -> list[str]:
    total = _number(quality_report.get("total_samples"))
    usable = _number(quality_report.get("usable_samples"))
    review = _number(quality_report.get("review_samples"))
    rejected = _number(quality_report.get("rejected_samples"))
    unresolved = max(total - usable - review - rejected, 0)
    return [
        f"- 导入候选样本: {total}",
        f"- 可直接或修复后采用: {usable}",
        f"- 需要人工复核: {review}",
        f"- 已拒绝/排除: {rejected}",
        f"- 其他状态: {unresolved}",
    ]


def render_portfolio_html(markdown: str) -> str:
    body = "\n".join(_markdown_line_to_html(line) for line in markdown.splitlines())
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>作品集展示报告</title>",
            "<style>",
            "body{font-family:Arial,'Microsoft YaHei',sans-serif;line-height:1.7;max-width:980px;margin:40px auto;padding:0 24px;color:#1f2937;}",
            "h1,h2,h3{line-height:1.3;color:#111827;} a{color:#2563eb;} li{margin:6px 0;}",
            "</style>",
            "</head>",
            "<body>",
            body,
            "</body>",
            "</html>",
        ]
    )


def _markdown_line_to_html(line: str) -> str:
    escaped = html.escape(line)
    if line.startswith("# "):
        return f"<h1>{html.escape(line[2:])}</h1>"
    if line.startswith("## "):
        return f"<h2>{html.escape(line[3:])}</h2>"
    if line.startswith("### "):
        return f"<h3>{html.escape(line[4:])}</h3>"
    if line.startswith("- "):
        return f"<li>{_linkify(escaped[2:])}</li>"
    if not line.strip():
        return ""
    return f"<p>{_linkify(escaped)}</p>"


def _linkify(value: str) -> str:
    import re

    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    return pattern.sub(r'<a href="\2">\1</a>', value)


def _batch_export_dir(batch: Batch) -> Path:
    report_path = batch.export_paths.get("dataset_quality_json") or batch.export_paths.get(
        "manifest"
    )
    if not report_path:
        raise ValueError("batch 导出路径中缺少 report 或 manifest")
    return Path(report_path).parent


def _read_quality_report(batch: Batch) -> dict[str, Any]:
    path_value = batch.export_paths.get("dataset_quality_json")
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _number(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0
