# 作品集上下文构建：从本地报告中提取给本地展示层使用的事实材料。
import json
from pathlib import Path
from typing import Any

from audio_dataset_agent.core.models import Batch
from audio_dataset_agent.knowledge.context import build_knowledge_context


def build_local_portfolio_research(batch: Batch) -> dict[str, Any]:
    """根据本地批次事实生成作品集展示层所需的说明数据。"""
    policy_labels = {
        "voice_clone": "声音克隆",
        "tts": "语音合成（TTS）",
        "asr": "语音识别（ASR）",
    }
    policy_label = policy_labels.get(batch.target_policy, batch.target_policy)
    quality_report = _read_json_path(batch.export_paths.get("dataset_quality_json"), {})
    references = quality_report.get("knowledge_references", [])
    if not isinstance(references, list):
        references = []

    return {
        "status": "succeeded",
        "message": "本地作品集报告已根据质量报告和知识库引用生成。",
        "problem_explanation": (
            f"本项目面向{policy_label}数据准备，解决原始音频格式不统一、无效片段、"
            "音频质量波动和人工筛选成本较高等问题。系统把确定性音频工具、"
            "用途化质量规则和受约束 Agent 组合成可回溯的处理流程，"
            f"当前批次候选样本数为 {batch.candidate_sample_count}。"
        ),
        "audio_processing_background": [
            {
                "topic": "用途化音频质量控制",
                "summary": "同一条音频在 TTS、ASR 和声音克隆中的准入标准不同，因此需要先确定数据用途，再解释指标和决策。",
                "why_it_matters_for_this_project": f"当前流程按{policy_label}画像组织文本、噪声、静音和音色相关判断。",
            },
            {
                "topic": "VAD 与候选样本生成",
                "summary": "VAD 用于定位语音活动区间，减少静音和无信息片段进入后续分析。",
                "why_it_matters_for_this_project": "候选样本由工具产生，之后才能进入统一的指标分析和 Agent 决策。",
            },
            {
                "topic": "修复后重新评估",
                "summary": "对可修复样本执行受控处理后，需要比较修复前后的指标和执行状态。",
                "why_it_matters_for_this_project": "修复不是自动通过条件，只有评估结果满足要求时才交给最终选择阶段。",
            },
        ],
        "tool_notes": [
            {
                "tool_name": "FFmpeg",
                "purpose": "统一音频格式、采样率和声道，并执行受控裁剪与基础修复。",
                "how_this_project_uses_it": "作为本地标准化和修复执行器，由流程传入已校验的参数。",
                "portfolio_value": "体现了音频数据管线与成熟命令行工具的工程集成能力。",
            },
            {
                "tool_name": "VAD / faster-whisper",
                "purpose": "VAD 负责语音活动检测，faster-whisper 负责在需要时生成 ASR 候选文本。",
                "how_this_project_uses_it": "VAD 先产生候选片段，ASR 结果作为 TTS/ASR 文本质量判断的证据。",
                "portfolio_value": "体现了音频处理、文本证据和用途化质量控制的组合。",
            },
            {
                "tool_name": "DeepFilterNet",
                "purpose": "用于可选的神经网络语音增强和降噪。",
                "how_this_project_uses_it": "只有策略和准入判断认为样本值得修复时，才进入受控修复链。",
                "portfolio_value": "展示了筛选、修复、复评和最终采用之间的闭环设计。",
            },
        ],
        "external_references": [],
        "local_references": references,
        "limitations": [
            "作品集报告只整理本地批次事实和本地知识库引用，不替代人耳听感或业务金标评估。",
            "本地知识库来源用于解释工具和规范，不会覆盖样本的 keep、reject、repair、review 决策。",
        ],
    }


def build_portfolio_context(batch: Batch, knowledge_retriever=None) -> str:
    """构造包含本地事实和知识库引用的 JSON 上下文。"""
    local_knowledge = build_knowledge_context([], max_chars=4000)
    if knowledge_retriever is not None:
        hits = knowledge_retriever.search(
            f"{batch.target_policy} 音频数据处理工具和数据集质量规范",
            profile=batch.target_policy,
            top_k=3,
            agent_name="local_portfolio_report",
        )
        local_knowledge = build_knowledge_context(hits, max_chars=4000)
    payload = {
        "batch": {
            "batch_id": batch.batch_id,
            "target_policy": batch.target_policy,
            "input_dir": str(batch.input_dir),
            "output_dir": str(batch.output_dir),
            "phase": batch.phase.value,
            "raw_asset_count": batch.raw_asset_count,
            "candidate_sample_count": batch.candidate_sample_count,
            "analysis_result_count": batch.analysis_result_count,
            "processing_strategy_count": batch.processing_strategy_count,
            "curator_decision_count": batch.curator_decision_count,
            "repair_plan_count": batch.repair_plan_count,
            "repair_execution_count": batch.repair_execution_count,
            "repair_evaluation_count": batch.repair_evaluation_count,
            "final_selection_count": batch.final_selection_count,
        },
        "dataset_quality_report": _read_json_path(
            batch.export_paths.get("dataset_quality_json")
        ),
        "manifest_records": _read_jsonl_path(batch.export_paths.get("manifest")),
        "review_queue": _read_json_path(batch.export_paths.get("review_queue_json"), []),
        "state": _read_selected_state_files(batch.export_paths.get("state_files")),
        "local_knowledge": {
            "context": local_knowledge.text,
            "references": local_knowledge.references,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _read_selected_state_files(state_files: object) -> dict[str, Any]:
    if not isinstance(state_files, dict):
        return {}

    selected_names = [
        "processing_strategies",
        "curator_decisions",
        "repair_evaluations",
        "final_selections",
    ]
    return {
        name: _read_json_path(state_files.get(name), [])
        for name in selected_names
        if state_files.get(name)
    }


def _read_json_path(path_value: object, default: Any | None = None) -> Any:
    if default is None:
        default = {}
    if not path_value:
        return default
    path = Path(str(path_value))
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_path(path_value: object) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(str(path_value))
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
