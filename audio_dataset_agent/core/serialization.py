# 序列化工具：把核心对象稳定写入 JSON 状态文件。
from datetime import datetime
from pathlib import Path
from typing import Any

from audio_dataset_agent.core.models import (
    AnalysisResult,
    Batch,
    BatchPhase,
    CandidateSample,
    CandidateSampleStatus,
    CuratorDecision,
    DecisionType,
    FinalSelection,
    ProcessingStrategy,
    RawAsset,
    RawAssetStatus,
    RepairEvaluation,
    RepairExecution,
    RepairPlan,
    RepairChain,
)


def batch_from_dict(payload: dict[str, Any]) -> Batch:
    """从 Harness checkpoint 恢复核心 Batch，不重新扫描已处理过的原始资产。"""
    assets = [
        RawAsset(
            asset_id=item["asset_id"], batch_id=item["batch_id"],
            source_path=Path(item["source_path"]), file_name=item["file_name"],
            file_size=item["file_size"], extension=item["extension"],
            work_path=Path(item["work_path"]) if item.get("work_path") else None,
            status=RawAssetStatus(item.get("status", "imported")),
            reference_text=item.get("reference_text"), reference_text_source=item.get("reference_text_source"),
            language=item.get("language"), speaker_id=item.get("speaker_id"),
            metadata_status=item.get("metadata_status", "missing"),
            metadata_warnings=list(item.get("metadata_warnings") or []),
        ) for item in payload.get("raw_assets", [])
    ]
    samples = [
        CandidateSample(
            sample_id=item["sample_id"], batch_id=item["batch_id"], asset_id=item["asset_id"],
            source_path=Path(item["source_path"]), start_time=item["start_time"], end_time=item["end_time"],
            duration=item["duration"], sample_path=Path(item["sample_path"]) if item.get("sample_path") else None,
            transcript=item.get("transcript"), transcript_source=item.get("transcript_source"),
            asr_confidence=item.get("asr_confidence"), status=CandidateSampleStatus(item.get("status", "created")),
            reference_text=item.get("reference_text"), reference_text_source=item.get("reference_text_source"),
            language=item.get("language"), speaker_id=item.get("speaker_id"),
            metadata_status=item.get("metadata_status", "missing"),
        ) for item in payload.get("candidate_samples", [])
    ]
    analyses = [AnalysisResult(
        sample_id=item["sample_id"], duration=item["duration"], transcript_length=item["transcript_length"],
        asr_confidence=item.get("asr_confidence"), issue_tags=list(item.get("issue_tags") or []),
        evidence=dict(item.get("evidence") or {}),
    ) for item in payload.get("analysis_results", [])]
    strategies = [ProcessingStrategy(
        sample_id=item["sample_id"], action=item["action"], recommended_operation=item["recommended_operation"],
        selected_tool_ids=list(item.get("selected_tool_ids") or []), reason=item.get("reason", ""),
        confidence=item.get("confidence", 0.0), agent_name=item.get("agent_name", "baseline_processing_strategy"),
    ) for item in payload.get("processing_strategies", [])]
    decisions = [CuratorDecision(
        sample_id=item["sample_id"], decision=DecisionType(item["decision"]), reason=item["reason"],
        confidence=item["confidence"], recommended_action=item["recommended_action"],
        policy_name=item.get("policy_name", "baseline"), strategy_consistency=item.get("strategy_consistency", "not_applicable"),
    ) for item in payload.get("curator_decisions", [])]
    plans = [RepairPlan(
        sample_id=item["sample_id"], operation=item["operation"], parameters=dict(item.get("parameters") or []),
        reason=item.get("reason", ""), status=item.get("status", "planned"), chain_id=item.get("chain_id"),
        step_index=item.get("step_index", 1),
    ) for item in payload.get("repair_plans", [])]
    chains = [RepairChain(
        chain_id=item["chain_id"], sample_id=item["sample_id"],
        plans=[RepairPlan(
            sample_id=plan["sample_id"], operation=plan["operation"], parameters=dict(plan.get("parameters") or {}),
            reason=plan.get("reason", ""), status=plan.get("status", "planned"), chain_id=plan.get("chain_id"),
            step_index=plan.get("step_index", 1),
        ) for plan in item.get("plans", [])],
        max_steps=item.get("max_steps", 3), status=item.get("status", "planned"),
    ) for item in payload.get("repair_chains", [])]
    executions = [RepairExecution(
        sample_id=item["sample_id"], operation=item["operation"], input_path=Path(item["input_path"]),
        output_path=Path(item["output_path"]), status=item["status"], error_message=item.get("error_message"),
        chain_id=item.get("chain_id"), step_index=item.get("step_index", 1),
    ) for item in payload.get("repair_executions", [])]
    evaluations = [RepairEvaluation(
        sample_id=item["sample_id"], operation=item["operation"], before_metrics=dict(item.get("before_metrics") or {}),
        after_metrics=dict(item.get("after_metrics") or {}), improved=item["improved"],
        recommendation=item["recommendation"], reason=item.get("reason", ""), chain_id=item.get("chain_id"),
        step_index=item.get("step_index", 1),
    ) for item in payload.get("repair_evaluations", [])]
    selections = [FinalSelection(
        sample_id=item["sample_id"], status=item["status"],
        selected_audio_path=Path(item["selected_audio_path"]) if item.get("selected_audio_path") else None,
        reason=item["reason"],
    ) for item in payload.get("final_selections", [])]
    return Batch(
        batch_id=payload["batch_id"], input_dir=Path(payload["input_dir"]), output_dir=Path(payload["output_dir"]),
        target_policy=payload["target_policy"], run_config=dict(payload.get("run_config") or {}),
        metadata_summary=dict(payload.get("metadata_summary") or {}), phase=BatchPhase(payload.get("phase", "created")),
        created_at=datetime.fromisoformat(payload["created_at"]), updated_at=datetime.fromisoformat(payload["updated_at"]),
        raw_assets=assets, candidate_samples=samples, analysis_results=analyses,
        processing_strategies=strategies, curator_decisions=decisions, repair_plans=plans,
        repair_chains=chains, repair_executions=executions, repair_evaluations=evaluations,
        final_selections=selections, export_paths=dict(payload.get("export_paths") or {}),
    )


def raw_asset_to_dict(asset: RawAsset) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id,
        "batch_id": asset.batch_id,
        "source_path": str(asset.source_path),
        "file_name": asset.file_name,
        "file_size": asset.file_size,
        "extension": asset.extension,
        "work_path": str(asset.work_path) if asset.work_path else None,
        "status": asset.status.value,
        "reference_text": asset.reference_text,
        "reference_text_source": asset.reference_text_source,
        "language": asset.language,
        "speaker_id": asset.speaker_id,
        "metadata_status": asset.metadata_status,
        "metadata_warnings": list(asset.metadata_warnings),
    }


def candidate_sample_to_dict(sample: CandidateSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "batch_id": sample.batch_id,
        "asset_id": sample.asset_id,
        "source_path": str(sample.source_path),
        "sample_path": str(sample.sample_path) if sample.sample_path else None,
        "start_time": sample.start_time,
        "end_time": sample.end_time,
        "duration": sample.duration,
        "transcript": sample.transcript,
        "transcript_source": sample.transcript_source,
        "asr_confidence": sample.asr_confidence,
        "status": sample.status.value,
        "reference_text": sample.reference_text,
        "reference_text_source": sample.reference_text_source,
        "language": sample.language,
        "speaker_id": sample.speaker_id,
        "metadata_status": sample.metadata_status,
    }


def analysis_result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "duration": result.duration,
        "transcript_length": result.transcript_length,
        "asr_confidence": result.asr_confidence,
        "issue_tags": list(result.issue_tags),
        "evidence": dict(result.evidence),
    }


def curator_decision_to_dict(decision: CuratorDecision) -> dict[str, Any]:
    return {
        "sample_id": decision.sample_id,
        "decision": decision.decision.value,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "recommended_action": decision.recommended_action,
        "policy_name": decision.policy_name,
        "strategy_consistency": decision.strategy_consistency,
    }


def processing_strategy_to_dict(strategy: ProcessingStrategy) -> dict[str, Any]:
    return {
        "sample_id": strategy.sample_id,
        "action": strategy.action,
        "recommended_operation": strategy.recommended_operation,
        "selected_tool_ids": list(strategy.selected_tool_ids),
        "reason": strategy.reason,
        "confidence": strategy.confidence,
        "agent_name": strategy.agent_name,
    }


def repair_plan_to_dict(plan: RepairPlan) -> dict[str, Any]:
    return {
        "sample_id": plan.sample_id,
        "operation": plan.operation,
        "parameters": dict(plan.parameters),
        "reason": plan.reason,
        "status": plan.status,
        "chain_id": plan.chain_id,
        "step_index": plan.step_index,
    }


def repair_execution_to_dict(execution: RepairExecution) -> dict[str, Any]:
    return {
        "sample_id": execution.sample_id,
        "operation": execution.operation,
        "input_path": str(execution.input_path),
        "output_path": str(execution.output_path),
        "status": execution.status,
        "error_message": execution.error_message,
        "chain_id": execution.chain_id,
        "step_index": execution.step_index,
    }


def repair_evaluation_to_dict(evaluation: RepairEvaluation) -> dict[str, Any]:
    return {
        "sample_id": evaluation.sample_id,
        "operation": evaluation.operation,
        "before_metrics": dict(evaluation.before_metrics),
        "after_metrics": dict(evaluation.after_metrics),
        "improved": evaluation.improved,
        "recommendation": evaluation.recommendation,
        "reason": evaluation.reason,
        "chain_id": evaluation.chain_id,
        "step_index": evaluation.step_index,
    }


def repair_chain_to_dict(chain: RepairChain) -> dict[str, Any]:
    return {
        "chain_id": chain.chain_id,
        "sample_id": chain.sample_id,
        "plans": [repair_plan_to_dict(plan) for plan in chain.plans],
        "max_steps": chain.max_steps,
        "status": chain.status,
    }


def final_selection_to_dict(selection: FinalSelection) -> dict[str, Any]:
    return {
        "sample_id": selection.sample_id,
        "status": selection.status,
        "selected_audio_path": (
            str(selection.selected_audio_path)
            if selection.selected_audio_path
            else None
        ),
        "reason": selection.reason,
    }


def batch_to_dict(batch: Batch) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "input_dir": str(batch.input_dir),
        "output_dir": str(batch.output_dir),
        "target_policy": batch.target_policy,
        "run_config": dict(batch.run_config),
        "metadata_summary": dict(batch.metadata_summary),
        "phase": batch.phase.value,
        "created_at": batch.created_at.isoformat(),
        "updated_at": batch.updated_at.isoformat(),
        "export_paths": dict(batch.export_paths),
        "raw_asset_count": batch.raw_asset_count,
        "raw_assets": [raw_asset_to_dict(asset) for asset in batch.raw_assets],
        "candidate_sample_count": batch.candidate_sample_count,
        "candidate_samples": [
            candidate_sample_to_dict(sample) for sample in batch.candidate_samples
        ],
        "analysis_result_count": batch.analysis_result_count,
        "analysis_results": [
            analysis_result_to_dict(result) for result in batch.analysis_results
        ],
        "processing_strategy_count": batch.processing_strategy_count,
        "processing_strategies": [
            processing_strategy_to_dict(strategy)
            for strategy in batch.processing_strategies
        ],
        "curator_decision_count": batch.curator_decision_count,
        "curator_decisions": [
            curator_decision_to_dict(decision) for decision in batch.curator_decisions
        ],
        "repair_plan_count": batch.repair_plan_count,
        "repair_plans": [repair_plan_to_dict(plan) for plan in batch.repair_plans],
        "repair_chain_count": batch.repair_chain_count,
        "repair_chains": [repair_chain_to_dict(chain) for chain in batch.repair_chains],
        "repair_execution_count": batch.repair_execution_count,
        "repair_executions": [
            repair_execution_to_dict(execution)
            for execution in batch.repair_executions
        ],
        "repair_evaluation_count": batch.repair_evaluation_count,
        "repair_evaluations": [
            repair_evaluation_to_dict(evaluation)
            for evaluation in batch.repair_evaluations
        ],
        "final_selection_count": batch.final_selection_count,
        "final_selections": [
            final_selection_to_dict(selection)
            for selection in batch.final_selections
        ],
    }
