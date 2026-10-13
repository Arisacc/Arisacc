# 序列化工具：把核心对象稳定写入 JSON 状态文件。
from typing import Any

from audio_dataset_agent.core.models import (
    AnalysisResult,
    Batch,
    CandidateSample,
    CuratorDecision,
    FinalSelection,
    ProcessingStrategy,
    RawAsset,
    RepairEvaluation,
    RepairExecution,
    RepairPlan,
    RepairChain,
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
