# manifest 导出：把最终样本写成训练/复核可消费的 JSONL 记录。
from typing import Any

from audio_dataset_agent.core.models import Batch


def build_manifest_records(batch: Batch) -> list[dict[str, Any]]:
    """为每个候选样本构造包含分析和决策信息的 manifest 记录。"""
    # 按 sample_id 将分析结果和决策结果拼回候选样本，形成最终 manifest 行。
    analysis_by_sample = {
        result.sample_id: result for result in batch.analysis_results
    }
    asset_by_id = {asset.asset_id: asset for asset in batch.raw_assets}
    decision_by_sample = {
        decision.sample_id: decision for decision in batch.curator_decisions
    }
    repair_plan_by_sample = {
        plan.sample_id: plan for plan in batch.repair_plans
    }
    repair_execution_by_sample = {
        execution.sample_id: execution for execution in batch.repair_executions
    }
    repair_evaluation_by_sample = {
        evaluation.sample_id: evaluation for evaluation in batch.repair_evaluations
    }
    final_selection_by_sample = {
        selection.sample_id: selection for selection in batch.final_selections
    }
    chain_by_sample = {chain.sample_id: chain for chain in batch.repair_chains}

    records = []
    for sample in batch.candidate_samples:
        analysis = analysis_by_sample.get(sample.sample_id)
        decision = decision_by_sample.get(sample.sample_id)
        repair_plan = repair_plan_by_sample.get(sample.sample_id)
        repair_execution = repair_execution_by_sample.get(sample.sample_id)
        repair_evaluation = repair_evaluation_by_sample.get(sample.sample_id)
        final_selection = final_selection_by_sample.get(sample.sample_id)
        asset = asset_by_id.get(sample.asset_id)
        chain = chain_by_sample.get(sample.sample_id)
        records.append(
            {
                "sample_id": sample.sample_id,
                "asset_id": sample.asset_id,
                "source_audio_path": str(asset.source_path) if asset else None,
                "work_audio_path": (
                    str(asset.work_path)
                    if asset and asset.work_path
                    else None
                ),
                "sample_audio_path": str(sample.sample_path) if sample.sample_path else None,
                "audio_path": str(sample.sample_path or sample.source_path),
                "text": sample.transcript,
                "duration": sample.duration,
                "start_time": sample.start_time,
                "end_time": sample.end_time,
                "transcript_source": sample.transcript_source,
                "reference_text": sample.reference_text,
                "reference_text_source": sample.reference_text_source,
                "language": sample.language,
                "speaker_id": sample.speaker_id,
                "metadata_status": sample.metadata_status,
                "asr_confidence": sample.asr_confidence,
                "audio_metrics": (
                    analysis.evidence.get("audio_metrics")
                    if analysis
                    else None
                ),
                "snr": analysis.evidence.get("snr_db") if analysis else None,
                "quality_score": (
                    analysis.evidence.get("quality_score")
                    if analysis
                    else None
                ),
                "quality_score_version": (
                    analysis.evidence.get("quality_score_version")
                    if analysis
                    else None
                ),
                "issue_tags": analysis.issue_tags if analysis else [],
                "text_alignment": (
                    analysis.evidence.get("text_alignment")
                    if analysis
                    else None
                ),
                "speaker_consistency": (
                    analysis.evidence.get("speaker_consistency")
                    if analysis
                    else None
                ),
                "decision": decision.decision.value if decision else None,
                "decision_reason": decision.reason if decision else None,
                "recommended_action": decision.recommended_action if decision else None,
                "repair_plan": {
                    "operation": repair_plan.operation,
                    "parameters": dict(repair_plan.parameters),
                    "status": repair_plan.status,
                }
                if repair_plan
                else None,
                "repair_chain_id": chain.chain_id if chain else None,
                "repair_chain": [
                    {
                        "step_index": plan.step_index,
                        "operation": plan.operation,
                        "parameters": dict(plan.parameters),
                        "status": plan.status,
                    }
                    for plan in chain.plans
                ] if chain else [],
                "repair_execution": {
                    "operation": repair_execution.operation,
                    "input_path": str(repair_execution.input_path),
                    "output_path": str(repair_execution.output_path),
                    "status": repair_execution.status,
                    "error_message": repair_execution.error_message,
                }
                if repair_execution
                else None,
                "repair_evaluation": {
                    "operation": repair_evaluation.operation,
                    "before_metrics": dict(repair_evaluation.before_metrics),
                    "after_metrics": dict(repair_evaluation.after_metrics),
                    "improved": repair_evaluation.improved,
                    "recommendation": repair_evaluation.recommendation,
                    "reason": repair_evaluation.reason,
                }
                if repair_evaluation
                else None,
                "final_status": final_selection.status if final_selection else None,
                "final_audio_path": (
                    str(final_selection.selected_audio_path)
                    if final_selection and final_selection.selected_audio_path
                    else None
                ),
                "final_selection": {
                    "status": final_selection.status,
                    "selected_audio_path": (
                        str(final_selection.selected_audio_path)
                        if final_selection.selected_audio_path
                        else None
                    ),
                    "reason": final_selection.reason,
                }
                if final_selection
                else None,
            }
        )

    return records
