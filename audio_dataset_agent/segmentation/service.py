"""候选样本切分服务：根据 VAD 结果生成 CandidateSample。

VAD 只负责给出语音区间；这里把每个区间包装成 CandidateSample。真正裁剪出 wav 文件的动作在 materialization_service 中完成。
"""
from dataclasses import replace
from datetime import datetime, timezone

from audio_dataset_agent.analyzers.vad import VADAnalyzer
from audio_dataset_agent.core.models import Batch, BatchPhase, CandidateSample


def generate_candidate_samples(batch: Batch, vad_analyzer: VADAnalyzer) -> Batch:
    """把标准化后的工作音频切成候选训练片段。

    VAD 只负责给出语音区间；这里把每个区间包装成 CandidateSample。真正裁剪出 wav 文件的动作在 materialization_service 中完成。
    """
    candidate_samples = []

    for asset in batch.raw_assets:
        if asset.work_path is None:
            raise ValueError(f"Raw asset is missing work_path: {asset.asset_id}")

        # detect 返回的是时间段，不是音频文件；这样 VAD 工具可以被替换，而下游仍然接收统一结构。
        segments = vad_analyzer.detect(asset.work_path)
        sample_metadata_status = (
            "requires_alignment"
            if asset.reference_text and len(segments) > 1
            else asset.metadata_status
        )
        for index, segment in enumerate(segments, start=1):
            sample_id = f"{batch.batch_id}_sample_{len(candidate_samples) + 1:04d}"
            # sample_id 按批次内顺序生成，后续 ASR、Inspector、Agent 决策都用它做关联键。
            candidate_samples.append(
                CandidateSample(
                    sample_id=sample_id,
                    batch_id=batch.batch_id,
                    asset_id=asset.asset_id,
                    source_path=asset.work_path,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    duration=segment.duration,
                    reference_text=asset.reference_text,
                    reference_text_source=asset.reference_text_source,
                    language=asset.language,
                    speaker_id=asset.speaker_id,
                    metadata_status=sample_metadata_status,
                )
            )

    return replace(
        batch,
        phase=BatchPhase.SEGMENTED,
        updated_at=datetime.now(timezone.utc),
        candidate_samples=candidate_samples,
    )
