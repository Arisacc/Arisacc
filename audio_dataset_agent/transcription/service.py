# 转写服务：按配置决定是否执行 ASR 并写回候选文本。
from dataclasses import replace
from datetime import datetime, timezone

from audio_dataset_agent.core.models import (
    Batch,
    BatchPhase,
    CandidateSampleStatus,
)
from audio_dataset_agent.transcription.transcriber import Transcriber


def transcribe_candidate_samples(batch: Batch, transcriber: Transcriber) -> Batch:
    """为每个候选样本写入转写信息。"""
    transcribed_samples = []

    for sample in batch.candidate_samples:
        # transcriber 只负责生成文本结果；service 负责写入 CandidateSample 状态。
        result = transcriber.transcribe(sample)
        transcribed_samples.append(
            replace(
                sample,
                transcript=result.text,
                transcript_source=result.source,
                asr_confidence=result.confidence,
                status=CandidateSampleStatus.TRANSCRIBED,
            )
        )

    return replace(
        batch,
        phase=BatchPhase.TRANSCRIBED,
        updated_at=datetime.now(timezone.utc),
        candidate_samples=transcribed_samples,
    )
