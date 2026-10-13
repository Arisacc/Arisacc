# 样本物化服务：协调切片落盘并更新候选样本路径。
from dataclasses import replace
from datetime import datetime, timezone

from audio_dataset_agent.core.models import Batch
from audio_dataset_agent.core.run_context import RunContext
from audio_dataset_agent.segmentation.materializer import (
    CopySampleMaterializer,
    SampleMaterializer,
)


def materialize_candidate_samples(
    batch: Batch,
    materializer: SampleMaterializer | None = None,
) -> Batch:
    """为所有候选样本创建实际音频文件。"""
    selected_materializer = materializer or CopySampleMaterializer()
    samples_dir = RunContext.from_batch(batch).samples_dir

    materialized_samples = []
    for sample in batch.candidate_samples:
        # service 只写回 sample_path；具体复制或 FFmpeg 裁剪交给 materializer。
        sample_path = selected_materializer.materialize(sample, samples_dir)
        materialized_samples.append(replace(sample, sample_path=sample_path))

    return replace(
        batch,
        updated_at=datetime.now(timezone.utc),
        candidate_samples=materialized_samples,
    )
