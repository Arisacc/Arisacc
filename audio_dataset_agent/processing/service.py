# 基础预处理服务：把原始音频转换成后续分析统一使用的工作音频。
from dataclasses import replace
from datetime import datetime, timezone

from audio_dataset_agent.core.models import Batch, BatchPhase, RawAssetStatus
from audio_dataset_agent.core.run_context import RunContext
from audio_dataset_agent.processing.normalizer import AudioNormalizer, CopyAudioNormalizer


def normalize_batch(batch: Batch, normalizer: AudioNormalizer | None = None) -> Batch:
    """为所有原始资产创建工作副本，并推进 batch 阶段。"""
    selected_normalizer = normalizer or CopyAudioNormalizer()
    work_dir = RunContext.from_batch(batch).normalized_dir

    normalized_assets = []
    for asset in batch.raw_assets:
        # normalizer 负责生成 work_path；service 负责把结果写回 Batch 状态。
        work_path = selected_normalizer.normalize(asset, work_dir)
        normalized_assets.append(
            replace(
                asset,
                work_path=work_path,
                status=RawAssetStatus.NORMALIZED,
            )
        )

    return replace(
        batch,
        phase=BatchPhase.NORMALIZED,
        updated_at=datetime.now(timezone.utc),
        raw_assets=normalized_assets,
    )
