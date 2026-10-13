"""导入服务：把输入目录扫描结果注册为 RawAsset。"""
from pathlib import Path
from typing import Any
from uuid import uuid4

from audio_dataset_agent.core.models import Batch, BatchPhase, RawAsset
from audio_dataset_agent.importer.scanner import scan_audio_files
from audio_dataset_agent.importer.metadata import (
    MetadataImportConfig,
    load_asset_metadata,
)


def create_import_batch(
    input_dir: Path | str,
    output_dir: Path | str,
    target_policy: str = "voice_clone",
    run_config: dict[str, Any] | None = None,
    metadata_config: MetadataImportConfig | None = None,
) -> Batch:
    """创建批次，并把支持的音频文件登记为原始资产。"""
    # 导入阶段只登记原始文件，不读取、不转码、不修改音频内容。
    # batch_id 是后续所有 asset/sample/state 文件的共同前缀，用来把一次处理的产物串起来。
    batch_id = f"batch_{uuid4().hex[:12]}"
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # scanner 只返回支持的音频后缀；这里开始把普通文件路径转换成系统内部的 RawAsset。
    audio_files = scan_audio_files(input_path)
    metadata_result = load_asset_metadata(input_path, audio_files, metadata_config)
    # 每个音频文件对应一个 RawAsset，后续阶段会基于它生成工作副本和候选样本。
    raw_assets = [
        RawAsset(
            asset_id=f"{batch_id}_asset_{index:04d}",
            batch_id=batch_id,
            source_path=path,
            file_name=path.name,
            file_size=path.stat().st_size,
            extension=path.suffix.lower(),
            reference_text=metadata_result.by_audio_path[path].reference_text,
            reference_text_source=(
                metadata_result.by_audio_path[path].reference_text_source
            ),
            language=metadata_result.by_audio_path[path].language,
            speaker_id=metadata_result.by_audio_path[path].speaker_id,
            metadata_status=metadata_result.by_audio_path[path].status,
            metadata_warnings=list(metadata_result.by_audio_path[path].warnings),
        )
        for index, path in enumerate(audio_files, start=1)
    ]

    return Batch(
        batch_id=batch_id,
        input_dir=input_path,
        output_dir=output_path,
        target_policy=target_policy,
        run_config=dict(run_config or {}),
        metadata_summary=dict(metadata_result.summary),
        phase=BatchPhase.IMPORTED,
        raw_assets=raw_assets,
    )
