"""生成批次运行配置、来源清单和工具版本信息。"""
import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from audio_dataset_agent.core.models import Batch
from audio_dataset_agent.core.run_context import RunContext


TRACKED_PACKAGES = (
    "faster-whisper",
    "silero-vad",
    "torch",
    "transformers",
    "imageio-ffmpeg",
)


def write_provenance_artifacts(batch: Batch, batch_export_dir: Path) -> dict[str, str]:
    """写出可复现和可追溯所需的三个批次级 JSON 文件。"""
    context = RunContext.from_batch(batch)
    paths = {
        "run_config": batch_export_dir / "run_config.json",
        "source_manifest": batch_export_dir / "source_manifest.json",
        "tool_versions": batch_export_dir / "tool_versions.json",
    }
    payloads = {
        "run_config": _build_run_config(batch, context, batch_export_dir),
        "source_manifest": _build_source_manifest(batch),
        "tool_versions": _build_tool_versions(batch.run_config),
    }
    for name, path in paths.items():
        path.write_text(
            json.dumps(payloads[name], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {name: str(path) for name, path in paths.items()}


def _build_run_config(
    batch: Batch,
    context: RunContext,
    batch_export_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "batch_id": batch.batch_id,
        "target_policy": batch.target_policy,
        "created_at": batch.created_at.isoformat(),
        "paths": {
            "input_dir": str(batch.input_dir),
            "output_dir": str(batch.output_dir),
            "normalized_dir": str(context.normalized_dir),
            "samples_dir": str(context.samples_dir),
            "repairs_dir": str(context.repairs_dir),
            "batch_export_dir": str(batch_export_dir),
        },
        "parameters": _json_safe(batch.run_config),
    }


def _build_source_manifest(batch: Batch) -> dict[str, Any]:
    input_root = batch.input_dir.resolve(strict=False)
    assets = []
    for asset in batch.raw_assets:
        source_path = asset.source_path.resolve(strict=False)
        try:
            relative_path = source_path.relative_to(input_root).as_posix()
        except ValueError:
            relative_path = asset.file_name
        assets.append(
            {
                "asset_id": asset.asset_id,
                "file_name": asset.file_name,
                "relative_path": relative_path,
                "source_path": str(asset.source_path),
                "file_size": asset.file_size,
                "extension": asset.extension,
                "sha256": _sha256_file(asset.source_path),
                "reference_text_source": asset.reference_text_source,
                "language": asset.language,
                "speaker_id": asset.speaker_id,
                "metadata_status": asset.metadata_status,
            }
        )
    return {
        "schema_version": 1,
        "batch_id": batch.batch_id,
        "input_dir": str(batch.input_dir),
        "asset_count": len(assets),
        "assets": assets,
    }


def _build_tool_versions(run_config: dict[str, Any]) -> dict[str, Any]:
    configured = run_config.get("tools", {})
    if not isinstance(configured, dict):
        configured = {}
    ffmpeg_path = configured.get("ffmpeg_path") or run_config.get("ffmpeg_path") or "ffmpeg"
    return {
        "schema_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "configured_tools": {
            "ffmpeg": {"path": str(ffmpeg_path)},
            "vad": run_config.get("vad"),
            "asr": run_config.get("asr"),
            "strategy_agent": run_config.get("strategy_agent") or run_config.get("strategy"),
            "curator_agent": run_config.get("curator_agent") or run_config.get("curator"),
            "report_agent": run_config.get("report_agent") or run_config.get("report"),
        },
        "packages": {
            package: _package_version(package)
            for package in TRACKED_PACKAGES
        },
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
