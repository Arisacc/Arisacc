"""外部参考文本与说话人元数据导入。"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MetadataImportConfig:
    """描述元数据文件位置和字段映射。"""

    metadata_path: Path | str | None = None
    audio_column: str = "audio_path"
    text_column: str = "text"
    language_column: str = "language"
    speaker_column: str = "speaker_id"
    use_sidecar_text: bool = True


@dataclass(frozen=True)
class AssetMetadata:
    reference_text: str | None = None
    reference_text_source: str | None = None
    language: str | None = None
    speaker_id: str | None = None
    status: str = "missing"
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetadataImportResult:
    by_audio_path: dict[Path, AssetMetadata]
    summary: dict[str, object]


def load_asset_metadata(
    input_dir: Path,
    audio_files: list[Path],
    config: MetadataImportConfig | None = None,
) -> MetadataImportResult:
    """读取表格/JSONL 和同名 TXT，并安全匹配到音频文件。"""
    active_config = config or MetadataImportConfig()
    metadata_path = _resolve_metadata_path(input_dir, active_config.metadata_path)
    records: list[dict[str, str]] = []
    source_type = "none"
    if metadata_path is not None:
        source_type = metadata_path.suffix.lower().lstrip(".")
        records = _read_records(metadata_path, active_config)

    relative_index: dict[str, Path] = {}
    basename_index: dict[str, list[Path]] = {}
    for audio_path in audio_files:
        relative_key = _normalize_audio_key(audio_path.relative_to(input_dir).as_posix())
        relative_index[relative_key] = audio_path
        basename_index.setdefault(audio_path.name.casefold(), []).append(audio_path)

    assigned: dict[Path, AssetMetadata] = {}
    orphan_records = 0
    ambiguous_records = 0
    empty_text_records = 0
    warnings: list[str] = []
    for row_number, record in enumerate(records, start=2):
        audio_key = _normalize_audio_key(record.get(active_config.audio_column, ""))
        matched_path, match_status = _match_audio_path(
            audio_key,
            relative_index,
            basename_index,
        )
        if match_status == "ambiguous":
            ambiguous_records += 1
            warnings.append(f"第 {row_number} 行音频路径不唯一：{audio_key}")
            continue
        if matched_path is None:
            orphan_records += 1
            warnings.append(f"第 {row_number} 行未找到音频：{audio_key or '<empty>'}")
            continue
        if matched_path in assigned:
            ambiguous_records += 1
            warnings.append(f"第 {row_number} 行与前一条记录重复：{audio_key}")
            assigned.pop(matched_path, None)
            continue

        text = str(record.get(active_config.text_column, "") or "").strip()
        status = "matched" if text else "empty_text"
        if not text:
            empty_text_records += 1
        assigned[matched_path] = AssetMetadata(
            reference_text=text or None,
            reference_text_source=source_type,
            language=_optional_text(record.get(active_config.language_column)),
            speaker_id=_optional_text(record.get(active_config.speaker_column)),
            status=status,
        )

    sidecar_matches = 0
    if active_config.use_sidecar_text:
        for audio_path in audio_files:
            if audio_path in assigned:
                continue
            sidecar_path = audio_path.with_suffix(".txt")
            if not sidecar_path.exists():
                continue
            text = _read_text_with_fallback(sidecar_path).strip()
            assigned[audio_path] = AssetMetadata(
                reference_text=text or None,
                reference_text_source="sidecar_txt",
                status="matched" if text else "empty_text",
            )
            if text:
                sidecar_matches += 1
            else:
                empty_text_records += 1

    result_by_path = {
        audio_path: assigned.get(audio_path, AssetMetadata())
        for audio_path in audio_files
    }
    matched_assets = sum(
        1 for metadata in result_by_path.values() if metadata.status == "matched"
    )
    missing_assets = sum(
        1 for metadata in result_by_path.values() if metadata.status == "missing"
    )
    if source_type == "none" and sidecar_matches:
        source_type = "sidecar_txt"
    return MetadataImportResult(
        by_audio_path=result_by_path,
        summary={
            "source_type": source_type,
            "metadata_path": str(metadata_path) if metadata_path else None,
            "total_assets": len(audio_files),
            "matched_assets": matched_assets,
            "missing_assets": missing_assets,
            "empty_text_assets": sum(
                1 for metadata in result_by_path.values() if metadata.status == "empty_text"
            ),
            "orphan_records": orphan_records,
            "ambiguous_records": ambiguous_records,
            "warnings": warnings,
        },
    )


def _resolve_metadata_path(
    input_dir: Path,
    configured_path: Path | str | None,
) -> Path | None:
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            # 优先兼容“相对于音频目录”的旧约定，再支持 CLI 常见的项目根目录相对路径。
            input_relative_path = input_dir / path
            project_relative_path = Path.cwd() / path
            path = (
                input_relative_path
                if input_relative_path.exists()
                else project_relative_path
            )
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"元数据文件不存在：{path}")
        if path.suffix.lower() not in {".csv", ".jsonl"}:
            raise ValueError("元数据文件只支持 CSV 或 JSONL")
        return path

    candidates = [
        path
        for path in (
            input_dir / "metadata.csv",
            input_dir / "metadata.jsonl",
            input_dir / "manifest.jsonl",
        )
        if path.exists()
    ]
    if len(candidates) > 1:
        raise ValueError("检测到多个元数据文件，请明确指定 metadata_path")
    return candidates[0] if candidates else None


def _read_records(path: Path, config: MetadataImportConfig) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        text = _read_text_with_fallback(path)
        reader = csv.DictReader(text.splitlines())
        fieldnames = set(reader.fieldnames or [])
        required = {config.audio_column, config.text_column}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"CSV 缺少必要列：{', '.join(sorted(missing))}")
        return [dict(row) for row in reader]

    records = []
    for line_number, line in enumerate(
        _read_text_with_fallback(path).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL 第 {line_number} 行不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 第 {line_number} 行必须是对象")
        records.append({str(key): value for key, value in value.items()})
    return records


def _match_audio_path(
    audio_key: str,
    relative_index: dict[str, Path],
    basename_index: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    if not audio_key:
        return None, "missing"
    if audio_key in relative_index:
        return relative_index[audio_key], "matched"
    candidates = basename_index.get(Path(audio_key).name.casefold(), [])
    if len(candidates) == 1:
        return candidates[0], "matched"
    if len(candidates) > 1:
        return None, "ambiguous"
    return None, "missing"


def _normalize_audio_key(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").removeprefix("./").casefold()


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _read_text_with_fallback(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别文本编码：{path}")
