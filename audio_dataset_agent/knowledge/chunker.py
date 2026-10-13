"""知识文档加载和按标题分段。"""
import hashlib
import re
from pathlib import Path

from audio_dataset_agent.knowledge.models import KnowledgeChunk


SUPPORTED_DOCUMENT_EXTENSIONS = {".md", ".txt"}


def load_knowledge_chunks(documents_root: Path | str) -> list[KnowledgeChunk]:
    root = Path(documents_root)
    if not root.exists():
        return []
    chunks: list[KnowledgeChunk] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS),
        key=lambda item: item.relative_to(root).as_posix().casefold(),
    ):
        chunks.extend(_chunk_file(root, path))
    return chunks


def _chunk_file(root: Path, path: Path) -> list[KnowledgeChunk]:
    raw_text = path.read_text(encoding="utf-8-sig")
    metadata, body = _parse_front_matter(raw_text)
    relative_source = path.relative_to(root).as_posix()
    title = metadata.get("title") or path.stem
    sections = _split_sections(body, title)
    chunks: list[KnowledgeChunk] = []
    for section_index, (section, content) in enumerate(sections, start=1):
        for part_index, part in enumerate(_split_long_content(content), start=1):
            if not part.strip():
                continue
            identity = f"{relative_source}:{section}:{section_index}:{part_index}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16],
                    source=relative_source,
                    title=title,
                    section=section,
                    content=part.strip(),
                    tool_ids=_as_list(metadata.get("tool_ids")),
                    profiles=_as_list(metadata.get("profiles")),
                    content_hash=hashlib.sha256(part.strip().encode("utf-8")).hexdigest(),
                )
            )
    return chunks


def _parse_front_matter(text: str) -> tuple[dict[str, str | list[str]], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    try:
        end_index = lines.index("---", 1)
    except ValueError:
        return {}, text
    metadata: dict[str, str | list[str]] = {}
    for line in lines[1:end_index]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("[]").strip('"').strip("'")
    return metadata, "\n".join(lines[end_index + 1 :])


def _split_sections(text: str, default_title: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = default_title
    current_lines: list[str] = []
    for line in text.splitlines():
        heading = re.match(r"^#{1,3}\s+(.+?)\s*$", line)
        if heading:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = heading.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines)))
    return sections


def _split_long_content(content: str, max_chars: int = 900) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    parts: list[str] = []
    current = ""
    for paragraph in paragraphs or [content]:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            parts.append(current)
            current = ""
        if len(paragraph) <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        for start in range(0, len(paragraph), max_chars):
            piece = paragraph[start : start + max_chars]
            if current:
                parts.append(current)
                current = ""
            parts.append(piece)
    if current:
        parts.append(current)
    return parts


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]
