"""知识片段、检索命中和 Prompt 上下文的数据结构。"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    title: str
    section: str
    content: str
    tool_ids: list[str] = field(default_factory=list)
    profiles: list[str] = field(default_factory=list)
    content_hash: str = ""


@dataclass(frozen=True)
class KnowledgeHit:
    chunk_id: str
    source: str
    title: str
    section: str
    content: str
    score: float
    tool_ids: list[str] = field(default_factory=list)
    profiles: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KnowledgeContext:
    text: str
    references: list[dict[str, object]] = field(default_factory=list)
