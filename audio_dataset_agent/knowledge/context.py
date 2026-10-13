"""将检索命中转换成受长度限制且带来源的 Prompt 上下文。"""
from audio_dataset_agent.knowledge.models import KnowledgeContext, KnowledgeHit


def build_knowledge_context(
    hits: list[KnowledgeHit],
    max_chars: int = 4000,
) -> KnowledgeContext:
    parts: list[str] = []
    references: list[dict[str, object]] = []
    used_sources: set[str] = set()
    for hit in hits:
        header = f"[来源：{hit.source} / {hit.section} / chunk_id={hit.chunk_id}]"
        part = f"{header}\n{hit.content}"
        remaining = max_chars - sum(len(item) + 2 for item in parts)
        if remaining <= 0:
            break
        parts.append(part[:remaining])
        if hit.source not in used_sources:
            references.append(
                {
                    "source": hit.source,
                    "title": hit.title,
                    "section": hit.section,
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                }
            )
            used_sources.add(hit.source)
    return KnowledgeContext(text="\n\n".join(parts)[:max_chars], references=references)
