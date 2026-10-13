"""本地知识库检索入口。"""
import json
from datetime import datetime, timezone
from pathlib import Path

from audio_dataset_agent.knowledge.chunker import load_knowledge_chunks
from audio_dataset_agent.knowledge.index import SQLiteKnowledgeIndex
from audio_dataset_agent.knowledge.models import KnowledgeHit


class LocalKnowledgeRetriever:
    def __init__(
        self,
        documents_root: Path | str,
        index_path: Path | str,
        auto_build: bool = True,
        audit_records: list[dict[str, object]] | None = None,
    ) -> None:
        self.documents_root = Path(documents_root)
        self.index = SQLiteKnowledgeIndex(index_path)
        self.audit_records = audit_records if audit_records is not None else []
        if auto_build and not self.index.index_path.exists():
            self.rebuild()

    def rebuild(self) -> int:
        chunks = load_knowledge_chunks(self.documents_root)
        self.index.rebuild(chunks)
        return len(chunks)

    def search(
        self,
        query: str,
        tool_ids: list[str] | None = None,
        profile: str | None = None,
        top_k: int = 3,
        agent_name: str = "unknown",
        sample_id: str | None = None,
    ) -> list[KnowledgeHit]:
        if not self.index.index_path.exists():
            self.rebuild()
        hits = self.index.search(query, tool_ids=tool_ids, profile=profile, top_k=top_k)
        self.audit_records.append(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "agent_name": agent_name,
                "sample_id": sample_id,
                "query": query,
                "profile": profile,
                "tool_ids": list(tool_ids or []),
                "hits": [
                    {
                        "chunk_id": hit.chunk_id,
                        "source": hit.source,
                        "section": hit.section,
                        "score": hit.score,
                    }
                    for hit in hits
                ],
            }
        )
        return hits

    def write_audit_files(
        self,
        export_dir: Path | str,
        append: bool = False,
    ) -> dict[str, str]:
        """将检索记录写入批次状态目录，不保存完整知识片段。"""
        state_dir = Path(export_dir) / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        queries_path = state_dir / "rag_queries.jsonl"
        retrievals_path = state_dir / "rag_retrievals.jsonl"
        mode = "a" if append else "w"
        with queries_path.open(mode, encoding="utf-8") as queries, retrievals_path.open(
            mode, encoding="utf-8"
        ) as retrievals:
            for record in self.audit_records:
                query_record = {
                    key: record.get(key)
                    for key in ("created_at", "agent_name", "sample_id", "query", "profile")
                }
                queries.write(json.dumps(query_record, ensure_ascii=False) + "\n")
                retrievals.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"rag_queries": str(queries_path), "rag_retrievals": str(retrievals_path)}
