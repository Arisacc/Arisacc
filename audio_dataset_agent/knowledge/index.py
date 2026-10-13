"""基于 SQLite 和字符 n-gram 的本地知识索引。"""
import json
import sqlite3
from pathlib import Path

from audio_dataset_agent.knowledge.models import KnowledgeChunk, KnowledgeHit


class SQLiteKnowledgeIndex:
    def __init__(self, index_path: Path | str):
        self.index_path = Path(index_path)

    def rebuild(self, chunks: list[KnowledgeChunk]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.index_path)
        try:
            connection.executescript(
                """
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS chunk_ngrams;
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    section TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_ids TEXT NOT NULL,
                    profiles TEXT NOT NULL,
                    content_hash TEXT NOT NULL
                );
                CREATE TABLE chunk_ngrams (
                    chunk_id TEXT NOT NULL,
                    gram TEXT NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                );
                CREATE INDEX idx_chunk_ngrams_gram ON chunk_ngrams(gram);
                """
            )
            for chunk in chunks:
                connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.source,
                        chunk.title,
                        chunk.section,
                        chunk.content,
                        json.dumps(chunk.tool_ids, ensure_ascii=False),
                        json.dumps(chunk.profiles, ensure_ascii=False),
                        chunk.content_hash,
                    ),
                )
                connection.executemany(
                    "INSERT INTO chunk_ngrams(chunk_id, gram) VALUES (?, ?)",
                    [(chunk.chunk_id, gram) for gram in _ngrams(chunk.title + "\n" + chunk.content)],
                )
            connection.commit()
        finally:
            connection.close()

    def search(
        self,
        query: str,
        tool_ids: list[str] | None = None,
        profile: str | None = None,
        top_k: int = 3,
    ) -> list[KnowledgeHit]:
        if not self.index_path.exists() or not query.strip():
            return []
        grams = sorted(set(_ngrams(query)))
        if not grams:
            return []
        placeholders = ",".join("?" for _ in grams)
        connection = sqlite3.connect(self.index_path)
        try:
            rows = connection.execute(
                f"""
                SELECT c.chunk_id, c.source, c.title, c.section, c.content,
                       c.tool_ids, c.profiles, COUNT(DISTINCT n.gram) AS matched
                FROM chunk_ngrams n
                JOIN chunks c ON c.chunk_id = n.chunk_id
                WHERE n.gram IN ({placeholders})
                GROUP BY c.chunk_id
                ORDER BY matched DESC
                LIMIT ?
                """,
                [*grams, max(top_k * 8, top_k)],
            ).fetchall()
        finally:
            connection.close()
        hits = []
        for row in rows:
            hit_tool_ids = json.loads(row[5])
            hit_profiles = json.loads(row[6])
            if tool_ids and not set(tool_ids).intersection(hit_tool_ids):
                continue
            if profile and profile not in hit_profiles:
                continue
            score = min(1.0, float(row[7]) / len(grams))
            hits.append(
                KnowledgeHit(
                    chunk_id=row[0],
                    source=row[1],
                    title=row[2],
                    section=row[3],
                    content=row[4],
                    score=round(score, 6),
                    tool_ids=hit_tool_ids,
                    profiles=hit_profiles,
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def _ngrams(text: str, size: int = 2) -> list[str]:
    normalized = "".join(
        character.casefold()
        for character in text
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )
    if not normalized:
        return []
    if len(normalized) <= size:
        return [normalized]
    return [normalized[index : index + size] for index in range(len(normalized) - size + 1)]
