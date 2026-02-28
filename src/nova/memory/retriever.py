"""Hybrid memory retriever — FTS5 keyword + optional vector search.

Searches NOVA's memory store using a combination of BM25 full-text search
and cosine similarity on embeddings. Falls back to FTS5-only when the
embedding API is unavailable.
"""

import logging
import math
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Hybrid FTS5 + vector search over NOVA's memory store.

    Combines keyword relevance (FTS5 BM25) with semantic similarity
    (vector cosine) using weighted score fusion and time decay.
    """

    VECTOR_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3
    DECAY_HALF_LIFE_DAYS = 30
    TOP_K = 5

    def __init__(
        self,
        memory_store: "MemoryStore",  # noqa: F821
        embedding_fn=None,
    ) -> None:
        """Initialize the retriever.

        Args:
            memory_store: The MemoryStore to search.
            embedding_fn: Optional async fn(text) -> list[float].
                          If None, only FTS5 keyword search is used.
        """
        from nova.memory.memory_store import MemoryStore

        self._store: MemoryStore = memory_store
        self._embedding_fn = embedding_fn

    async def search(self, query: str) -> list[dict]:
        """Search memories using hybrid FTS5 + vector approach.

        Args:
            query: The search query.

        Returns:
            Top K results sorted by final score, each dict containing
            id, key, value, final_score.
        """
        results: dict[int, dict] = {}

        # FTS5 keyword search — skip for very short queries (noise)
        cleaned_query = query.strip()
        if len(cleaned_query) > 1:
            for row in self._store.search_memories_fts(query):
                score = 1.0 / (1.0 + abs(row["rank"]))
                results[row["id"]] = {
                    **row,
                    "keyword_score": score,
                    "vector_score": 0.0,
                }

        # Vector search (if embedding function available)
        if self._embedding_fn:
            try:
                query_vec = await self._embedding_fn(query)
                for row in self._store.get_memories_with_embeddings():
                    sim = self._cosine_similarity(
                        query_vec, row["embedding"],
                    )
                    rid = row["id"]
                    if rid in results:
                        results[rid]["vector_score"] = sim
                    else:
                        results[rid] = {
                            "id": rid,
                            "key": row["key"],
                            "value": row["value"],
                            "updated_at": row["updated_at"],
                            "keyword_score": 0.0,
                            "vector_score": sim,
                        }
            except Exception:
                logger.warning(
                    "Vector search failed, using FTS5 only",
                    exc_info=True,
                )

        # Score fusion + time decay
        now = datetime.now()
        scored = []
        for r in results.values():
            raw = (
                self.VECTOR_WEIGHT * r["vector_score"]
                + self.KEYWORD_WEIGHT * r["keyword_score"]
            )
            updated = r.get("updated_at", now.isoformat())
            age = (now - datetime.fromisoformat(updated)).days
            decay = math.exp(
                -0.693 * age / self.DECAY_HALF_LIFE_DAYS,
            )
            scored.append({**r, "final_score": raw * decay})

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[: self.TOP_K]

    def format_for_prompt(self, results: list[dict]) -> str:
        """Format search results for injection into the system prompt.

        Args:
            results: Search results from search().

        Returns:
            Formatted string, or empty string if no results.
        """
        if not results:
            return ""

        lines = []
        for r in results:
            lines.append(f"- {r['key']}: {r['value']}")
        return "\n".join(lines)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
