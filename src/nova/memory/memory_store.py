"""SQLite-backed memory store — replaces flat JSON with searchable database.

Stores long-term facts (memories), interaction logs, and session tracking
in ~/.nova/memory/nova.db. Provides FTS5 full-text search and optional
vector embeddings for hybrid retrieval.
"""

import asyncio
import json
import logging
import re
import sqlite3
import struct
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_MEMORY_DIR = Path.home() / ".nova" / "memory"
_DB_PATH = _MEMORY_DIR / "nova.db"
_LEGACY_JSON = Path.home() / ".nova" / "memory.json"
_MEMORY_MD = _MEMORY_DIR / "MEMORY.md"

# Module-level singleton
_instance: "MemoryStore | None" = None


def _sanitize_fts_query(query: str) -> str:
    """Remove FTS5 special characters, keep words only.

    FTS5 MATCH syntax treats characters like ?, !, ", *, ( ) as
    operators, causing syntax errors on raw user input.

    Args:
        query: Raw user query string.

    Returns:
        Cleaned query safe for FTS5 MATCH, or empty string.
    """
    # Remove everything except letters, numbers, spaces
    cleaned = re.sub(r"[^\w\s]", " ", query)
    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# --- Schema ---

_SCHEMA = """
-- Long-term memory facts (replaces memory.json)
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT DEFAULT 'user',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    expires_at  TEXT,
    embedding   BLOB
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_key ON memories(key);

-- Daily interaction log
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    tool_calls  TEXT,
    tokens_est  INTEGER,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interactions_date ON interactions(date);

-- Session tracking
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    summary     TEXT,
    token_count INTEGER DEFAULT 0
);
"""

# FTS5 tables created separately (can't use IF NOT EXISTS with virtual tables)
_FTS_SCHEMA = [
    """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        key, value,
        content=memories,
        content_rowid=id,
        tokenize='unicode61 remove_diacritics 2'
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS interactions_fts USING fts5(
        content,
        content=interactions,
        content_rowid=id,
        tokenize='unicode61 remove_diacritics 2'
    )""",
]

# Triggers to keep FTS5 in sync with content tables
_FTS_TRIGGERS = [
    # memories → memories_fts
    """CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories
    BEGIN
        INSERT INTO memories_fts(rowid, key, value)
        VALUES (new.id, new.key, new.value);
    END""",
    """CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories
    BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, key, value)
        VALUES('delete', old.id, old.key, old.value);
    END""",
    """CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
    BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, key, value)
        VALUES('delete', old.id, old.key, old.value);
        INSERT INTO memories_fts(rowid, key, value)
        VALUES (new.id, new.key, new.value);
    END""",
    # interactions → interactions_fts
    """CREATE TRIGGER IF NOT EXISTS interactions_ai
    AFTER INSERT ON interactions
    BEGIN
        INSERT INTO interactions_fts(rowid, content)
        VALUES (new.id, new.content);
    END""",
    """CREATE TRIGGER IF NOT EXISTS interactions_ad
    AFTER DELETE ON interactions
    BEGIN
        INSERT INTO interactions_fts(interactions_fts, rowid, content)
        VALUES('delete', old.id, old.content);
    END""",
]


class MemoryStore:
    """SQLite-backed memory store with FTS5 search.

    Provides CRUD for long-term facts, interaction logging, and
    session tracking. Automatically migrates from legacy memory.json.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the memory store.

        Args:
            db_path: Path to the SQLite database file.
                     Defaults to ~/.nova/memory/nova.db.
        """
        if db_path is None:
            self._db_path = _DB_PATH
        else:
            self._db_path = Path(db_path)

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Optional async embedding function: async fn(text) -> list[float] | None
        self._embedding_fn = None

        self._init_schema()
        self._migrate_legacy_json()

    def _init_schema(self) -> None:
        """Create tables, indexes, FTS5 virtual tables, and triggers."""
        self._conn.executescript(_SCHEMA)
        for stmt in _FTS_SCHEMA:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Already exists
        for stmt in _FTS_TRIGGERS:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Already exists
        self._conn.commit()
        logger.info("Memory store initialized: %s", self._db_path)

    def _migrate_legacy_json(self) -> None:
        """Migrate facts from ~/.nova/memory.json to SQLite if present."""
        json_path = _LEGACY_JSON
        if self._db_path != _DB_PATH:
            return  # Don't migrate in test mode
        if not json_path.exists():
            return

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return

            count = self._conn.execute(
                "SELECT COUNT(*) FROM memories",
            ).fetchone()[0]
            if count > 0:
                logger.debug("Memory DB already has data, skipping migration")
                return

            now = datetime.now().isoformat()
            for key, value in data.items():
                self._conn.execute(
                    "INSERT OR IGNORE INTO memories "
                    "(key, value, source, created_at, updated_at) "
                    "VALUES (?, ?, 'migrated', ?, ?)",
                    (str(key).strip().lower(), str(value).strip(), now, now),
                )
            self._conn.commit()

            # Rename legacy file
            backup = json_path.with_suffix(".json.bak")
            json_path.rename(backup)
            logger.info(
                "Migrated %d facts from memory.json → nova.db (backup: %s)",
                len(data), backup,
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Legacy JSON migration failed: %s", e)

    # --- Memory CRUD ---

    def set_embedding_fn(self, fn) -> None:
        """Set the async embedding function for auto-embedding.

        Args:
            fn: Async callable (text) -> list[float] | None.
        """
        self._embedding_fn = fn

    def store_memory(
        self, key: str, value: str, source: str = "user",
    ) -> None:
        """Store or update a fact in the memory database.

        If an embedding function is set, schedules background embedding
        generation for the stored value.

        Args:
            key: Fact identifier (e.g. "name", "hobby").
            value: Fact value.
            source: Origin — "user", "auto", or "system".
        """
        key = key.strip().lower()
        value = value.strip()
        now = datetime.now().isoformat()

        existing = self._conn.execute(
            "SELECT id FROM memories WHERE key = ?", (key,),
        ).fetchone()

        if existing:
            self._conn.execute(
                "UPDATE memories SET value=?, source=?, updated_at=? WHERE key=?",
                (value, source, now, key),
            )
        else:
            self._conn.execute(
                "INSERT INTO memories (key, value, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, value, source, now, now),
            )
        self._conn.commit()
        self._sync_memory_md()
        logger.info("Memory stored: %s=%s (source=%s)", key, value, source)

        # Schedule async embedding if available
        if self._embedding_fn is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._embed_memory(key, value))
            except RuntimeError:
                pass  # No running loop — skip embedding

    async def _embed_memory(self, key: str, value: str) -> None:
        """Generate and store embedding for a memory value."""
        try:
            vec = await self._embedding_fn(value)
            if vec is not None:
                blob = struct.pack(f"{len(vec)}f", *vec)
                self._conn.execute(
                    "UPDATE memories SET embedding = ? WHERE key = ?",
                    (blob, key),
                )
                self._conn.commit()
                logger.info("Embedded memory: %s (%d dimensions)", key, len(vec))
        except Exception:
            logger.warning("Failed to embed memory %s", key, exc_info=True)

    def get_memory(self, key: str) -> str | None:
        """Get a single memory by key.

        Args:
            key: The fact key to look up.

        Returns:
            The fact value, or None if not found.
        """
        row = self._conn.execute(
            "SELECT value FROM memories WHERE key = ?",
            (key.strip().lower(),),
        ).fetchone()
        return row["value"] if row else None

    def get_all_memories(self) -> dict[str, str]:
        """Get all stored memories as a dict.

        Returns:
            Dict of key-value fact pairs.
        """
        rows = self._conn.execute(
            "SELECT key, value FROM memories ORDER BY updated_at DESC",
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def delete_memory(self, key: str) -> bool:
        """Delete a memory by key.

        Args:
            key: The fact key to remove.

        Returns:
            True if the memory existed and was removed.
        """
        key = key.strip().lower()
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE key = ?", (key,),
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            self._sync_memory_md()
            logger.info("Memory deleted: %s", key)
            return True
        return False

    def search_memories_fts(self, query: str, limit: int = 20) -> list[dict]:
        """Search memories using FTS5 full-text search.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of dicts with id, key, value, updated_at, rank.
        """
        cleaned = _sanitize_fts_query(query)
        if not cleaned:
            return []

        try:
            rows = self._conn.execute(
                """
                SELECT m.id, m.key, m.value, m.updated_at, f.rank
                FROM memories_fts f JOIN memories m ON m.id = f.rowid
                WHERE memories_fts MATCH ? ORDER BY f.rank LIMIT ?
                """,
                (cleaned, limit),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "key": r["key"],
                    "value": r["value"],
                    "updated_at": r["updated_at"],
                    "rank": r["rank"],
                }
                for r in rows
            ]
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 search failed: %s", e)
            return []

    def memory_count(self) -> int:
        """Return the total number of stored memories."""
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # --- Interaction Logging ---

    def log_interaction(
        self,
        role: str,
        content: str,
        tool_calls: list | None = None,
    ) -> int:
        """Log a conversation turn to the interactions table.

        Args:
            role: "user" or "assistant".
            content: Message content.
            tool_calls: Optional list of tool call dicts.

        Returns:
            The row ID of the inserted interaction.
        """
        now = datetime.now()
        cursor = self._conn.execute(
            "INSERT INTO interactions (date, role, content, tool_calls, tokens_est, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                now.strftime("%Y-%m-%d"),
                role,
                content,
                json.dumps(tool_calls) if tool_calls else None,
                len(content) // 4,
                now.isoformat(),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_recent_interactions(
        self, date: str | None = None, limit: int = 10,
    ) -> list[dict]:
        """Get the most recent interactions, optionally filtered by date.

        Args:
            date: Date filter (YYYY-MM-DD). Defaults to today.
            limit: Max number of interactions to return.

        Returns:
            List of dicts with role and content, in chronological order.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        rows = self._conn.execute(
            "SELECT role, content FROM interactions "
            "WHERE date = ? ORDER BY id DESC LIMIT ?",
            (date, limit),
        ).fetchall()
        return [
            {"role": r["role"], "content": r["content"]}
            for r in reversed(rows)
        ]

    def search_interactions_fts(
        self, query: str, limit: int = 20,
    ) -> list[dict]:
        """Search interactions using FTS5.

        Args:
            query: Search query string.
            limit: Maximum results.

        Returns:
            List of dicts with id, date, role, content, rank.
        """
        cleaned = _sanitize_fts_query(query)
        if not cleaned:
            return []

        try:
            rows = self._conn.execute(
                """
                SELECT i.id, i.date, i.role, i.content, i.created_at, f.rank
                FROM interactions_fts f JOIN interactions i ON i.id = f.rowid
                WHERE interactions_fts MATCH ? ORDER BY f.rank LIMIT ?
                """,
                (cleaned, limit),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "date": r["date"],
                    "role": r["role"],
                    "content": r["content"],
                    "created_at": r["created_at"],
                    "rank": r["rank"],
                }
                for r in rows
            ]
        except sqlite3.OperationalError as e:
            logger.warning("Interaction FTS5 search failed: %s", e)
            return []

    # --- Sessions ---

    def start_session(self) -> int:
        """Start a new session and return its ID."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?)", (now,),
        )
        self._conn.commit()
        return cursor.lastrowid

    def end_session(
        self, session_id: int, summary: str = "", token_count: int = 0,
    ) -> None:
        """End a session with optional summary."""
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE sessions SET ended_at=?, summary=?, token_count=? WHERE id=?",
            (now, summary, token_count, session_id),
        )
        self._conn.commit()

    # --- Embedding support ---

    def store_embedding(self, key: str, embedding: list[float]) -> None:
        """Store an embedding vector for a memory.

        Args:
            key: Memory key to attach the embedding to.
            embedding: List of float values (e.g. 768-dim vector).
        """
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        self._conn.execute(
            "UPDATE memories SET embedding = ? WHERE key = ?",
            (blob, key.strip().lower()),
        )
        self._conn.commit()

    def get_memories_with_embeddings(self) -> list[dict]:
        """Get all memories that have embeddings stored.

        Returns:
            List of dicts with id, key, value, updated_at, embedding (as list[float]).
        """
        rows = self._conn.execute(
            "SELECT id, key, value, updated_at, embedding "
            "FROM memories WHERE embedding IS NOT NULL",
        ).fetchall()
        results = []
        for row in rows:
            emb_blob = row["embedding"]
            emb = list(struct.unpack(f"{len(emb_blob) // 4}f", emb_blob))
            results.append({
                "id": row["id"],
                "key": row["key"],
                "value": row["value"],
                "updated_at": row["updated_at"],
                "embedding": emb,
            })
        return results

    async def backfill_embeddings(
        self, embedding_fn=None,
    ) -> int:
        """Embed all memories that don't have embeddings yet.

        Args:
            embedding_fn: Async callable (text) -> list[float] | None.
                          Falls back to self._embedding_fn if not provided.

        Returns:
            Number of memories successfully embedded.
        """
        fn = embedding_fn or self._embedding_fn
        if fn is None:
            return 0

        rows = self._conn.execute(
            "SELECT key, value FROM memories WHERE embedding IS NULL",
        ).fetchall()

        if not rows:
            return 0

        count = 0
        for row in rows:
            try:
                vec = await fn(row["value"])
                if vec is not None:
                    blob = struct.pack(f"{len(vec)}f", *vec)
                    self._conn.execute(
                        "UPDATE memories SET embedding = ? WHERE key = ?",
                        (blob, row["key"]),
                    )
                    self._conn.commit()
                    count += 1
            except Exception:
                logger.warning(
                    "Backfill failed for %s", row["key"], exc_info=True,
                )

        logger.info("Backfilled %d memories with embeddings", count)
        return count

    # --- MEMORY.md sync ---

    def _sync_memory_md(self) -> None:
        """Write a human-readable mirror of all memories to MEMORY.md."""
        try:
            memories = self.get_all_memories()
            if not memories:
                return

            lines = ["# NOVA Memory\n", ""]
            for key, value in memories.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

            md_path = self._db_path.parent / "MEMORY.md"
            md_path.write_text("\n".join(lines), encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to sync MEMORY.md: %s", e)

    # --- Cleanup ---

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


def get_memory_store() -> MemoryStore:
    """Get the singleton MemoryStore instance.

    Returns:
        The shared MemoryStore.
    """
    global _instance
    if _instance is None:
        _instance = MemoryStore()
    return _instance


def reset_memory_store() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    if _instance is not None:
        _instance.close()
    _instance = None
