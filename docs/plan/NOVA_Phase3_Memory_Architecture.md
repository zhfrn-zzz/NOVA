# NOVA Phase 3 — Memory & Prompt Architecture

## Design Philosophy

Terinspirasi oleh OpenClaw (file-first memory), ZeroClaw/NullClaw (SQLite hybrid search pada hardware minimal), diadaptasi untuk voice assistant yang jalan 24/7.

**Prinsip utama:**
- Files are the source of truth — semua memory bisa dibaca manusia
- Only load what's relevant — jangan dump semua ke prompt
- Survive restarts — conversation context persist across sessions
- Graceful degradation — kalau embedding API down, fallback ke keyword search
- Voice-optimized — memory retrieval harus <100ms, tidak boleh tambah latency

---

## Current State (Phase 2)

```
System Prompt: hardcoded string di gemini.py (~800 tokens)
Memory:        ~/.nova/memory.json (flat key-value, 2 facts)
History:       ConversationContext sliding window 10 turns (in-memory, lost on restart)
Token usage:   ~500-1500 tokens/request (prompt + history + response)
Problem:       24/7 operation → history accumulates → token usage climbs → API limit hit
```

---

## Target Architecture (Phase 3)

```
~/.nova/
├── prompts/
│   ├── SOUL.md              # Personality (JARVIS character)
│   ├── RULES.md             # Response rules, tool usage rules
│   └── USER.md              # User profile (Zhafran, preferences)
├── memory/
│   ├── nova.db              # SQLite: FTS5 index + vector BLOBs + conversation log
│   ├── MEMORY.md            # Curated long-term facts (human-readable mirror)
│   └── daily/
│       ├── 2026-02-26.md    # Daily interaction log
│       ├── 2026-02-27.md
│       └── ...
├── memory.json              # [DEPRECATED] migrated to nova.db
├── tts_usage.json
├── google-cloud-tts-key.json
└── logs/
```

---

## Component 1: File-Based Prompt System

### Prompt Files

**`~/.nova/prompts/SOUL.md`** — Personality & character
```markdown
You are NOVA, a personal AI assistant created by Zhafran.
You are modeled after JARVIS — calm, composed, and quietly competent.
You speak with a refined, slightly formal tone but never stiff or robotic.

Address the user as "Sir" or "Pak" naturally.
Be efficient and precise — deliver information, not filler.
Show quiet confidence. State things, don't hedge.
Light sarcasm is acceptable when appropriate, but always respectful.
You are loyal and proactive.
```

**`~/.nova/prompts/RULES.md`** — Behavioral rules
```markdown
Response rules:
- Keep responses between 20-50 words unless user asks for detail.
- Responses will be spoken aloud — plain text only.
- No markdown, bullet points, asterisks, emoji, exclamation marks.
- Default to Indonesian unless user speaks English.
- Never start with "Tentu" or "Baik" — answer or act directly.

Tool usage:
- Use tools immediately. Don't ask confirmation unless destructive.
- Only call web_search once per question.
- Answer from search results directly, never say "saya menemukan hasil."
```

**`~/.nova/prompts/USER.md`** — User context
```markdown
Name: Zhafran
Location: Bekasi, Indonesia
Occupation: 11th grade vocational student, Computer Network Engineering
Currently interning at Wantimpres
Interests: AI, men's fashion (old money/Victorian), fragrances
```

### Prompt Assembly

```python
class PromptAssembler:
    """Assembles system prompt from file components + dynamic context."""

    PROMPT_DIR = Path("~/.nova/prompts").expanduser()

    def build(self, memory_context: str = "", datetime_str: str = "") -> str:
        """Assemble final system prompt. Called every LLM request."""
        sections = []

        # 1. Core identity (SOUL.md) — always loaded
        sections.append(self._read("SOUL.md"))

        # 2. Rules (RULES.md) — always loaded
        sections.append(self._read("RULES.md"))

        # 3. User profile (USER.md) — always loaded
        user = self._read("USER.md")
        if user:
            sections.append(f"About the user:\n{user}")

        # 4. Current datetime — injected every call
        if datetime_str:
            sections.append(f"Current date and time: {datetime_str}")

        # 5. Relevant memories — from hybrid search, NOT full dump
        if memory_context:
            sections.append(f"Relevant memories:\n{memory_context}")

        return "\n\n".join(filter(None, sections))

    def _read(self, filename: str) -> str:
        path = self.PROMPT_DIR / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""
```

**Keuntungan vs current hardcoded string:**
- Edit personality → edit file → langsung aktif, no restart
- Token budget visible: `wc -w ~/.nova/prompts/*.md`
- Version control: `git diff` untuk track perubahan prompt
- USER.md bisa di-update oleh NOVA sendiri (via tool)

---

## Component 2: SQLite Memory Store

### Schema

```sql
-- Long-term memory facts (replaces memory.json)
CREATE TABLE memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT DEFAULT 'user',     -- 'user' | 'auto' | 'system'
    created_at  TEXT NOT NULL,           -- ISO 8601
    updated_at  TEXT NOT NULL,
    expires_at  TEXT,                    -- optional TTL for temporary facts
    embedding   BLOB                    -- float32 vector, nullable
);

CREATE UNIQUE INDEX idx_memories_key ON memories(key);

-- Daily interaction log (for retrieval and compaction)
CREATE TABLE interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,           -- YYYY-MM-DD
    role        TEXT NOT NULL,           -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    tool_calls  TEXT,                    -- JSON, nullable
    tokens_est  INTEGER,
    created_at  TEXT NOT NULL
);

CREATE INDEX idx_interactions_date ON interactions(date);

-- FTS5 for keyword search over memories
CREATE VIRTUAL TABLE memories_fts USING fts5(
    key, value,
    content=memories,
    content_rowid=id,
    tokenize='unicode61 remove_diacritics 2'
);

-- FTS5 for interaction search
CREATE VIRTUAL TABLE interactions_fts USING fts5(
    content,
    content=interactions,
    content_rowid=id,
    tokenize='unicode61 remove_diacritics 2'
);

-- Session tracking
CREATE TABLE sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    summary     TEXT,
    token_count INTEGER DEFAULT 0
);
```

### Why SQLite, not JSON?

| | JSON (current) | SQLite (target) |
|--|--|--|
| Search | Linear scan all entries | FTS5 BM25 + vector cosine |
| Scale | ~20 facts before prompt explosion | 10,000+ facts |
| Crash safety | Can corrupt on power loss | ACID transactions |
| Dependencies | None | Python built-in `sqlite3` |
| Backup | `cp memory.json` | `cp nova.db` |
| Human readable | Direct (it IS the file) | Mirror to MEMORY.md |

---

## Component 3: Hybrid Memory Retrieval

### Architecture

```
User says: "jadwal saya besok gimana?"
                    │
                    ▼
        ┌───────────────────┐
        │  Query Analyzer   │
        │  "jadwal besok"   │
        └────────┬──────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  FTS5 Search │  │Vector Search │
│  (BM25)      │  │  (Cosine)    │
│  weight: 0.3 │  │  weight: 0.7 │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
     ┌─────────────────┐
     │  Score Fusion   │
     │  + Time Decay   │
     │  + Dedup        │
     └────────┬────────┘
              ▼
     Top 5 results → injected into prompt
```

### Implementation

```python
import sqlite3, struct, math
from datetime import datetime


class MemoryRetriever:
    """Hybrid FTS5 + vector search over NOVA's memory store."""

    VECTOR_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3
    DECAY_HALF_LIFE_DAYS = 30
    TOP_K = 5

    def __init__(self, db_path: str, embedding_fn=None):
        self.db = sqlite3.connect(db_path)
        self.embedding_fn = embedding_fn  # async fn(text) -> list[float]

    async def search(self, query: str) -> list[dict]:
        results = {}

        # FTS5 keyword search
        for row in self._fts5_search(query):
            score = 1.0 / (1.0 + abs(row["rank"]))
            results[row["id"]] = {**row, "keyword_score": score, "vector_score": 0.0}

        # Vector search (if embedding available)
        if self.embedding_fn:
            query_vec = await self.embedding_fn(query)
            for row in self._vector_search(query_vec):
                rid = row["id"]
                if rid in results:
                    results[rid]["vector_score"] = row["cosine_sim"]
                else:
                    results[rid] = {**row, "keyword_score": 0.0, "vector_score": row["cosine_sim"]}

        # Score fusion + time decay
        now = datetime.now()
        scored = []
        for r in results.values():
            raw = self.VECTOR_WEIGHT * r["vector_score"] + self.KEYWORD_WEIGHT * r["keyword_score"]
            age = (now - datetime.fromisoformat(r["updated_at"])).days
            decay = math.exp(-0.693 * age / self.DECAY_HALF_LIFE_DAYS)
            scored.append({**r, "final_score": raw * decay})

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:self.TOP_K]

    def _fts5_search(self, query: str) -> list[dict]:
        cursor = self.db.execute("""
            SELECT m.id, m.key, m.value, m.updated_at, f.rank
            FROM memories_fts f JOIN memories m ON m.id = f.rowid
            WHERE memories_fts MATCH ? ORDER BY f.rank LIMIT 20
        """, (query,))
        return [{"id": r[0], "key": r[1], "value": r[2], "updated_at": r[3], "rank": r[4]}
                for r in cursor.fetchall()]

    def _vector_search(self, query_vec: list[float]) -> list[dict]:
        cursor = self.db.execute(
            "SELECT id, key, value, updated_at, embedding FROM memories WHERE embedding IS NOT NULL"
        )
        results = []
        for row in cursor.fetchall():
            stored = struct.unpack(f"{len(row[4])//4}f", row[4])
            dot = sum(a*b for a, b in zip(query_vec, stored))
            na = math.sqrt(sum(a*a for a in query_vec))
            nb = math.sqrt(sum(b*b for b in stored))
            sim = dot / (na * nb) if na and nb else 0.0
            results.append({"id": row[0], "key": row[1], "value": row[2],
                            "updated_at": row[3], "cosine_sim": sim})
        results.sort(key=lambda x: x["cosine_sim"], reverse=True)
        return results[:20]
```

### Embedding Strategy

```
Primary:   Gemini Embedding API (free tier, 1500 RPD)
           model: text-embedding-004, 768 dimensions

Fallback:  FTS5-only (zero API cost, zero latency)

Logic:     API available → embed + store BLOB → hybrid search
           API down      → pure FTS5 keyword search
           Same circuit breaker as TTS quota
```

Storage: 768 floats × 4 bytes = 3KB per memory. 10K memories = 30MB.

---

## Component 4: Conversation Window Management

### The Problem

```
Turn 1:   system(800) + user(50) + assistant(100) = 950 tokens
Turn 50:  system(800) + 50×150 = 8,300 tokens
Turn 200: system(800) + 200×150 = 30,800 tokens   ← approaches limit

24h @ 5/hr = 120 turns → ~18,800 tokens per request
```

### Solution: Sliding Window + Auto-Compaction

```
┌────────────────────────────────────────────────┐
│  [Turn 1] [Turn 2] ... [Turn 20]  ← Window    │
│                                                │
│  When window full (20 turns):                  │
│    1. LLM summarizes turns 1-15                │
│    2. Summary → daily log + interactions table │
│    3. Auto-extract facts → memories table      │
│    4. Drop turns 1-15, keep 16-20              │
│    5. Inject summary as context bridge         │
│                                                │
│  On restart:                                   │
│    1. Load today's daily log summary           │
│    2. Load last 5 turns from DB                │
│    3. Resume seamlessly                        │
└────────────────────────────────────────────────┘
```

### Implementation

```python
class ConversationManager:
    MAX_TURNS = 20
    COMPACT_AT = 15       # compact oldest 15, keep newest 5
    KEEP_RECENT = 5

    def __init__(self, db, llm_fn):
        self.db = db
        self.llm_fn = llm_fn
        self.history = []
        self._load_recent()

    def _load_recent(self):
        """On startup: load last N turns from today."""
        today = datetime.now().strftime("%Y-%m-%d")
        rows = self.db.execute(
            "SELECT role, content FROM interactions WHERE date=? ORDER BY id DESC LIMIT ?",
            (today, self.KEEP_RECENT)
        ).fetchall()
        self.history = [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    async def add_turn(self, role, content, tool_calls=None):
        self.history.append({"role": role, "content": content})
        now = datetime.now()
        self.db.execute(
            "INSERT INTO interactions (date,role,content,tool_calls,tokens_est,created_at) VALUES (?,?,?,?,?,?)",
            (now.strftime("%Y-%m-%d"), role, content,
             json.dumps(tool_calls) if tool_calls else None,
             len(content)//4, now.isoformat())
        )
        self.db.commit()

        if len(self.history) >= self.MAX_TURNS:
            await self._compact()

    async def _compact(self):
        old = self.history[:self.COMPACT_AT]
        recent = self.history[self.COMPACT_AT:]

        text = "\n".join(f"{t['role']}: {t['content']}" for t in old)

        # Summarize
        summary = await self.llm_fn(
            f"Summarize this conversation concisely. Key facts, decisions, requests. "
            f"Indonesian. Max 100 words.\n\n{text}"
        )
        self._append_daily_log(summary)

        # Auto-extract facts
        await self._extract_facts(text)

        # Replace history
        self.history = [
            {"role": "system", "content": f"[Ringkasan percakapan sebelumnya: {summary}]"},
            *recent
        ]

    async def _extract_facts(self, text):
        result = await self.llm_fn(
            "Extract important long-term facts about the user from this conversation. "
            "key: value format, one per line. If nothing worth remembering, output NONE.\n\n" + text
        )
        if result.strip().upper() != "NONE":
            for line in result.strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    self._upsert_memory(k.strip(), v.strip(), "auto")
```

---

## Component 5: Token Budget

### Per-Request Allocation

```
Total budget: ~8,000 tokens (conservative for speed)

  System prompt (SOUL + RULES + USER + datetime):  ~600 tok (fixed)
  Retrieved memories (top 5 × ~50 words):           ~250 tok (variable)
  Conversation history (≤20 turns):                ~3,000 tok (capped)
  Tool schemas:                                    ~1,500 tok (fixed)
  LLM response:                                      ~150 tok (target)
  Buffer:                                          ~2,500 tok

With compaction: token usage per request stays FLAT regardless of uptime.
```

### 24/7 Daily Estimate

```
120 interactions/day × ~4,500 tokens = 540K tokens/day

Gemini 2.0 Flash free tier:
  1,500 RPD   → we use ~120      ✓
  1M TPM      → we peak ~22K TPM ✓

WELL WITHIN FREE TIER.
```

---

## Component 6: Updated LLM Tools

```
OLD                          NEW
─────────────────────────────────────────
remember_fact(key, value)  → memory_store(key, value)
recall_facts()             → memory_search(query)
                           → memory_forget(key)       [new]
                           → update_user_profile(info) [new]
```

---

## Migration Plan

| Phase | Scope | Duration |
|-------|-------|----------|
| 3a | File-based prompts (SOUL/RULES/USER.md, PromptAssembler) | Day 1 |
| 3b | SQLite memory store + FTS5 indexing | Day 2-3 |
| 3c | Conversation manager + compaction + daily logs | Day 3-4 |
| 3d | Hybrid search + Gemini embeddings (optional) | Day 5 |
| 3e | Hardening: restart recovery, CLI tools, integration tests | Day 6-7 |

---

## Performance Targets

| Metric | Phase 2 | Phase 3 Target |
|--------|---------|----------------|
| Prompt assembly | ~0ms (hardcoded) | <10ms (file read) |
| Memory retrieval | ~0ms (dump all) | <100ms (hybrid search) |
| Token/request growth | Unbounded | Capped ~5K |
| Max continuous uptime | ~2h | Unlimited |
| Memory capacity | ~20 facts | 10,000+ facts |
| Restart recovery | Zero context | Last 5 turns + today's log |

---

## Summary

Phase 3 transforms NOVA's memory from "dump semua JSON ke prompt" menjadi "intelligent retrieval system" — hybrid FTS5 + vector search, auto-compaction, file-based prompts yang editable tanpa restart. Token usage flat regardless of uptime, memory scalable ke puluhan ribu fakta, dan semua bisa dibaca manusia via Markdown mirror.
