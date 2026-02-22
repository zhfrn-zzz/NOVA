# NOVA — Neural-Orchestrated Voice Assistant

## What is this project?

NOVA is a personal AI voice assistant that runs on extremely low-spec hardware (ASUS E410MA, Celeron N4020, 4GB RAM, Ubuntu Linux). All AI processing (STT, LLM, TTS) is offloaded to **free-tier cloud APIs**. The local machine acts purely as a thin-client orchestrator.

The pipeline: **Mic → Cloud STT → Cloud LLM → Cloud TTS → Speaker**

Wake word detection uses [openwakeword](https://github.com/dscripka/openWakeWord) with a custom `hey_nova.onnx` model for always-listening activation. Falls back to keyboard hotkey (`Ctrl+Space`) if openwakeword is unavailable.

## Hardware Constraints (CRITICAL — read before every implementation decision)

| Component | Spec | What this means |
|-----------|------|-----------------|
| CPU | Celeron N4020 (2C/2T, 1.1GHz) | **No local ML inference. Zero. Not even tiny models.** |
| RAM | 4GB DDR4 (soldered) | Ubuntu uses ~900MB. NOVA budget: **< 300MB total** |
| Storage | 128GB eMMC | Plenty for code, not for models |
| Network | Wi-Fi 802.11ac | **All AI is cloud-dependent. Internet = oxygen.** |
| OS | Ubuntu Linux (latest LTS) | No Windows overhead |

## Architecture Rules

1. **Every AI operation goes to cloud APIs.** No local inference, no local models, no exceptions.
2. **All providers must have fallbacks.** Free tiers have rate limits. When one dies, the next one picks up automatically.
3. **Async everything.** Use `httpx` (not `requests`). Use `asyncio` for the main loop. Blocking calls kill responsiveness.
4. **Memory is sacred.** Profile RAM usage. If it creeps past 250MB, something is wrong.
5. **Responses must be concise.** System prompt enforces <100 word responses for voice delivery. Nobody wants to listen to a 500-word essay.
6. **Bilingual: Bahasa Indonesia + English.** Auto-detect language, respond in kind.

## Provider Priority Chain

### STT (Speech-to-Text)
1. **Groq Whisper** (whisper-large-v3-turbo) — fastest, ~0.3s
2. **Google Cloud STT** (free 60 min/month) — fallback
3. **Cloudflare Workers AI** (Whisper) — fallback 2

### LLM (Brain)
1. **Google Gemini** (2.5 Flash / 2.0 Flash Lite) — 15 RPM, 1500 req/day
2. **Groq** (Llama 3.3 70B) — 30 RPM, ultra-fast
3. **Cloudflare Workers AI** (Llama/Mistral) — 10K neurons/day
4. **OpenRouter** (various free models) — last resort

### TTS (Text-to-Speech)
1. **Edge TTS** (edge-tts lib) — unlimited, no API key, very natural
2. **Cloudflare Workers AI** — fallback

## Code Conventions

- Python 3.10+ with type hints on ALL public functions
- Async/await for all I/O (httpx, not requests)
- Abstract base classes for all provider types (easy to swap)
- `ruff` for linting and formatting
- Tests in `tests/` — run with `pytest`
- API keys in `.env` — **never hardcoded, never committed**
- Docstrings on all classes and public methods
- Logging via Python `logging` + `rich` for terminal output

## Project Structure

```
nova/
├── pyproject.toml
├── .env.example
├── .env                          # NOT committed (in .gitignore)
├── CLAUDE.md                     # This file
├── PLAN.md                       # Implementation plan
├── PRD.md                        # Product requirements
├── README.md
├── src/
│   └── nova/
│       ├── __init__.py
│       ├── main.py               # Entry point, CLI args, async loop
│       ├── config.py             # Settings from .env + defaults
│       ├── orchestrator.py       # Core pipeline coordinator
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── capture.py        # Mic input + VAD
│       │   ├── playback.py       # Speaker output (mpv)
│       │   ├── wake_word.py      # Hotkey fallback detector (Ctrl+Space)
│       │   └── wake_word_oww.py  # OpenWakeWord detector (default)
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py           # Abstract: STTProvider, LLMProvider, TTSProvider
│       │   ├── router.py         # Failover logic + smart routing
│       │   ├── stt/
│       │   │   ├── __init__.py
│       │   │   ├── groq_whisper.py
│       │   │   ├── google_stt.py
│       │   │   └── cloudflare_stt.py
│       │   ├── llm/
│       │   │   ├── __init__.py
│       │   │   ├── gemini.py
│       │   │   ├── groq_llm.py
│       │   │   ├── cloudflare_llm.py
│       │   │   └── openrouter.py
│       │   └── tts/
│       │       ├── __init__.py
│       │       ├── edge_tts_provider.py
│       │       └── cloudflare_tts.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── system_control.py # Volume, brightness, app launch
│       │   ├── web_search.py     # DuckDuckGo / SearXNG
│       │   └── time_date.py      # Local time/date
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── context.py        # Sliding window conversation manager
│       │   ├── persistent.py     # User facts stored in ~/.nova/memory.json
│       │   └── cache.py          # SQLite response cache
│       └── utils/
│           ├── __init__.py
│           ├── logger.py
│           └── audio_utils.py
├── tests/
│   ├── test_orchestrator.py
│   ├── test_providers.py
│   ├── test_router.py
│   └── test_tools.py
└── scripts/
    ├── setup.sh                  # Full dependency install script
    └── nova.service              # systemd auto-start
```

## Current Progress

- [x] Task 1: Project scaffolding
- [x] Task 2: Provider base interfaces
- [x] Task 3: Configuration system
- [x] Task 4: Edge TTS provider (primary TTS)
- [x] Task 5: Gemini LLM provider (primary LLM)
- [x] Task 6: Groq Whisper STT provider (primary STT)
- [x] Task 7: Audio capture with VAD
- [x] Task 8: Text-only pipeline integration ← **Integration Milestone**
- [x] Task 9: Full voice pipeline ← **MVP Milestone**
- [x] Task 10: Fallback providers + router wiring
- [x] Task 11: Error handling & resilience
- [x] Task 12: Logging & monitoring
- [x] Task 13: Wake word detection — hotkey fallback (Phase 2)
- [x] Task 13.5: OpenWakeWord integration — always-listening with hey_nova.onnx
- [x] Task 14: System control tools (Phase 2)
- [ ] Task 15: Response caching (Phase 2)
- [x] Task 15.5: Persistent user memory — remember_fact/recall_facts (Phase 2)
- [x] Task 16: Web search tool (Phase 2)
- [ ] Task 17: Audio streaming TTS (Phase 2)
- [ ] Task 18: Systemd service + setup script (Phase 2)

## Important Notes

- After completing each task, update the checklist above.
- Run `ruff check src/` and `pytest` after every task.
- Test on the actual ASUS E410MA, not just a dev machine.
- Monitor RAM with `htop` during testing — never exceed 300MB.
