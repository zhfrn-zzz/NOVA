# NOVA — Neural-Orchestrated Voice Assistant

A personal AI voice assistant that runs on low-spec hardware, powered by free-tier cloud APIs.

**Pipeline:** Mic → Cloud STT → Cloud LLM → Cloud TTS → Speaker

## Quick Start

```bash
cp .env.example .env
# Edit .env with your API keys
pip install -e ".[dev]"
python -m nova.main --text-only
```

See `CLAUDE.md` for full project documentation.
