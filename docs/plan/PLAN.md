# NOVA Implementation Plan

> This document is the step-by-step execution plan for building NOVA.
> Read `PRD.md` for full context on requirements. Read `CLAUDE.md` for project conventions.
> Execute tasks **in order**. Each task must be tested before moving to the next.

---

## Development Principles

1. **One task at a time.** Build, test, commit. Then next task.
2. **Fail fast.** Test API connectivity early. If a provider doesn't work, discover it in isolation, not after integrating 5 components.
3. **Interface-first.** Abstract base classes before implementations. This makes swapping providers trivial.
4. **Test on real hardware.** RAM and CPU matter. Profile on the E410MA.
5. **Commit after every task.** Use descriptive messages: `feat: implement edge-tts provider (task 4)`.

---

## Pre-Flight Requirements

Before starting Task 1, ensure these are available:

```bash
# System dependencies
sudo apt update && sudo apt install -y mpv portaudio19-dev python3-dev python3-pip python3-venv git libffi-dev

# Verify microphone works
arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav

# API keys needed (get these first):
# - Google Gemini: https://ai.google.dev
# - Groq: https://console.groq.com
# - Cloudflare (optional for Phase 1): https://dash.cloudflare.com
```

---

## Phase 1: MVP (Tasks 1–12)

Goal: Speak a question → hear a spoken answer. That's it. Everything else is Phase 2.

---

### Task 1: Project Scaffolding

**Time:** 15–30 min
**Depends on:** Nothing
**Deliverable:** Complete directory structure, pyproject.toml, .env.example, .gitignore

**Instructions:**

1. Create the full directory structure exactly as defined in `CLAUDE.md` under "Project Structure".
2. Create `pyproject.toml` using `hatchling` as build backend. Define project metadata and dependencies:
   - Runtime deps: `httpx`, `edge-tts`, `sounddevice`, `numpy`, `python-dotenv`, `google-generativeai`, `python-mpv`, `rich`, `pydantic-settings`
   - Dev deps: `pytest`, `pytest-asyncio`, `ruff`
3. Create `.env.example` with all API key placeholders:
   ```
   GEMINI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   CLOUDFLARE_ACCOUNT_ID=your_id_here
   CLOUDFLARE_API_TOKEN=your_token_here
   ```
4. Create `.gitignore` (include `.env`, `__pycache__`, `*.pyc`, `.ruff_cache`, `dist/`, `*.egg-info`).
5. Create empty `__init__.py` files in all package directories.
6. Initialize git repo and make initial commit.

**Validation:**
```bash
pip install -e ".[dev]"    # Must succeed
python -c "import nova"    # Must not error
ruff check src/            # Must pass (no files to lint yet, but should not error)
```

---

### Task 2: Provider Base Interfaces

**Time:** 30–45 min
**Depends on:** Task 1
**Deliverable:** `providers/base.py` with abstract classes, `providers/router.py` with failover logic

**Instructions:**

1. In `src/nova/providers/base.py`, create three abstract base classes:

   ```python
   class STTProvider(ABC):
       name: str
       async def transcribe(self, audio_bytes: bytes) -> str: ...
       async def is_available(self) -> bool: ...

   class LLMProvider(ABC):
       name: str
       async def generate(self, prompt: str, context: list[dict]) -> str: ...
       async def generate_stream(self, prompt: str, context: list[dict]) -> AsyncIterator[str]: ...
       async def is_available(self) -> bool: ...

   class TTSProvider(ABC):
       name: str
       async def synthesize(self, text: str, language: str = "id") -> bytes: ...
       async def is_available(self) -> bool: ...
   ```

   Also define custom exceptions:
   ```python
   class ProviderError(Exception): ...
   class RateLimitError(ProviderError): ...
   class ProviderTimeoutError(ProviderError): ...
   ```

2. In `src/nova/providers/router.py`, create `ProviderRouter`:
   - Accepts an ordered list of providers (any type that extends a base class).
   - `async def execute(self, method_name, *args, **kwargs)` — tries each provider in order.
   - On `RateLimitError` or `ProviderTimeoutError`: log warning, try next provider.
   - On all providers exhausted: raise `AllProvidersFailedError`.
   - Implement exponential backoff: 1s, 2s, 4s between retries on same provider.
   - Log which provider is being used for each request.

**Validation:**
```bash
pytest tests/test_router.py  # Write tests with mock providers that simulate failures
```

Test scenarios to implement:
- Primary provider succeeds → returns result
- Primary fails with RateLimitError → fallback succeeds → returns result
- All providers fail → raises AllProvidersFailedError
- Timeout handling works correctly

---

### Task 3: Configuration System

**Time:** 20–30 min
**Depends on:** Task 1
**Deliverable:** `config.py` with typed settings loaded from `.env`

**Instructions:**

1. Create `src/nova/config.py` using `pydantic-settings`:

   ```python
   class NovaConfig(BaseSettings):
       # API Keys
       gemini_api_key: str = ""
       groq_api_key: str = ""
       cloudflare_account_id: str = ""
       cloudflare_api_token: str = ""

       # Audio
       sample_rate: int = 16000
       channels: int = 1
       silence_threshold: float = 0.03
       silence_duration: float = 1.5
       max_recording_seconds: float = 15.0

       # Provider priorities
       stt_providers: list[str] = ["groq", "cloudflare"]
       llm_providers: list[str] = ["gemini", "groq", "cloudflare"]
       tts_providers: list[str] = ["edge", "cloudflare"]

       # Timeouts (seconds)
       stt_timeout: float = 10.0
       llm_timeout: float = 15.0
       tts_timeout: float = 10.0

       # Conversation
       max_context_turns: int = 10
       default_language: str = "auto"  # "auto", "id", "en"

       # System
       log_level: str = "INFO"
       cache_ttl_hours: int = 24

       model_config = SettingsConfigDict(env_file=".env", env_prefix="NOVA_")
   ```

2. Add a `get_config()` singleton function.
3. Validate that at least one LLM API key is provided on startup.

**Validation:**
```bash
# Create .env with test values, then:
python -c "from nova.config import get_config; c = get_config(); print(c.sample_rate)"
```

---

### Task 4: Edge TTS Provider (Primary TTS)

**Time:** 30–45 min
**Depends on:** Task 2
**Deliverable:** Working TTS that converts text to spoken audio

**Why TTS first:** It's the simplest component (no API key, text in → audio out). Building this first gives you audio feedback for testing everything else.

**Instructions:**

1. Implement `EdgeTTSProvider` in `src/nova/providers/tts/edge_tts_provider.py`:
   - Extends `TTSProvider` from base.py.
   - Uses `edge-tts` library.
   - Voice mapping:
     - Indonesian: `id-ID-ArdiNeural` (male) or `id-ID-GadisNeural` (female)
     - English: `en-US-GuyNeural` (male) or `en-US-JennyNeural` (female)
   - `synthesize(text, language="id")` → returns MP3 bytes.
   - Auto-detect language if language="auto" (simple heuristic: check for common Indonesian words, or use the LLM's language detection from previous turn).
   - `is_available()` → always True (no API key needed).

2. Create audio playback utility in `src/nova/audio/playback.py`:
   - `async def play_audio(audio_bytes: bytes)` — saves to temp file, plays via `mpv --no-video --really-quiet`.
   - Clean up temp file after playback.
   - Handle case where mpv is not installed.

**Validation:**
```bash
# Add a __main__ block to edge_tts_provider.py for testing:
python -m nova.providers.tts.edge_tts_provider
# Should speak: "Halo, saya Nova, asisten suara pribadi Anda" in Indonesian
# Then: "Hello, I am Nova, your personal voice assistant" in English
# Both should sound natural and play through speakers
```

Measure and print latency for each synthesis.

---

### Task 5: Gemini LLM Provider (Primary LLM)

**Time:** 45–60 min
**Depends on:** Task 2, Task 3
**Deliverable:** Working LLM that generates conversational responses

**Instructions:**

1. Implement `GeminiProvider` in `src/nova/providers/llm/gemini.py`:
   - Extends `LLMProvider` from base.py.
   - Uses `google-generativeai` SDK.
   - Model: `gemini-2.0-flash` (check availability, fall back to `gemini-1.5-flash` if needed).
   - System prompt (set this as the system instruction):
     ```
     You are NOVA, a personal voice assistant. You run on a low-spec laptop and communicate via voice.

     Rules:
     - Keep responses under 100 words unless the user explicitly asks for detail.
     - Be conversational and natural — your responses will be spoken aloud.
     - Detect the user's language (Indonesian or English) and respond in the same language.
     - Don't use markdown formatting, bullet points, or special characters — plain spoken text only.
     - Don't say "as a voice assistant" or reference your nature unless asked.
     - Be helpful, direct, and friendly.
     - For questions you can't answer, say so briefly rather than making things up.
     ```
   - `generate(prompt, context)` — sends conversation history + new prompt, returns response text.
   - `is_available()` — makes a minimal API call to verify key works.
   - On HTTP 429: raise `RateLimitError` with retry-after info if available.
   - On HTTP 500/503: raise `ProviderError`.
   - Timeout: use config's `llm_timeout`.

**Validation:**
```bash
python -m nova.providers.llm.gemini
# Test 1: "Siapa presiden pertama Indonesia?" → should answer in Indonesian
# Test 2: "What is the speed of light?" → should answer in English
# Test 3: Send 3 exchanges to test context works
# Print response + latency for each
```

---

### Task 6: Groq Whisper STT Provider (Primary STT)

**Time:** 45–60 min
**Depends on:** Task 2, Task 3
**Deliverable:** Working STT that converts speech audio to text

**Instructions:**

1. Implement `GroqWhisperProvider` in `src/nova/providers/stt/groq_whisper.py`:
   - Extends `STTProvider` from base.py.
   - Uses `httpx` to call Groq API directly (endpoint: `https://api.groq.com/openai/v1/audio/transcriptions`).
   - Model: `whisper-large-v3-turbo`.
   - `transcribe(audio_bytes)` — accepts WAV bytes, sends as multipart form, returns transcript text.
   - Set `language` parameter based on config (or omit for auto-detect).
   - Handle rate limits (429) → `RateLimitError`.
   - Timeout: use config's `stt_timeout`.

2. Create a test recording utility for validation:
   - Record 3 seconds from mic using `sounddevice`.
   - Convert to WAV bytes (16kHz, mono, 16-bit PCM).
   - Send to Groq, print transcript.

**Validation:**
```bash
python -m nova.providers.stt.groq_whisper
# Speak into mic: "Halo apa kabar" → should print Indonesian transcript
# Speak: "Hello how are you" → should print English transcript
# Print latency for each transcription (target: <1 second)
```

---

### Task 7: Audio Capture with VAD

**Time:** 30–45 min
**Depends on:** Task 6
**Deliverable:** Smart audio recording that starts/stops based on voice activity

**Instructions:**

1. Create `AudioCapture` in `src/nova/audio/capture.py`:
   - Uses `sounddevice` for recording at 16kHz, mono.
   - **Voice Activity Detection (VAD):** Energy-based detection.
     - Calculate RMS energy of each audio chunk.
     - State machine: WAITING → RECORDING → DONE.
     - WAITING: listen for energy above `silence_threshold` → transition to RECORDING.
     - RECORDING: record audio. If energy below threshold for `silence_duration` seconds → transition to DONE.
     - Safety: max recording time from config (`max_recording_seconds`).
   - Returns WAV bytes (with proper WAV header).
   - `async def capture() -> bytes` — main method.

2. Include configurable parameters from NovaConfig:
   - `silence_threshold` (default: 0.03)
   - `silence_duration` (default: 1.5 seconds)
   - `max_recording_seconds` (default: 15)

**Validation:**
```bash
python -m nova.audio.capture
# Should print "Listening..." then wait for speech
# Speak a sentence → should stop recording ~1.5s after you stop
# Should save to test.wav and report: duration, file size
# Play back with: aplay test.wav
```

Test edge cases:
- Speaking very softly (should still capture if above threshold)
- Long pause mid-sentence (should not cut off if under silence_duration)
- Background noise without speech (should not trigger false recording)

---

### Task 8: Text-Only Pipeline Integration

**Time:** 45–60 min
**Depends on:** Task 4, Task 5
**Deliverable:** Type a question → hear the answer spoken. Full LLM + TTS loop.

> **This is the first integration milestone.** It proves the core loop works before adding audio input complexity.

**Instructions:**

1. Create `Orchestrator` in `src/nova/orchestrator.py`:
   - Initializes all providers based on config.
   - Creates `ProviderRouter` instances for STT, LLM, and TTS.
   - Main methods:
     - `async def process_text(self, text: str) -> str` — sends text to LLM, gets response.
     - `async def speak(self, text: str)` — converts text to audio via TTS, plays it.
     - `async def handle_interaction(self, user_input: str)` — process_text + speak.

2. Create conversation context manager in `src/nova/memory/context.py`:
   - `ConversationContext` class.
   - Stores list of `{"role": "user" | "assistant", "content": str}`.
   - Sliding window: keeps last `max_context_turns` exchanges (from config).
   - `add_exchange(user_msg, assistant_msg)` — appends and trims.
   - `get_context() -> list[dict]` — returns current window.
   - `clear()` — resets context.

3. Wire everything in `src/nova/main.py`:
   - Parse CLI args: `--text-only` (force text mode), `--verbose` (debug logging).
   - Async main loop:
     ```
     while True:
         user_input = input("You: ")
         if user_input in ("exit", "quit", "bye"):
             break
         response = await orchestrator.handle_interaction(user_input)
         print(f"Nova: {response}")
     ```
   - Handle `Ctrl+C` gracefully (print goodbye, cleanup).

**Validation:**
```bash
python -m nova.main --text-only
# Type: "Halo, siapa kamu?"
# → Should print response AND speak it aloud
# Type: "Nama saya Zhafran"
# Type: "Siapa nama saya?"
# → Should correctly recall "Zhafran" from context
# Type: "exit"
# → Should exit cleanly
```

Check:
- [ ] Responses are concise (under 100 words)
- [ ] Audio plays while text is printed
- [ ] Context works across multiple turns
- [ ] Memory usage: run `htop` and verify <200MB
- [ ] Ctrl+C doesn't leave orphan processes

---

### Task 9: Full Voice Pipeline (MVP)

**Time:** 60–90 min
**Depends on:** Task 6, Task 7, Task 8
**Deliverable:** Speak a question → hear the answer. **This is the MVP.**

**Instructions:**

1. Add voice mode to the orchestrator:
   - `async def handle_voice_interaction(self)`:
     1. Print "🎤 Listening..." (or play a subtle beep)
     2. Call `AudioCapture.capture()` → get WAV bytes
     3. Call STT router → get transcript text
     4. Print `f"You: {transcript}"`
     5. Call LLM router with transcript + context → get response
     6. Print `f"Nova: {response}"`
     7. Call TTS router → get audio bytes
     8. Play audio
     9. Add exchange to conversation context
     10. Log total latency breakdown: STT took Xs, LLM took Xs, TTS took Xs, total Xs

2. Update `main.py`:
   - Default mode: voice (press Enter to start listening, or use push-to-talk).
   - `--text-only` flag: text mode as built in Task 8.
   - After each voice interaction, wait for Enter key to start next (push-to-talk pattern).
   - This is simpler and more reliable than always-on listening for MVP.

3. Add latency measurement:
   - Wrap each stage (STT, LLM, TTS) with `time.perf_counter()`.
   - Print summary: `[STT: 0.4s | LLM: 1.8s | TTS: 0.5s | Total: 2.7s]`

**Validation:**
```bash
python -m nova.main
# Press Enter
# Speak: "Jam berapa sekarang?"
# → Should hear a spoken answer about the current time
# → Latency breakdown printed

# Press Enter
# Speak: "Tell me a fun fact"
# → Should hear an English response
# → Language auto-detected correctly
```

Critical checks:
- [ ] Full pipeline works end-to-end via voice
- [ ] Total latency printed for every interaction
- [ ] Average latency < 5 seconds on stable connection
- [ ] Transcript accuracy is acceptable (>90% for clear speech)
- [ ] Text-only fallback still works with --text-only
- [ ] RAM usage stays under 300MB

**🎉 If this task passes, NOVA MVP is complete.**

---

### Task 10: Fallback Providers

**Time:** 60–90 min
**Depends on:** Task 9
**Deliverable:** Backup LLM and TTS providers wired into the router

**Instructions:**

1. Implement `GroqLLMProvider` in `src/nova/providers/llm/groq_llm.py`:
   - Model: `llama-3.3-70b-versatile` (or latest available on Groq).
   - Use httpx to call `https://api.groq.com/openai/v1/chat/completions`.
   - Same system prompt as Gemini.
   - Same interface: `generate(prompt, context)`, `is_available()`.

2. Implement `CloudflareTTSProvider` in `src/nova/providers/tts/cloudflare_tts.py`:
   - Use httpx to call Cloudflare Workers AI TTS endpoint.
   - Endpoint: `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/m2m100-1.2b` (or appropriate TTS model — check Cloudflare docs for current TTS model availability).
   - Fallback for when Edge TTS fails.

3. Wire fallback providers into the router:
   - LLM router: [GeminiProvider, GroqLLMProvider]
   - TTS router: [EdgeTTSProvider, CloudflareTTSProvider]
   - STT router: [GroqWhisperProvider] (add more later if needed)

4. Test failover by simulating failures.

**Validation:**
```bash
# Test 1: Normal operation (primary providers)
python -m nova.main --text-only
# Type a question → verify Gemini is used (check log)

# Test 2: Force failover
# Temporarily set an invalid Gemini API key in .env
python -m nova.main --text-only
# Type a question → verify Groq LLM is used as fallback
# Log should show: "Gemini failed (RateLimitError), falling back to Groq"

# Test 3: Restore valid keys, verify primary is used again
```

---

### Task 11: Error Handling & Resilience

**Time:** 30–45 min
**Depends on:** Task 10
**Deliverable:** NOVA never crashes from a single bad interaction

**Instructions:**

1. Add error handling to the orchestrator for every failure mode:

   | Failure | User Feedback | Behavior |
   |---------|--------------|----------|
   | Network timeout | "Koneksi lambat, mohon tunggu..." / "Connection is slow, please wait..." | Retry once, then fail gracefully |
   | All STT providers failed | "Maaf, saya tidak bisa mendengar sekarang. Coba ketik saja." | Switch to text input mode |
   | All LLM providers failed | "Semua layanan sedang sibuk, coba lagi sebentar." | Wait 5 seconds, allow retry |
   | All TTS providers failed | Print response to terminal only | Continue without audio |
   | No speech detected (empty audio) | "Saya tidak mendengar apa-apa, bisa diulang?" | Return to listening |
   | Audio device not found | "Mikrofon tidak ditemukan, beralih ke mode teks." | Auto-switch to text-only |
   | Unexpected exception | "Terjadi kesalahan, tapi saya masih berjalan." | Log traceback, continue running |

2. Add `--check` CLI command to `main.py`:
   ```bash
   python -m nova.main --check
   # Output:
   # ✅ Gemini API: connected (key valid)
   # ✅ Groq API: connected (key valid)
   # ❌ Cloudflare: not configured
   # ✅ Edge TTS: available
   # ✅ Microphone: detected (default device)
   # ✅ mpv: installed
   ```

**Validation:**
- Disconnect WiFi → NOVA informs user, doesn't crash
- Remove all API keys → health check reports all failed, NOVA exits gracefully with message
- Send empty audio → NOVA asks to repeat, continues listening
- Run 20+ interactions → no crashes, no memory leaks

---

### Task 12: Logging & Monitoring

**Time:** 20–30 min
**Depends on:** Task 11
**Deliverable:** Structured logging with performance tracking

**Instructions:**

1. Create `src/nova/utils/logger.py`:
   - Use Python `logging` module with `rich.logging.RichHandler` for pretty terminal output.
   - Log format: `[TIME] [LEVEL] [COMPONENT] message`
   - Components: `STT`, `LLM`, `TTS`, `AUDIO`, `ROUTER`, `ORCHESTRATOR`
   - File logging: write to `~/.nova/logs/nova-YYYY-MM-DD.log` with daily rotation.
   - Keep last 7 days of logs.

2. For each interaction, log a summary:
   ```
   [INFO] [ORCHESTRATOR] Interaction #42 complete
     STT: groq_whisper (0.34s) | LLM: gemini (1.82s) | TTS: edge (0.45s) | Total: 2.61s
     Input: "siapa presiden indonesia" | Response: 127 chars
   ```

3. Add `--verbose` flag for DEBUG level (shows full API payloads, audio chunk sizes, etc.).

**Validation:**
```bash
python -m nova.main --verbose
# Run a few interactions
# Check ~/.nova/logs/ for log files
# Verify latency is logged for every stage
```

---

## Phase 2: Enhanced Features (Tasks 13–18)

> **Do not start Phase 2 until Phase 1 is fully stable and tested.**
> Phase 2 adds convenience features on top of the working MVP.

---

### Task 13: Wake Word Detection

**Time:** 45–60 min
**Depends on:** Phase 1 complete
**Deliverable:** Say "Hey Nova" to activate listening (hands-free)

**Instructions:**

1. Implement `WakeWordDetector` in `src/nova/audio/wake_word.py`:
   - Use `pvporcupine` library.
   - Free built-in keyword: use "Porcupine" (free) or create a custom "Nova" keyword if Porcupine allows.
   - Run in a separate asyncio task (non-blocking).
   - On detection: play a short activation beep (generate a sine wave or use a small audio file), then trigger `AudioCapture`.
   - Configurable sensitivity via config (`wake_word_sensitivity`).

2. Update `main.py`:
   - Default mode: wake word (always listening for hotword).
   - `--push-to-talk` flag: press Enter to record (as in Phase 1).
   - `--text-only` flag: text mode.

3. **Monitor CPU usage carefully.** Porcupine should use <10% CPU in continuous listening mode on the Celeron. If it uses more, this feature may not be viable on this hardware.

**Validation:**
- Say "Hey Nova" → activation beep plays, recording starts
- Say a question → full pipeline runs
- CPU usage during idle listening: <10%
- False positive rate: say random words for 2 minutes, count false activations (target: <5%)

---

### Task 14: System Control Tools

**Time:** 45–60 min
**Depends on:** Phase 1 complete
**Deliverable:** Control the laptop via voice commands

**Instructions:**

1. Create `src/nova/tools/system_control.py` with these commands:

   | Voice Command | Action | Implementation |
   |---------------|--------|----------------|
   | "volume up/down" | Adjust volume ±10% | `pactl set-sink-volume` |
   | "mute/unmute" | Toggle mute | `pactl set-sink-mute` |
   | "brightness up/down" | Adjust brightness | `brightnessctl` or write to `/sys/class/backlight/` |
   | "open browser" | Launch Firefox/Chrome | `subprocess.Popen(["firefox"])` |
   | "open terminal" | Launch terminal | `subprocess.Popen(["gnome-terminal"])` |
   | "open file manager" | Launch Nautilus/Thunar | `subprocess.Popen(["nautilus"])` |
   | "lock screen" | Lock | `loginctl lock-session` |
   | "what time is it" | Speak current time | `datetime.now()` → TTS |

2. Register these as **function calls** in the LLM provider:
   - Define tool schemas that the LLM can call.
   - When LLM returns a function call, execute the corresponding system command.
   - Return the result to the user via TTS.

3. Create `src/nova/tools/time_date.py` for time/date queries (these don't need system commands, just return formatted text).

**Validation:**
- "Nova, naikkan volume" → volume increases
- "What time is it?" → speaks the current time
- "Buka browser" → Firefox opens
- LLM correctly identifies when to use tools vs. when to just respond

---

### Task 15: Response Caching

**Time:** 30–45 min
**Depends on:** Phase 1 complete
**Deliverable:** Frequently asked questions answered instantly from cache

**Instructions:**

1. Create `src/nova/memory/cache.py`:
   - SQLite database at `~/.nova/cache.db`.
   - Schema: `(prompt_hash TEXT PRIMARY KEY, response TEXT, language TEXT, created_at TIMESTAMP, ttl_hours INT)`.
   - `prompt_hash`: normalized lowercase, stripped, hashed with SHA-256.
   - `get(prompt) -> str | None` — returns cached response if exists and not expired.
   - `set(prompt, response, ttl_hours)` — store response.
   - `clear_expired()` — remove entries past TTL.
   - Default TTL: 24 hours for factual queries, skip cache for conversational queries.

2. Integrate into orchestrator:
   - Before calling LLM, check cache.
   - Cache hit → skip LLM, go directly to TTS.
   - Cache miss → normal LLM call, then cache the response.
   - Conversational queries (containing "I", "my", "me", context-dependent) should bypass cache.

**Validation:**
- Ask "What is the capital of Japan?" → first time: full pipeline. Second time: cache hit, faster response.
- Cache hit should skip LLM latency entirely.
- Track and log cache hit rate.

---

### Task 16: Web Search Tool

**Time:** 30–45 min
**Depends on:** Phase 1 complete
**Deliverable:** Answer current-events questions using web search

**Instructions:**

1. Create `src/nova/tools/web_search.py`:
   - Use `duckduckgo-search` Python library (no API key needed).
   - `async def search(query: str, max_results: int = 3) -> list[dict]` — returns title + snippet + URL.
   - Register as LLM function call tool.

2. LLM integration:
   - When user asks about current events, news, or things the LLM can't know, it calls the search tool.
   - Search results are fed back to LLM as context.
   - LLM summarizes the results in a concise spoken response.

**Validation:**
- "What's the weather in Jakarta today?" → should search and summarize
- "Who won the latest football match?" → should search and answer
- Results are summarized concisely for voice delivery

---

### Task 17: Audio Streaming TTS

**Time:** 45–60 min
**Depends on:** Task 4
**Deliverable:** Audio starts playing before full TTS generation completes

**Instructions:**

1. Modify `EdgeTTSProvider` to support streaming:
   - `edge-tts` supports streaming audio chunks.
   - Instead of waiting for full MP3, yield chunks as they arrive.
   - `async def synthesize_stream(self, text: str, language: str) -> AsyncIterator[bytes]`

2. Modify `playback.py` to support streaming input:
   - Feed audio chunks to mpv as they arrive via stdin pipe.
   - First audio should start playing within ~200ms of TTS call.

3. This significantly reduces **perceived** latency — user hears audio before the full response is synthesized.

**Validation:**
- Long response (50+ words): audio starts within 500ms, plays smoothly
- Compare perceived latency vs. non-streaming: should feel noticeably faster

---

### Task 18: Systemd Service & Setup Script

**Time:** 20–30 min
**Depends on:** All Phase 2 tasks
**Deliverable:** NOVA starts on boot, restarts on crash

**Instructions:**

1. Create `scripts/nova.service`:
   ```ini
   [Unit]
   Description=NOVA Voice Assistant
   After=network-online.target sound.target
   Wants=network-online.target

   [Service]
   Type=simple
   User=%u
   WorkingDirectory=/home/%u/nova
   ExecStart=/home/%u/nova/.venv/bin/python -m nova.main
   Restart=on-failure
   RestartSec=5
   Environment=DISPLAY=:0
   Environment=PULSE_SERVER=unix:/run/user/1000/pulse/native

   [Install]
   WantedBy=default.target
   ```

2. Create `scripts/setup.sh`:
   - Install system deps (apt).
   - Create Python venv.
   - Install Python deps.
   - Copy `.env.example` to `.env` if not exists.
   - Install and enable systemd service.
   - Run `--check` to verify everything.

3. Update `README.md` with complete setup instructions.

**Validation:**
```bash
bash scripts/setup.sh
sudo systemctl start nova
sudo systemctl status nova  # Should be active (running)
# Speak to NOVA → should respond
sudo systemctl stop nova
```

---

## Quick Reference: Task Dependency Graph

```
Task 1 (scaffolding)
├── Task 2 (base interfaces)
│   ├── Task 4 (Edge TTS)
│   ├── Task 5 (Gemini LLM) ← also needs Task 3
│   └── Task 6 (Groq STT) ← also needs Task 3
├── Task 3 (config)
│
Task 4 + Task 5 → Task 8 (text pipeline) ← INTEGRATION MILESTONE
Task 6 + Task 7 (audio capture) + Task 8 → Task 9 (voice pipeline) ← MVP MILESTONE
Task 9 → Task 10 (fallback providers)
Task 10 → Task 11 (error handling)
Task 11 → Task 12 (logging)
│
Phase 1 complete → Phase 2 (Tasks 13–18, can be done in any order)
```

---

## API Quick Reference

### Groq (STT + LLM)
- Base URL: `https://api.groq.com/openai/v1/`
- Auth header: `Authorization: Bearer {GROQ_API_KEY}`
- STT endpoint: `POST /audio/transcriptions` (multipart form: file + model)
- LLM endpoint: `POST /chat/completions` (JSON: model, messages, max_tokens)

### Google Gemini (LLM)
- SDK: `google-generativeai` Python package
- Auth: `genai.configure(api_key=GEMINI_API_KEY)`
- Model: `genai.GenerativeModel("gemini-2.0-flash")`

### Edge TTS
- Library: `edge-tts` (no API key)
- Usage: `edge_tts.Communicate(text, voice).save(path)` or stream

### Cloudflare Workers AI
- Base URL: `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/`
- Auth header: `Authorization: Bearer {CLOUDFLARE_API_TOKEN}`
- Models: check `https://developers.cloudflare.com/workers-ai/models/` for current availability
