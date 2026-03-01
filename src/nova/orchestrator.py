"""Core pipeline orchestrator — coordinates STT, LLM, TTS, and context.

Supports two response paths:
- Streaming: LLM streams sentences → TTS synthesizes each immediately → play.
  Used for conversational queries. Lowest time-to-first-audio.
- Tool: LLM generates full response with function calling → TTS streams.
  Used when the query likely needs tools (time, volume, search, etc.).
"""

import asyncio
import logging
import os
import re
import shutil
import time

from nova.audio.streaming_tts import StreamingTTSPlayer
from nova.config import get_config
from nova.memory.conversation import ConversationManager
from nova.memory.embeddings import get_embedder
from nova.memory.memory_store import get_memory_store
from nova.memory.prompt_assembler import get_prompt_assembler
from nova.memory.retriever import MemoryRetriever
from nova.providers.base import AllProvidersFailedError
from nova.providers.llm.gemini import GeminiProvider
from nova.providers.llm.groq_llm import GroqLLMProvider
from nova.providers.router import ProviderRouter
from nova.providers.stt.groq_whisper import GroqWhisperProvider
from nova.providers.tts.cloudflare_tts import CloudflareTTSProvider
from nova.providers.tts.edge_tts_provider import EdgeTTSProvider, detect_language
from nova.providers.tts.google_cloud_tts import GoogleCloudTTSProvider
from nova.tools.registry import get_tool_declarations

logger = logging.getLogger(__name__)

# Keywords that indicate the query likely needs tool/function calling.
# When detected, the orchestrator uses the non-streaming tool path.
# Matching uses prefix/stem comparison so Indonesian suffixed words
# like "lagunya", "putarkan", "musiknya" still match their roots.
_TOOL_KEYWORDS = {
    # Time/date
    "jam", "waktu", "time", "tanggal", "date",
    # System control
    "volume", "mute", "unmute", "screenshot", "timer",
    "shutdown", "restart", "lock", "sleep", "hibernate",
    "matikan", "kunci", "tidurkan", "tangkap",
    # App control
    "buka", "tutup", "open",
    # System info
    "baterai", "battery", "ram", "storage", "disk", "uptime", "ip",
    "penyimpanan", "menyala", "address",
    # Notes & reminders
    "catat", "catatan", "notes", "note", "ingatkan", "reminder",
    "hapus",
    # Display & network
    "brightness", "kecerahan", "wifi", "terang", "redup",
    # Web search
    "cari", "search", "google", "berita", "cuaca", "news", "weather",
    # Memory
    "ingat", "remember", "lupakan", "forget", "profil",
    # Media
    "play", "pause", "next", "previous", "stop",
    # Music
    "putar", "puterin", "lagu", "musik", "music", "song", "skip",
    "judul", "mainkan", "nyalakan",
    # Dictation
    "ketik", "dictate", "tulis",
}

# Simple greetings — skip memory retrieval for these
_GREETINGS = {
    "halo", "hai", "hi", "hey", "hello",
    "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "pagi", "siang", "sore", "malam",
    "good morning", "good afternoon", "good evening", "good night",
    "assalamualaikum", "apa kabar", "how are you",
}


class Orchestrator:
    """Coordinates the full NOVA pipeline: STT -> LLM -> TTS -> playback."""

    def __init__(self) -> None:
        config = get_config()

        # --- Build provider lists based on available credentials ---
        # STT providers
        stt_providers = [GroqWhisperProvider()]

        # LLM providers: Gemini (primary), Groq LLM (fallback)
        llm_providers = []
        if config.gemini_api_key:
            llm_providers.append(GeminiProvider())
        if config.groq_api_key:
            llm_providers.append(GroqLLMProvider())
        if not llm_providers:
            # At least need one — will fail at runtime with clear error
            llm_providers.append(GeminiProvider())

        # TTS providers: Google Cloud TTS (primary, if configured),
        # Edge TTS (default), Cloudflare (fallback)
        tts_providers: list = []
        gcp_key = os.path.expanduser(config.google_cloud_tts_key_path)
        if gcp_key and os.path.isfile(gcp_key):
            tts_providers.append(GoogleCloudTTSProvider())
        tts_providers.append(EdgeTTSProvider())
        if config.cloudflare_account_id and config.cloudflare_api_token:
            tts_providers.append(CloudflareTTSProvider())

        # Create routers
        self._stt_router = ProviderRouter("STT", stt_providers)
        self._llm_router = ProviderRouter("LLM", llm_providers)
        self._tts_router = ProviderRouter("TTS", tts_providers)

        # Audio capture (lazy — created on first voice interaction)
        self._audio_capture = None

        # Memory store and conversation manager
        self._memory_store = get_memory_store()
        self._context = ConversationManager(
            memory_store=self._memory_store,
            llm_fn=self._summarize_for_compaction,
        )
        self._context.start_session()
        self._default_language = config.default_language

        # Tool declarations for function calling
        self._tools = get_tool_declarations()

        # Streaming TTS player for reduced latency
        self._streaming_tts = StreamingTTSPlayer()

        # Interaction counter for logging
        self._interaction_count = 0

        # TTS warmup state
        self._tts_warmed_up = False

        # --- Embedding & retrieval ---
        embedder = get_embedder()
        self._embedding_fn = embedder.embed if embedder else None

        # Wire embedding into memory store for auto-embedding on store
        if self._embedding_fn:
            self._memory_store.set_embedding_fn(self._embedding_fn)

        # Create retriever with embedding function
        self._retriever = MemoryRetriever(
            memory_store=self._memory_store,
            embedding_fn=self._embedding_fn,
        )

        # Backfill existing memories without embeddings
        if self._embedding_fn:
            asyncio.ensure_future(self._backfill_startup())

        logger.info(
            "Orchestrator initialized — LLM: %s | TTS: %s | STT: %s | Embedder: %s",
            [p.name for p in llm_providers],
            [p.name for p in tts_providers],
            [p.name for p in stt_providers],
            "enabled" if embedder else "disabled",
        )

    async def _summarize_for_compaction(self, prompt: str) -> str:
        """Use the primary LLM to summarize conversation for compaction.

        Args:
            prompt: The summarization prompt with conversation text.

        Returns:
            Summary text from the LLM.
        """
        return await self._llm_router.execute(
            "generate", prompt, context=[], tools=None,
        )

    def _get_audio_capture(self):
        """Lazy-init AudioCapture to avoid import errors when mic is missing."""
        if self._audio_capture is None:
            from nova.audio.capture import AudioCapture
            self._audio_capture = AudioCapture()
        return self._audio_capture

    async def _backfill_startup(self) -> None:
        """Backfill embeddings for memories that don't have them yet."""
        try:
            count = await self._memory_store.backfill_embeddings()
            if count > 0:
                logger.info("Backfilled %d memories with embeddings", count)
        except Exception:
            logger.warning("Startup embedding backfill failed", exc_info=True)

    def _is_simple_greeting(self, text: str) -> bool:
        """Check if the text is a simple greeting (skip memory retrieval).

        Args:
            text: The user's input text.

        Returns:
            True if the text is a short greeting.
        """
        cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
        if len(cleaned.split()) >= 5:
            return False
        return cleaned in _GREETINGS

    async def _get_memory_context(self, user_input: str) -> str:
        """Retrieve relevant memory context for the current query.

        Args:
            user_input: The user's input text.

        Returns:
            Formatted memory context string, or empty string.
        """
        if self._is_simple_greeting(user_input):
            return ""
        try:
            results = await self._retriever.search(user_input)
            return self._retriever.format_for_prompt(results)
        except Exception:
            logger.warning("Memory retrieval failed", exc_info=True)
            return ""

    async def _warmup_tts(self) -> None:
        """Pre-initialize Edge TTS connection for faster first request."""
        if self._tts_warmed_up:
            return
        try:
            provider = self._tts_router.providers[0]
            if hasattr(provider, "warmup"):
                await provider.warmup()
            self._tts_warmed_up = True
        except Exception:
            logger.debug("TTS warmup failed (non-critical)")

    def _likely_needs_tools(self, text: str) -> bool:
        """Heuristic: does this query likely need tool/function calls?

        Checks for keywords associated with NOVA's tools (time, volume,
        search, etc.) using prefix matching so Indonesian suffixed words
        like "lagunya", "putarkan", "musiknya" are correctly detected.

        Errs toward False — conversational queries get the faster
        streaming path; missed tool queries still work via the model's
        text response (just without tool results).
        """
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        for word in words:
            for keyword in _TOOL_KEYWORDS:
                # Prefix match: "lagunya".startswith("lagu") → True
                # Also exact: "lagu".startswith("lagu") → True
                if word.startswith(keyword) and len(keyword) >= 3:
                    return True
                # Exact match for short keywords (2 chars like "ip")
                if len(keyword) < 3 and word == keyword:
                    return True
        return False

    async def _respond_streaming(
        self, user_input: str,
    ) -> tuple[str, float] | None:
        """Streaming path: LLM stream → TTS stream.

        Returns (response_text, total_time) or None if streaming fails.
        Falls back gracefully so the caller can retry with the tool path.
        """
        # Inject memory context into prompt assembler
        memory_context = await self._get_memory_context(user_input)
        if memory_context:
            get_prompt_assembler().set_memory_context(memory_context)

        context = self._context.get_context()
        provider = self._llm_router.providers[0]

        try:
            start = time.perf_counter()
            sentence_stream = provider.generate_stream(user_input, context)
            full_text, tts_time = await self._streaming_tts.stream_from_llm(
                sentence_stream, self._tts_router, language="auto",
            )
            total = time.perf_counter() - start

            if not full_text:
                return None

            logger.info(
                "Streaming response: %.2fs total (TTS: %.2fs), %d chars",
                total, tts_time, len(full_text),
            )
            return full_text, total
        except Exception:
            logger.warning("Streaming path failed, will fall back", exc_info=True)
            return None

    async def _respond_with_tools(
        self, user_input: str,
    ) -> tuple[str, float]:
        """Standard tool path: LLM with function calling → streaming TTS.

        Returns (response_text, total_time). Raises on total failure.
        """
        start = time.perf_counter()

        response, llm_time = await self.process_text(user_input)
        language = detect_language(response)
        tts_time = await self.speak(response, language=language)
        total = time.perf_counter() - start

        logger.info(
            "Tool response: %.2fs total (LLM: %.2fs, TTS: %.2fs), %d chars",
            total, llm_time, tts_time, len(response),
        )
        return response, total

    async def process_text(self, text: str) -> tuple[str, float]:
        """Send text to the LLM and return the response with timing.

        Args:
            text: User's input text.

        Returns:
            Tuple of (response text, elapsed seconds).
        """
        context = self._context.get_context()

        # Inject memory context into prompt assembler
        memory_context = await self._get_memory_context(text)
        if memory_context:
            get_prompt_assembler().set_memory_context(memory_context)

        start = time.perf_counter()
        response = await self._llm_router.execute(
            "generate", text, context, self._tools,
        )
        elapsed = time.perf_counter() - start

        logger.info("LLM responded in %.2fs (%d chars)", elapsed, len(response))
        return response, elapsed

    async def speak(self, text: str, language: str | None = None) -> float:
        """Convert text to speech and play through speakers.

        Uses streaming TTS: splits into sentences, synthesizes and plays
        the first sentence immediately while synthesizing the rest in
        the background for gapless playback.

        Args:
            text: Text to speak aloud.
            language: Language code, or None to auto-detect.

        Returns:
            TTS time in seconds (0.0 on failure).
        """
        lang = language or self._default_language

        try:
            tts_elapsed = await self._streaming_tts.synthesize_and_play(
                text, self._tts_router, language=lang,
            )
            return tts_elapsed
        except AllProvidersFailedError:
            logger.error("All TTS providers failed — response printed only")
            return 0.0
        except Exception:
            logger.exception("TTS playback error — response printed only")
            return 0.0

    async def handle_interaction(self, user_input: str) -> str:
        """Process a full text interaction: LLM response + TTS playback.

        Routes between two paths:
        - Streaming: LLM streams sentences → TTS plays each immediately.
          Used for conversational queries. Lowest time-to-first-audio.
        - Tool: Full LLM response with function calling → streaming TTS.
          Used when the query likely needs tools.

        Args:
            user_input: The user's text input.

        Returns:
            The assistant's response text, or error message.
        """
        self._interaction_count += 1
        interaction_id = self._interaction_count

        # Warmup TTS on first interaction (non-blocking)
        if not self._tts_warmed_up:
            asyncio.ensure_future(self._warmup_tts())

        use_tools = self._likely_needs_tools(user_input)

        try:
            if use_tools:
                logger.info(
                    "[#%d] Tool path (keywords detected)", interaction_id,
                )
                response, total_time = await self._respond_with_tools(user_input)
            else:
                # Try streaming first; fall back to tools on failure
                result = await self._respond_streaming(user_input)
                if result is not None:
                    response, total_time = result
                    logger.info(
                        "[#%d] Streaming path succeeded", interaction_id,
                    )
                else:
                    logger.info(
                        "[#%d] Streaming failed, falling back to tool path",
                        interaction_id,
                    )
                    response, total_time = await self._respond_with_tools(user_input)

        except AllProvidersFailedError:
            logger.error("[#%d] All LLM providers failed", interaction_id)
            return "Semua layanan sedang sibuk, coba lagi sebentar."
        except Exception:
            logger.exception("[#%d] Unexpected error", interaction_id)
            return "Terjadi kesalahan, tapi saya masih berjalan."

        # Store in context (async)
        await self._context.add_exchange(user_input, response)

        logger.info(
            "Interaction #%d complete — %.2fs | %r → %d chars",
            interaction_id, total_time,
            user_input[:80], len(response),
        )

        return response

    async def handle_voice_interaction(self) -> str | None:
        """Process a full voice interaction: capture -> STT -> LLM -> TTS -> playback.

        After STT, routes through the same dual-path as handle_interaction:
        streaming for conversational queries, tool path for tool queries.

        Returns:
            The assistant's response text, None if no speech, or error string.
        """
        self._interaction_count += 1
        interaction_id = self._interaction_count
        total_start = time.perf_counter()

        # Warmup TTS on first interaction (non-blocking)
        if not self._tts_warmed_up:
            asyncio.ensure_future(self._warmup_tts())

        # 1. Capture audio from microphone
        try:
            audio_capture = self._get_audio_capture()
            wav_bytes = await audio_capture.capture()
        except OSError as e:
            logger.error("[#%d] Audio device error: %s", interaction_id, e)
            return "__AUDIO_DEVICE_ERROR__"
        except Exception as e:
            logger.exception("[#%d] Audio capture error: %s", interaction_id, e)
            return "__AUDIO_DEVICE_ERROR__"

        # Check for empty audio (no speech detected)
        if len(wav_bytes) <= 44:
            return None

        # 2. STT: transcribe audio
        stt_start = time.perf_counter()
        try:
            transcript = await self._stt_router.execute("transcribe", wav_bytes)
            stt_time = time.perf_counter() - stt_start
        except AllProvidersFailedError:
            logger.error("[#%d] All STT providers failed", interaction_id)
            return "__STT_FAILED__"
        except Exception:
            logger.exception("[#%d] STT error", interaction_id)
            return "__STT_FAILED__"

        if not transcript or not transcript.strip():
            return None

        transcript = transcript.strip()

        # 3. LLM → TTS: route through streaming or tool path
        use_tools = self._likely_needs_tools(transcript)

        try:
            if use_tools:
                response, _ = await self._respond_with_tools(transcript)
            else:
                result = await self._respond_streaming(transcript)
                if result is not None:
                    response, _ = result
                else:
                    response, _ = await self._respond_with_tools(transcript)
        except AllProvidersFailedError:
            logger.error("[#%d] All LLM providers failed", interaction_id)
            return "Semua layanan sedang sibuk, coba lagi sebentar."
        except Exception:
            logger.exception("[#%d] Unexpected LLM/TTS error", interaction_id)
            return "Terjadi kesalahan, tapi saya masih berjalan."

        # 4. Store in context (async)
        await self._context.add_exchange(transcript, response)

        total_time = time.perf_counter() - total_start
        logger.info(
            "Voice #%d complete — STT: %.2fs | Total: %.2fs | %r → %d chars",
            interaction_id, stt_time, total_time,
            transcript[:80], len(response),
        )

        return response

    @property
    def last_transcript(self) -> str | None:
        """Return the last user message from context, if any."""
        context = self._context.get_context()
        for msg in reversed(context):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def clear_context(self) -> None:
        """Reset the conversation history."""
        self._context.clear()

    async def check_providers(self) -> dict[str, dict[str, bool | str]]:
        """Check connectivity to all providers, mic, and audio player.

        Returns:
            Dict mapping component names to their status info.
        """
        results: dict[str, dict[str, bool | str]] = {}

        # Check each provider in all routers
        for router in [self._stt_router, self._llm_router, self._tts_router]:
            for provider in router.providers:
                try:
                    available = await provider.is_available()
                    results[f"{router.provider_type}/{provider.name}"] = {
                        "available": available,
                        "status": "connected" if available else "not configured",
                    }
                except Exception as e:
                    results[f"{router.provider_type}/{provider.name}"] = {
                        "available": False,
                        "status": f"error: {e}",
                    }

        # Check microphone
        try:
            import sounddevice as sd
            default_input = sd.query_devices(kind="input")
            results["microphone"] = {
                "available": True,
                "status": f"detected ({default_input['name']})",
            }
        except Exception as e:
            results["microphone"] = {
                "available": False,
                "status": f"not found: {e}",
            }

        # Check audio player
        for player in ["mpv", "ffplay", "aplay"]:
            if shutil.which(player):
                results["audio_player"] = {
                    "available": True,
                    "status": f"{player} installed",
                }
                break
        else:
            results["audio_player"] = {
                "available": False,
                "status": "no player found (install mpv)",
            }

        return results
