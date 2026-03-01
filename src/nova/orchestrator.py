"""Core pipeline orchestrator — coordinates STT, LLM, TTS, and context.

Uses a single unified streaming path: the LLM always streams with tools
enabled.  When the model decides to call a function, tool execution
happens inline inside ``generate_stream()`` — no keyword heuristics
needed.
"""

import asyncio
import logging
import os
import re
import shutil
import time
from collections.abc import Callable

from nova.audio.streaming_tts import StreamingTTSPlayer
from nova.config import get_config
from nova.heartbeat.queue import Notification, NotificationQueue
from nova.heartbeat.scheduler import HeartbeatScheduler
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

        # --- Heartbeat scheduler ---
        self._notification_queue = NotificationQueue()
        self._heartbeat_scheduler = HeartbeatScheduler(
            memory_store=self._memory_store,
            notification_queue=self._notification_queue,
        )
        if config.heartbeat_enabled:
            self._heartbeat_scheduler.start()

        # Text-only flag (set by main.py when in text mode)
        self._text_only = False

        logger.info(
            "Orchestrator initialized — LLM: %s | TTS: %s | STT: %s | Embedder: %s | Heartbeat: %s",
            [p.name for p in llm_providers],
            [p.name for p in tts_providers],
            [p.name for p in stt_providers],
            "enabled" if embedder else "disabled",
            "enabled" if config.heartbeat_enabled else "disabled",
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

    def _inject_passive_notifications(self) -> None:
        """Check for passive notifications and inject into prompt context."""
        passive = self._notification_queue.get_passive()
        if passive:
            note_text = self._format_notifications(passive)
            get_prompt_assembler().set_notification_context(note_text)
            logger.info(
                "Injected %d passive notification(s) into LLM context",
                len(passive),
            )

    @staticmethod
    def _format_notifications(notifications: list[Notification]) -> str:
        """Format notifications for LLM context injection.

        Args:
            notifications: List of Notification objects.

        Returns:
            Formatted string for the LLM to incorporate naturally.
        """
        lines: list[str] = []
        for n in notifications:
            if n.message == "__morning_greeting__":
                lines.append("Deliver a brief morning greeting to the user.")
            elif n.message == "__sleep_reminder__":
                lines.append(
                    "Gently remind the user it's late and they should rest."
                )
            else:
                lines.append(f"Remind the user: {n.message}")
        return "\n".join(lines)

    async def _respond(
        self, user_input: str,
    ) -> tuple[str, float] | None:
        """Unified streaming path: LLM stream (with tools) → TTS stream.

        The LLM provider streams with tools enabled.  If the model emits a
        function call, it is executed inline inside ``generate_stream()`` and
        a new stream is started with the tool result — all transparent to
        the caller.

        Returns (response_text, total_time) or None if streaming fails.
        Falls back gracefully so the caller can retry with the fallback path.
        """
        # Inject memory context into prompt assembler
        memory_context = await self._get_memory_context(user_input)
        if memory_context:
            get_prompt_assembler().set_memory_context(memory_context)

        # Inject passive notifications (heartbeat)
        self._inject_passive_notifications()

        context = self._context.get_context()
        provider = self._llm_router.providers[0]

        try:
            start = time.perf_counter()
            sentence_stream = provider.generate_stream(
                user_input, context, tools=self._tools,
            )
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

    async def _respond_fallback(
        self, user_input: str,
    ) -> tuple[str, float]:
        """Non-streaming fallback: LLM with function calling → streaming TTS.

        Used when streaming fails (e.g., provider doesn't support streaming).

        Returns (response_text, total_time). Raises on total failure.
        """
        start = time.perf_counter()

        response, llm_time = await self.process_text(user_input)
        language = detect_language(response)
        tts_time = await self.speak(response, language=language)
        total = time.perf_counter() - start

        logger.info(
            "Fallback response: %.2fs total (LLM: %.2fs, TTS: %.2fs), %d chars",
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

        # Inject passive notifications (heartbeat)
        self._inject_passive_notifications()

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

        Uses a single unified streaming path with tools always enabled.
        The LLM decides whether to call tools — no keyword heuristics.
        Falls back to non-streaming if streaming fails.

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

        try:
            # Primary path: unified streaming with tools
            result = await self._respond(user_input)
            if result is not None:
                response, total_time = result
                logger.info(
                    "[#%d] Streaming path succeeded", interaction_id,
                )
            else:
                # Streaming failed — fall back to non-streaming
                logger.info(
                    "[#%d] Streaming failed, falling back to non-streaming",
                    interaction_id,
                )
                response, total_time = await self._respond_fallback(user_input)

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

        Uses the same unified streaming path as handle_interaction.

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

        # 3. LLM → TTS: unified streaming path with tools
        try:
            result = await self._respond(transcript)
            if result is not None:
                response, _ = result
            else:
                response, _ = await self._respond_fallback(transcript)
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

    def stop(self) -> None:
        """Shut down background systems (heartbeat scheduler)."""
        self._heartbeat_scheduler.stop()

    def set_ambient_fn(self, fn: Callable[[], float]) -> None:
        """Set the ambient RMS function for the heartbeat scheduler.

        Called by main.py after the wake word detector is initialized,
        so the scheduler can use ambient noise as a presence heuristic.

        Args:
            fn: Callable returning current ambient RMS level.
        """
        self._heartbeat_scheduler._ambient_fn = fn

    @property
    def notification_queue(self) -> NotificationQueue:
        """Return the notification queue for main loop access."""
        return self._notification_queue

    async def deliver_notification(self, notification: Notification) -> str:
        """Generate and speak a notification message via LLM + TTS.

        Used for ACTIVE notifications that need to be spoken immediately.

        Args:
            notification: The notification to deliver.

        Returns:
            The generated notification text.
        """
        if notification.message == "__morning_greeting__":
            prompt = "Deliver a brief, warm morning greeting."
        elif notification.message == "__sleep_reminder__":
            prompt = "Gently remind the user it's late and time to rest."
        else:
            prompt = f"Deliver this reminder concisely: {notification.message}"

        # Use LLM to generate natural wording, then TTS
        try:
            provider = self._llm_router.providers[0]
            start = time.perf_counter()
            sentence_stream = provider.generate_stream(
                prompt, context=[], tools=None,
            )
            full_text, tts_time = await self._streaming_tts.stream_from_llm(
                sentence_stream, self._tts_router, language="auto",
            )
            elapsed = time.perf_counter() - start
            logger.info(
                "Notification delivered: %.2fs (TTS: %.2fs) — %r",
                elapsed, tts_time, (full_text or "")[:80],
            )
            return full_text or prompt
        except Exception:
            logger.exception("Failed to deliver notification via LLM+TTS")
            # Fallback: just speak the raw message
            try:
                await self.speak(notification.message)
            except Exception:
                logger.warning("TTS fallback also failed", exc_info=True)
            return notification.message

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
