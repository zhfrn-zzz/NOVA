"""Core pipeline orchestrator — coordinates STT, LLM, TTS, and context."""

import logging
import shutil
import time

from nova.audio.playback import play_audio
from nova.config import get_config
from nova.memory.context import ConversationContext
from nova.providers.base import AllProvidersFailedError
from nova.providers.llm.gemini import GeminiProvider
from nova.providers.llm.groq_llm import GroqLLMProvider
from nova.providers.router import ProviderRouter
from nova.providers.stt.groq_whisper import GroqWhisperProvider
from nova.providers.tts.cloudflare_tts import CloudflareTTSProvider
from nova.providers.tts.edge_tts_provider import EdgeTTSProvider, detect_language

logger = logging.getLogger(__name__)


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

        # TTS providers: Edge TTS (primary), Cloudflare (fallback, if configured)
        tts_providers: list = [EdgeTTSProvider()]
        if config.cloudflare_account_id and config.cloudflare_api_token:
            tts_providers.append(CloudflareTTSProvider())

        # Create routers
        self._stt_router = ProviderRouter("STT", stt_providers)
        self._llm_router = ProviderRouter("LLM", llm_providers)
        self._tts_router = ProviderRouter("TTS", tts_providers)

        # Audio capture (lazy — created on first voice interaction)
        self._audio_capture = None

        # Conversation memory
        self._context = ConversationContext(max_turns=config.max_context_turns)
        self._default_language = config.default_language

        # Interaction counter for logging
        self._interaction_count = 0

        logger.info(
            "Orchestrator initialized — LLM: %s | TTS: %s | STT: %s",
            [p.name for p in llm_providers],
            [p.name for p in tts_providers],
            [p.name for p in stt_providers],
        )

    def _get_audio_capture(self):
        """Lazy-init AudioCapture to avoid import errors when mic is missing."""
        if self._audio_capture is None:
            from nova.audio.capture import AudioCapture
            self._audio_capture = AudioCapture()
        return self._audio_capture

    async def process_text(self, text: str) -> tuple[str, float]:
        """Send text to the LLM and return the response with timing.

        Args:
            text: User's input text.

        Returns:
            Tuple of (response text, elapsed seconds).
        """
        context = self._context.get_context()

        start = time.perf_counter()
        response = await self._llm_router.execute("generate", text, context)
        elapsed = time.perf_counter() - start

        logger.info("LLM responded in %.2fs (%d chars)", elapsed, len(response))
        return response, elapsed

    async def speak(self, text: str, language: str | None = None) -> float:
        """Convert text to speech and play through speakers.

        Args:
            text: Text to speak aloud.
            language: Language code, or None to auto-detect.

        Returns:
            TTS synthesis time in seconds (0.0 on failure).
        """
        lang = language or self._default_language

        start = time.perf_counter()
        try:
            audio_bytes = await self._tts_router.execute("synthesize", text, lang)
            tts_elapsed = time.perf_counter() - start
            logger.info("TTS synthesized in %.2fs (%d bytes)", tts_elapsed, len(audio_bytes))
            await play_audio(audio_bytes)
            return tts_elapsed
        except AllProvidersFailedError:
            logger.error("All TTS providers failed — response printed only")
            return 0.0
        except Exception:
            logger.exception("TTS playback error — response printed only")
            return 0.0

    async def handle_interaction(self, user_input: str) -> str:
        """Process a full text interaction: LLM response + TTS playback.

        Handles all failure modes gracefully so NOVA never crashes.

        Args:
            user_input: The user's text input.

        Returns:
            The assistant's response text, or error message.
        """
        self._interaction_count += 1
        interaction_id = self._interaction_count

        try:
            # Get LLM response
            response, llm_time = await self.process_text(user_input)
        except AllProvidersFailedError:
            logger.error("[Interaction #%d] All LLM providers failed", interaction_id)
            return "Semua layanan sedang sibuk, coba lagi sebentar."
        except Exception:
            logger.exception("[Interaction #%d] Unexpected LLM error", interaction_id)
            return "Terjadi kesalahan, tapi saya masih berjalan."

        # Detect language from the response for TTS
        language = detect_language(response)

        # Speak the response (non-blocking failure — prints only if TTS fails)
        tts_time = await self.speak(response, language=language)

        # Store in context
        self._context.add_exchange(user_input, response)

        # Per-interaction summary log
        logger.info(
            "Interaction #%d complete\n"
            "  LLM: %.2fs | TTS: %.2fs | Total: %.2fs\n"
            "  Input: %r | Response: %d chars",
            interaction_id,
            llm_time, tts_time, llm_time + tts_time,
            user_input[:80], len(response),
        )

        return response

    async def handle_voice_interaction(self) -> str | None:
        """Process a full voice interaction: capture -> STT -> LLM -> TTS -> playback.

        Handles all failure modes:
        - Audio device not found → returns special sentinel
        - No speech detected → returns None
        - STT failure → returns sentinel for text fallback
        - LLM/TTS failures → graceful error messages

        Returns:
            The assistant's response text, None if no speech, or error string.
        """
        self._interaction_count += 1
        interaction_id = self._interaction_count
        total_start = time.perf_counter()

        # 1. Capture audio from microphone
        try:
            audio_capture = self._get_audio_capture()
            wav_bytes = await audio_capture.capture()
        except OSError as e:
            logger.error("[Interaction #%d] Audio device error: %s", interaction_id, e)
            return "__AUDIO_DEVICE_ERROR__"
        except Exception as e:
            logger.exception("[Interaction #%d] Audio capture error: %s", interaction_id, e)
            return "__AUDIO_DEVICE_ERROR__"

        # Check for empty audio (no speech detected)
        # WAV header is 44 bytes; anything near that size means no real audio
        if len(wav_bytes) <= 44:
            return None

        # 2. STT: transcribe audio
        stt_start = time.perf_counter()
        try:
            transcript = await self._stt_router.execute("transcribe", wav_bytes)
            stt_time = time.perf_counter() - stt_start
        except AllProvidersFailedError:
            logger.error("[Interaction #%d] All STT providers failed", interaction_id)
            return "__STT_FAILED__"
        except Exception:
            logger.exception("[Interaction #%d] STT error", interaction_id)
            return "__STT_FAILED__"

        if not transcript or not transcript.strip():
            return None

        transcript = transcript.strip()

        # 3. LLM: generate response
        try:
            response, llm_time = await self.process_text(transcript)
        except AllProvidersFailedError:
            logger.error("[Interaction #%d] All LLM providers failed", interaction_id)
            return "Semua layanan sedang sibuk, coba lagi sebentar."
        except Exception:
            logger.exception("[Interaction #%d] Unexpected LLM error", interaction_id)
            return "Terjadi kesalahan, tapi saya masih berjalan."

        # 4. Detect language for TTS
        language = detect_language(response)

        # 5. TTS: synthesize and play
        tts_time = await self.speak(response, language=language)

        # 6. Store in context
        self._context.add_exchange(transcript, response)

        total_time = time.perf_counter() - total_start
        logger.info(
            "Voice interaction #%d complete\n"
            "  STT: %.2fs | LLM: %.2fs | TTS: %.2fs | Total: %.2fs\n"
            "  Input: %r | Response: %d chars",
            interaction_id,
            stt_time, llm_time, tts_time, total_time,
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
