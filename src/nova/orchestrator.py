"""Core pipeline orchestrator — coordinates STT, LLM, TTS, and context."""

import logging
import time

from nova.audio.capture import AudioCapture
from nova.audio.playback import play_audio
from nova.config import get_config
from nova.memory.context import ConversationContext
from nova.providers.base import AllProvidersFailedError
from nova.providers.llm.gemini import GeminiProvider
from nova.providers.router import ProviderRouter
from nova.providers.stt.groq_whisper import GroqWhisperProvider
from nova.providers.tts.edge_tts_provider import EdgeTTSProvider, detect_language

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the full NOVA pipeline: STT -> LLM -> TTS -> playback."""

    def __init__(self) -> None:
        config = get_config()

        # Initialize providers
        self._stt_router = ProviderRouter("STT", [GroqWhisperProvider()])
        self._llm_router = ProviderRouter("LLM", [GeminiProvider()])
        self._tts_router = ProviderRouter("TTS", [EdgeTTSProvider()])

        # Audio capture
        self._audio_capture = AudioCapture()

        # Conversation memory
        self._context = ConversationContext(max_turns=config.max_context_turns)
        self._default_language = config.default_language

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

        Args:
            user_input: The user's text input.

        Returns:
            The assistant's response text.
        """
        # Get LLM response
        response, llm_time = await self.process_text(user_input)

        # Detect language from the response for TTS
        language = detect_language(response)

        # Speak the response (non-blocking failure — prints only if TTS fails)
        tts_time = await self.speak(response, language=language)

        # Store in context
        self._context.add_exchange(user_input, response)

        logger.info(
            "Interaction complete [LLM: %.2fs | TTS: %.2fs | Total: %.2fs] | context: %d turns",
            llm_time, tts_time, llm_time + tts_time, self._context.turn_count,
        )

        return response

    async def handle_voice_interaction(self) -> str | None:
        """Process a full voice interaction: capture -> STT -> LLM -> TTS -> playback.

        Returns:
            The assistant's response text, or None if no speech was detected.
        """
        total_start = time.perf_counter()

        # 1. Capture audio from microphone
        wav_bytes = await self._audio_capture.capture()

        # Check for empty audio (no speech detected)
        # WAV header is 44 bytes; anything near that size means no real audio
        if len(wav_bytes) <= 44:
            return None

        # 2. STT: transcribe audio
        stt_start = time.perf_counter()
        transcript = await self._stt_router.execute("transcribe", wav_bytes)
        stt_time = time.perf_counter() - stt_start

        if not transcript or not transcript.strip():
            return None

        transcript = transcript.strip()

        # 3. LLM: generate response
        response, llm_time = await self.process_text(transcript)

        # 4. Detect language for TTS
        language = detect_language(response)

        # 5. TTS: synthesize and play
        tts_time = await self.speak(response, language=language)

        # 6. Store in context
        self._context.add_exchange(transcript, response)

        total_time = time.perf_counter() - total_start
        logger.info(
            "Voice interaction complete "
            "[STT: %.2fs | LLM: %.2fs | TTS: %.2fs | Total: %.2fs] | context: %d turns",
            stt_time, llm_time, tts_time, total_time, self._context.turn_count,
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
