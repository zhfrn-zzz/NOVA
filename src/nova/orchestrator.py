"""Core pipeline orchestrator — coordinates STT, LLM, TTS, and context."""

import logging
import time

from nova.audio.playback import play_audio
from nova.config import get_config
from nova.memory.context import ConversationContext
from nova.providers.base import AllProvidersFailedError
from nova.providers.llm.gemini import GeminiProvider
from nova.providers.router import ProviderRouter
from nova.providers.tts.edge_tts_provider import EdgeTTSProvider, detect_language

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the full NOVA pipeline: STT -> LLM -> TTS -> playback."""

    def __init__(self) -> None:
        config = get_config()

        # Initialize providers
        self._llm_router = ProviderRouter("LLM", [GeminiProvider()])
        self._tts_router = ProviderRouter("TTS", [EdgeTTSProvider()])

        # Conversation memory
        self._context = ConversationContext(max_turns=config.max_context_turns)
        self._default_language = config.default_language

    async def process_text(self, text: str) -> str:
        """Send text to the LLM and return the response.

        Args:
            text: User's input text.

        Returns:
            LLM response string.
        """
        context = self._context.get_context()

        start = time.perf_counter()
        response = await self._llm_router.execute("generate", text, context)
        elapsed = time.perf_counter() - start

        logger.info("LLM responded in %.2fs (%d chars)", elapsed, len(response))
        return response

    async def speak(self, text: str, language: str | None = None) -> None:
        """Convert text to speech and play through speakers.

        Args:
            text: Text to speak aloud.
            language: Language code, or None to auto-detect.
        """
        lang = language or self._default_language

        start = time.perf_counter()
        try:
            audio_bytes = await self._tts_router.execute("synthesize", text, lang)
            tts_elapsed = time.perf_counter() - start
            logger.info("TTS synthesized in %.2fs (%d bytes)", tts_elapsed, len(audio_bytes))
            await play_audio(audio_bytes)
        except AllProvidersFailedError:
            logger.error("All TTS providers failed — response printed only")
        except Exception:
            logger.exception("TTS playback error — response printed only")

    async def handle_interaction(self, user_input: str) -> str:
        """Process a full text interaction: LLM response + TTS playback.

        Args:
            user_input: The user's text input.

        Returns:
            The assistant's response text.
        """
        start = time.perf_counter()

        # Get LLM response
        response = await self.process_text(user_input)

        # Detect language from the response for TTS
        language = detect_language(response)

        # Speak the response (non-blocking failure — prints only if TTS fails)
        await self.speak(response, language=language)

        # Store in context
        self._context.add_exchange(user_input, response)

        total = time.perf_counter() - start
        logger.info(
            "Interaction complete: %.2fs total | context: %d turns",
            total, self._context.turn_count,
        )

        return response

    def clear_context(self) -> None:
        """Reset the conversation history."""
        self._context.clear()
