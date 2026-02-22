"""Edge TTS provider — free, unlimited, no API key required."""

import io
import logging
import time

import edge_tts

from nova.providers.base import ProviderError, TTSProvider

logger = logging.getLogger(__name__)

# Voice mapping per language
VOICES = {
    "id": "id-ID-ArdiNeural",
    "en": "en-US-GuyNeural",
}

# Common Indonesian words for simple language detection
_ID_WORDS = {
    "apa", "siapa", "bagaimana", "mengapa", "kenapa", "dimana", "kapan",
    "saya", "kamu", "anda", "dia", "kami", "mereka", "ini", "itu",
    "dan", "atau", "yang", "dengan", "untuk", "dari", "ke", "di",
    "adalah", "bisa", "tidak", "sudah", "akan", "sedang", "telah",
    "halo", "terima", "kasih", "tolong", "mohon", "selamat",
}


def detect_language(text: str) -> str:
    """Detect whether text is Indonesian or English using word heuristics.

    Args:
        text: Input text to classify.

    Returns:
        "id" for Indonesian, "en" for English.
    """
    words = set(text.lower().split())
    id_count = len(words & _ID_WORDS)
    # If 2+ Indonesian words found, assume Indonesian
    if id_count >= 2:
        return "id"
    # Single Indonesian word with short text — still likely Indonesian
    if id_count >= 1 and len(words) <= 5:
        return "id"
    return "en"


class EdgeTTSProvider(TTSProvider):
    """Text-to-Speech using Microsoft Edge TTS (free, unlimited)."""

    name = "edge_tts"

    async def synthesize(self, text: str, language: str = "id") -> bytes:
        """Convert text to MP3 audio bytes.

        Args:
            text: Text to speak.
            language: "id", "en", or "auto" for auto-detection.

        Returns:
            MP3 audio bytes.
        """
        if language == "auto":
            language = detect_language(text)

        voice = VOICES.get(language, VOICES["en"])
        logger.info("EdgeTTS: voice=%s, language=%s, text_len=%d", voice, language, len(text))

        try:
            communicate = edge_tts.Communicate(text, voice)
            buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.write(chunk["data"])
            audio_bytes = buffer.getvalue()
            if not audio_bytes:
                raise ProviderError(self.name, "Edge TTS returned empty audio")
            logger.debug("EdgeTTS: synthesized %d bytes", len(audio_bytes))
            return audio_bytes
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.name, f"Synthesis failed: {e}") from e

    async def warmup(self) -> None:
        """Pre-initialize DNS/SSL caching for faster first real request."""
        try:
            communicate = edge_tts.Communicate("ok", VOICES["en"])
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    break  # Got first audio chunk, connection is warm
            logger.debug("EdgeTTS: connection warmed up")
        except Exception:
            logger.debug("EdgeTTS: warmup failed (non-critical)")

    async def is_available(self) -> bool:
        """Edge TTS is always available (no API key needed)."""
        return True


if __name__ == "__main__":
    import asyncio

    from nova.audio.playback import play_audio

    async def main() -> None:
        provider = EdgeTTSProvider()

        tests = [
            ("Halo, saya Nova, asisten suara pribadi Anda.", "id"),
            ("Hello, I am Nova, your personal voice assistant.", "en"),
        ]

        for text, lang in tests:
            print(f"\n--- Synthesizing ({lang}): {text}")
            start = time.perf_counter()
            audio = await provider.synthesize(text, language=lang)
            elapsed = time.perf_counter() - start
            print(f"    Synthesized {len(audio):,} bytes in {elapsed:.2f}s")

            print("    Playing...")
            await play_audio(audio)
            print("    Done.")

    asyncio.run(main())
