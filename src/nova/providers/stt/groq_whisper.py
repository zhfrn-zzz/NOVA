"""Groq Whisper STT provider — primary speech-to-text using Groq API."""

import io
import logging
import wave

import httpx

from nova.config import get_config
from nova.providers.base import (
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    STTProvider,
)

logger = logging.getLogger(__name__)

_GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_MODEL = "whisper-large-v3-turbo"


class GroqWhisperProvider(STTProvider):
    """Speech-to-Text using Groq's Whisper API."""

    name = "groq_whisper"

    def __init__(self) -> None:
        config = get_config()
        self._api_key = config.groq_api_key
        self._timeout = config.stt_timeout
        self._language = config.default_language

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Convert WAV audio bytes to text via Groq Whisper API.

        Args:
            audio_bytes: Raw WAV audio data (16kHz, mono, 16-bit PCM).

        Returns:
            Transcribed text string.

        Raises:
            RateLimitError: On HTTP 429.
            ProviderError: On server or API errors.
            ProviderTimeoutError: When request exceeds stt_timeout.
        """
        if not self._api_key:
            raise ProviderError(self.name, "Groq API key not configured")

        # Build multipart form data
        form_data: dict[str, str] = {"model": _MODEL}
        if self._language and self._language != "auto":
            form_data["language"] = self._language

        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    _GROQ_STT_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    data=form_data,
                    files=files,
                )

            if response.status_code == 429:
                retry_after = _parse_retry_after_header(response)
                raise RateLimitError(self.name, retry_after=retry_after)

            if response.status_code >= 500:
                raise ProviderError(
                    self.name,
                    f"Server error {response.status_code}: {response.text}",
                )

            if response.status_code != 200:
                raise ProviderError(
                    self.name,
                    f"API error {response.status_code}: {response.text}",
                )

            data = response.json()
            text = data.get("text", "").strip()

            if not text:
                logger.warning("Groq Whisper: returned empty transcript")

            logger.debug("Groq Whisper: transcribed %d chars", len(text))
            return text

        except (RateLimitError, ProviderError, ProviderTimeoutError):
            raise
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(self.name, self._timeout) from e
        except Exception as e:
            raise ProviderError(self.name, f"Transcription failed: {e}") from e

    async def is_available(self) -> bool:
        """Check if the Groq API key is configured."""
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            return response.status_code == 200
        except Exception:
            logger.warning("Groq Whisper: availability check failed")
            return False


def _parse_retry_after_header(response: httpx.Response) -> float | None:
    """Extract retry-after seconds from response headers."""
    value = response.headers.get("retry-after")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return None


def _record_audio(duration: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Record audio from microphone and return WAV bytes.

    Args:
        duration: Recording duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        WAV-encoded audio bytes (mono, 16-bit PCM).
    """
    import sounddevice as sd

    print(f"  Recording {duration}s of audio...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("  Recording complete.")

    # Convert to WAV bytes using wave module
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return buffer.getvalue()


if __name__ == "__main__":
    import asyncio
    import time

    async def main() -> None:
        provider = GroqWhisperProvider()

        available = await provider.is_available()
        print(f"Groq Whisper available: {available}")
        if not available:
            print("ERROR: Groq API key not configured or invalid.")
            print("Set NOVA_GROQ_API_KEY in your .env file.")
            return

        print("\n=== Groq Whisper STT Test ===")
        print("Speak into your microphone when prompted.\n")

        # Record and transcribe
        print("--- Recording ---")
        audio_bytes = _record_audio(duration=3.0)
        print(f"  Captured {len(audio_bytes):,} bytes of WAV audio")

        print("--- Transcribing ---")
        start = time.perf_counter()
        transcript = await provider.transcribe(audio_bytes)
        elapsed = time.perf_counter() - start

        print(f"  Transcript: {transcript!r}")
        print(f"  Latency: {elapsed:.2f}s")

        print("\n=== Test complete ===")

    asyncio.run(main())
