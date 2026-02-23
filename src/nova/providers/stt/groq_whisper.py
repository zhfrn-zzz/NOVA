"""Groq Whisper STT provider — primary speech-to-text using Groq API.

Includes anti-hallucination measures:
  1. RMS energy gate — skip API call if audio is near-silent
  2. Hallucination phrase filter — reject known Whisper false positives
  3. verbose_json response format — filter by no_speech_prob metadata
"""

import io
import logging
import struct
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

# Minimum RMS energy to consider audio as containing speech.
# Below this threshold the API call is skipped entirely.
_MIN_RMS_THRESHOLD = 200

# Maximum allowed no_speech_prob from Whisper verbose_json segments.
# Segments above this are treated as hallucinations.
_NO_SPEECH_PROB_THRESHOLD = 0.7

# Common Whisper hallucination phrases (lowercased).
# Matches are checked as exact full-string OR first-3-word prefix.
HALLUCINATION_PHRASES: list[str] = [
    "terima kasih",
    "thank you",
    "thanks for watching",
    "subscribe",
    "like and subscribe",
    "see you next time",
    "sampai jumpa",
    "selamat tinggal",
    "bye bye",
    "subtitles by",
    "translated by",
    "amara.org",
    "salam",
    "assalamualaikum",
]


def _compute_rms(audio_bytes: bytes) -> float:
    """Compute RMS energy of raw WAV audio bytes.

    Reads the WAV header to find PCM data, then calculates root-mean-square
    of 16-bit samples.

    Returns:
        RMS value (float). Returns 0.0 on any parse error.
    """
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return 0.0
            raw = wf.readframes(n_frames)

        # Unpack as signed 16-bit little-endian samples
        n_samples = len(raw) // 2
        if n_samples == 0:
            return 0.0
        samples = struct.unpack(f"<{n_samples}h", raw)
        sum_sq = sum(s * s for s in samples)
        return (sum_sq / n_samples) ** 0.5
    except Exception:
        logger.warning("RMS computation failed, proceeding with API call")
        return float("inf")  # fail-open: don't block the call


def _is_hallucination(text: str) -> bool:
    """Check whether *text* matches a known Whisper hallucination.

    Two matching strategies:
      1. Exact match after strip+lower.
      2. First 3 words of the transcript match any phrase exactly.
    """
    normalised = text.strip().lower()
    if not normalised:
        return False

    # Exact full-string match
    if normalised in HALLUCINATION_PHRASES:
        return True

    # First-3-word prefix match
    prefix = " ".join(normalised.split()[:3])
    for phrase in HALLUCINATION_PHRASES:
        if prefix == phrase:
            return True

    return False


class GroqWhisperProvider(STTProvider):
    """Speech-to-Text using Groq's Whisper API.

    Anti-hallucination pipeline:
      1. RMS gate — reject near-silent audio before the API call.
      2. verbose_json — request segment-level no_speech_prob.
      3. Phrase filter — catch common Whisper phantom outputs.
    """

    name = "groq_whisper"

    def __init__(self) -> None:
        config = get_config()
        self._api_key = config.groq_api_key
        self._timeout = config.stt_timeout
        self._language = config.stt_language
        self._prompt = (
            "Ini adalah percakapan dalam bahasa Indonesia dan English."
        )

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Convert WAV audio bytes to text via Groq Whisper API.

        Args:
            audio_bytes: Raw WAV audio data (16kHz, mono, 16-bit PCM).

        Returns:
            Transcribed text string (empty string when no speech detected).

        Raises:
            RateLimitError: On HTTP 429.
            ProviderError: On server or API errors.
            ProviderTimeoutError: When request exceeds stt_timeout.
        """
        if not self._api_key:
            raise ProviderError(self.name, "Groq API key not configured")

        # --- Gate 1: minimum audio energy ---
        rms = _compute_rms(audio_bytes)
        if rms < _MIN_RMS_THRESHOLD:
            logger.debug(
                "Groq Whisper: audio RMS %.1f < %d — skipping API call",
                rms, _MIN_RMS_THRESHOLD,
            )
            return ""

        # Build multipart form data
        form_data: dict[str, str] = {
            "model": _MODEL,
            "prompt": self._prompt,
            "response_format": "verbose_json",
        }
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

            # --- Gate 2: no_speech_prob filtering ---
            segments = data.get("segments", [])
            if segments:
                avg_no_speech = sum(
                    s.get("no_speech_prob", 0.0) for s in segments
                ) / len(segments)
                if avg_no_speech > _NO_SPEECH_PROB_THRESHOLD:
                    logger.debug(
                        "Groq Whisper: avg no_speech_prob %.2f > %.2f — "
                        "rejecting transcript %r",
                        avg_no_speech, _NO_SPEECH_PROB_THRESHOLD, text,
                    )
                    return ""

            # --- Gate 3: hallucination phrase filter ---
            if text and _is_hallucination(text):
                logger.info(
                    "Groq Whisper: rejected hallucination %r", text,
                )
                return ""

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
