"""Google Cloud TTS provider — Chirp 3 HD voices with quota-based circuit breaker.

Primary TTS provider when credentials are configured. Uses high-quality
Chirp 3 HD voices for Indonesian and English. Falls back automatically
when the monthly quota is near its limit or on API errors.
"""

import asyncio
import logging
import os

from nova.config import get_config
from nova.providers.base import ProviderError, TTSProvider
from nova.providers.tts.edge_tts_provider import detect_language
from nova.utils.tts_quota import TTSQuotaTracker

logger = logging.getLogger(__name__)

# Voice mapping: Chirp 3 HD voices
_VOICES = {
    "id": {"voice_name": "id-ID-Chirp3-HD-Algenib", "language_code": "id-ID"},
    "en": {"voice_name": "en-US-Chirp3-HD-Algenib", "language_code": "en-US"},
}


class GoogleCloudTTSProvider(TTSProvider):
    """Text-to-Speech using Google Cloud TTS with Chirp 3 HD voices.

    Includes a quota-based circuit breaker that tracks monthly character usage
    and raises ProviderError when the quota is near its limit, allowing the
    router to fall back to Edge TTS.
    """

    name = "google_cloud_tts"

    def __init__(self) -> None:
        config = get_config()
        self._key_path = os.path.expanduser(config.google_cloud_tts_key_path)
        self._quota_tracker = TTSQuotaTracker(monthly_limit=config.google_tts_monthly_quota)
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Google Cloud TTS client.

        Returns:
            google.cloud.texttospeech_v1.TextToSpeechClient instance.
        """
        if self._client is None:
            from google.cloud import texttospeech_v1
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                self._key_path,
            )
            self._client = texttospeech_v1.TextToSpeechClient(credentials=credentials)
            logger.info("Google Cloud TTS client initialized")
        return self._client

    def _synthesize_sync(self, text: str, language: str) -> bytes:
        """Synchronous TTS synthesis (runs in thread via asyncio.to_thread).

        Args:
            text: Text to speak.
            language: Language code ("id" or "en").

        Returns:
            MP3 audio bytes.
        """
        from google.cloud import texttospeech_v1

        voice_config = _VOICES.get(language, _VOICES["en"])

        synthesis_input = texttospeech_v1.SynthesisInput(text=text)
        voice = texttospeech_v1.VoiceSelectionParams(
            language_code=voice_config["language_code"],
            name=voice_config["voice_name"],
        )
        audio_config = texttospeech_v1.AudioConfig(
            audio_encoding=texttospeech_v1.AudioEncoding.MP3,
        )

        client = self._get_client()
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config,
        )
        return response.audio_content

    async def synthesize(self, text: str, language: str = "id") -> bytes:
        """Convert text to MP3 audio bytes with quota-based circuit breaker.

        Args:
            text: Text to speak.
            language: "id", "en", or "auto" for auto-detection.

        Returns:
            MP3 audio bytes.

        Raises:
            ProviderError: On quota exceeded or API failure.
        """
        if language == "auto":
            language = detect_language(text)

        # Circuit breaker: check quota
        self._quota_tracker.reset_if_new_month()
        if not self._quota_tracker.can_use(len(text)):
            chars_used, limit, _ = self._quota_tracker.get_usage()
            logger.warning(
                "Google TTS quota near limit (%d/%d), falling back",
                chars_used, limit,
            )
            raise ProviderError(self.name, "quota_exceeded")

        voice_config = _VOICES.get(language, _VOICES["en"])
        logger.info(
            "GoogleCloudTTS: voice=%s, language=%s, text_len=%d",
            voice_config["voice_name"], language, len(text),
        )

        try:
            # Run sync SDK call in a thread to avoid blocking the event loop
            audio_bytes = await asyncio.to_thread(
                self._synthesize_sync, text, language,
            )

            if not audio_bytes:
                raise ProviderError(self.name, "Google TTS returned empty audio")

            # Record successful usage
            self._quota_tracker.record_usage(len(text))
            logger.debug("GoogleCloudTTS: synthesized %d bytes", len(audio_bytes))
            return audio_bytes

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(self.name, f"Synthesis failed: {e}") from e

    async def is_available(self) -> bool:
        """Check if credentials exist and quota is not exceeded."""
        if not self._key_path or not os.path.isfile(self._key_path):
            return False
        self._quota_tracker.reset_if_new_month()
        # Consider unavailable if less than 1000 chars remaining
        return self._quota_tracker.get_remaining() > 1000

    def get_quota_status(self) -> dict[str, object]:
        """Return quota status for --check and --quota CLI commands.

        Returns:
            Dict with keys: configured, chars_used, limit, remaining, month.
        """
        configured = bool(self._key_path and os.path.isfile(self._key_path))
        chars_used, limit, month = self._quota_tracker.get_usage()
        return {
            "configured": configured,
            "chars_used": chars_used,
            "limit": limit,
            "remaining": limit - chars_used,
            "month": month,
        }


if __name__ == "__main__":
    import time

    async def main() -> None:
        provider = GoogleCloudTTSProvider()

        available = await provider.is_available()
        print(f"Available: {available}")

        if not available:
            print("Google Cloud TTS not configured. Set NOVA_GOOGLE_CLOUD_TTS_KEY_PATH.")
            return

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

            from nova.audio.playback import play_audio
            print("    Playing...")
            await play_audio(audio)
            print("    Done.")

        status = provider.get_quota_status()
        print(f"\nQuota: {status['chars_used']:,} / {status['limit']:,} chars used")

    asyncio.run(main())
