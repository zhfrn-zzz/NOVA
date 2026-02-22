"""Cloudflare Workers AI TTS provider — fallback TTS.

This provider is optional — only used if Cloudflare credentials are configured.
Uses Cloudflare's Workers AI text-to-speech endpoint.
"""

import logging

import httpx

from nova.config import get_config
from nova.providers.base import (
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    TTSProvider,
)

logger = logging.getLogger(__name__)

# Cloudflare Workers AI TTS model
_TTS_MODEL = "@cf/myshell-ai/melotts"


class CloudflareTTSProvider(TTSProvider):
    """Text-to-Speech using Cloudflare Workers AI (optional fallback)."""

    name = "cloudflare_tts"

    def __init__(self) -> None:
        config = get_config()
        self._account_id = config.cloudflare_account_id
        self._api_token = config.cloudflare_api_token
        self._timeout = config.tts_timeout

    def _get_url(self) -> str:
        """Build the Cloudflare Workers AI endpoint URL."""
        return (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{_TTS_MODEL}"
        )

    async def synthesize(self, text: str, language: str = "id") -> bytes:
        """Convert text to audio bytes via Cloudflare Workers AI.

        Args:
            text: Text to speak.
            language: Language code ("id" or "en").

        Returns:
            Audio bytes.

        Raises:
            RateLimitError: On HTTP 429.
            ProviderError: On server or API errors.
            ProviderTimeoutError: When request exceeds tts_timeout.
        """
        if not self._account_id or not self._api_token:
            raise ProviderError(self.name, "Cloudflare credentials not configured")

        # Map language to a speaker/lang parameter
        lang_map = {"id": "id", "en": "en"}
        target_lang = lang_map.get(language, "en")

        payload = {
            "text": text,
            "lang": target_lang,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._get_url(),
                    headers={
                        "Authorization": f"Bearer {self._api_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                retry_val = float(retry_after) if retry_after else None
                raise RateLimitError(self.name, retry_after=retry_val)

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

            audio_bytes = response.content
            if not audio_bytes or len(audio_bytes) < 100:
                raise ProviderError(self.name, "Cloudflare TTS returned empty audio")

            logger.debug("Cloudflare TTS: synthesized %d bytes", len(audio_bytes))
            return audio_bytes

        except (RateLimitError, ProviderError, ProviderTimeoutError):
            raise
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(self.name, self._timeout) from e
        except Exception as e:
            raise ProviderError(self.name, f"Synthesis failed: {e}") from e

    async def is_available(self) -> bool:
        """Check if Cloudflare credentials are configured."""
        if not self._account_id or not self._api_token:
            return False
        # Quick connectivity check
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.cloudflare.com/client/v4/user/tokens/verify",
                    headers={"Authorization": f"Bearer {self._api_token}"},
                )
            return response.status_code == 200
        except Exception:
            logger.warning("Cloudflare TTS: availability check failed")
            return False
