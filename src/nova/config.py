"""NOVA configuration system — typed settings loaded from .env."""

import logging

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_config_instance: "NovaConfig | None" = None


class NovaConfig(BaseSettings):
    """All NOVA settings, loaded from environment variables with NOVA_ prefix."""

    # API Keys
    gemini_api_key: str = ""
    groq_api_key: str = ""
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""

    # Audio
    sample_rate: int = 16000
    channels: int = 1
    silence_threshold: float = 0.03
    silence_duration: float = 1.5
    max_recording_seconds: float = 15.0

    # Provider priorities
    stt_providers: list[str] = ["groq", "cloudflare"]
    llm_providers: list[str] = ["gemini", "groq", "cloudflare"]
    tts_providers: list[str] = ["edge", "cloudflare"]

    # Timeouts (seconds)
    stt_timeout: float = 10.0
    llm_timeout: float = 15.0
    tts_timeout: float = 10.0

    # Conversation
    max_context_turns: int = 10
    default_language: str = "auto"

    # System
    log_level: str = "INFO"
    cache_ttl_hours: int = 24

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NOVA_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def validate_api_keys(self) -> None:
        """Validate that at least one LLM API key is configured.

        Raises:
            ValueError: If no LLM API key is set.
        """
        has_gemini = bool(self.gemini_api_key)
        has_groq = bool(self.groq_api_key)
        has_cloudflare = bool(self.cloudflare_account_id and self.cloudflare_api_token)

        if not (has_gemini or has_groq or has_cloudflare):
            raise ValueError(
                "No LLM API key configured. Set at least one of: "
                "NOVA_GEMINI_API_KEY, NOVA_GROQ_API_KEY, or "
                "NOVA_CLOUDFLARE_ACCOUNT_ID + NOVA_CLOUDFLARE_API_TOKEN"
            )

        configured = []
        if has_gemini:
            configured.append("Gemini")
        if has_groq:
            configured.append("Groq")
        if has_cloudflare:
            configured.append("Cloudflare")
        logger.info("Configured LLM providers: %s", ", ".join(configured))


def get_config() -> NovaConfig:
    """Get the singleton NovaConfig instance.

    Returns:
        The shared NovaConfig loaded from environment.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = NovaConfig()
    return _config_instance
