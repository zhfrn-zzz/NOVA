"""Tests for the NOVA configuration system."""

import pytest

from nova.config import NovaConfig


class TestNovaConfigDefaults:
    def test_default_values(self):
        config = NovaConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.silence_threshold == 0.03
        assert config.silence_duration == 1.5
        assert config.max_recording_seconds == 15.0
        assert config.stt_timeout == 10.0
        assert config.llm_timeout == 15.0
        assert config.tts_timeout == 10.0
        assert config.max_context_turns == 10
        assert config.default_language == "auto"
        assert config.log_level == "INFO"
        assert config.cache_ttl_hours == 24

    def test_default_provider_priorities(self):
        config = NovaConfig()
        assert config.stt_providers == ["groq", "cloudflare"]
        assert config.llm_providers == ["gemini", "groq", "cloudflare"]
        assert config.tts_providers == ["edge", "cloudflare"]

    def test_default_api_keys_are_empty(self):
        config = NovaConfig()
        assert config.gemini_api_key == ""
        assert config.groq_api_key == ""
        assert config.cloudflare_account_id == ""
        assert config.cloudflare_api_token == ""


class TestNovaConfigFromEnv:
    def test_loads_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("NOVA_GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.setenv("NOVA_GROQ_API_KEY", "test-groq-key")
        monkeypatch.setenv("NOVA_SAMPLE_RATE", "44100")
        monkeypatch.setenv("NOVA_LOG_LEVEL", "DEBUG")

        config = NovaConfig()
        assert config.gemini_api_key == "test-gemini-key"
        assert config.groq_api_key == "test-groq-key"
        assert config.sample_rate == 44100
        assert config.log_level == "DEBUG"


class TestAPIKeyValidation:
    def test_no_keys_raises_error(self):
        config = NovaConfig()
        with pytest.raises(ValueError, match="No LLM API key configured"):
            config.validate_api_keys()

    def test_gemini_key_alone_is_sufficient(self, monkeypatch):
        monkeypatch.setenv("NOVA_GEMINI_API_KEY", "some-key")
        config = NovaConfig()
        config.validate_api_keys()  # should not raise

    def test_groq_key_alone_is_sufficient(self, monkeypatch):
        monkeypatch.setenv("NOVA_GROQ_API_KEY", "some-key")
        config = NovaConfig()
        config.validate_api_keys()  # should not raise

    def test_cloudflare_needs_both_id_and_token(self):
        config = NovaConfig(cloudflare_account_id="id-only")
        with pytest.raises(ValueError, match="No LLM API key configured"):
            config.validate_api_keys()

    def test_cloudflare_both_fields_is_sufficient(self):
        config = NovaConfig(
            cloudflare_account_id="my-id",
            cloudflare_api_token="my-token",
        )
        config.validate_api_keys()  # should not raise


class TestGetConfigSingleton:
    def test_returns_same_instance(self):
        import nova.config as cfg

        # Reset singleton
        cfg._config_instance = None
        c1 = cfg.get_config()
        c2 = cfg.get_config()
        assert c1 is c2

        # Cleanup
        cfg._config_instance = None
