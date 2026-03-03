"""Tests for TTS audio cache — LRU eviction, byte limits, key isolation."""

from nova.audio.tts_cache import TTSCache


class TestTTSCache:
    def test_cache_miss_returns_none(self):
        cache = TTSCache()
        assert cache.get("hello", "voice-a", "en") is None

    def test_cache_hit_returns_audio(self):
        cache = TTSCache()
        audio = b"\xff" * 100
        cache.put("hello", "voice-a", "en", audio)
        assert cache.get("hello", "voice-a", "en") == audio

    def test_different_voice_is_different_entry(self):
        cache = TTSCache()
        audio_a = b"\xaa" * 50
        audio_b = b"\xbb" * 50
        cache.put("hello", "voice-a", "en", audio_a)
        cache.put("hello", "voice-b", "en", audio_b)
        assert cache.get("hello", "voice-a", "en") == audio_a
        assert cache.get("hello", "voice-b", "en") == audio_b

    def test_different_language_is_different_entry(self):
        cache = TTSCache()
        audio_en = b"\x01" * 50
        audio_id = b"\x02" * 50
        cache.put("hello", "voice-a", "en", audio_en)
        cache.put("hello", "voice-a", "id", audio_id)
        assert cache.get("hello", "voice-a", "en") == audio_en
        assert cache.get("hello", "voice-a", "id") == audio_id

    def test_lru_eviction_by_max_entries(self):
        cache = TTSCache(max_entries=2, max_bytes=10 * 1024 * 1024)
        cache.put("a", "v", "en", b"\x01")
        cache.put("b", "v", "en", b"\x02")
        cache.put("c", "v", "en", b"\x03")  # evicts "a"
        assert cache.get("a", "v", "en") is None
        assert cache.get("b", "v", "en") == b"\x02"
        assert cache.get("c", "v", "en") == b"\x03"

    def test_lru_access_promotes_entry(self):
        cache = TTSCache(max_entries=2, max_bytes=10 * 1024 * 1024)
        cache.put("a", "v", "en", b"\x01")
        cache.put("b", "v", "en", b"\x02")
        # Access "a" to promote it
        cache.get("a", "v", "en")
        cache.put("c", "v", "en", b"\x03")  # evicts "b" (LRU)
        assert cache.get("a", "v", "en") == b"\x01"
        assert cache.get("b", "v", "en") is None
        assert cache.get("c", "v", "en") == b"\x03"

    def test_byte_limit_eviction(self):
        cache = TTSCache(max_entries=100, max_bytes=200)
        cache.put("a", "v", "en", b"\x00" * 100)
        cache.put("b", "v", "en", b"\x00" * 100)
        assert cache.size == 2
        assert cache.bytes_used == 200
        # Adding 100 more bytes should evict "a"
        cache.put("c", "v", "en", b"\x00" * 100)
        assert cache.get("a", "v", "en") is None
        assert cache.size == 2
        assert cache.bytes_used == 200

    def test_overwrite_existing_entry(self):
        cache = TTSCache()
        cache.put("hello", "v", "en", b"\x01" * 50)
        cache.put("hello", "v", "en", b"\x02" * 80)
        assert cache.get("hello", "v", "en") == b"\x02" * 80
        assert cache.size == 1
        assert cache.bytes_used == 80

    def test_size_and_bytes_tracking(self):
        cache = TTSCache()
        assert cache.size == 0
        assert cache.bytes_used == 0
        cache.put("a", "v", "en", b"\x00" * 50)
        assert cache.size == 1
        assert cache.bytes_used == 50
        cache.put("b", "v", "en", b"\x00" * 75)
        assert cache.size == 2
        assert cache.bytes_used == 125
