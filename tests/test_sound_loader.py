"""Tests for custom sound loader — file loading, fallback, caching."""

import io
import wave

import numpy as np
import pytest

from nova.audio.sound_loader import (
    audio_to_wav_bytes,
    clear_cache,
    load_sound,
)


@pytest.fixture(autouse=True)
def _clear_sound_cache():
    """Clear the sound cache before each test."""
    clear_cache()
    yield
    clear_cache()


def _dummy_fallback(sample_rate: int = 22050, **kwargs) -> np.ndarray:
    """Generate a simple sine wave as fallback."""
    t = np.linspace(0, 0.1, int(sample_rate * 0.1), dtype=np.float64)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    return (audio * 32767).astype(np.int16)


class TestLoadSoundFallback:
    def test_fallback_when_no_custom_file(self, tmp_path, monkeypatch):
        """Should use fallback_fn when no custom file exists."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)
        audio = load_sound("beep", _dummy_fallback)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16
        assert len(audio) > 0

    def test_fallback_passes_sample_rate(self, tmp_path, monkeypatch):
        """Fallback function should receive sample_rate."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)
        received = {}

        def capture_fn(sample_rate: int = 22050, **kwargs):
            received["sr"] = sample_rate
            return _dummy_fallback(sample_rate=sample_rate)

        load_sound("test", capture_fn, sample_rate=44100)
        assert received["sr"] == 44100

    def test_fallback_passes_extra_kwargs(self, tmp_path, monkeypatch):
        """Extra kwargs should be forwarded to fallback_fn."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)
        received = {}

        def capture_fn(sample_rate: int = 22050, volume: float = 0.5):
            received["volume"] = volume
            return _dummy_fallback(sample_rate=sample_rate)

        load_sound("test", capture_fn, sample_rate=22050, volume=0.8)
        assert received["volume"] == 0.8


class TestLoadSoundCustomFile:
    def test_loads_wav_file(self, tmp_path, monkeypatch):
        """Should load a custom .wav file."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)

        # Create a valid WAV file
        sr = 22050
        samples = (np.sin(np.linspace(0, 1, sr)) * 32767).astype(np.int16)
        wav_path = tmp_path / "beep.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())

        audio = load_sound("beep", _dummy_fallback, sample_rate=sr)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16
        assert len(audio) > 0

    def test_stereo_converted_to_mono(self, tmp_path, monkeypatch):
        """Stereo files should be mixed down to mono."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)

        sr = 22050
        n = sr  # 1 second
        stereo = np.column_stack([
            (np.sin(np.linspace(0, 10, n)) * 0.5).astype(np.float32),
            (np.sin(np.linspace(0, 15, n)) * 0.5).astype(np.float32),
        ])

        wav_path = tmp_path / "chime.wav"

        try:
            import soundfile as sf
            sf.write(str(wav_path), stereo, sr)
        except ImportError:
            pytest.skip("soundfile not installed")

        audio = load_sound("chime", _dummy_fallback, sample_rate=sr)
        assert audio.ndim == 1  # mono

    def test_resamples_different_rate(self, tmp_path, monkeypatch):
        """Files with different sample rate should be resampled."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)

        source_sr = 44100
        target_sr = 22050
        n = source_sr  # 1 second at 44100

        audio_data = (np.sin(np.linspace(0, 10, n)) * 0.5).astype(np.float32)
        wav_path = tmp_path / "alert.wav"

        try:
            import soundfile as sf
            sf.write(str(wav_path), audio_data, source_sr)
        except ImportError:
            pytest.skip("soundfile not installed")

        audio = load_sound("alert", _dummy_fallback, sample_rate=target_sr)
        # Should be roughly half the length (44100→22050)
        expected_len = int(n * target_sr / source_sr)
        assert abs(len(audio) - expected_len) <= 1

    def test_corrupt_file_falls_back(self, tmp_path, monkeypatch):
        """Corrupt audio file should fall back to generated sound."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)

        corrupt_path = tmp_path / "beep.wav"
        corrupt_path.write_bytes(b"not a wav file at all")

        audio = load_sound("beep", _dummy_fallback)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0


class TestCaching:
    def test_cached_on_second_call(self, tmp_path, monkeypatch):
        """Second call should return cached result without re-loading."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)
        call_count = {"n": 0}

        def counting_fallback(sample_rate: int = 22050, **kwargs):
            call_count["n"] += 1
            return _dummy_fallback(sample_rate=sample_rate)

        load_sound("beep", counting_fallback)
        load_sound("beep", counting_fallback)
        assert call_count["n"] == 1

    def test_clear_cache_reloads(self, tmp_path, monkeypatch):
        """After clear_cache, sound should be re-loaded."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)
        call_count = {"n": 0}

        def counting_fallback(sample_rate: int = 22050, **kwargs):
            call_count["n"] += 1
            return _dummy_fallback(sample_rate=sample_rate)

        load_sound("beep", counting_fallback)
        clear_cache()
        load_sound("beep", counting_fallback)
        assert call_count["n"] == 2


class TestConfigDisabled:
    def test_skips_file_when_disabled(self, tmp_path, monkeypatch):
        """When custom_sounds_enabled=False, should skip file loading."""
        monkeypatch.setattr("nova.audio.sound_loader._get_sounds_dir", lambda: tmp_path)

        # Create a custom file that would normally be loaded
        sr = 22050
        samples = (np.sin(np.linspace(0, 1, sr)) * 32767).astype(np.int16)
        wav_path = tmp_path / "beep.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())

        # Disable custom sounds via config
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.custom_sounds_enabled = False
        with patch("nova.config.get_config", return_value=mock_config):
            fallback_called = {"yes": False}

            def tracked_fallback(sample_rate: int = 22050, **kwargs):
                fallback_called["yes"] = True
                return _dummy_fallback(sample_rate=sample_rate)

            load_sound("beep", tracked_fallback)
            assert fallback_called["yes"]


class TestAudioToWavBytes:
    def test_valid_wav_output(self):
        """audio_to_wav_bytes should produce valid WAV data."""
        audio = _dummy_fallback(sample_rate=16000)
        wav_data = audio_to_wav_bytes(audio, sample_rate=16000)

        # Verify it's valid WAV
        buf = io.BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == len(audio)

    def test_roundtrip(self):
        """WAV bytes should roundtrip back to same audio."""
        original = _dummy_fallback(sample_rate=22050)
        wav_data = audio_to_wav_bytes(original, sample_rate=22050)

        buf = io.BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        recovered = np.frombuffer(raw, dtype=np.int16)

        np.testing.assert_array_equal(original, recovered)


class TestDirectoryCreation:
    def test_sounds_dir_created(self, tmp_path, monkeypatch):
        """load_sound should create the sounds dir if it doesn't exist."""
        sounds_dir = tmp_path / "nova_sounds"
        monkeypatch.setattr(
            "nova.audio.sound_loader._get_sounds_dir", lambda: sounds_dir,
        )

        assert not sounds_dir.exists()
        load_sound("beep", _dummy_fallback)
        assert sounds_dir.exists()
