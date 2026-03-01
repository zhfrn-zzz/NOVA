"""Tests for heartbeat audio generation — chime and alert sounds."""

import numpy as np

from nova.heartbeat.audio import generate_alert, generate_chime


class TestGenerateChime:
    def test_returns_int16_array(self):
        audio = generate_chime()
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16

    def test_correct_duration(self):
        sample_rate = 22050
        audio = generate_chime(sample_rate=sample_rate)
        expected_samples = int(sample_rate * 0.6)
        assert len(audio) == expected_samples

    def test_values_in_int16_range(self):
        audio = generate_chime()
        assert audio.min() >= -32768
        assert audio.max() <= 32767

    def test_not_silent(self):
        audio = generate_chime()
        assert np.abs(audio).max() > 0

    def test_volume_affects_amplitude(self):
        quiet = generate_chime(volume=0.1)
        loud = generate_chime(volume=0.9)
        assert np.abs(loud).max() > np.abs(quiet).max()

    def test_custom_sample_rate(self):
        audio = generate_chime(sample_rate=44100)
        expected = int(44100 * 0.6)
        assert len(audio) == expected


class TestGenerateAlert:
    def test_returns_int16_array(self):
        audio = generate_alert()
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16

    def test_correct_duration(self):
        sample_rate = 22050
        audio = generate_alert(sample_rate=sample_rate)
        expected_samples = int(sample_rate * 1.0)
        assert len(audio) == expected_samples

    def test_values_in_int16_range(self):
        audio = generate_alert()
        assert audio.min() >= -32768
        assert audio.max() <= 32767

    def test_not_silent(self):
        audio = generate_alert()
        assert np.abs(audio).max() > 0

    def test_volume_affects_amplitude(self):
        quiet = generate_alert(volume=0.1)
        loud = generate_alert(volume=0.9)
        assert np.abs(loud).max() > np.abs(quiet).max()

    def test_alert_longer_than_chime(self):
        chime = generate_chime()
        alert = generate_alert()
        assert len(alert) > len(chime)
