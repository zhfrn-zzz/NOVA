"""Tests for DeepTalk continuous conversation mode."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nova.deeptalk.session import (
    DEEPTALK_TRIGGERS,
    EXIT_PHRASES,
    DeepTalkSession,
    is_deeptalk_trigger,
    is_exit_phrase,
)
from nova.providers.stt.groq_whisper import _has_mixed_scripts

# ── Exit phrase detection ────────────────────────────────────────────


class TestExitPhrase:
    def test_exact_match(self):
        assert is_exit_phrase("selesai") is True
        assert is_exit_phrase("stop") is True
        assert is_exit_phrase("keluar") is True
        assert is_exit_phrase("berhenti") is True

    def test_case_insensitive(self):
        assert is_exit_phrase("STOP") is True
        assert is_exit_phrase("Selesai") is True
        assert is_exit_phrase("KELUAR") is True

    def test_with_prefix(self):
        assert is_exit_phrase("Nova selesai") is True
        assert is_exit_phrase("nova selesai") is True

    def test_with_trailing_words(self):
        assert is_exit_phrase("selesai dong") is True
        assert is_exit_phrase("stop please") is True

    def test_whitespace(self):
        assert is_exit_phrase("  selesai  ") is True
        assert is_exit_phrase("  stop  ") is True

    def test_empty_string(self):
        assert is_exit_phrase("") is False

    def test_not_exit_command(self):
        # "selesaikan" contains "selesai" but is NOT a word-boundary match
        assert is_exit_phrase("tolong selesaikan tugas ini") is False
        assert is_exit_phrase("selesaikan pekerjaan") is False

    def test_regular_sentences(self):
        assert is_exit_phrase("apa cuaca hari ini") is False
        assert is_exit_phrase("nyalakan AC") is False
        assert is_exit_phrase("ceritakan tentang python") is False

    def test_exit_deep_talk_phrase(self):
        assert is_exit_phrase("exit deep talk") is True
        assert is_exit_phrase("exit deeptalk") is True
        assert is_exit_phrase("mode biasa") is True

    def test_all_phrases_recognized(self):
        for phrase in EXIT_PHRASES:
            assert is_exit_phrase(phrase) is True, f"Missed: {phrase}"


# ── DeepTalk trigger detection ───────────────────────────────────────


class TestDeepTalkTrigger:
    def test_triggers_detected(self):
        assert is_deeptalk_trigger("mode deeptalk") is True
        assert is_deeptalk_trigger("mulai deep talk") is True
        assert is_deeptalk_trigger("deep talk mode") is True

    def test_case_insensitive(self):
        assert is_deeptalk_trigger("Mode DeepTalk") is True
        assert is_deeptalk_trigger("MULAI DEEPTALK") is True

    def test_with_surrounding_text(self):
        assert is_deeptalk_trigger("hey nova mode deeptalk please") is True

    def test_non_triggers(self):
        assert is_deeptalk_trigger("apa itu deeptalk") is False
        assert is_deeptalk_trigger("halo nova") is False
        assert is_deeptalk_trigger("nyalakan AC") is False

    def test_all_triggers_recognized(self):
        for trigger in DEEPTALK_TRIGGERS:
            assert is_deeptalk_trigger(trigger) is True, f"Missed: {trigger}"


# ── Barge-in detector lifecycle ──────────────────────────────────────


class TestBargeInDetector:
    @pytest.fixture
    def mock_detector(self):
        """Create a BargeInDetector with mocked ONNX session."""
        mock_session = MagicMock()
        mock_input_h = MagicMock(name="h_input")
        mock_input_h.name = "h"
        mock_input_c = MagicMock(name="c_input")
        mock_input_c.name = "c"
        mock_input_sr = MagicMock(name="sr_input")
        mock_input_sr.name = "sr"
        mock_input_in = MagicMock(name="audio_input")
        mock_input_in.name = "input"
        mock_session.get_inputs.return_value = [
            mock_input_in, mock_input_h, mock_input_c, mock_input_sr,
        ]

        with patch("nova.deeptalk.barge_in._ensure_vad_model", return_value="fake.onnx"):
            with patch(
                "nova.deeptalk.barge_in.onnxruntime.InferenceSession",
                return_value=mock_session,
            ):
                from nova.deeptalk.barge_in import BargeInDetector

                detector = BargeInDetector()
                return detector

    def test_start_stop_lifecycle(self, mock_detector):
        """Start and stop monitoring without errors."""
        # Override _monitor_loop to avoid opening real audio stream
        mock_detector._monitor_loop = lambda: None
        mock_detector.start_monitoring()
        assert mock_detector._monitoring is True
        mock_detector.stop_monitoring()
        assert mock_detector._monitoring is False
        assert mock_detector._thread is None

    def test_not_monitoring_by_default(self, mock_detector):
        assert mock_detector._monitoring is False

    def test_double_start_is_safe(self, mock_detector):
        mock_detector._monitor_loop = lambda: None
        mock_detector.start_monitoring()
        mock_detector.start_monitoring()  # Should not start a second thread
        mock_detector.stop_monitoring()


# ── DeepTalk session ─────────────────────────────────────────────────


class TestDeepTalkSession:
    @pytest.fixture
    def mock_orchestrator(self):
        orch = AsyncMock()
        orch.speak = AsyncMock()
        orch.handle_interaction = AsyncMock(return_value="Test response")
        orch.stop_speaking = MagicMock()
        orch.reset_speaking = MagicMock()
        orch._stt_router = AsyncMock()
        return orch

    @pytest.fixture
    def mock_barge_in(self):
        bi = MagicMock()
        bi.start_monitoring = MagicMock()
        bi.stop_monitoring = MagicMock()
        bi.stop_capture = MagicMock()
        return bi

    def test_initial_state(self, mock_orchestrator, mock_barge_in):
        session = DeepTalkSession(mock_orchestrator, mock_barge_in)
        assert session.is_active is False

    @pytest.mark.asyncio
    async def test_exit_phrase_ends_session(self, mock_orchestrator, mock_barge_in):
        """Session ends when an exit phrase is transcribed."""
        session = DeepTalkSession(mock_orchestrator, mock_barge_in)

        # Mock capture to return audio, STT to return exit phrase
        with patch.object(
            session, "_capture_audio", new_callable=AsyncMock,
            return_value=b"\x00" * 100,
        ):
            with patch.object(
                session, "_transcribe", new_callable=AsyncMock,
                return_value="selesai",
            ):
                await session.start()

        assert session.is_active is False
        # Should announce entry and exit
        assert mock_orchestrator.speak.call_count == 2

    @pytest.mark.asyncio
    async def test_normal_turn_processes_interaction(
        self, mock_orchestrator, mock_barge_in,
    ):
        """A normal turn calls handle_interaction then loops."""
        session = DeepTalkSession(mock_orchestrator, mock_barge_in)
        call_count = 0

        async def _mock_capture():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                # After 2 turns, simulate exit
                return b"\x00" * 100
            return b"\x00" * 100

        async def _mock_transcribe(wav):
            nonlocal call_count
            if call_count >= 3:
                return "selesai"
            return "apa kabar"

        with patch.object(session, "_capture_audio", side_effect=_mock_capture):
            with patch.object(session, "_transcribe", side_effect=_mock_transcribe):
                await session.start()

        # 2 normal interactions before exit
        assert mock_orchestrator.handle_interaction.call_count == 2
        assert mock_barge_in.start_monitoring.call_count == 2
        assert mock_barge_in.stop_monitoring.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_audio_skipped(self, mock_orchestrator, mock_barge_in):
        """Empty audio (no speech) is skipped without error."""
        session = DeepTalkSession(mock_orchestrator, mock_barge_in)
        call_count = 0

        async def _capture():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"\x00" * 10  # Too short (< 44 byte WAV header)
            return b"\x00" * 100

        async def _transcribe(wav):
            return "selesai"

        with patch.object(session, "_capture_audio", side_effect=_capture):
            with patch.object(session, "_transcribe", side_effect=_transcribe):
                await session.start()

        # First capture was empty, only one transcribe call
        mock_orchestrator.handle_interaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_transcript_skipped(self, mock_orchestrator, mock_barge_in):
        """Empty transcript (STT returned nothing) is skipped."""
        session = DeepTalkSession(mock_orchestrator, mock_barge_in)
        call_count = 0

        async def _capture():
            return b"\x00" * 100

        async def _transcribe(wav):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ""
            return "selesai"

        with patch.object(session, "_capture_audio", side_effect=_capture):
            with patch.object(session, "_transcribe", side_effect=_transcribe):
                await session.start()

        mock_orchestrator.handle_interaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_capture_skipped(self, mock_orchestrator, mock_barge_in):
        """None from VAD capture (no speech detected) is skipped."""
        session = DeepTalkSession(mock_orchestrator, mock_barge_in)
        call_count = 0

        async def _capture():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # VAD found no speech
            return b"\x00" * 100

        async def _transcribe(wav):
            return "selesai"

        with patch.object(session, "_capture_audio", side_effect=_capture):
            with patch.object(session, "_transcribe", side_effect=_transcribe):
                await session.start()

        mock_orchestrator.handle_interaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_barge_in_calls_stop_speaking(
        self, mock_orchestrator, mock_barge_in,
    ):
        """Barge-in callback calls stop_speaking on orchestrator."""
        session = DeepTalkSession(mock_orchestrator, mock_barge_in)
        session._handle_barge_in()
        mock_orchestrator.stop_speaking.assert_called_once()


# ── Mixed-script hallucination detection ─────────────────────────────


class TestMixedScripts:
    """Tests for _has_mixed_scripts (Whisper hallucination Gate 4)."""

    def test_latin_only(self):
        assert _has_mixed_scripts("Hello, how are you?") is False

    def test_indonesian_latin(self):
        assert _has_mixed_scripts("Selamat pagi, apa kabar hari ini?") is False

    def test_mixed_three_scripts(self):
        # Latin + Korean + CJK
        text = "Hello 겠죠 那个"
        assert _has_mixed_scripts(text) is True

    def test_whisper_hallucination_sample(self):
        text = (
            "Ayo ouya...inhos ot predisahkan tak expecting it to be easy... "
            "Ayo, siellä ol겠죠 et akhirnya tidak akan mir lesukan langsung... "
            "Whereas b чет- underneath那"
        )
        assert _has_mixed_scripts(text) is True

    def test_two_scripts_ok(self):
        # Latin + Cyrillic only = 2 scripts, should not trigger
        assert _has_mixed_scripts("hello мир") is False

    def test_empty_string(self):
        assert _has_mixed_scripts("") is False

    def test_numbers_and_punctuation_only(self):
        assert _has_mixed_scripts("123-456, 789!") is False

    def test_arabic_mix(self):
        # Latin + CJK + Arabic = 3 scripts
        assert _has_mixed_scripts("test 那 ام") is True


# ── VAD capture_speech ───────────────────────────────────────────────


class TestCaptureSpeech:
    """Tests for BargeInDetector.capture_speech()."""

    @pytest.fixture
    def detector(self):
        """BargeInDetector with mocked ONNX session."""
        mock_session = MagicMock()
        mock_input_h = MagicMock()
        mock_input_h.name = "h"
        mock_input_c = MagicMock()
        mock_input_c.name = "c"
        mock_input_sr = MagicMock()
        mock_input_sr.name = "sr"
        mock_input_in = MagicMock()
        mock_input_in.name = "input"
        mock_session.get_inputs.return_value = [
            mock_input_in, mock_input_h, mock_input_c, mock_input_sr,
        ]

        with patch("nova.deeptalk.barge_in._ensure_vad_model", return_value="fake.onnx"):
            with patch(
                "nova.deeptalk.barge_in.onnxruntime.InferenceSession",
                return_value=mock_session,
            ):
                from nova.deeptalk.barge_in import BargeInDetector

                det = BargeInDetector()
                return det

    def test_stop_capture_flag(self, detector):
        """stop_capture sets the flag that ends capture_speech."""
        assert detector._capturing is False
        detector._capturing = True
        detector.stop_capture()
        assert detector._capturing is False

    def test_capture_returns_none_on_no_speech(self, detector):
        """capture_speech returns None when stopped before speech."""
        chunk_count = 0
        fake_audio = np.zeros((512, 1), dtype=np.int16)

        def _mock_read(n):
            nonlocal chunk_count
            chunk_count += 1
            if chunk_count > 5:
                detector._capturing = False
            return fake_audio, False

        detector._predict = MagicMock(return_value=0.0)

        mock_stream = MagicMock()
        mock_stream.read = _mock_read
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        with patch("nova.deeptalk.barge_in.sd.InputStream", return_value=mock_stream):
            result = detector.capture_speech()

        assert result is None

    def test_capture_returns_wav_on_speech(self, detector):
        """capture_speech returns WAV bytes when speech is detected."""
        chunk_count = 0
        fake_audio = np.full((512, 1), 1000, dtype=np.int16)

        def _mock_read(n):
            nonlocal chunk_count
            chunk_count += 1
            return fake_audio, False

        def _mock_predict(chunk):
            # First 15 chunks: speech (enough for min_speech_duration=0.3s)
            # Then 50+ chunks silence (enough for silence_duration=1.5s)
            if chunk_count <= 15:
                return 0.9
            return 0.0

        detector._predict = _mock_predict

        mock_stream = MagicMock()
        mock_stream.read = _mock_read
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        with patch("nova.deeptalk.barge_in.sd.InputStream", return_value=mock_stream):
            result = detector.capture_speech(silence_duration=1.0)

        assert result is not None
        assert len(result) > 44  # WAV header + audio data

    def test_capture_resets_state(self, detector):
        """State is reset after capture completes."""
        detector._capturing = True
        detector._predict = MagicMock(return_value=0.0)

        fake_audio = np.zeros((512, 1), dtype=np.int16)
        mock_stream = MagicMock()

        call_count = 0

        def _mock_read(n):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                detector._capturing = False
            return fake_audio, False

        mock_stream.read = _mock_read
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        with patch("nova.deeptalk.barge_in.sd.InputStream", return_value=mock_stream):
            detector.capture_speech()

        assert detector._capturing is False
