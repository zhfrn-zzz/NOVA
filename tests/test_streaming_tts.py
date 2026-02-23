"""Tests for streaming TTS — sentence splitting and overlapped playback."""

from unittest.mock import AsyncMock, patch

import pytest

from nova.audio.streaming_tts import StreamingTTSPlayer, split_sentences


class TestSplitSentences:
    def test_empty_string(self):
        assert split_sentences("") == []

    def test_single_sentence_no_period(self):
        assert split_sentences("Halo saya Nova") == ["Halo saya Nova"]

    def test_single_sentence_with_period(self):
        assert split_sentences("Halo saya Nova.") == ["Halo saya Nova."]

    def test_two_short_sentences_merged(self):
        # Both sentences are < 40 chars, so they get merged into one
        result = split_sentences("Halo saya Nova. Saya bisa membantu Anda.")
        assert len(result) == 1
        assert "Halo saya Nova." in result[0]
        assert "Saya bisa membantu Anda." in result[0]

    def test_exclamation_and_question(self):
        result = split_sentences("Halo! Apa kabar? Saya baik.")
        # All fragments are < 40 chars — everything merges into one
        assert len(result) == 1
        assert "Halo!" in result[0]
        assert "Apa kabar?" in result[0]
        assert "Saya baik." in result[0]

    def test_abbreviation_not_split(self):
        result = split_sentences("Dr. Budi mengatakan hal penting.")
        # "Dr." should NOT be a sentence break
        assert len(result) == 1
        assert "Dr." in result[0]

    def test_dll_abbreviation(self):
        result = split_sentences(
            "Ada buku, pensil, dll. yang perlu dibeli. Jangan lupa."
        )
        # "dll." should NOT be a sentence break, and total < 40 chars per part
        # so everything merges into one
        assert len(result) == 1
        assert "dll." in result[0]

    def test_short_fragment_merged(self):
        result = split_sentences("Ok. Saya akan membantu Anda sekarang.")
        # "Ok." is too short (<10 chars), should be merged
        assert len(result) == 1
        assert "Ok." in result[0]

    def test_multiple_long_sentences(self):
        text = (
            "Baterai Anda saat ini di 75 persen. "
            "Sedang mengisi daya melalui kabel USB. "
            "Diperkirakan penuh dalam satu jam."
        )
        result = split_sentences(text)
        # Each sentence is ~35 chars (< 40), so first merges with second (now > 40),
        # but the merge logic keeps accumulating into the buffer.
        # The buffer after first two merges is > 70 chars, so third merges as
        # trailing fragment (< 40 chars). Result: 1 sentence.
        assert len(result) == 1

    def test_preserves_content(self):
        text = "Halo! Saya Nova. Senang berkenalan!"
        result = split_sentences(text)
        joined = " ".join(result)
        # All original words should be present
        for word in ["Halo!", "Saya", "Nova.", "Senang", "berkenalan!"]:
            assert word in joined

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_decimal_number_not_split(self):
        result = split_sentences("Harganya 3.500 rupiah. Cukup murah.")
        # "3.500" should NOT cause a split, and both parts are < 40 chars
        # so they merge into one
        assert len(result) == 1
        assert "3.500" in result[0]

    def test_short_sentences_merged_under_40(self):
        """Short sentences (< 40 chars) merge to reduce TTS API calls."""
        result = split_sentences(
            "Mohon maaf, Tuan. Anda benar, ini sudah sore."
        )
        # Both fragments are < 40 chars → merged into one
        assert len(result) == 1
        assert result[0] == "Mohon maaf, Tuan. Anda benar, ini sudah sore."


class TestStreamingTTSPlayer:
    @pytest.mark.asyncio
    async def test_empty_text_returns_zero(self):
        player = StreamingTTSPlayer()
        mock_router = AsyncMock()
        result = await player.synthesize_and_play("", mock_router, "id")
        assert result == 0.0
        mock_router.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_sentence_calls_tts_once(self):
        player = StreamingTTSPlayer()
        mock_router = AsyncMock()
        mock_router.execute.return_value = b"fake-audio-bytes"

        with patch("nova.audio.streaming_tts.play_audio", new_callable=AsyncMock):
            result = await player.synthesize_and_play(
                "Halo saya Nova.", mock_router, "id",
            )

        assert result > 0.0
        mock_router.execute.assert_called_once_with(
            "synthesize", "Halo saya Nova.", "id",
        )

    @pytest.mark.asyncio
    async def test_multi_sentence_calls_tts_per_sentence(self):
        player = StreamingTTSPlayer()
        mock_router = AsyncMock()
        mock_router.execute.return_value = b"fake-audio-bytes"

        # Use long sentences (>= 40 chars each) so they don't merge
        with patch("nova.audio.streaming_tts.play_audio", new_callable=AsyncMock):
            result = await player.synthesize_and_play(
                "Baterai laptop Anda saat ini sedang terisi penuh. "
                "Sedang mengisi daya melalui kabel USB yang terhubung.",
                mock_router, "id",
            )

        assert result > 0.0
        # Both sentences are >= 40 chars, so they stay separate → 2 TTS calls
        assert mock_router.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_synthesis_failure_skips_sentence(self):
        player = StreamingTTSPlayer()
        mock_router = AsyncMock()
        # First call fails, second succeeds
        mock_router.execute.side_effect = [
            Exception("TTS error"),
            b"fake-audio-bytes",
        ]

        # Use long sentences (>= 40 chars) so they don't merge
        with patch(
            "nova.audio.streaming_tts.play_audio", new_callable=AsyncMock,
        ) as mock_play:
            await player.synthesize_and_play(
                "Kalimat pertama ini akan mengalami kegagalan. "
                "Kalimat kedua ini berhasil disintesis dengan baik.",
                mock_router, "id",
            )

        # Only the second sentence should have been played
        mock_play.assert_called_once_with(b"fake-audio-bytes")
