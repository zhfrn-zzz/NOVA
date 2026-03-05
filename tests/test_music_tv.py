"""Tests for TV music playback via WebOS YouTube deep-link."""

from unittest.mock import AsyncMock, patch

import pytest

from nova.tools.music_player import _search_youtube, play_music


# ── _search_youtube helper ──────────────────────────────────────────


@pytest.mark.asyncio
class TestSearchYouTube:
    """Tests for the yt-dlp search helper."""

    @patch("nova.tools.music_player.asyncio.create_subprocess_exec")
    async def test_returns_first_video_id(self, mock_exec):
        proc = AsyncMock()
        proc.communicate.return_value = (b"dQw4w9WgXcQ\nabc123\n", b"")
        mock_exec.return_value = proc

        result = await _search_youtube("Never Gonna Give You Up")
        assert result == "dQw4w9WgXcQ"

    @patch("nova.tools.music_player.asyncio.create_subprocess_exec")
    async def test_returns_none_on_empty(self, mock_exec):
        proc = AsyncMock()
        proc.communicate.return_value = (b"", b"no results")
        mock_exec.return_value = proc

        result = await _search_youtube("xyznonexistent")
        assert result is None

    @patch(
        "nova.tools.music_player.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError,
    )
    async def test_returns_none_when_ytdlp_missing(self, _mock):
        result = await _search_youtube("test")
        assert result is None


# ── play_music with TV target ───────────────────────────────────────


@pytest.mark.asyncio
class TestPlayMusicTV:
    """Tests for play_music(target='tv_atas'/'tv_bawah')."""

    @patch("nova.tools.music_player._search_youtube", return_value="abc123")
    async def test_tv_atas_plays_youtube(self, _mock_search):
        """Happy path: search → connect → play on TV Atas."""
        driver = AsyncMock()
        driver.play_youtube.return_value = "Memutar YouTube di TV Atas (video: abc123)."

        with patch("nova.tools.iot.get_tv_atas_webos", return_value=driver):
            result = await play_music("About You", target="tv_atas")

        assert "abc123" in result
        driver.play_youtube.assert_awaited_once_with("abc123")

    @patch("nova.tools.music_player._search_youtube", return_value="abc123")
    async def test_tv_bawah_plays_youtube(self, _mock_search):
        """TV Bawah happy path."""
        driver = AsyncMock()
        driver.play_youtube.return_value = "Memutar YouTube di TV Bawah (video: abc123)."

        with patch("nova.tools.iot.get_tv_bawah_webos", return_value=driver):
            result = await play_music("About You", target="tv_bawah")

        assert "abc123" in result

    @patch("nova.tools.music_player._search_youtube", return_value=None)
    async def test_tv_search_fails(self, _mock):
        """If yt-dlp returns nothing, report not found."""
        result = await play_music("xyznonexist", target="tv_atas")
        assert "Tidak menemukan" in result

    @patch("nova.tools.music_player._search_youtube", return_value="abc123")
    async def test_tv_atas_no_driver_configured(self, _mock):
        """If WebOS driver env vars are missing, report config error."""
        with patch("nova.tools.iot.get_tv_atas_webos", return_value=None):
            result = await play_music("test", target="tv_atas")
        assert "belum dikonfigurasi" in result

    @patch("nova.tools.music_player._search_youtube", return_value="abc123")
    async def test_tv_atas_auto_power_on(self, _mock_search):
        """If TV Atas is unreachable, IR power-on → retry."""
        driver = AsyncMock()
        driver.play_youtube.side_effect = [
            ConnectionError("TV off"),
            "Memutar YouTube di TV Atas (video: abc123).",
        ]
        mock_ir = AsyncMock(return_value="Power sent")

        with (
            patch("nova.tools.iot.get_tv_atas_webos", return_value=driver),
            patch("nova.tools.iot.tv_atas_ir", mock_ir),
            patch("nova.tools.music_player.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await play_music("test", target="tv_atas")

        assert "abc123" in result
        mock_ir.assert_awaited_once_with("Power")
        assert driver.play_youtube.await_count == 2

    @patch("nova.tools.music_player._search_youtube", return_value="abc123")
    async def test_tv_bawah_cannot_auto_power_on(self, _mock_search):
        """TV Bawah has no IR hub — must tell user to power on manually."""
        driver = AsyncMock()
        driver.play_youtube.side_effect = ConnectionError("TV off")

        with patch("nova.tools.iot.get_tv_bawah_webos", return_value=driver):
            result = await play_music("test", target="tv_bawah")

        assert "tidak bisa dinyalakan otomatis" in result.lower() or "manual" in result.lower()


# ── play_music local (unchanged behavior) ───────────────────────────


@pytest.mark.asyncio
class TestPlayMusicLocal:
    """Ensure local playback still works as before."""

    @patch("nova.tools.music_player._search_youtube", return_value="dQw4w9WgXcQ")
    @patch("nova.tools.music_player.webbrowser.open")
    async def test_local_opens_browser(self, mock_open, _mock_search):
        result = await play_music("Never Gonna Give You Up")
        assert "music.youtube.com" in result
        mock_open.assert_called_once()

    async def test_empty_query(self):
        result = await play_music("")
        assert "Tidak ada lagu" in result
