"""Tests for the weather tool (Open-Meteo API)."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nova.tools.weather import _WEATHER_CODES, _day_name_id, _geocode, get_weather

# Sample API responses for mocking

_MOCK_FORECAST_RESPONSE = {
    "current": {
        "temperature_2m": 30.5,
        "relative_humidity_2m": 75,
        "weathercode": 3,
        "wind_speed_10m": 12.0,
    },
    "daily": {
        "time": ["2026-03-01", "2026-03-02", "2026-03-03"],
        "temperature_2m_max": [32.0, 31.5, 33.0],
        "temperature_2m_min": [24.0, 23.5, 25.0],
        "precipitation_sum": [0.0, 5.2, 0.0],
        "precipitation_probability_max": [10, 80, 5],
        "weathercode": [3, 63, 1],
        "wind_speed_10m_max": [15.0, 20.0, 10.0],
    },
}

_MOCK_GEOCODE_RESPONSE = {
    "results": [
        {
            "name": "Tokyo",
            "country": "Japan",
            "latitude": 35.6895,
            "longitude": 139.6917,
        }
    ]
}

_MOCK_GEOCODE_EMPTY = {}


def _make_mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "https://example.com"),
    )
    return resp


class TestDayNameId:
    """Test Indonesian day name helper."""

    def test_all_days(self):
        expected = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        for i, name in enumerate(expected):
            # 2026-03-02 is Monday (weekday=0)
            dt = datetime(2026, 3, 2 + i)
            assert _day_name_id(dt) == name

    def test_known_date(self):
        # 2026-03-01 is Sunday
        dt = datetime(2026, 3, 1)
        assert _day_name_id(dt) == "Minggu"


class TestWeatherCodes:
    """Test weather code mapping."""

    def test_common_codes_exist(self):
        assert _WEATHER_CODES[0] == "cerah"
        assert _WEATHER_CODES[63] == "hujan"
        assert _WEATHER_CODES[95] == "badai petir"

    def test_all_codes_are_strings(self):
        for code, desc in _WEATHER_CODES.items():
            assert isinstance(code, int)
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestGeocode:
    """Test geocoding function."""

    @pytest.mark.asyncio
    async def test_geocode_success(self):
        mock_resp = _make_mock_response(_MOCK_GEOCODE_RESPONSE)

        with patch("nova.tools.weather.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _geocode("Tokyo")

        assert result is not None
        lat, lon, name = result
        assert lat == 35.6895
        assert lon == 139.6917
        assert "Tokyo" in name
        assert "Japan" in name

    @pytest.mark.asyncio
    async def test_geocode_not_found(self):
        mock_resp = _make_mock_response(_MOCK_GEOCODE_EMPTY)

        with patch("nova.tools.weather.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _geocode("xyznonexistentcity123")

        assert result is None

    @pytest.mark.asyncio
    async def test_geocode_timeout(self):
        with patch("nova.tools.weather.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await _geocode("Tokyo")

        assert result is None


class TestGetWeather:
    """Test the main get_weather function."""

    @pytest.mark.asyncio
    async def test_default_location(self):
        """Default location (Bekasi) should work without geocoding."""
        mock_resp = _make_mock_response(_MOCK_FORECAST_RESPONSE)

        with patch("nova.tools.weather.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_weather()

        assert "Cuaca Bekasi:" in result
        assert "Sekarang:" in result
        assert "berawan" in result  # weathercode 3
        assert "30.5" in result
        assert "75%" in result

    @pytest.mark.asyncio
    async def test_with_daily_forecast(self):
        """Should include daily forecast lines."""
        mock_resp = _make_mock_response(_MOCK_FORECAST_RESPONSE)

        with patch("nova.tools.weather.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_weather(days=3)

        # Should have 3 daily entries
        assert "2026-03-01" in result
        assert "2026-03-02" in result
        assert "2026-03-03" in result
        # Rain probability for day 2
        assert "80%" in result
        assert "5.2mm" in result

    @pytest.mark.asyncio
    async def test_with_location_geocode(self):
        """Should geocode when location is provided."""
        geo_resp = _make_mock_response(_MOCK_GEOCODE_RESPONSE)
        forecast_resp = _make_mock_response(_MOCK_FORECAST_RESPONSE)

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "geocoding" in url:
                return geo_resp
            return forecast_resp

        with patch("nova.tools.weather.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=mock_get)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_weather(location="Tokyo")

        assert "Tokyo, Japan" in result

    @pytest.mark.asyncio
    async def test_unknown_location(self):
        """Should return error for unknown location."""
        mock_resp = _make_mock_response(_MOCK_GEOCODE_EMPTY)

        with patch("nova.tools.weather.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_weather(location="xyznonexistent123")

        assert "tidak ditemukan" in result

    @pytest.mark.asyncio
    async def test_api_timeout(self):
        """Should return error message on API timeout."""
        with patch("nova.tools.weather.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await get_weather()

        assert "Gagal" in result

    @pytest.mark.asyncio
    async def test_days_clamped(self):
        """Days parameter should be clamped to 1-7."""
        mock_resp = _make_mock_response(_MOCK_FORECAST_RESPONSE)

        with patch("nova.tools.weather.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            # Should not raise even with out-of-range days
            result = await get_weather(days=0)
            assert isinstance(result, str)

            result = await get_weather(days=99)
            assert isinstance(result, str)


class TestWeatherInRegistry:
    """Test that get_weather is properly registered."""

    def test_get_weather_in_tool_names(self):
        from nova.tools.registry import get_all_tool_names

        names = get_all_tool_names()
        assert "get_weather" in names

    def test_get_weather_declaration_exists(self):
        from nova.tools.registry import get_tool_declarations

        tools = get_tool_declarations()
        found = False
        for tool in tools:
            for fn_decl in tool.function_declarations:
                if fn_decl.name == "get_weather":
                    found = True
                    schema = fn_decl.parameters_json_schema
                    assert "location" in schema["properties"]
                    assert "days" in schema["properties"]
        assert found, "get_weather declaration not found in registry"

    def test_all_declared_tools_have_implementations(self):
        from nova.tools.registry import get_all_tool_names, get_tool_declarations

        tools = get_tool_declarations()
        names = get_all_tool_names()
        for tool in tools:
            for fn_decl in tool.function_declarations:
                assert fn_decl.name in names, (
                    f"Declared function {fn_decl.name!r} has no implementation"
                )
