"""Weather tool using Open-Meteo free API."""

import logging
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

# Open-Meteo WMO weather codes → Indonesian descriptions
_WEATHER_CODES = {
    0: "cerah",
    1: "sebagian cerah",
    2: "berawan sebagian",
    3: "berawan",
    45: "berkabut",
    48: "berkabut tebal",
    51: "gerimis ringan",
    53: "gerimis",
    55: "gerimis lebat",
    61: "hujan ringan",
    63: "hujan",
    65: "hujan lebat",
    71: "salju ringan",
    73: "salju",
    75: "salju lebat",
    80: "hujan singkat ringan",
    81: "hujan singkat",
    82: "hujan singkat lebat",
    95: "badai petir",
    96: "badai petir dengan hujan es ringan",
    99: "badai petir dengan hujan es lebat",
}

# Default location: Bekasi, Indonesia
_DEFAULT_LAT = -6.2383
_DEFAULT_LON = 106.9756
_DEFAULT_TIMEZONE = "Asia/Jakarta"

_API_BASE = "https://api.open-meteo.com/v1/forecast"


async def get_weather(
    location: str = "",
    days: int = 3,
) -> str:
    """Fetch weather forecast.

    Args:
        location: City name (optional). If empty, uses Bekasi.
                  If provided, geocode first via Open-Meteo geocoding API.
        days: Number of forecast days (1-7, default 3).

    Returns:
        Formatted weather string for LLM to relay to user.
    """
    days = max(2, min(7, days))

    # Geocode if location provided
    if location.strip():
        coords = await _geocode(location.strip())
        if coords is None:
            return f"Lokasi '{location}' tidak ditemukan."
        lat, lon, resolved_name = coords
    else:
        lat, lon, resolved_name = _DEFAULT_LAT, _DEFAULT_LON, "Bekasi"

    # Fetch forecast
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "weathercode",
            "wind_speed_10m_max",
        ]),
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "weathercode",
            "wind_speed_10m",
        ]),
        "timezone": _DEFAULT_TIMEZONE,
        "forecast_days": days,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_API_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error("Weather API error: %s", e)
        return f"Gagal mengambil data cuaca: {e}"

    # Format response
    lines = [f"Cuaca {resolved_name}:"]

    # Current conditions
    current = data.get("current", {})
    if current:
        code = current.get("weathercode", 0)
        desc = _WEATHER_CODES.get(code, f"kode {code}")
        temp = current.get("temperature_2m", "?")
        humidity = current.get("relative_humidity_2m", "?")
        wind = current.get("wind_speed_10m", "?")
        lines.append(
            f"Sekarang: {desc}, {temp}\u00b0C, kelembapan {humidity}%, "
            f"angin {wind} km/j"
        )

    # Daily forecast
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    for i, date_str in enumerate(dates):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_name = _day_name_id(dt)

        code = daily["weathercode"][i]
        desc = _WEATHER_CODES.get(code, f"kode {code}")
        t_max = daily["temperature_2m_max"][i]
        t_min = daily["temperature_2m_min"][i]
        rain = daily["precipitation_sum"][i]
        rain_prob = daily.get("precipitation_probability_max", [None] * len(dates))[i]

        rain_info = ""
        if rain_prob is not None and rain_prob > 0:
            rain_info = f", kemungkinan hujan {rain_prob}%"
            if rain > 0:
                rain_info += f" ({rain}mm)"

        lines.append(
            f"{day_name} ({date_str}): {desc}, {t_min}-{t_max}\u00b0C{rain_info}"
        )

    return "\n".join(lines)


async def _geocode(query: str) -> tuple[float, float, str] | None:
    """Geocode a city name using Open-Meteo geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url, params={"name": query, "count": 1})
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        r = results[0]
        name = r.get("name", query)
        country = r.get("country", "")
        display = f"{name}, {country}" if country else name
        return r["latitude"], r["longitude"], display
    except Exception as e:
        logger.error("Geocoding error: %s", e)
        return None


def _day_name_id(dt: datetime) -> str:
    """Get Indonesian day name."""
    days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    return days[dt.weekday()]
