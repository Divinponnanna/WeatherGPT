"""
modules/weather_api.py – Live Weather Data from Open-Meteo
===========================================================
Responsibilities
----------------
* Geocode a city name → (latitude, longitude, display name)
* Request a 7-day weather forecast from the Open-Meteo Forecast API
* Normalize the JSON response into a consistent Python dict
* Handle every foreseeable failure (unknown city, timeout, offline, bad data)
* Never fabricate weather values

Public API
----------
get_forecast(city: str, days: int = 7) -> dict
    Returns a normalized forecast dict or raises WeatherAPIError.

get_demo_forecast(city: str) -> dict
    Returns clearly-labelled sample data for offline / demo use.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from config import (
    OPEN_METEO_GEOCODING_URL,
    OPEN_METEO_FORECAST_URL,
    REQUEST_TIMEOUT,
    MAX_FORECAST_DAYS,
    MIN_FORECAST_DAYS,
    SAMPLE_WEATHER_PATH,
)

logger = logging.getLogger(__name__)

# ── Custom exception ───────────────────────────────────────────────────────────

class WeatherAPIError(Exception):
    """Raised when the weather API cannot fulfil the request."""


# ── WMO weather code → human-readable condition ───────────────────────────────

WMO_CODES: dict[int, str] = {
    0:  "Clear sky",
    1:  "Mainly clear",  2: "Partly cloudy",  3: "Overcast",
    45: "Fog",          48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain",  63: "Moderate rain",   65: "Heavy rain",
    71: "Slight snow",  73: "Moderate snow",   75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm + slight hail",
    99: "Thunderstorm + heavy hail",
}


def _wmo_description(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return WMO_CODES.get(int(code), f"Weather code {code}")


# ── In-memory TTL Caches & Rate Protection ────────────────────────────────────
_geo_cache: dict[str, tuple[float, dict]] = {}
_forecast_cache: dict[tuple, tuple[float, dict]] = {}
GEO_CACHE_TTL = 3600.0      # 1 hour
FORECAST_CACHE_TTL = 300.0  # 5 minutes

def clear_weather_cache() -> None:
    """Clear in-memory weather and geocoding caches (useful for testing and reset)."""
    _geo_cache.clear()
    _forecast_cache.clear()


# ── Geocoding ─────────────────────────────────────────────────────────────────

def geocode_city(city: str) -> dict[str, Any]:
    """
    Convert a city name into geographic coordinates.
    Includes input length validation and in-memory TTL caching.

    Returns
    -------
    dict with keys: latitude, longitude, display_name, country, timezone
    """
    if not city or not str(city).strip():
        raise WeatherAPIError("City name must not be empty.")

    clean_city = str(city).strip()
    if len(clean_city) > 100:
        raise WeatherAPIError("City name exceeds maximum length of 100 characters.")

    import time
    now = time.time()
    cache_key = clean_city.lower()
    if cache_key in _geo_cache:
        ts, cached_res = _geo_cache[cache_key]
        if now - ts < GEO_CACHE_TTL:
            return cached_res

    params = {"name": clean_city, "count": 1, "language": "en", "format": "json"}
    try:
        resp = requests.get(
            OPEN_METEO_GEOCODING_URL, params=params, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        raise WeatherAPIError(
            "Cannot reach geocoding service. Check your internet connection."
        )
    except requests.exceptions.Timeout:
        raise WeatherAPIError(
            f"Geocoding request timed out after {REQUEST_TIMEOUT} seconds."
        )
    except requests.exceptions.HTTPError as exc:
        raise WeatherAPIError(f"Geocoding HTTP error: {exc}") from exc
    except ValueError as exc:
        raise WeatherAPIError(f"Invalid JSON from geocoding API: {exc}") from exc

    results = data.get("results")
    if not results:
        # Progressively trim words from the right (e.g. "Madikeri Llomorrow" -> "Madikeri")
        words = city.strip().split()
        import re
        for i in range(len(words) - 1, 0, -1):
            candidate = " ".join(words[:i]).strip()
            candidate = re.sub(r"['’]s\b", "", candidate)
            candidate = re.sub(r"[^\w\s-]", "", candidate).strip()
            if len(candidate) >= 2:
                try:
                    retry_resp = requests.get(
                        OPEN_METEO_GEOCODING_URL,
                        params={"name": candidate, "count": 1, "language": "en", "format": "json"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    if retry_resp.ok:
                        retry_data = retry_resp.json()
                        if retry_data.get("results"):
                            results = retry_data["results"]
                            logger.info("Geocoding resolved '%s' to '%s'", city, candidate)
                            break
                except Exception:
                    continue

    if not results:
        raise WeatherAPIError(
            f"City '{city}' not found. Please check the spelling or try a nearby city."
        )

    r = results[0]
    res = {
        "latitude":     float(r["latitude"]),
        "longitude":    float(r["longitude"]),
        "display_name": r.get("name", city),
        "country":      r.get("country", ""),
        "timezone":     r.get("timezone", "UTC"),
    }
    _geo_cache[cache_key] = (now, res)
    return res


# ── Forecast retrieval ────────────────────────────────────────────────────────

def _clamp_days(days: int) -> int:
    return max(MIN_FORECAST_DAYS, min(days, MAX_FORECAST_DAYS))


def fetch_forecast_raw(lat: float, lon: float, days: int, timezone: str) -> dict:
    """
    Call the Open-Meteo Forecast API and return the raw JSON dict.
    Includes in-memory TTL caching to prevent rate-limit exhaustion.
    """
    days = _clamp_days(days)
    import time
    now = time.time()
    cache_key = (round(float(lat), 4), round(float(lon), 4), days, str(timezone))
    if cache_key in _forecast_cache:
        ts, cached_raw = _forecast_cache[cache_key]
        if now - ts < FORECAST_CACHE_TTL:
            return cached_raw
    params = {
        "latitude":    lat,
        "longitude":   lon,
        "forecast_days": days,
        "timezone":    timezone,
        # Current weather
        "current": ",".join([
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "weather_code",
            "precipitation",
            "surface_pressure",
            "visibility",
        ]),
        # Daily forecast
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "weather_code",
            "sunrise",
            "sunset",
            "uv_index_max",
        ]),
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit":   "celsius",
    }

    try:
        resp = requests.get(
            OPEN_METEO_FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        _forecast_cache[cache_key] = (now, data)
        return data
    except requests.exceptions.ConnectionError:
        raise WeatherAPIError(
            "Cannot reach weather service. Check your internet connection."
        )
    except requests.exceptions.Timeout:
        raise WeatherAPIError(
            f"Forecast request timed out after {REQUEST_TIMEOUT} seconds."
        )
    except requests.exceptions.HTTPError as exc:
        raise WeatherAPIError(f"Forecast HTTP error: {exc}") from exc
    except ValueError as exc:
        raise WeatherAPIError(f"Invalid JSON from forecast API: {exc}") from exc


# ── Response normalization ────────────────────────────────────────────────────

def _safe(value: Any, default: Any = None) -> Any:
    """Return value or default if value is None."""
    return default if value is None else value


def normalize_forecast(raw: dict, geo: dict, requested_days: int) -> dict:
    """
    Convert Open-Meteo raw JSON into a stable, well-typed Python dict
    that all downstream modules (alerts, charts, agent) can rely on.
    """
    current_raw = raw.get("current", {})
    daily_raw   = raw.get("daily", {})

    # ── Current conditions ──────────────────────────────────────────────────
    current = {
        "temperature_c":      _safe(current_raw.get("temperature_2m")),
        "feels_like_c":       _safe(current_raw.get("apparent_temperature")),
        "humidity_pct":       _safe(current_raw.get("relative_humidity_2m")),
        "wind_speed_kmh":     _safe(current_raw.get("wind_speed_10m")),
        "wind_direction_deg": _safe(current_raw.get("wind_direction_10m")),
        "precipitation_mm":   _safe(current_raw.get("precipitation"), 0.0),
        "pressure_hpa":       _safe(current_raw.get("surface_pressure")),
        "visibility_km":      round(current_raw["visibility"] / 1000.0, 1) if current_raw.get("visibility") is not None else None,
        "weather_code":       _safe(current_raw.get("weather_code")),
        "condition":          _wmo_description(current_raw.get("weather_code")),
        "timestamp":          current_raw.get("time", ""),
    }

    # ── Daily forecast ──────────────────────────────────────────────────────
    dates          = daily_raw.get("time", [])
    temp_max       = daily_raw.get("temperature_2m_max", [])
    temp_min       = daily_raw.get("temperature_2m_min", [])
    precip_sum     = daily_raw.get("precipitation_sum", [])
    precip_prob    = daily_raw.get("precipitation_probability_max", [])
    wind_max       = daily_raw.get("wind_speed_10m_max", [])
    wind_gusts     = daily_raw.get("wind_gusts_10m_max", [])
    weather_codes  = daily_raw.get("weather_code", [])
    sunrises       = daily_raw.get("sunrise", [])
    sunsets        = daily_raw.get("sunset", [])
    uv_indices     = daily_raw.get("uv_index_max", [])

    daily = []
    for i, date in enumerate(dates):
        daily.append({
            "date":               date,
            "temp_max_c":         _safe(temp_max[i] if i < len(temp_max) else None),
            "temp_min_c":         _safe(temp_min[i] if i < len(temp_min) else None),
            "precipitation_mm":   _safe(precip_sum[i] if i < len(precip_sum) else None, 0.0),
            "rain_probability_pct": _safe(precip_prob[i] if i < len(precip_prob) else None, 0.0),
            "wind_speed_kmh":     _safe(wind_max[i] if i < len(wind_max) else None),
            "wind_gusts_kmh":     _safe(wind_gusts[i] if i < len(wind_gusts) else None),
            "weather_code":       _safe(weather_codes[i] if i < len(weather_codes) else None),
            "condition":          _wmo_description(weather_codes[i] if i < len(weather_codes) else None),
            "sunrise":            sunrises[i] if i < len(sunrises) else None,
            "sunset":             sunsets[i] if i < len(sunsets) else None,
            "uv_index":           _safe(uv_indices[i] if i < len(uv_indices) else None),
        })

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "city":           geo["display_name"],
        "country":        geo["country"],
        "latitude":       geo["latitude"],
        "longitude":      geo["longitude"],
        "timezone":       geo["timezone"],
        "requested_days": requested_days,
        "retrieved_at":   retrieved_at,
        "is_demo":        False,
        "source":         "Open-Meteo (https://open-meteo.com/)",
        "current":        current,
        "daily":          daily,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_forecast(city: str, days: int = 7) -> dict:
    """
    Retrieve a live weather forecast for *city* covering *days* days.

    Parameters
    ----------
    city : str
        The city name in English.
    days : int
        Number of forecast days (1–7).

    Returns
    -------
    dict
        Normalized forecast dictionary.

    Raises
    ------
    WeatherAPIError
        On network failure, unknown city or API error.
    """
    days = _clamp_days(days)
    geo  = geocode_city(city)
    raw  = fetch_forecast_raw(geo["latitude"], geo["longitude"], days, geo["timezone"])
    return normalize_forecast(raw, geo, days)


def get_demo_forecast(city: str = "Bengaluru") -> dict:
    """
    Return clearly-labelled sample forecast data for offline / demo use.
    Reads from data/sample_weather.json if present, otherwise generates inline data.
    """
    # Try loading from the bundled sample file first
    if SAMPLE_WEATHER_PATH.exists():
        try:
            data = json.loads(SAMPLE_WEATHER_PATH.read_text(encoding="utf-8"))
            data["is_demo"] = True
            data["city"]    = f"{city} (Demo)"
            data["retrieved_at"] = "DEMO DATA – not live"
            return data
        except (json.JSONDecodeError, KeyError):
            pass   # Fall through to inline sample

    # Inline sample – 7 days, synthetic but plausible for Bengaluru
    daily = []
    base_dates = [
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-05", "2025-01-06", "2025-01-07",
    ]
    for i, d in enumerate(base_dates):
        daily.append({
            "date":               d,
            "temp_max_c":         28.0 + i * 0.5,
            "temp_min_c":         18.0 + i * 0.3,
            "precipitation_mm":   5.0 * i,
            "rain_probability_pct": 10.0 * i,
            "wind_speed_kmh":     12.0 + i,
            "wind_gusts_kmh":     20.0 + i * 1.5,
            "weather_code":       1,
            "condition":          "Mainly clear",
            "sunrise":            f"{d}T06:30",
            "sunset":             f"{d}T18:30",
            "uv_index":           6.0 + i * 0.2,
        })

    return {
        "city":           f"{city} (Demo)",
        "country":        "India",
        "latitude":       12.97,
        "longitude":      77.59,
        "timezone":       "Asia/Kolkata",
        "requested_days": 7,
        "retrieved_at":   "DEMO DATA: not live",
        "is_demo":        True,
        "source":         "Built-in sample data (not from Open-Meteo)",
        "current": {
            "temperature_c":      26.5,
            "feels_like_c":       28.0,
            "humidity_pct":       65,
            "wind_speed_kmh":     14.0,
            "wind_direction_deg": 220,
            "precipitation_mm":   0.0,
            "pressure_hpa":       1013.2,
            "visibility_km":      10.0,
            "weather_code":       1,
            "condition":          "Mainly clear",
            "timestamp":          "2025-01-01T12:00",
        },
        "daily": daily,
    }
