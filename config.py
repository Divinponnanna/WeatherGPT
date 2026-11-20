"""
config.py - WeatherGPT Application Configuration
=================================================
Loads environment variables, defines constants, alert thresholds and
application-wide settings. All secrets are read exclusively from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
CLIMATE_DOCS_DIR = DATA_DIR / "climate_documents"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"
SAMPLE_WEATHER_PATH = DATA_DIR / "sample_weather.json"

# Ensure directories exist at import time
CLIMATE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Load environment variables (server-side only) ─────────────────────────────
load_dotenv(ROOT_DIR / ".env.local")
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY", "").strip() or None

# ── Mode detection ─────────────────────────────────────────────────────────────
# If the key is missing or still the placeholder value, switch to demo mode.
_PLACEHOLDER = "your_gemini_api_key_here"
DEMO_MODE: bool = (
    GEMINI_API_KEY is None or GEMINI_API_KEY == _PLACEHOLDER
)

# ── Gemini model ───────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-3.1-flash-lite"

# ── Open-Meteo endpoints ───────────────────────────────────────────────────────
OPEN_METEO_GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"

# ── API request timeouts (seconds) ────────────────────────────────────────────
REQUEST_TIMEOUT: int = 10

# ── Forecast limits ───────────────────────────────────────────────────────────
MAX_FORECAST_DAYS: int = 7
MIN_FORECAST_DAYS: int = 1

# ── Agentic loop cap ──────────────────────────────────────────────────────────
MAX_TOOL_ROUNDS: int = 5

# ── RAG settings ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHROMA_COLLECTION: str = "climate_knowledge"
CHUNK_SIZE: int = 500        # characters per chunk
CHUNK_OVERLAP: int = 100     # character overlap between consecutive chunks
RAG_TOP_K: int = 4

# ── Alert thresholds (DEMO values – NOT official IMD/WMO criteria) ────────────
ALERT_THRESHOLDS: dict[str, float] = {
    "rain_probability_pct": 70.0,   # %
    "daily_rainfall_mm":    50.0,   # mm
    "max_temperature_c":    40.0,   # °C
    "min_temperature_c":     5.0,   # °C
    "wind_gusts_kmh":       50.0,   # km/h
}

# ── UI copy ───────────────────────────────────────────────────────────────────
APP_NAME: str = "WeatherGPT"
APP_TAGLINE: str = "Your Agentic AI-Powered Weather Assistant"
ALERT_DISCLAIMER: str = (
    "These are informational project alerts based on demonstration thresholds, "
    "not official meteorological warnings. Please verify safety-critical information "
    "with IMD and relevant local authorities."
)
DEMO_BANNER: str = (
    "**Demo Mode**: No Gemini API key detected. "
    "AI features show pre-written responses. "
    "Live weather from Open-Meteo remains active."
)
DATA_ATTRIBUTION: str = (
    "Weather data provided by [Open-Meteo](https://open-meteo.com/) "
    "(free for non-commercial use)."
)
