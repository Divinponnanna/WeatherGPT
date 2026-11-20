"""
modules/alerts.py – Deterministic Weather Alert Engine
======================================================
Responsibilities
----------------
* Apply configurable threshold rules to a normalized forecast dict.
* Return structured alert objects: never guesses, always rule-based.
* Thresholds are DEMO values; labels say so explicitly.

Public API
----------
check_alerts(forecast: dict, thresholds: dict | None = None) -> list[dict]
"""

from __future__ import annotations

import logging
from typing import Any

from config import ALERT_THRESHOLDS, ALERT_DISCLAIMER

logger = logging.getLogger(__name__)

# ── Severity mapping ──────────────────────────────────────────────────────────

def _severity(alert_type: str, value: float, threshold: float) -> str:
    ratio = value / threshold if threshold else 1.0
    if ratio >= 1.5:
        return "HIGH"
    if ratio >= 1.2:
        return "MODERATE"
    return "LOW"


# ── Individual alert checkers ─────────────────────────────────────────────────

def _check_rain_probability(day: dict, thresh: float) -> dict | None:
    val = day.get("rain_probability_pct")
    if val is None:
        return None
    if val >= thresh:
        return {
            "type":        "Rain Likely",
            "observed":    val,
            "unit":        "%",
            "threshold":   thresh,
            "date":        day["date"],
            "severity":    _severity("rain_prob", val, thresh),
            "explanation": (
                f"Rain probability is {val:.0f}% on {day['date']}, "
                f"which meets or exceeds the {thresh:.0f}% demo threshold."
            ),
        }
    return None


def _check_heavy_rain(day: dict, thresh: float) -> dict | None:
    val = day.get("precipitation_mm")
    if val is None:
        return None
    if val >= thresh:
        return {
            "type":        "Heavy Rain",
            "observed":    val,
            "unit":        "mm",
            "threshold":   thresh,
            "date":        day["date"],
            "severity":    _severity("heavy_rain", val, thresh),
            "explanation": (
                f"Daily precipitation is {val:.1f} mm on {day['date']}, "
                f"which meets or exceeds the {thresh:.0f} mm demo threshold."
            ),
        }
    return None


def _check_heat(day: dict, thresh: float) -> dict | None:
    val = day.get("temp_max_c")
    if val is None:
        return None
    if val >= thresh:
        return {
            "type":        "Heat Alert",
            "observed":    val,
            "unit":        "°C (max)",
            "threshold":   thresh,
            "date":        day["date"],
            "severity":    _severity("heat", val, thresh),
            "explanation": (
                f"Maximum temperature is {val:.1f}°C on {day['date']}, "
                f"which meets or exceeds the {thresh:.0f}°C demo threshold."
            ),
        }
    return None


def _check_cold(day: dict, thresh: float) -> dict | None:
    val = day.get("temp_min_c")
    if val is None:
        return None
    if val <= thresh:
        return {
            "type":        "Cold Alert",
            "observed":    val,
            "unit":        "°C (min)",
            "threshold":   thresh,
            "date":        day["date"],
            "severity":    _severity("cold", abs(val - thresh) + thresh, thresh),
            "explanation": (
                f"Minimum temperature is {val:.1f}°C on {day['date']}, "
                f"at or below the {thresh:.0f}°C demo threshold."
            ),
        }
    return None


def _check_wind_gusts(day: dict, thresh: float) -> dict | None:
    val = day.get("wind_gusts_kmh")
    if val is None:
        return None
    if val >= thresh:
        return {
            "type":        "Strong Wind",
            "observed":    val,
            "unit":        "km/h (gusts)",
            "threshold":   thresh,
            "date":        day["date"],
            "severity":    _severity("wind", val, thresh),
            "explanation": (
                f"Wind gusts reach {val:.0f} km/h on {day['date']}, "
                f"which meets or exceeds the {thresh:.0f} km/h demo threshold."
            ),
        }
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def check_alerts(
    forecast: dict,
    thresholds: dict[str, float] | None = None,
) -> list[dict]:
    """
    Scan every day in *forecast['daily']* and return a list of alert dicts.

    Parameters
    ----------
    forecast   : normalized dict from weather_api.get_forecast()
    thresholds : override default thresholds (keys from ALERT_THRESHOLDS)

    Returns
    -------
    list[dict]  – may be empty if no thresholds are exceeded.
                  Every dict has: type, observed, unit, threshold, date,
                                  severity, explanation, disclaimer.
    """
    if not isinstance(forecast, dict):
        logger.warning("check_alerts received a non-dict forecast; returning [].")
        return []

    t = {**ALERT_THRESHOLDS, **(thresholds or {})}
    daily: list[dict] = forecast.get("daily", [])

    alerts: list[dict] = []
    for day in daily:
        checkers = [
            _check_rain_probability(day, t["rain_probability_pct"]),
            _check_heavy_rain(day,      t["daily_rainfall_mm"]),
            _check_heat(day,            t["max_temperature_c"]),
            _check_cold(day,            t["min_temperature_c"]),
            _check_wind_gusts(day,      t["wind_gusts_kmh"]),
        ]
        for alert in checkers:
            if alert is not None:
                alert["disclaimer"] = ALERT_DISCLAIMER
                alerts.append(alert)

    logger.info("check_alerts: %d alerts generated for %s.", len(alerts), forecast.get("city", "?"))
    return alerts
