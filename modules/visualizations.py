"""
modules/visualizations.py – Plotly Chart Builders
==================================================
Responsibilities
----------------
* Build interactive Plotly figures from a normalized forecast dict.
* Return figure objects; let the caller render them (e.g. st.plotly_chart).
* Handle None / missing values gracefully.

Public API
----------
build_temperature_chart(forecast: dict) -> go.Figure
build_precipitation_chart(forecast: dict) -> go.Figure
build_rain_probability_chart(forecast: dict) -> go.Figure
build_wind_chart(forecast: dict) -> go.Figure
build_forecast_chart(forecast: dict) -> dict[str, go.Figure]  # all-in-one
"""

from __future__ import annotations

import logging
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Shared styling ─────────────────────────────────────────────────────────────
_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Arial, sans-serif", size=13, color="#E0E0E0"),
    margin=dict(l=40, r=20, t=50, b=40),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

_COLOR_MAX  = "#FF6B6B"   # red-ish for max temp
_COLOR_MIN  = "#74C0FC"   # blue for min temp
_COLOR_RAIN = "#4DABF7"   # blue for rain
_COLOR_PROB = "#A9E34B"   # green for probability
_COLOR_WIND = "#FFA94D"   # orange for wind


def _extract_daily(forecast: dict, key: str) -> list:
    return [day.get(key) for day in forecast.get("daily", [])]


def _dates(forecast: dict) -> list[str]:
    return _extract_daily(forecast, "date")


# ── Temperature chart ─────────────────────────────────────────────────────────

def build_temperature_chart(forecast: dict) -> go.Figure:
    """Line chart: daily max vs min temperature."""
    dates    = _dates(forecast)
    temp_max = _extract_daily(forecast, "temp_max_c")
    temp_min = _extract_daily(forecast, "temp_min_c")
    city     = forecast.get("city", "")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=temp_max,
        mode="lines+markers",
        name="Max Temp (°C)",
        line=dict(color=_COLOR_MAX, width=2.5),
        marker=dict(size=7),
        hovertemplate="%{y:.1f}°C<extra>Max Temp</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=temp_min,
        mode="lines+markers",
        name="Min Temp (°C)",
        line=dict(color=_COLOR_MIN, width=2.5),
        marker=dict(size=7),
        fill="tonexty",
        fillcolor="rgba(116,192,252,0.12)",
        hovertemplate="%{y:.1f}°C<extra>Min Temp</extra>",
    ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=f"Temperature Forecast: {city}", font=dict(size=16)),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Temperature (°C)"),
    )
    return fig


# ── Precipitation chart ───────────────────────────────────────────────────────

def build_precipitation_chart(forecast: dict) -> go.Figure:
    """Bar chart: daily precipitation sum."""
    dates  = _dates(forecast)
    precip = _extract_daily(forecast, "precipitation_mm")
    city   = forecast.get("city", "")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=precip,
        name="Precipitation (mm)",
        marker_color=_COLOR_RAIN,
        hovertemplate="%{y:.1f} mm<extra>Precipitation</extra>",
    ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=f"Daily Precipitation: {city}", font=dict(size=16)),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Precipitation (mm)"),
        bargap=0.3,
    )
    return fig


# ── Rain probability chart ────────────────────────────────────────────────────

def build_rain_probability_chart(forecast: dict) -> go.Figure:
    """Bar + threshold-line chart: rain probability % with 70% alert line."""
    dates = _dates(forecast)
    probs = _extract_daily(forecast, "rain_probability_pct")
    city  = forecast.get("city", "")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=probs,
        name="Rain Probability (%)",
        marker_color=_COLOR_PROB,
        hovertemplate="%{y:.0f}%<extra>Rain Probability</extra>",
    ))
    # Alert threshold line
    fig.add_hline(
        y=70, line_dash="dash", line_color="#FF6B6B",
        annotation_text="Alert threshold (70%)",
        annotation_position="top right",
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=f"Rain Probability: {city}", font=dict(size=16)),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Probability (%)", range=[0, 105]),
        bargap=0.3,
    )
    return fig


# ── Wind chart ────────────────────────────────────────────────────────────────

def build_wind_chart(forecast: dict) -> go.Figure:
    """Line chart: max wind speed and gusts."""
    dates  = _dates(forecast)
    wind   = _extract_daily(forecast, "wind_speed_kmh")
    gusts  = _extract_daily(forecast, "wind_gusts_kmh")
    city   = forecast.get("city", "")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=wind,
        mode="lines+markers",
        name="Wind Speed (km/h)",
        line=dict(color=_COLOR_WIND, width=2.5),
        marker=dict(size=7),
        hovertemplate="%{y:.0f} km/h<extra>Wind Speed</extra>",
    ))
    if any(g is not None for g in gusts):
        fig.add_trace(go.Scatter(
            x=dates, y=gusts,
            mode="lines+markers",
            name="Wind Gusts (km/h)",
            line=dict(color="#FF6B6B", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="%{y:.0f} km/h<extra>Gusts</extra>",
        ))
    # Alert threshold line
    fig.add_hline(
        y=50, line_dash="dash", line_color="#FF6B6B",
        annotation_text="Gust alert (50 km/h)",
        annotation_position="top right",
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=f"Wind Forecast: {city}", font=dict(size=16)),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Speed (km/h)"),
    )
    return fig


# ── Combined convenience function ─────────────────────────────────────────────

def build_forecast_chart(forecast: dict) -> dict[str, go.Figure]:
    """
    Build all four forecast charts and return them in a dict.

    Keys: 'temperature', 'precipitation', 'rain_probability', 'wind'
    """
    if not forecast or not forecast.get("daily"):
        logger.warning("build_forecast_chart received empty forecast.")
        return {}

    return {
        "temperature":      build_temperature_chart(forecast),
        "precipitation":    build_precipitation_chart(forecast),
        "rain_probability": build_rain_probability_chart(forecast),
        "wind":             build_wind_chart(forecast),
    }
