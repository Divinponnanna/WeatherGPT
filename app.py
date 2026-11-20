"""
app.py – WeatherGPT Streamlit Application
==========================================
Entry point: streamlit run app.py
Sections:
  🏠 Dashboard    – Live weather + current conditions
  📅 Forecast     – 7-day forecast cards + charts
  💬 AI Chat      – Agentic Gemini conversation
  🚨 Alerts       – Rule-based weather alerts
  📚 Climate KB   – PDF upload + RAG search
  ⚙️ Settings     – Config status + controls
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="WeatherGPT - Agentic AI Weather Assistant",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Security headers / meta tags ──────────────────────────────────────────────
st.markdown("""
<meta http-equiv="X-Content-Type-Options" content="nosniff">
<meta http-equiv="X-Frame-Options" content="SAMEORIGIN">
<meta name="referrer" content="strict-origin-when-cross-origin">
""", unsafe_allow_html=True)

# ── Inject custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography & base ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Dark gradient background (pure slate/navy - no purple) ── */
.stApp { background: linear-gradient(135deg, #0b0f19, #111827, #0f172a); }

/* ── Buttons (subtly rounded - no pill shapes) ── */
.stButton > button {
    border-radius: 6px !important;
    font-weight: 500;
}

/* ── Metric cards ─────────────────────────────── */
.weather-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    backdrop-filter: blur(12px);
    margin-bottom: 0.8rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.weather-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(100,200,255,0.12);
}
.metric-label { color: #94A3B8; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.3rem; }
.metric-value { color: #F1F5F9; font-size: 1.8rem; font-weight: 700; line-height: 1.1; }
.metric-unit  { color: #94A3B8; font-size: 0.82rem; }

/* ── Forecast day cards ───────────────────────── */
.forecast-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem 0.8rem;
    text-align: center;
    transition: transform 0.2s;
}
.forecast-card:hover { transform: translateY(-3px); }
.forecast-day  { color: #64B5F6; font-size: 0.8rem; font-weight: 600; }
.forecast-cond { color: #90CAF9; font-size: 0.75rem; margin: 0.3rem 0; }
.forecast-high { color: #FF8A65; font-weight: 700; font-size: 1.1rem; }
.forecast-low  { color: #64B5F6; font-size: 0.95rem; }

/* ── Alert cards ──────────────────────────────── */
.alert-high     { background: rgba(239,68,68,0.15);  border-left: 4px solid #EF4444; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
.alert-moderate { background: rgba(245,158,11,0.15); border-left: 4px solid #F59E0B; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
.alert-low      { background: rgba(59,130,246,0.15); border-left: 4px solid #3B82F6; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }

/* ── Chat messages ────────────────────────────── */
.chat-metadata { color: #64748B; font-size: 0.75rem; margin-top: 0.3rem; }

/* ── RAG passage cards ────────────────────────── */
.rag-passage {
    background: rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.rag-source { color: #38BDF8; font-size: 0.78rem; font-weight: 600; }

/* ── Status badges (subtly rounded, not pill) ── */
.status-ok   { background: rgba(34,197,94,0.15); color: #86EFAC; border: 1px solid rgba(34,197,94,0.3); border-radius: 4px; padding: 0.15rem 0.6rem; font-size: 0.75rem; display: inline-block; }
.status-warn { background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); border-radius: 4px; padding: 0.15rem 0.6rem; font-size: 0.75rem; display: inline-block; }
.status-err  { background: rgba(239,68,68,0.15);  color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3);  border-radius: 4px; padding: 0.15rem 0.6rem; font-size: 0.75rem; display: inline-block; }

/* ── Demo banner ──────────────────────────────── */
.demo-banner {
    background: linear-gradient(90deg, rgba(245,158,11,0.15), rgba(239,68,68,0.1));
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 8px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 1rem;
}

/* ── Section headers ──────────────────────────── */
.section-title { color: #E2E8F0; font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; }
.section-sub   { color: #94A3B8; font-size: 0.85rem; }

/* ── Divider ──────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.08); }

/* ── Footer ───────────────────────────────────── */
.site-footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    color: #64748B;
    font-size: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}
.site-footer a {
    color: #94A3B8;
    text-decoration: none;
    margin-right: 1.2rem;
}
.site-footer a:hover {
    color: #38BDF8;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)


# ── Imports (after CSS injection) ────────────────────────────────────────────
from config import (
    APP_NAME, APP_TAGLINE, ALERT_DISCLAIMER, DEMO_BANNER,
    DATA_ATTRIBUTION, DEMO_MODE
)


# ── Session state initialisation ─────────────────────────────────────────────
def _init_session():
    defaults = {
        "chat_history":    [],
        "current_forecast": None,
        "current_city":    "",
        "recent_searches": [],
        "alerts":          [],
        "rag_indexed":     False,
        "rag_doc_names":   [],
        "active_tab":      "Dashboard",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"# {APP_NAME}")
        st.markdown(f"*{APP_TAGLINE}*")
        st.divider()

        # City search
        st.markdown("### Location")
        city_input = st.text_input(
            "Enter city name",
            value=st.session_state.current_city,
            placeholder="e.g. Bengaluru, London, Tokyo",
            key="sidebar_city_input",
        )
        days_input = st.slider("Forecast days", 1, 7, 7, key="sidebar_days")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Get Weather", use_container_width=True, type="primary"):
                if city_input.strip():
                    _load_forecast(city_input.strip(), days_input)
                else:
                    st.warning("Please enter a city name.")
        with col2:
            if st.button("Demo Data", use_container_width=True):
                _load_demo_forecast(city_input.strip() or "Bengaluru")

        if st.session_state.current_forecast:
            if st.button("Refresh Weather", use_container_width=True):
                _load_forecast(st.session_state.current_city or "Bengaluru", days_input)

        # Quick location suggestions
        st.markdown("<div style='color:#94A3B8; font-size:0.75rem; margin-top:0.6rem; margin-bottom:0.2rem;'>Popular Cities:</div>", unsafe_allow_html=True)
        sug_cols = st.columns(3)
        suggestions = ["Bengaluru", "Chennai", "Delhi", "Mumbai", "Madikeri", "London"]
        for idx, s_city in enumerate(suggestions):
            with sug_cols[idx % 3]:
                if st.button(s_city, key=f"sug_{s_city}", use_container_width=True):
                    _load_forecast(s_city, days_input)

        # Recent searches
        if st.session_state.get("recent_searches"):
            st.markdown("<div style='color:#94A3B8; font-size:0.75rem; margin-top:0.6rem; margin-bottom:0.2rem;'>Recent Searches:</div>", unsafe_allow_html=True)
            recent_list = st.session_state.recent_searches[:3]
            rec_cols = st.columns(len(recent_list))
            for idx, r_city in enumerate(recent_list):
                with rec_cols[idx]:
                    if st.button(r_city, key=f"rec_{r_city}_{idx}", use_container_width=True):
                        _load_forecast(r_city, days_input)

        st.divider()

        # Navigation
        st.markdown("### Navigation")
        tabs = ["Dashboard", "Forecast", "AI Chat", "Alerts", "Climate KB", "Settings"]
        for tab in tabs:
            is_active = st.session_state.active_tab == tab
            btn_type = "primary" if is_active else "secondary"
            if st.button(tab, use_container_width=True, key=f"nav_{tab}"):
                st.session_state.active_tab = tab

        st.divider()

        # Quick status
        from modules.ai_agent import get_agent_status
        status = get_agent_status()
        ai_pill = '<span class="status-ok">AI Active</span>' if status["available"] else '<span class="status-warn">Demo Mode</span>'
        st.markdown(f"**AI Engine:** {ai_pill}", unsafe_allow_html=True)
        st.markdown(DATA_ATTRIBUTION)


# ── Forecast loading helpers ──────────────────────────────────────────────────
def _load_forecast(city: str, days: int):
    city = str(city).strip()
    if len(city) > 100:
        st.error("City name is too long (maximum 100 characters).")
        return

    with st.spinner(f"Fetching live weather for {city}..."):
        try:
            from modules.weather_api import get_forecast, WeatherAPIError
            from modules.alerts import check_alerts
            forecast = get_forecast(city, days)
            alerts   = check_alerts(forecast)
            st.session_state.current_forecast = forecast
            st.session_state.current_city     = city
            st.session_state.alerts           = alerts

            # Track recent searches
            if "recent_searches" not in st.session_state:
                st.session_state.recent_searches = []
            if city not in st.session_state.recent_searches:
                st.session_state.recent_searches.insert(0, city)
                st.session_state.recent_searches = st.session_state.recent_searches[:5]

            st.success(f"Live weather loaded for {forecast['city']} at {forecast['retrieved_at']}")
        except Exception as exc:
            st.error(f"Could not load weather: {exc}")


def _load_demo_forecast(city: str):
    city = str(city).strip()[:100]
    with st.spinner("Loading demo data..."):
        from modules.weather_api import get_demo_forecast
        from modules.alerts import check_alerts
        forecast = get_demo_forecast(city)
        alerts   = check_alerts(forecast)
        st.session_state.current_forecast = forecast
        st.session_state.current_city     = city
        st.session_state.alerts           = alerts
        st.info("Demo data loaded. This is sample data, not live weather.")


# ── Helpers ───────────────────────────────────────────────────────────────────
import html
from typing import Any

def _escape(text: Any) -> str:
    """Sanitize user or API supplied strings before embedding into HTML to prevent XSS."""
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def _fmt(val, unit="", decimals=1, default="N/A"):
    if val is None:
        return default
    return f"{val:.{decimals}f}{unit}"


def _wind_dir_arrow(deg: float | None) -> str:
    if deg is None:
        return "?"
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[round(deg / 45) % 8]


WEATHER_EMOJI = {
    "Clear sky": "☀️", "Mainly clear": "🌤️", "Partly cloudy": "⛅",
    "Overcast": "☁️", "Fog": "🌫️", "Light drizzle": "🌦️",
    "Moderate drizzle": "🌧️", "Dense drizzle": "🌧️",
    "Slight rain": "🌦️", "Moderate rain": "🌧️", "Heavy rain": "🌧️",
    "Slight snow": "🌨️", "Moderate snow": "❄️", "Heavy snow": "❄️",
    "Slight showers": "🌦️", "Moderate showers": "🌧️", "Violent showers": "⛈️",
    "Thunderstorm": "⛈️", "Thunderstorm + slight hail": "⛈️",
}

def _weather_emoji(condition: str) -> str:
    for k, v in WEATHER_EMOJI.items():
        if k.lower() in condition.lower():
            return v
    return "🌡️"


# ── DASHBOARD TAB ─────────────────────────────────────────────────────────────
def render_dashboard():
    st.markdown('<div class="section-title">Dashboard</div>', unsafe_allow_html=True)

    forecast = st.session_state.current_forecast

    if not forecast:
        st.markdown("""
        <div class="weather-card" style="text-align:center; padding:3rem;">
            <h2 style="color:#E2E8F0; margin-bottom:0.5rem;">WeatherGPT Meteorological Platform</h2>
            <p style="color:#94A3B8; max-width:600px; margin:0 auto 1.5rem auto;">
                Select a location from the sidebar or click one of the quick suggestions 
                to inspect live weather metrics, high-resolution forecasts, and agentic AI climate analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    cur = forecast.get("current", {})
    is_demo = forecast.get("is_demo", False)
    daily = forecast.get("daily", [])
    d0 = daily[0] if daily else {}

    # Demo banner
    if is_demo:
        st.markdown(f'<div class="demo-banner"><strong>Demo Mode</strong>: {forecast["retrieved_at"]}</div>', unsafe_allow_html=True)

    # Header card
    city_esc = _escape(forecast.get('city', ''))
    country_esc = _escape(forecast.get('country', ''))
    cond_esc = _escape(cur.get('condition', 'N/A'))
    retrieved_esc = _escape(forecast.get('retrieved_at', ''))

    st.markdown(f"""
    <div class="weather-card">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
            <div>
                <div style="font-size:2.8rem; font-weight:800; color:#F1F5F9; line-height:1;">
                    {_fmt(cur.get('temperature_c'), '°C', 1)}
                </div>
                <div style="color:#F8FAFC; font-size:1.2rem; font-weight:600; margin-top:0.4rem;">
                    {city_esc}, {country_esc}
                </div>
                <div style="color:#38BDF8; font-size:0.95rem; margin-top:0.2rem;">
                    {cond_esc} · Feels like {_fmt(cur.get('feels_like_c'), '°C', 1)}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="color:#94A3B8; font-size:0.85rem;">Coordinates</div>
                <div style="color:#E2E8F0; font-size:0.95rem; font-family:monospace;">
                    {forecast.get('latitude', '?')}°N, {forecast.get('longitude', '?')}°E
                </div>
                <div style="color:#94A3B8; font-size:0.8rem; margin-top:0.3rem;">
                    Timezone: {_escape(forecast.get('timezone', 'UTC'))}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric grid - Row 1
    c1, c2, c3, c4 = st.columns(4)
    row1 = [
        (c1, "Humidity", _fmt(cur.get("humidity_pct"), "%", 0), "Relative humidity"),
        (c2, "Wind Speed", _fmt(cur.get("wind_speed_kmh"), " km/h", 0), f"Direction: {_wind_dir_arrow(cur.get('wind_direction_deg'))} ({_fmt(cur.get('wind_direction_deg'),'°',0)})"),
        (c3, "Precipitation", _fmt(cur.get("precipitation_mm"), " mm", 1), "Observed precipitation"),
        (c4, "High / Low Today", f"{_fmt(d0.get('temp_max_c'), '°C', 0)} / {_fmt(d0.get('temp_min_c'), '°C', 0)}", "Daily range"),
    ]
    for col, label, val, sub in row1:
        with col:
            st.markdown(f"""
            <div class="weather-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-unit">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Metric grid - Row 2
    c5, c6, c7, c8 = st.columns(4)
    row2 = [
        (c5, "Atmospheric Pressure", _fmt(cur.get("pressure_hpa"), " hPa", 1), "Surface pressure"),
        (c6, "Visibility", _fmt(cur.get("visibility_km"), " km", 1), "Horizontal visibility"),
        (c7, "UV Index", _fmt(d0.get("uv_index"), "", 1), "Max UV index"),
        (c8, "Rain Probability", _fmt(d0.get("rain_probability_pct"), "%", 0), "Peak precipitation probability"),
    ]
    for col, label, val, sub in row2:
        with col:
            st.markdown(f"""
            <div class="weather-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-unit">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Sunrise / sunset from day 0
    sunrise = d0.get("sunrise", "")
    sunset  = d0.get("sunset", "")
    if sunrise or sunset:
        cs1, cs2 = st.columns(2)
        with cs1:
            st.markdown(f"""<div class="weather-card">
                <div class="metric-label">Sunrise</div>
                <div class="metric-value" style="font-size:1.4rem;">{sunrise.split("T")[-1] if sunrise else "N/A"}</div>
                <div class="metric-unit">Local time</div>
            </div>""", unsafe_allow_html=True)
        with cs2:
            st.markdown(f"""<div class="weather-card">
                <div class="metric-label">Sunset</div>
                <div class="metric-value" style="font-size:1.4rem;">{sunset.split("T")[-1] if sunset else "N/A"}</div>
                <div class="metric-unit">Local time</div>
            </div>""", unsafe_allow_html=True)

    # Source attribution
    st.markdown(f"""
    <div style="color:#64748B; font-size:0.78rem; margin-top:0.8rem;">
        Source: {forecast.get('source', 'Open-Meteo API')} · 
        Updated: {forecast.get('retrieved_at', 'N/A')}
    </div>""", unsafe_allow_html=True)


# ── FORECAST TAB ──────────────────────────────────────────────────────────────
def render_forecast():
    st.markdown('<div class="section-title">7-Day Forecast</div>', unsafe_allow_html=True)
    forecast = st.session_state.current_forecast

    if not forecast:
        st.info("Enter a city in the sidebar and click Get Weather to load forecasts.")
        return

    daily = forecast.get("daily", [])
    if not daily:
        st.warning("No daily forecast data available.")
        return

    # Forecast cards
    cols = st.columns(len(daily))
    for col, day in zip(cols, daily):
        emoji = _weather_emoji(day.get("condition", ""))
        with col:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="forecast-day">{day['date']}</div>
                <div style="font-size:1.8rem; margin:0.4rem 0;">{emoji}</div>
                <div class="forecast-cond">{day.get('condition','N/A')}</div>
                <div class="forecast-high">{_fmt(day.get('temp_max_c'),'°',0)}</div>
                <div class="forecast-low">{_fmt(day.get('temp_min_c'),'°',0)}</div>
                <div style="color:#38BDF8; font-size:0.72rem; margin-top:0.5rem;">
                    Rain: {_fmt(day.get('rain_probability_pct'),'%',0)}<br>
                    Precip: {_fmt(day.get('precipitation_mm'),' mm',1)}<br>
                    Wind: {_fmt(day.get('wind_speed_kmh'),' km/h',0)}
                </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Charts
    st.markdown("### Meteorological Visualizations")
    from modules.visualizations import build_forecast_chart
    charts = build_forecast_chart(forecast)

    if charts:
        tab_t, tab_p, tab_r, tab_w = st.tabs(
            ["Temperature", "Precipitation", "Rain Probability", "Wind Speed"]
        )
        with tab_t:
            st.plotly_chart(charts["temperature"], use_container_width=True)
        with tab_p:
            st.plotly_chart(charts["precipitation"], use_container_width=True)
        with tab_r:
            st.plotly_chart(charts["rain_probability"], use_container_width=True)
        with tab_w:
            st.plotly_chart(charts["wind"], use_container_width=True)
    else:
        st.warning("Could not generate charts from forecast data.")


# ── AI CHAT TAB ───────────────────────────────────────────────────────────────
def render_chat():
    st.markdown('<div class="section-title">AI Weather Chat</div>', unsafe_allow_html=True)

    from config import DEMO_MODE as dm
    if dm:
        st.markdown(f'<div class="demo-banner">{DEMO_BANNER}</div>', unsafe_allow_html=True)

    # Sample questions
    with st.expander("Sample questions you can ask"):
        samples = [
            "What is the weather in Bengaluru today?",
            "Will it rain in Mumbai tomorrow?",
            "Show me the 7-day forecast for London.",
            "Is heavy rain expected in Chennai this week?",
            "What are the effects of heavy rainfall on agriculture?",
            "Should I carry an umbrella in Delhi tomorrow?",
            "Compare today's and tomorrow's temperature in Pune.",
            "Explain the drought conditions in simple language.",
        ]
        for s in samples:
            if st.button(s, key=f"sample_{s[:30]}"):
                st.session_state._prefill_question = s

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                clean_tools = _escape(", ".join(msg["tool_calls"]))
                st.markdown(
                    f'<div class="chat-metadata">Tools used: {clean_tools}</div>',
                    unsafe_allow_html=True
                )

    # Chat input
    prefill = st.session_state.pop("_prefill_question", None)
    question = st.chat_input("Ask anything about weather or climate…") or prefill

    if question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get AI answer
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking…"):
                import importlib
                import modules.weather_api
                import modules.ai_agent
                importlib.reload(modules.weather_api)
                importlib.reload(modules.ai_agent)
                from modules.ai_agent import answer_question
                context = {}
                if st.session_state.current_forecast:
                    context["current_forecast"] = {
                        "city": st.session_state.current_forecast["city"],
                        "retrieved_at": st.session_state.current_forecast["retrieved_at"],
                        "current": st.session_state.current_forecast["current"],
                    }

                result = answer_question(question, context)
                answer       = result.get("answer", "No response generated.")
                tool_calls   = result.get("tool_calls", [])
                tool_results = result.get("tool_results", [])

            st.markdown(answer)

            # Show tool results inline
            for tr in tool_results:
                tool_name = tr.get("tool", "")
                if tool_name == "get_forecast" and tr.get("forecast"):
                    fc = tr["forecast"]
                    if not fc.get("is_demo"):
                        st.session_state.current_forecast = fc
                        st.session_state.current_city     = fc["city"]
                    with st.expander(f"📡 Weather data: {fc.get('city', '')}", expanded=False):
                        cur = fc.get("current", {})
                        st.json({
                            "city":          fc.get("city"),
                            "retrieved_at":  fc.get("retrieved_at"),
                            "temperature_c": cur.get("temperature_c"),
                            "condition":     cur.get("condition"),
                            "humidity_pct":  cur.get("humidity_pct"),
                            "wind_speed_kmh": cur.get("wind_speed_kmh"),
                        })

                elif tool_name == "check_alerts" and tr.get("alerts") is not None:
                    alerts = tr["alerts"]
                    if alerts:
                        st.session_state.alerts = alerts
                        with st.expander(f"{len(alerts)} alert(s) detected", expanded=True):
                            for a in alerts:
                                sev = str(a.get("severity", "LOW"))
                                css = f"alert-{sev.lower()}" if sev.lower() in ("high", "moderate", "low") else "alert-low"
                                a_type = _escape(a.get('type', ''))
                                a_date = _escape(a.get('date', ''))
                                a_obs = _escape(a.get('observed', ''))
                                a_unit = _escape(a.get('unit', ''))
                                a_thresh = _escape(a.get('threshold', ''))
                                a_sev = _escape(sev)
                                st.markdown(f"""<div class="{css}">
                                    <strong>{a_type}</strong> ({a_date}): 
                                    {a_obs} {a_unit} 
                                    [threshold: {a_thresh} {a_unit}]: 
                                    <em>{a_sev}</em>
                                </div>""", unsafe_allow_html=True)
                    else:
                        st.success("No weather alerts for the requested period.")

                elif tool_name == "rag_retrieve" and tr.get("passages"):
                    with st.expander(f"Climate knowledge retrieved ({len(tr['passages'])} passages)"):
                        for p in tr["passages"]:
                            src_e = _escape(p.get('source', ''))
                            pg_e = _escape(p.get('page_number', '1'))
                            rel_e = _escape(p.get('relevance_score', '?'))
                            txt_e = _escape(p.get('text', '')[:400])
                            st.markdown(f"""<div class="rag-passage">
                                <div class="rag-source">{src_e} · Page {pg_e} · Score: {rel_e}</div>
                                <div style="color:#CBD5E1; font-size:0.88rem; margin-top:0.4rem; white-space:pre-wrap;">{txt_e}...</div>
                            </div>""", unsafe_allow_html=True)

            if tool_calls:
                clean_tc = _escape(", ".join(tool_calls))
                st.markdown(
                    f'<div class="chat-metadata">Tools executed: {clean_tc}</div>',
                    unsafe_allow_html=True
                )

        # Store assistant message
        st.session_state.chat_history.append({
            "role":       "assistant",
            "content":    answer,
            "tool_calls": tool_calls,
        })

    # Clear chat
    st.divider()
    if st.button("Clear conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── ALERTS TAB ────────────────────────────────────────────────────────────────
def render_alerts():
    st.markdown('<div class="section-title">Weather Alerts</div>', unsafe_allow_html=True)

    forecast = st.session_state.current_forecast
    alerts   = st.session_state.alerts

    if not forecast:
        st.info("Load a city's weather first to see alerts.")
        return

    city_esc = _escape(forecast.get("city", ""))

    if not alerts:
        st.markdown(f"""
        <div class="weather-card" style="text-align:center; padding:1.5rem;">
            <div style="color:#86EFAC; font-size:1.1rem; font-weight:600;">
                No active weather alerts for {city_esc}
            </div>
            <div style="color:#94A3B8; font-size:0.85rem; margin-top:0.3rem;">
                All monitored weather metrics are within demonstration thresholds.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"**{len(alerts)} alert(s)** found for **{city_esc}**")

        for alert in alerts:
            sev = alert.get("severity", "LOW")
            css = f"alert-{sev.lower()}"
            sev_esc = _escape(sev)
            type_esc = _escape(alert.get("type", ""))
            date_esc = _escape(alert.get("date", ""))
            exp_esc = _escape(alert.get("explanation", ""))
            obs_esc = _escape(alert.get("observed", ""))
            unit_esc = _escape(alert.get("unit", ""))
            thresh_esc = _escape(alert.get("threshold", ""))
            st.markdown(f"""
            <div class="{css}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong>[{sev_esc}] {type_esc}</strong>
                    <span style="color:#94A3B8; font-size:0.8rem;">{date_esc}</span>
                </div>
                <div style="color:#E2E8F0; margin-top:0.4rem;">{exp_esc}</div>
                <div style="color:#94A3B8; font-size:0.8rem; margin-top:0.4rem;">
                    Observed: <strong>{obs_esc} {unit_esc}</strong> · 
                    Threshold: {thresh_esc} {unit_esc} · 
                    Severity: <strong>{sev_esc}</strong>
                </div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="background:rgba(245,158,11,0.08); border-radius:6px; padding:0.8rem 1rem; border:1px solid rgba(245,158,11,0.3);">
        {ALERT_DISCLAIMER}
    </div>""", unsafe_allow_html=True)

    # Threshold table
    with st.expander("Current alert thresholds"):
        from config import ALERT_THRESHOLDS
        import pandas as pd
        df = pd.DataFrame([
            {"Alert Type": "Rain Probability", "Threshold": f"≥ {ALERT_THRESHOLDS['rain_probability_pct']:.0f}%"},
            {"Alert Type": "Daily Rainfall",   "Threshold": f"≥ {ALERT_THRESHOLDS['daily_rainfall_mm']:.0f} mm"},
            {"Alert Type": "Max Temperature",  "Threshold": f"≥ {ALERT_THRESHOLDS['max_temperature_c']:.0f}°C"},
            {"Alert Type": "Min Temperature",  "Threshold": f"≤ {ALERT_THRESHOLDS['min_temperature_c']:.0f}°C"},
            {"Alert Type": "Wind Gusts",        "Threshold": f"≥ {ALERT_THRESHOLDS['wind_gusts_kmh']:.0f} km/h"},
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)


# ── CLIMATE KB TAB ────────────────────────────────────────────────────────────
def render_climate_kb():
    st.markdown('<div class="section-title">Climate Knowledge Base</div>', unsafe_allow_html=True)

    from modules.rag_system import (
        is_rag_available, ingest_documents, ingest_pdf_bytes,
        rag_retrieve, get_indexed_documents, get_collection_count
    )

    if not is_rag_available():
        st.error(
            "RAG dependencies not available. "
            "Install `sentence-transformers` and `chromadb`, then restart."
        )
        return

    # ── Indexing ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Document Indexing")

        # Upload PDF
        uploaded = st.file_uploader(
            "Upload a climate-related PDF document",
            type=["pdf"],
            key="pdf_uploader",
        )
        if uploaded:
            if st.button("Index uploaded PDF", type="primary"):
                with st.spinner(f"Indexing '{uploaded.name}'..."):
                    try:
                        n = ingest_pdf_bytes(uploaded.read(), uploaded.name)
                        if uploaded.name not in st.session_state.rag_doc_names:
                            st.session_state.rag_doc_names.append(uploaded.name)
                        st.success(f"Indexed {n} new chunks from '{uploaded.name}'.")
                        st.session_state.rag_indexed = True
                    except Exception as exc:
                        st.error(f"Indexing failed: {exc}")

        # Index built-in documents
        if st.button("Index built-in climate notes"):
            with st.spinner("Indexing built-in documents..."):
                try:
                    n = ingest_documents()
                    st.success(f"Indexed {n} new chunks from built-in documents.")
                    st.session_state.rag_indexed = True
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    with col2:
        st.markdown("### Index Status")
        try:
            count = get_collection_count()
            docs  = get_indexed_documents()
            st.metric("Total chunks", count)
            st.metric("Documents", len(docs))
            if docs:
                for d in docs:
                    st.markdown(f"`{d}`")
        except Exception as exc:
            st.error(f"Status error: {exc}")

    st.divider()

    # ── Semantic search & AI Q&A ──────────────────────────────────────────────
    st.markdown("### Knowledge Base Q&A & Search")
    st.markdown(
        '<span style="color:#94A3B8; font-size:0.85rem;">'
        'Ask questions about your uploaded PDFs or built-in climate documents. '
        'WeatherGPT will retrieve relevant passages and synthesize an answer citing exact pages.'
        '</span>',
        unsafe_allow_html=True,
    )

    # Sample query suggestions
    sample_queries = [
        "What are the main vehicle emissions causing air pollution?",
        "What are the effects of heavy rainfall on agriculture?",
        "Explain the difference between meteorological and agricultural drought.",
        "What solutions are proposed to reduce vehicle emissions?",
    ]
    st.markdown('<div style="margin: 0.6rem 0 0.4rem 0; font-size:0.82rem; color:#64B5F6;">Try a sample question:</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    chosen_sample = None
    for i, sq in enumerate(sample_queries):
        with cols[i % 2]:
            if st.button(sq, key=f"kb_sample_{i}", use_container_width=True):
                chosen_sample = sq

    search_q = st.text_input(
        "Enter your question",
        value=chosen_sample or "",
        placeholder="e.g. What are the main vehicle emissions in the report?",
        key="rag_search_input",
    )
    col_k, col_btn = st.columns([1, 2])
    with col_k:
        top_k = st.slider("Passages to retrieve", 1, 6, 3, key="rag_top_k")
    with col_btn:
        st.write("")
        st.write("")
        search_clicked = st.button("Ask Knowledge Base", type="primary", use_container_width=True, key="rag_search_btn")

    if search_clicked or chosen_sample:
        query_to_run = (search_q or chosen_sample or "").strip()
        if not query_to_run:
            st.warning("Please enter a question.")
        else:
            count = get_collection_count()
            if count == 0:
                st.warning("The knowledge base is empty. Index some documents first.")
            else:
                with st.spinner("Searching documents & generating answer..."):
                    passages = rag_retrieve(query_to_run, top_k)

                if not passages:
                    st.info("No relevant passages found. Try a different question or index more documents.")
                else:
                    # Generate synthesized answer
                    from modules.ai_agent import _call_gemini_once
                    from config import DEMO_MODE

                    ai_answer = ""
                    if not DEMO_MODE:
                        context_str = "\n\n".join([
                            f"[Document: {p['source']} | Page {p['page_number']}]\n{p['text']}"
                            for p in passages
                        ])
                        rag_prompt = (
                            "You are WeatherGPT's Climate Knowledge Assistant.\n"
                            "Answer the user's question clearly, directly, and accurately using ONLY the retrieved document excerpts below.\n"
                            "Cite the source document name and page number for facts and figures.\n\n"
                            f"Question: {query_to_run}\n\n"
                            f"Retrieved document passages:\n{context_str}\n\n"
                            "Synthesized answer with citations:"
                        )
                        try:
                            ai_answer = _call_gemini_once(rag_prompt)
                        except Exception as exc:
                            logger.warning("Gemini synthesis failed: %s", exc)

                    if not ai_answer:
                        # Fallback synthesis for demo / offline mode
                        p_bullets = [
                            f"• **{p['source']} (Page {p['page_number']})**: {p['text'][:250]}..."
                            for p in passages
                        ]
                        ai_answer = "Key findings from the indexed documents:\n\n" + "\n\n".join(p_bullets)

                    # Display AI Answer Card
                    clean_answer = _escape(ai_answer)
                    st.markdown(f"""
                    <div style="background:rgba(59,130,246,0.12); border-left:4px solid #3B82F6; border-radius:8px; padding:1.2rem; margin:1rem 0;">
                        <div style="color:#64B5F6; font-size:1.05rem; font-weight:700; margin-bottom:0.5rem;">
                            AI Answer from Indexed Documents
                        </div>
                        <div style="color:#F1F5F9; font-size:0.95rem; line-height:1.6; white-space:pre-wrap;">{clean_answer}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display cited source excerpts
                    with st.expander(f"View {len(passages)} Cited Document Excerpt(s)", expanded=True):
                        for i, p in enumerate(passages, 1):
                            src_esc = _escape(p.get('source', 'Unknown'))
                            pg_esc = _escape(p.get('page_number', '1'))
                            rel_esc = _escape(p.get('relevance_score', '?'))
                            text_esc = _escape(p.get('text', ''))
                            st.markdown(f"""<div class="rag-passage">
                                <div class="rag-source">
                                    [{i}] Document: {src_esc} · Page {pg_esc} · 
                                    Relevance: {rel_esc}
                                </div>
                                <div style="color:#CBD5E1; font-size:0.9rem; margin-top:0.5rem; line-height:1.6; white-space:pre-wrap;">
                                    {text_esc}
                                </div>
                            </div>""", unsafe_allow_html=True)


# ── PRIVACY POLICY TAB ────────────────────────────────────────────────────────
def render_privacy_policy():
    st.markdown('<div class="section-title">Privacy Policy</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="weather-card">
        <h3 style="color:#F1F5F9; font-size:1.1rem; margin-bottom:0.8rem;">1. Information We Process</h3>
        <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">
            WeatherGPT processes geographic search terms (city names or coordinates) strictly to query public meteorological data via the Open-Meteo API. 
            When asking conversational climate or weather questions, queries are processed through Google Gemini API.
        </p>

        <h3 style="color:#F1F5F9; font-size:1.1rem; margin-top:1.5rem; margin-bottom:0.8rem;">2. Document Indexing and Local Storage</h3>
        <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">
            PDF documents uploaded to the Climate Knowledge Base are parsed and embedded locally using Sentence Transformers. Vector embeddings and text chunks are stored in your local ChromaDB directory. No uploaded documents are transmitted to external ad networks or third-party trackers.
        </p>

        <h3 style="color:#F1F5F9; font-size:1.1rem; margin-top:1.5rem; margin-bottom:0.8rem;">3. Data Retention and Control</h3>
        <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">
            All chat histories and cached session states reside purely in temporary memory during your active browser session. You can purge conversational history, reset loaded forecasts, and clear your vector database index at any time from the Settings tab.
        </p>

        <h3 style="color:#F1F5F9; font-size:1.1rem; margin-top:1.5rem; margin-bottom:0.8rem;">4. Third-Party Meteorological Services</h3>
        <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">
            Live weather conditions and forecasts are provided by Open-Meteo under standard non-commercial open data licenses.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── TERMS AND CONDITIONS TAB ──────────────────────────────────────────────────
def render_terms():
    st.markdown('<div class="section-title">Terms and Conditions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="weather-card">
        <h3 style="color:#F1F5F9; font-size:1.1rem; margin-bottom:0.8rem;">1. Purpose and Demonstration Nature</h3>
        <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">
            WeatherGPT is an agentic AI engineering demonstration and meteorological reference platform. It provides automated weather data retrieval, RAG-augmented climate analysis, and threshold-based informational alerts.
        </p>

        <h3 style="color:#F1F5F9; font-size:1.1rem; margin-top:1.5rem; margin-bottom:0.8rem;">2. Not Official Meteorological Advice</h3>
        <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">
            All automated alerts (including rain probability, high heat, and wind gust thresholds) are computed using project demonstration rules. They do not constitute official meteorological warnings. For safety-critical decisions, severe weather warnings, and disaster management, always consult official national agencies such as the India Meteorological Department (IMD) or your local meteorological authority.
        </p>

        <h3 style="color:#F1F5F9; font-size:1.1rem; margin-top:1.5rem; margin-bottom:0.8rem;">3. Acceptable Use</h3>
        <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">
            Users agree not to submit malicious content or abuse underlying API endpoints. The software is provided on an "as-is" basis without warranties regarding uninterrupted uptime or data perfection.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── SETTINGS TAB ─────────────────────────────────────────────────────────────
def render_settings():
    st.markdown('<div class="section-title">Settings & Status</div>', unsafe_allow_html=True)

    # API Status
    st.markdown("### Integration Status")
    from modules.ai_agent import get_agent_status
    from modules.rag_system import is_rag_available, get_collection_count
    status = get_agent_status()

    rows = [
        ("Gemini AI", "Active" if status["available"] else "Demo mode (no key)", status["model"]),
        ("Open-Meteo", "Available (free, no key needed)", "Weather + geocoding API"),
        ("Sentence Transformers", "Available" if is_rag_available() else "Not installed", "all-MiniLM-L6-v2"),
        ("ChromaDB", "Available" if is_rag_available() else "Not installed", f"{get_collection_count()} chunks indexed"),
    ]
    import pandas as pd
    df = pd.DataFrame(rows, columns=["Integration", "Status", "Details"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    if not status["available"]:
        st.markdown("""
        > **To enable Gemini AI:**
        > 1. Get a free API key from [Google AI Studio](https://aistudio.google.com/)
        > 2. Add it to your `.env` file: `GEMINI_API_KEY=your_key_here`
        > 3. Restart the Streamlit app
        """)

    st.divider()

    # Controls
    st.markdown("### Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat cleared.")
    with c2:
        if st.button("Reset forecast", use_container_width=True):
            st.session_state.current_forecast = None
            st.session_state.alerts = []
            st.success("Forecast reset.")
    with c3:
        from modules.rag_system import clear_collection
        if st.button("Clear RAG index", use_container_width=True):
            st.session_state._confirm_clear_rag = True

    if st.session_state.get("_confirm_clear_rag"):
        st.warning("Action required: Confirm clearing all indexed vector knowledge chunks.")
        cb1, cb2 = st.columns(2)
        with cb1:
            if st.button("Confirm Delete", type="primary", use_container_width=True, key="confirm_clear_btn"):
                try:
                    clear_collection()
                    st.session_state.rag_indexed = False
                    st.session_state.rag_doc_names = []
                    st.session_state._confirm_clear_rag = False
                    st.success("RAG index cleared.")
                    st.rerun()
                except Exception as exc:
                    st.error("Could not clear index.")
        with cb2:
            if st.button("Cancel", use_container_width=True, key="cancel_clear_btn"):
                st.session_state._confirm_clear_rag = False
                st.rerun()

    st.divider()
    st.markdown("### Application Info")
    st.markdown(f"""
    | Property | Value |
    |---|---|
    | Application | WeatherGPT |
    | Version | 1.0.0 |
    | Python | {sys.version.split()[0]} |
    | AI Model | {status['model']} |
    | Mode | {'Demo' if status['demo_mode'] else 'Live'} |
    | Weather Source | Open-Meteo (free) |
    """)


# ── Main router ───────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    tab = st.session_state.get("active_tab", "Dashboard")

    # Remove emoji prefix if present
    tab_clean = tab.strip()
    for prefix in ["🏠 ", "📅 ", "💬 ", "🚨 ", "📚 ", "⚙️ "]:
        tab_clean = tab_clean.replace(prefix, "")

    renderers = {
        "Dashboard":           render_dashboard,
        "Forecast":            render_forecast,
        "AI Chat":             render_chat,
        "Alerts":              render_alerts,
        "Climate KB":          render_climate_kb,
        "Settings":            render_settings,
        "Privacy Policy":      render_privacy_policy,
        "Terms & Conditions":  render_terms,
    }

    # Default to Dashboard on first load
    renderer = renderers.get(tab_clean, render_dashboard)
    renderer()

    # Persistent global footer
    st.divider()
    fc1, fc2, fc3 = st.columns([3, 1, 1])
    with fc1:
        st.markdown(
            "<span style='color:#64748B; font-size:0.8rem;'>"
            "WeatherGPT v1.0.0 · Meteorological data provided by <a href='https://open-meteo.com/' target='_blank' style='color:#94A3B8;'>Open-Meteo</a>"
            "</span>",
            unsafe_allow_html=True
        )
    with fc2:
        if st.button("Privacy Policy", key="footer_privacy", use_container_width=True):
            st.session_state.active_tab = "Privacy Policy"
            st.rerun()
    with fc3:
        if st.button("Terms & Conditions", key="footer_terms", use_container_width=True):
            st.session_state.active_tab = "Terms & Conditions"
            st.rerun()


if __name__ == "__main__":
    main()
