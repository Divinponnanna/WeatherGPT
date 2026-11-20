"""
modules/ai_agent.py – Gemini Agentic AI Core
=============================================
Responsibilities
----------------
* Initialise the google-genai client from GEMINI_API_KEY.
* Define three Python tools for Gemini function calling:
    1. get_forecast(city, days)
    2. check_alerts(forecast)      – takes the city + days, calls weather API internally
    3. rag_retrieve(question, top_k)
* Run an agentic loop: Gemini picks tools → Python executes → Gemini synthesises.
* Cap rounds at MAX_TOOL_ROUNDS to prevent runaway calls.
* Return a human-readable answer string plus structured tool_results.
* Fall back gracefully to demo answers when GEMINI_API_KEY is missing.

Public API
----------
answer_question(question: str, context: dict | None = None) -> dict
    Returns {"answer": str, "tool_results": list, "error": str | None}

get_agent_status() -> dict
    Returns {"available": bool, "model": str, "demo_mode": bool}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    DEMO_MODE,
    MAX_TOOL_ROUNDS,
)

logger = logging.getLogger(__name__)

# ── Lazy Gemini client ────────────────────────────────────────────────────────
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if DEMO_MODE:
        raise RuntimeError("Gemini API key not configured. Running in demo mode.")
    try:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialised (model=%s).", GEMINI_MODEL)
        return _client
    except ImportError as exc:
        raise RuntimeError(
            "google-genai package not installed. "
            "Run: pip install google-genai"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Gemini client: {exc}") from exc


# ── Tool declarations (sent to Gemini so it knows what it can call) ───────────

def _build_tools():
    """Build the google.genai Tool objects from function declarations."""
    from google.genai import types

    get_forecast_decl = types.FunctionDeclaration(
        name="get_forecast",
        description=(
            "Retrieve a live weather forecast from Open-Meteo for the specified city. "
            "Returns current conditions and a daily forecast for the requested number of days."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "city": types.Schema(
                    type=types.Type.STRING,
                    description="City name in English, e.g. 'Bengaluru' or 'London'.",
                ),
                "days": types.Schema(
                    type=types.Type.INTEGER,
                    description="Number of forecast days (1-7). Default is 7.",
                ),
            },
            required=["city"],
        ),
    )

    check_alerts_decl = types.FunctionDeclaration(
        name="check_alerts",
        description=(
            "Evaluate the weather forecast for the given city and return any alert conditions "
            "based on demonstration thresholds (heavy rain, heat, cold, strong wind). "
            "This tool internally fetches the forecast; just provide city and days."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "city": types.Schema(
                    type=types.Type.STRING,
                    description="City name.",
                ),
                "days": types.Schema(
                    type=types.Type.INTEGER,
                    description="Number of days to check (1-7). Default is 7.",
                ),
            },
            required=["city"],
        ),
    )

    rag_retrieve_decl = types.FunctionDeclaration(
        name="rag_retrieve",
        description=(
            "Search the indexed climate knowledge base and return relevant passages "
            "about climate, rainfall, agriculture, drought, weather phenomena, etc. "
            "Use this tool for climate background questions, NOT for live weather data."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "question": types.Schema(
                    type=types.Type.STRING,
                    description="The climate question to search for.",
                ),
                "top_k": types.Schema(
                    type=types.Type.INTEGER,
                    description="Number of passages to return (default 4).",
                ),
            },
            required=["question"],
        ),
    )

    return [types.Tool(function_declarations=[
        get_forecast_decl,
        check_alerts_decl,
        rag_retrieve_decl,
    ])]


# ── Python-side tool execution ────────────────────────────────────────────────

def _execute_tool(name: str, args: dict) -> tuple[Any, dict]:
    """
    Execute the named Python tool with the given arguments.

    Returns (result_for_gemini, structured_result_for_ui)
    """
    # ── get_forecast ─────────────────────────────────────────────────────────
    if name == "get_forecast":
        from modules.weather_api import get_forecast, WeatherAPIError
        city = str(args.get("city", "")).strip()
        days = int(args.get("days", 7))
        days = max(1, min(days, 7))
        if not city:
            result = {"error": "City name is required."}
        else:
            try:
                forecast = get_forecast(city, days)
                # Slim down for Gemini context window (keep essential fields)
                slim = {
                    "city":         forecast["city"],
                    "country":      forecast["country"],
                    "retrieved_at": forecast["retrieved_at"],
                    "current":      forecast["current"],
                    "daily_summary": [
                        {
                            "date":               d["date"],
                            "condition":          d["condition"],
                            "temp_max_c":         d["temp_max_c"],
                            "temp_min_c":         d["temp_min_c"],
                            "rain_probability_pct": d["rain_probability_pct"],
                            "precipitation_mm":   d["precipitation_mm"],
                            "wind_speed_kmh":     d["wind_speed_kmh"],
                        }
                        for d in forecast["daily"]
                    ],
                }
                return slim, {"tool": "get_forecast", "forecast": forecast}
            except WeatherAPIError as exc:
                result = {"error": str(exc)}
        return result, {"tool": "get_forecast", "error": result.get("error")}

    # ── check_alerts ─────────────────────────────────────────────────────────
    elif name == "check_alerts":
        from modules.weather_api import get_forecast, WeatherAPIError
        from modules.alerts import check_alerts
        city = str(args.get("city", "")).strip()
        days = int(args.get("days", 7))
        days = max(1, min(days, 7))
        if not city:
            result = {"error": "City name is required."}
            return result, {"tool": "check_alerts", "error": result["error"]}
        try:
            forecast = get_forecast(city, days)
            alerts   = check_alerts(forecast)
            slim_alerts = [
                {
                    "type":      a["type"],
                    "date":      a["date"],
                    "observed":  a["observed"],
                    "unit":      a["unit"],
                    "threshold": a["threshold"],
                    "severity":  a["severity"],
                }
                for a in alerts
            ]
            result_data = {"city": forecast["city"], "alerts": slim_alerts}
            return result_data, {
                "tool": "check_alerts",
                "forecast": forecast,
                "alerts": alerts,
            }
        except WeatherAPIError as exc:
            err = {"error": str(exc)}
            return err, {"tool": "check_alerts", "error": str(exc)}

    # ── rag_retrieve ──────────────────────────────────────────────────────────
    elif name == "rag_retrieve":
        from modules.rag_system import rag_retrieve, is_rag_available
        question = str(args.get("question", "")).strip()
        top_k    = int(args.get("top_k", 4))
        if not question:
            result = {"error": "Question is required."}
            return result, {"tool": "rag_retrieve", "error": result["error"]}
        if not is_rag_available():
            msg = (
                "RAG knowledge base is not available. "
                "sentence-transformers or chromadb may not be installed."
            )
            return {"passages": [], "note": msg}, {
                "tool": "rag_retrieve", "passages": [], "note": msg
            }
        passages = rag_retrieve(question, top_k)
        slim = [
            {
                "text":    p["text"][:600],
                "source":  p["source"],
                "page":    p["page_number"],
            }
            for p in passages
        ]
        return {"passages": slim}, {"tool": "rag_retrieve", "passages": passages}

    else:
        err = f"Unknown tool '{name}'."
        logger.warning(err)
        return {"error": err}, {"tool": name, "error": err}


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are WeatherGPT, an AI-powered weather assistant.

RULES:
1. Use the provided tools to get real data before answering live-weather questions.
2. Never invent current weather conditions, temperatures or forecasts.
3. When asked about climate background (effects of rain on agriculture, drought, etc.), 
   use the rag_retrieve tool.
4. When asked about live weather, current conditions or forecasts, use get_forecast.
5. When asked about weather alerts or dangerous conditions, use check_alerts.
6. If a tool returns an error, explain the error clearly to the user.
7. Always cite where information came from (live API, RAG document, or your own knowledge).
8. Use clear, friendly language suitable for a general audience.
9. Keep answers concise but informative.
10. Alert thresholds are project demonstrations: always remind users to check official sources.
"""


# ── City Extraction with known cities & cleanup ───────────────────────────────

KNOWN_CITIES = {
    "bengaluru", "bangalore", "mumbai", "delhi", "new delhi", "chennai", "kolkata",
    "hyderabad", "pune", "ahmedabad", "jaipur", "surat", "lucknow", "kanpur",
    "nagpur", "indore", "bhopal", "patna", "vadodara", "ghaziabad", "ludhiana",
    "agra", "nashik", "ranchi", "varanasi", "srinagar", "aurangabad", "dhanbad",
    "amritsar", "navi mumbai", "allahabad", "prayagraj", "howrah", "gwalior",
    "jabalpur", "coimbatore", "vijayawada", "jodhpur", "madurai", "raipur",
    "kota", "chandigarh", "guwahati", "solapur", "hubli", "dharwad", "mysore",
    "mysuru", "tiruchirappalli", "bareilly", "aligarh", "tiruppur", "moradabad",
    "jalandhar", "bhubaneswar", "salem", "warangal", "jalgaon", "guntur",
    "thiruvananthapuram", "trivandrum", "kochi", "cochin", "goa", "shimla",
    "dehradun", "manali", "darjeeling", "gangtok", "ooty", "nainital",
    "madikeri", "virajpet", "coorg", "kodagu", "somwarpet", "kushalnagar",
    "chikmagalur", "chikkamagaluru", "wayanad", "munnar", "udupi", "mangalore",
    "mangaluru", "hassan", "belagavi", "belgaum", "bellary", "ballari",
    "davangere", "shimoga", "shivamogga", "tumkur", "tumakuru", "bidar",
    "gulbarga", "kalaburagi", "bijapur", "vijayapura", "raichur", "bagalkot",
    "gadag", "chitradurga", "kolar", "mandya", "ramanagara", "chamarajanagar",
    "yadgir", "koppal", "haveri",
    "london", "new york", "tokyo", "paris", "dubai", "singapore", "sydney",
    "san francisco", "berlin", "toronto", "seattle", "chicago", "los angeles",
    "zurich", "rome", "madrid", "amsterdam", "beijing", "shanghai", "hong kong",
}

TIME_AND_FILLER_WORDS = [
    r"\b(?:right\s+now|now|currently)\b",
    r"\b(?:today|todays|tomorrow|11omorrow|llomorrow|tmrw|tommorow|tomorow|tommorrow|tonight|yesterday)\b",
    r"\b(?:this|next)\s+(?:week|weekend|month|year|morning|afternoon|evening|night|day)\b",
    r"\bnext\s+\d+\s+days?\b",
    r"\b\d+\s+days?\b",
    r"\bthese\s+days\b",
    r"\bfor\s+me\b",
    r"\bplease\b",
    r"\bthis\b",
    r"\bnext\b",
]

STOP_CITY_WORDS = {
    "weather", "forecast", "climate", "temp", "temperature", "rain", "heavy",
    "city", "area", "for", "in", "at", "on", "next", "this", "the", "now", "today", "tomorrow"
}


def _clean_city_candidate(c: str) -> str:
    import re
    c = re.sub(r"['’]s\b", "", c)
    for pat in TIME_AND_FILLER_WORDS:
        c = re.sub(pat, "", c, flags=re.IGNORECASE)
    c = re.sub(r"[^\w\s-]", "", c).strip()
    return c


def _extract_city(text: str) -> str | None:
    """Robust city extraction with known-city lookup and time-word cleanup."""
    if not text:
        return None
    import re
    lower_text = text.lower()

    # 1. Check known multi-word & single-word cities first
    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(city) + r"\b", lower_text):
            return city.title()

    # 2. Look for prepositions: in, for, at, of, near, around
    match = re.search(
        r"\b(?:in|for|at|of|near|around)\s+([A-Za-z][A-Za-z\s'-]{1,30})",
        text,
        re.IGNORECASE,
    )
    if match:
        cand = _clean_city_candidate(match.group(1))
        words = cand.split()
        if words:
            filtered = [w for w in words if w.lower() not in STOP_CITY_WORDS]
            if filtered:
                cand = " ".join(filtered[:3]).strip()
                if len(cand) >= 2:
                    return cand.title()

    # 3. Look for "<City> weather" or "<City> forecast"
    match2 = re.search(
        r"\b([A-Za-z][A-Za-z\s'-]{2,20})\s+(?:weather|forecast|temperature|climate)\b",
        text,
        re.IGNORECASE,
    )
    if match2:
        cand = _clean_city_candidate(match2.group(1))
        if len(cand) >= 2:
            return cand.title()

    return None


# ── Question intent classifiers ───────────────────────────────────────────────

_WEATHER_KEYWORDS = {
    "weather", "forecast", "rain", "raining", "rainfall", "temperature", "temp",
    "hot", "cold", "wind", "windy", "humid", "humidity", "sunny", "cloud",
    "cloudy", "storm", "umbrella", "today", "tomorrow", "week", "degrees",
    "celsius", "heat", "chill", "drizzle", "shower", "alert", "warning",
    "safe", "travel", "gust", "gusts", "monsoon",
}

_KNOWLEDGE_KEYWORDS = {
    "climate", "environment", "pollution", "vehicle", "vehicles", "emission",
    "emissions", "air quality", "aqi", "smog", "particulate", "pm2.5", "pm10",
    "drought", "flood", "agriculture", "crop", "crops", "soil", "farming",
    "heatwave", "heat wave", "cyclone", "carbon", "greenhouse", "warming",
    "ozone", "report", "pdf", "document", "notes", "paper", "study",
    "e-waste", "waste", "battery", "toxic", "bs-vi", "electric vehicle", "ev",
    "knowledge base", "rag", "summary", "summarize",
}


def _is_weather_question(q: str) -> bool:
    q_low = q.lower()
    return any(w in q_low for w in _WEATHER_KEYWORDS)


def _is_knowledge_question(q: str) -> bool:
    q_low = q.lower()
    return any(w in q_low for w in _KNOWLEDGE_KEYWORDS)


# Backward compatibility
_is_climate_question = _is_knowledge_question


# ── Candidate models & resilient Gemini caller ───────────────────────────────

CANDIDATE_MODELS = [
    GEMINI_MODEL,
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]


def _call_gemini_once(prompt: str) -> str:
    """
    Make a single text call with automatic candidate model fallbacks.
    Tries GEMINI_MODEL first; if 503/404, automatically tries next working model.
    """
    from google.genai import types
    client = _get_client()

    # Deduplicate while preserving order
    models_to_try = []
    for m in CANDIDATE_MODELS:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_exc = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
                if response and response.candidates:
                    parts = response.candidates[0].content.parts or []
                    text = " ".join(p.text for p in parts if p.text).strip()
                    if text:
                        return text
                return "No response generated."
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                is_transient = any(
                    kw in err_str for kw in
                    ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "timed out", "timeout")
                )
                if is_transient and attempt == 0:
                    time.sleep(0.5)
                else:
                    logger.warning("Model '%s' failed (%s), trying next candidate...", model_name, exc)
                    break  # try next model

    raise RuntimeError(str(last_exc))


# ── Prompt Builder ────────────────────────────────────────────────────────────

def _build_grounded_prompt(
    question: str,
    forecast: dict | None,
    alerts: list,
    rag_passages: list,
    context: dict | None,
) -> str:
    """Build a grounded prompt containing live forecast data and/or RAG excerpts."""
    parts = [_SYSTEM_PROMPT, "\n\n---\n\n"]

    if context and not forecast:
        parts.append("## Already-loaded forecast context\n")
        parts.append(json.dumps(context, indent=2, default=str)[:3000])
        parts.append("\n\n")

    if forecast:
        parts.append("## Live Weather Forecast Data (from Open-Meteo API)\n")
        cur = forecast.get("current", {})
        parts.append(
            f"City: {forecast.get('city')}  |  "
            f"Retrieved: {forecast.get('retrieved_at')}  |  "
            f"Source: {forecast.get('source', 'Open-Meteo')}\n"
        )
        parts.append(
            f"Current conditions: {cur.get('temperature_c')}°C, {cur.get('condition')}, "
            f"humidity {cur.get('humidity_pct')}%, wind {cur.get('wind_speed_kmh')} km/h\n\n"
        )
        parts.append("Daily forecast (next 7 days):\n")
        for day in forecast.get("daily", []):
            parts.append(
                f"  - {day.get('date')}: max {day.get('temp_max_c')}°C / "
                f"min {day.get('temp_min_c')}°C, "
                f"precipitation {day.get('precipitation_mm')} mm, "
                f"rain probability {day.get('rain_probability_pct')}%, "
                f"wind gusts {day.get('wind_gusts_kmh')} km/h, "
                f"condition: {day.get('condition')}\n"
            )
        parts.append("\n")

    if alerts:
        parts.append("## Triggered Weather Alerts\n")
        for a in alerts:
            parts.append(
                f"  - [{a.get('severity','?')}] {a.get('type')} on {a.get('date')}: "
                f"observed {a.get('observed')} {a.get('unit')} "
                f"(threshold {a.get('threshold')} {a.get('unit')})\n"
            )
        parts.append("\n")

    if rag_passages:
        parts.append("## Retrieved Excerpts from Indexed Documents / Knowledge Base\n")
        for p in rag_passages:
            parts.append(
                f"[Source: {p.get('source','?')} | Page {p.get('page_number','?')}]\n"
                f"{p.get('text','')[:600]}\n\n"
            )

    parts.append(f"---\n\n## User Question\n{question}\n\n")
    parts.append(
        "Instructions:\n"
        "- Answer the question directly and conversationally using the real data provided above.\n"
        "- When asked about weather, state the actual numbers, dates, and probabilities.\n"
        "- When referencing document excerpts, cite the document name and page number.\n"
        "- Never say live data is unavailable if forecast data is provided above.\n"
        "- Be concise, accurate, and helpful.\n"
    )
    return "".join(parts)


# ── Python Weather Formatter (instant & offline fallback) ─────────────────────

def _python_weather_answer(question: str, forecast: dict, alerts: list) -> str:
    """
    Generate a natural-language weather answer in Python with zero API calls.
    Covers heavy rain, rain probability, umbrellas, temperatures, wind, and weekly forecast.
    """
    q = question.lower()
    city = forecast.get("city", "the city")
    cur  = forecast.get("current", {})
    daily = forecast.get("daily", [])
    src  = forecast.get("source", "Open-Meteo")
    retrieved = forecast.get("retrieved_at", "")
    is_demo = forecast.get("is_demo", False)

    demo_note = " *(demo data)*" if is_demo else ""

    temp    = cur.get("temperature_c")
    feels   = cur.get("feels_like_c")
    cond    = cur.get("condition", "Unknown")
    humid   = cur.get("humidity_pct")
    wind    = cur.get("wind_speed_kmh")

    tomorrow = daily[1] if len(daily) > 1 else (daily[0] if daily else {})
    t_max    = tomorrow.get("temp_max_c")
    t_min    = tomorrow.get("temp_min_c")
    t_rain   = tomorrow.get("precipitation_mm", 0)
    t_prob   = tomorrow.get("rain_probability_pct", 0)
    t_cond   = tomorrow.get("condition", "")
    t_date   = tomorrow.get("date", "tomorrow")
    t_gusts  = tomorrow.get("wind_gusts_kmh", 0)

    alert_lines = ""
    if alerts:
        lines = [f"  🔴 **{a['type']}** on {a['date']} ({a['severity']})" for a in alerts[:3]]
        if len(alerts) > 3:
            lines.append(f"  … and {len(alerts) - 3} more alert(s)")
        alert_lines = "\n\n**⚠️ Weather Alerts:**\n" + "\n".join(lines)

    # 1. Heavy rain specific question
    if "heavy" in q and ("rain" in q or "rainfall" in q or "shower" in q):
        heavy_days = [
            d for d in daily
            if d.get("precipitation_mm", 0) >= 50 or d.get("rain_probability_pct", 0) >= 70
        ]
        if heavy_days:
            lines = [f"⚠️ **Yes, heavy rain conditions are expected in {city}:**\n"]
            for d in heavy_days:
                lines.append(
                    f"- **{d.get('date')}**: rain **{d.get('precipitation_mm', 0):.1f} mm**, "
                    f"probability **{d.get('rain_probability_pct', 0):.0f}%**, {d.get('condition', '')}"
                )
            return "\n".join(lines) + alert_lines + f"\n\n*Source: {src}{demo_note}*"
        else:
            max_p = max((d.get("rain_probability_pct", 0) for d in daily), default=0)
            max_r = max((d.get("precipitation_mm", 0) for d in daily), default=0)
            return (
                f"☀️ **No heavy rain is expected in {city} this week.**\n\n"
                f"- Maximum rain probability over the next 7 days is **{max_p:.0f}%**\n"
                f"- Maximum expected daily precipitation is **{max_r:.1f} mm**\n"
                f"- Current condition: {cond} at {temp}°C.\n"
                f"{alert_lines}\n\n*Source: {src}{demo_note}*"
            )

    # 2. Umbrella advice
    if any(w in q for w in ["umbrella", "carry", "bring"]):
        if t_prob >= 70:
            advice = f"**Yes, definitely carry an umbrella!** 🌧️ Rain probability tomorrow in {city} is **{t_prob:.0f}%** with {t_rain:.1f} mm expected ({t_cond})."
        elif t_prob >= 40:
            advice = f"**Maybe bring an umbrella.** 🌦️ Rain probability tomorrow in {city} is **{t_prob:.0f}%** with {t_rain:.1f} mm expected ({t_cond})."
        else:
            advice = f"**No umbrella needed.** ☀️ Rain probability tomorrow in {city} is only **{t_prob:.0f}%** ({t_cond})."
        return f"{advice}{alert_lines}\n\n*Source: {src}{demo_note}*"

    # 3. Rain tomorrow / general rain
    if any(w in q for w in ["rain", "raining", "rainfall", "drizzle", "shower", "wet"]):
        if t_prob >= 70:
            msg = f"**Yes, it is very likely to rain in {city} tomorrow ({t_date}).** 🌧️\n\n- Rain probability: **{t_prob:.0f}%**\n- Expected rainfall: **{t_rain:.1f} mm**\n- Condition: {t_cond}\n- Temperature: {t_max}°C / {t_min}°C"
        elif t_prob >= 40:
            msg = f"**There is a moderate chance of rain in {city} tomorrow ({t_date}).** 🌦️\n\n- Rain probability: **{t_prob:.0f}%**\n- Expected rainfall: **{t_rain:.1f} mm**\n- Condition: {t_cond}"
        else:
            msg = f"**Rain is unlikely in {city} tomorrow ({t_date}).** ☀️\n\n- Rain probability: **{t_prob:.0f}%**\n- Expected rainfall: **{t_rain:.1f} mm**\n- Condition: {t_cond}\n- Temperature: {t_max}°C / {t_min}°C"
        return f"{msg}{alert_lines}\n\n*Source: {src}{demo_note}*"

    # 4. Temperature / heat / cold
    if any(w in q for w in ["temperature", "temp", "hot", "cold", "heat", "degrees", "celsius", "warm", "chill"]):
        lines = [
            f"🌡️ **Temperature in {city}: {temp}°C** (feels like {feels}°C){demo_note}",
            f"- Current condition: {cond}",
            f"- Humidity: {humid}%",
            f"- Wind: {wind} km/h",
            f"- Tomorrow: {t_max}°C high / {t_min}°C low",
        ]
        return "\n".join(lines) + alert_lines + f"\n\n*Source: {src}*"

    # 5. Wind
    if any(w in q for w in ["wind", "windy", "gust", "breeze"]):
        lines = [
            f"💨 **Wind in {city}: {wind} km/h**{demo_note}",
            f"- Tomorrow wind gusts: up to {t_gusts} km/h",
            f"- Current condition: {cond}",
        ]
        return "\n".join(lines) + alert_lines + f"\n\n*Source: {src}*"

    # 6. Weekly overview
    if any(w in q for w in ["week", "7 day", "7-day", "forecast", "days"]) and daily:
        lines = [f"📅 **7-Day Forecast for {city}**{demo_note}\n"]
        for d in daily[:7]:
            emoji = "🌧️" if d.get("rain_probability_pct", 0) >= 70 else \
                    "🌦️" if d.get("rain_probability_pct", 0) >= 40 else "☀️"
            lines.append(
                f"{emoji} **{d.get('date')}**: "
                f"{d.get('temp_max_c')}°C / {d.get('temp_min_c')}°C, "
                f"rain {d.get('rain_probability_pct', 0):.0f}%, "
                f"{d.get('condition', '')}"
            )
        return "\n".join(lines) + alert_lines + f"\n\n*Source: {src}*"

    # 7. Alert check
    if any(w in q for w in ["alert", "warning", "dangerous", "safe", "risk"]):
        if alerts:
            lines = [f"🚨 **{len(alerts)} weather alert(s) for {city}:**\n"]
            for a in alerts:
                lines.append(
                    f"- **{a['type']}** [{a['severity']}] on {a['date']}: "
                    f"{a['observed']} {a['unit']} (threshold {a['threshold']} {a['unit']})"
                )
            return "\n".join(lines) + f"\n\n*Source: {src}{demo_note}*"
        return f"✅ **No weather alerts for {city}.** Conditions look safe.\n\n*Source: {src}{demo_note}*"

    # Default conditions summary
    lines = [
        f"⛅ **Current Weather in {city}**{demo_note}\n",
        f"- **Temperature:** {temp}°C (feels like {feels}°C)",
        f"- **Condition:** {cond}",
        f"- **Humidity:** {humid}%",
        f"- **Wind:** {wind} km/h",
        f"\n**Tomorrow ({t_date}):** {t_max}°C / {t_min}°C, rain {t_prob:.0f}%, {t_cond}",
    ]
    return "\n".join(lines) + alert_lines + f"\n\n*Source: {src} | Retrieved: {retrieved}*"


# ── Public API ────────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    context: dict | None = None,
) -> dict:
    """
    Answer a weather, climate, or document knowledge question.

    Data is fetched directly using Python modules (Open-Meteo API, ChromaDB RAG),
    then passed to Gemini for a grounded natural answer. If Gemini is unavailable,
    it falls back cleanly to the Python formatter so answers are always provided.
    """
    if not question or not str(question).strip():
        return {
            "answer": "Please enter a question.",
            "tool_results": [],
            "tool_calls": [],
            "error": None,
            "demo_mode": DEMO_MODE,
        }

    question = str(question).strip()
    if len(question) > 1000:
        logger.warning("Truncating oversized question from %d to 1000 chars.", len(question))
        question = question[:1000]

    tool_results: list[dict] = []
    tool_calls_made: list[str] = []

    # ── Step 1: Pre-fetch live weather if query is weather-related ────────────
    forecast = None
    alerts: list = []

    if _is_weather_question(question):
        city = _extract_city(question)
        if not city and context:
            ctx_fc = context.get("current_forecast") or context
            city = ctx_fc.get("city")
        city = city or "Bengaluru"

        try:
            from modules.weather_api import get_forecast
            logger.info("Fetching forecast for '%s'", city)
            forecast = get_forecast(city, 7)
            tool_results.append({"tool": "get_forecast", "forecast": forecast})
            tool_calls_made.append("get_forecast")
        except Exception as exc:
            logger.warning("Weather fetch failed for '%s': %s", city, exc)
            tool_results.append({"tool": "get_forecast", "error": str(exc)})

        if forecast:
            try:
                from modules.alerts import check_alerts
                from config import ALERT_THRESHOLDS
                alerts = check_alerts(forecast, ALERT_THRESHOLDS)
                tool_results.append({"tool": "check_alerts", "forecast": forecast, "alerts": alerts})
                tool_calls_made.append("check_alerts")
            except Exception as exc:
                logger.warning("Alert check failed: %s", exc)

    # ── Step 2: Pre-fetch RAG passages if query touches knowledge/docs ────────
    rag_passages: list = []
    try:
        from modules.rag_system import rag_retrieve, is_rag_available, get_collection_count
        if is_rag_available() and get_collection_count() > 0:
            # Query RAG if question touches climate, environment, or indexed documents
            if _is_knowledge_question(question):
                logger.info("Querying RAG for: %s", question[:60])
                rag_passages = rag_retrieve(question, top_k=3)
                if rag_passages:
                    tool_results.append({"tool": "rag_retrieve", "passages": rag_passages})
                    tool_calls_made.append("rag_retrieve")
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)

    # ── Step 3: In Demo Mode, return demo response ────────────────────────────
    if DEMO_MODE:
        return _demo_answer(question, context, forecast, alerts, rag_passages, tool_results, tool_calls_made)

    # ── Step 4: Live Gemini mode with grounded prompt ─────────────────────────
    prompt = _build_grounded_prompt(question, forecast, alerts, rag_passages, context)
    logger.info("Calling Gemini for grounded answer...")

    try:
        answer = _call_gemini_once(prompt)
        return {
            "answer":       answer,
            "tool_results": tool_results,
            "tool_calls":   tool_calls_made,
            "error":        None,
            "demo_mode":    False,
        }
    except Exception as exc:
        logger.warning("Gemini call failed (%s); falling back to Python formatter", exc)
        # Fallback gracefully to Python answer so the user is never left without an answer
        if forecast:
            fallback = _python_weather_answer(question, forecast, alerts)
            return {
                "answer":       fallback,
                "tool_results": tool_results,
                "tool_calls":   tool_calls_made,
                "error":        None,
                "demo_mode":    False,
            }
        elif rag_passages:
            p_lines = [f"- **{p['source']} (Page {p['page_number']})**: {p['text'][:250]}…" for p in rag_passages]
            fallback = "📚 **Information from Knowledge Base:**\n\n" + "\n\n".join(p_lines)
            return {
                "answer":       fallback,
                "tool_results": tool_results,
                "tool_calls":   tool_calls_made,
                "error":        None,
                "demo_mode":    False,
            }
        return {
            "answer":       f"⚠️ Could not process request: {exc}",
            "tool_results": tool_results,
            "tool_calls":   tool_calls_made,
            "error":        str(exc),
            "demo_mode":    False,
        }


# ── Demo mode fallback ────────────────────────────────────────────────────────

def _demo_answer(
    question: str,
    context: dict | None,
    forecast: dict | None = None,
    alerts: list | None = None,
    rag_passages: list | None = None,
    tool_results: list | None = None,
    tool_calls_made: list | None = None,
) -> dict:
    """Return a clean formatted response in Demo Mode."""
    tool_results = tool_results or []
    tool_calls_made = tool_calls_made or []

    if forecast:
        answer = _python_weather_answer(question, forecast, alerts or [])
    elif rag_passages:
        p_lines = [f"- **{p['source']} (Page {p['page_number']})**: {p['text'][:300]}…" for p in rag_passages]
        answer = "📚 **Relevant Passages from Knowledge Base:**\n\n" + "\n\n".join(p_lines)
    else:
        answer = (
            f"**Demo Mode**: Gemini API key not configured.\n\n"
            f"I understood your question: *'{question}'*\n\n"
            f"Live weather data from Open-Meteo and RAG knowledge base search remain active. "
            f"Add `GEMINI_API_KEY` to your `.env` to enable full conversational AI."
        )

    return {
        "answer":       answer,
        "tool_results": tool_results,
        "tool_calls":   tool_calls_made,
        "error":        None,
        "demo_mode":    True,
    }


# ── Status ────────────────────────────────────────────────────────────────────

def get_agent_status() -> dict:
    """Return the current agent configuration status."""
    return {
        "available": not DEMO_MODE,
        "model":     GEMINI_MODEL,
        "demo_mode": DEMO_MODE,
        "api_key_set": GEMINI_API_KEY is not None and GEMINI_API_KEY != "your_gemini_api_key_here",
    }

