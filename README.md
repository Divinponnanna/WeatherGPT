# WeatherGPT – Agentic AI Weather Assistant

> A complete, production-quality Agentic AI weather application built with Python,
> Streamlit, Google Gemini, Open-Meteo, Sentence Transformers and ChromaDB.

---

## 🌟 Project Introduction

WeatherGPT is an academic engineering project demonstrating:

- **Live weather retrieval** via the Open-Meteo API (free, no account needed)
- **Agentic AI** using Google Gemini with actual Python function calling
- **Retrieval-Augmented Generation (RAG)** over climate documents with page citations
- **Deterministic weather alerts** with configurable thresholds
- **Interactive Plotly charts** for forecasts
- **Conversational chat history** in Streamlit session state
- **Offline / demo mode** when APIs are unavailable

---

## 🎯 Objectives

1. Retrieve and display real-time weather for any city.
2. Let users ask natural-language weather questions.
3. Have Gemini automatically select and execute the right Python tools.
4. Ground AI answers in real weather data — never invented values.
5. Enable climate document search with source and page citations.
6. Alert on dangerous weather using rule-based, transparent thresholds.
7. Visualise forecasts with interactive Plotly charts.
8. Handle every failure gracefully with clear user messages.

---

## ✨ Features

| Feature | Detail |
|---|---|
| Live weather | Open-Meteo API (7-day forecast, current conditions) |
| Geocoding | City name → coordinates automatically |
| AI chat | Gemini 2.0 Flash with function calling |
| Agentic tools | get_forecast, check_alerts, rag_retrieve |
| RAG | PDF + TXT indexing, semantic search, page citations |
| Alerts | 5 configurable threshold rules |
| Charts | 4 Plotly charts (temp, precip, rain prob, wind) |
| Demo mode | Full offline fallback with sample data |
| Tests | 60+ automated pytest tests |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Frontend | Streamlit 1.35+ |
| AI | Google Gemini (google-genai SDK) |
| Weather API | Open-Meteo (free, non-commercial) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| PDF extraction | PyPDF |
| Charts | Plotly |
| Data | Pandas |
| Testing | Pytest + pytest-mock |

---

## 🏗️ Architecture

```
User Input (Streamlit)
        │
        ▼
   app.py (UI orchestration)
        │
        ├──► modules/weather_api.py ──► Open-Meteo Geocoding + Forecast API
        │
        ├──► modules/ai_agent.py ─────► Google Gemini (function calling loop)
        │          │
        │          ├──► Tool: get_forecast   → weather_api.py → Open-Meteo
        │          ├──► Tool: check_alerts   → alerts.py (deterministic)
        │          └──► Tool: rag_retrieve   → rag_system.py → ChromaDB
        │
        ├──► modules/alerts.py ───────► Threshold-based alert engine
        │
        ├──► modules/rag_system.py ───► PDF ingest → Embeddings → ChromaDB
        │
        └──► modules/visualizations.py ► Plotly chart builders

Persistence:
  vector_db/   ← ChromaDB local storage
  data/        ← Sample data + uploaded climate documents
```

---

## 📁 Folder Structure

```
WeatherGPT/
├── app.py                   # Streamlit UI entry point
├── config.py                # Configuration, constants, thresholds
├── requirements.txt         # All Python dependencies
├── .env                     # API keys (not committed)
├── .env.example             # Template for .env
├── .gitignore               # Excludes secrets, venv, vector_db
├── README.md                # This file
│
├── modules/
│   ├── __init__.py
│   ├── weather_api.py       # Open-Meteo geocoding + forecast
│   ├── ai_agent.py          # Gemini client + agentic loop
│   ├── rag_system.py        # PDF ingest, embeddings, ChromaDB
│   ├── alerts.py            # Alert threshold engine
│   └── visualizations.py   # Plotly chart builders
│
├── data/
│   ├── climate_documents/   # Seed documents for RAG
│   └── sample_weather.json  # Demo mode fallback data
│
├── vector_db/               # ChromaDB persistence (auto-created)
│
├── tests/
│   ├── test_weather_api.py
│   ├── test_ai_agent.py
│   ├── test_rag_system.py
│   ├── test_alerts.py
│   └── test_visualizations.py
│
└── docs/
    ├── architecture.md
    ├── project_report.md
    └── demo_guide.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11 or 3.12
- Internet access (for first-time model download and live weather)
- A Google Gemini API key (free from [Google AI Studio](https://aistudio.google.com/))

### Step 1 – Clone or navigate to the project

```bash
cd C:\Users\PONNANNA MB\PycharmProjects\Agentic_AI
```

### Step 2 – Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First-time install downloads PyTorch and sentence-transformers (~1.5 GB).
> This only happens once.

### Step 4 – Configure your Gemini API key

```bash
# Copy the template
copy .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=your_actual_key_here
```

> Without the key, the app runs in **Demo Mode** — weather data is live but AI explanations are pre-written.

---

## ▶️ Running the Application

```bash
# Activate venv first
.\.venv\Scripts\Activate.ps1

# Start the Streamlit app
streamlit run app.py
```

The browser will open at `http://localhost:8501`.

---

## 📖 How to Use Each Feature

### Dashboard
- Enter a city in the sidebar → click **Get Weather**
- See current temperature, humidity, wind, precipitation
- Or click **Demo Data** for offline sample data

### 7-Day Forecast
- Navigate to **Forecast** in the sidebar
- View forecast cards for each day
- Explore 4 interactive Plotly charts

### AI Chat
- Navigate to **AI Chat**
- Type any weather or climate question
- Gemini will call the appropriate Python tools and explain the data
- Sample questions are provided for reference

### Weather Alerts
- Navigate to **Alerts** after loading weather data
- See all triggered threshold-based alerts
- Each alert shows type, observed value, threshold and severity

### Climate Knowledge Base
- Navigate to **Climate KB**
- Click **Index built-in climate notes** to load sample documents
- Or upload your own PDF files
- Search semantically and see source + page citations

### Settings
- Navigate to **Settings** to view integration status
- Clear chat history, reset forecast or clear RAG index

---

## 🧪 Running Tests

```bash
# Activate venv
.\.venv\Scripts\Activate.ps1

# Run all tests with verbose output
python -m pytest tests/ -v

# Run only a specific test file
python -m pytest tests/test_alerts.py -v

# Run with coverage (if coverage is installed)
python -m pytest tests/ -v --tb=short
```

> All tests mock external API calls. No internet required. No API quota consumed.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in the activated venv |
| AI Chat shows Demo Mode | Add your Gemini API key to `.env` and restart |
| Weather not loading | Check internet connection; try Demo Data mode |
| RAG indexing fails | Ensure `sentence-transformers` and `chromadb` are installed |
| Port 8501 already in use | Use `streamlit run app.py --server.port 8502` |
| Slow first startup | Sentence Transformer model downloads ~90 MB on first use |

---

## ⚠️ Limitations

- Alert thresholds are **demonstration values**, not official IMD/WMO warnings.
- Gemini may not always choose the optimal tool on the first attempt.
- RAG quality depends on the quality of indexed documents.
- Open-Meteo forecast accuracy varies by region and season.
- Demo mode provides pre-written AI responses, not real Gemini inference.

---

## 🔒 Security

- **API keys** are loaded from `.env`, never hardcoded.
- `.env` is excluded from version control via `.gitignore`.
- All tool arguments are validated before execution.
- No personal data is stored beyond session state.

---

## 📄 Attribution

- Weather data: **Open-Meteo** – https://open-meteo.com/ (free for non-commercial use)
- AI: **Google Gemini** – https://ai.google.dev/
- Embeddings: **Sentence Transformers** – https://sbert.net/
- Charts: **Plotly** – https://plotly.com/

---

## 👨‍💻 Project Information

| Property | Value |
|---|---|
| Project | WeatherGPT |
| Type | Academic Engineering Mini Project |
| Language | Python 3.11 |
| Framework | Streamlit |
| AI Model | Gemini 2.0 Flash |
