# modules/__init__.py
# Exposes the public API of each module for convenient imports.

from modules.weather_api import get_forecast
from modules.alerts import check_alerts
from modules.rag_system import rag_retrieve
from modules.ai_agent import answer_question
from modules import visualizations

__all__ = [
    "get_forecast",
    "check_alerts",
    "rag_retrieve",
    "answer_question",
    "visualizations",
]
