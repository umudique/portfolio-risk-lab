"""Streamlit entry point for Portfolio Risk Lab."""

from __future__ import annotations

from app.dashboard import render_dashboard
from app.orchestrator import ConcreteAnalysisOrchestrator


def main() -> None:
    render_dashboard(ConcreteAnalysisOrchestrator())


if __name__ == "__main__":
    main()
