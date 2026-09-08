"""UI package for Blutbild-Buddy.

Holds the Streamlit rendering helpers and per-tab render functions, split out
from app.py to keep each concern in its own module. app.py remains the entry
point (page config, session setup, main() orchestration) and re-exports the
shared helpers so existing imports/tests keep working.
"""
