"""LLM Settings tab (moved verbatim from app.py, WP2).

Renders the LLM Settings tab: connection settings (base URL, model, API key,
timeout), PDF rendering settings (DPI, batch size), and a connection test.
"""

import streamlit as st

from translations import t
from ui.helpers import _job_button, _show_flash


def render_llm_settings_tab(session):
    """Render the LLM Settings tab."""
    st.subheader(t("subheader_llm_settings"))
    st.caption(t("caption_llm_settings"))

    # LLM connection settings
    # Note: no `value=` here — the session-state keys are initialized in app.py,
    # and passing both would trigger Streamlit's "default + session state" warning.
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Base URL", key="llm_base_url")
        st.text_input("Model", key="llm_model")
    with col2:
        st.text_input("API Key", type="password", key="llm_api_key")
        st.number_input("Timeout (s)", min_value=1, max_value=2000, key="llm_timeout")

    st.divider()
    st.subheader(t("subheader_pdf_settings_tab"))
    st.caption(t("caption_pdf_settings_tab"))

    # No `index=` — the session-state keys are initialized in app.py and take
    # precedence; passing both triggers Streamlit's "default + session state" warning.
    dpi_options = [75, 100, 125, 150, 200]
    st.selectbox("DPI", options=dpi_options, format_func=lambda x: f"{x} {'(default)' if x == 150 else ''}", key="pdf_dpi")

    # 0 = every page of a report in ONE call ("one report = one LLM call").
    # Measured on the local endpoint this is both faster and safer than
    # per-page calls; 1-4 split big reports for weaker models.
    batch_options = [0, 1, 2, 3, 4]
    st.selectbox(
        t("label_batch_size"), options=batch_options,
        format_func=lambda x: (f"{t('label_batch_all')} (default)" if x == 0
                               else str(x)),
        key="pdf_batch_size",
    )

    # Settings are applied live (session state is read on every run) — no save needed.
    st.caption(t("caption_settings_live"))

    # Test connection button (LLM call → primary, per app-wide color convention).
    # Runs as a background job: an inline call blocks the whole Streamlit script,
    # so an unreachable endpoint froze the app for the full configured timeout
    # (up to 2000s) with no feedback. The job's verdict is flashed back into this
    # tab by the LLM-Jobs poll fragment (see _FINISH_FLASH_KEYS["test_connection"]).
    _show_flash("llm_settings")
    _job_button(
        session,
        label_key="btn_test_connection",
        kind="test_connection",
        job_label_key="job_kind_test_connection",
        key="btn_test_connection_action",
        flash_key="llm_settings",
        button_type="primary",
        help_key="help_test_connection",
    )
