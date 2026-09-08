"""LLM Settings tab (moved verbatim from app.py, WP2).

Renders the LLM Settings tab: connection settings (base URL, model, API key,
timeout), PDF rendering settings (DPI, batch size), and a connection test.
"""

import streamlit as st
from openai import APIConnectionError, APITimeoutError

import config
from llm_client import _build_client
from translations import t


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

    batch_options = [1, 2, 3, 4]
    st.selectbox(t("label_batch_size"), options=batch_options, format_func=lambda x: f"{x} {'(default)' if x == 1 else ''}", key="pdf_batch_size")

    # Settings are applied live (session state is read on every run) — no save needed.
    st.caption(t("caption_settings_live"))

    # Test connection button (LLM call → primary, per app-wide color convention)
    st.divider()
    if st.button(t("btn_test_connection"), type="primary", width='stretch'):
        try:
            test_llm = _build_client()
            model = st.session_state.get("llm_model", config.get_unsloth_model())
            timeout = int(st.session_state.get("llm_timeout", config.get_llm_timeout()))
            # Lightweight ping — just list models or do a tiny completion
            test_llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "OK"}],
                max_tokens=1,
                timeout=timeout,
            )
            st.success(t("msg_llm_connected"))
        except (APIConnectionError, APITimeoutError) as e:
            st.error(f"{t('msg_llm_disconnected')} {e}")
            st.warning(
                t("tooltip_docker_llm")
            )
        except Exception as e:
            st.error(f"{t('msg_llm_error')} {type(e).__name__}: {e}")
