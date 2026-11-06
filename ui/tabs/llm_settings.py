"""Settings tab (consolidated in the UX rework).

The single "⚙️ Einstellungen" tab: LLM connection settings (base URL, model,
API key, timeout), PDF rendering settings (DPI, batch size), a connection
test, and the biomarker registry editor (moved here from the old
Biomarker-Einstellingen/Mapping tab — reference-data maintenance belongs with
the other settings, not inside the per-patient data review).

The filename stays llm_settings.py: the test suite asserts on its source.
"""

import streamlit as st

from database import get_custom_biomarkers, get_registry_names, update_biomarker_registry
from translations import t
from ui.helpers import _flash, _job_button, _show_flash


def render_llm_settings_tab(session, patient=None):
    """Render the Settings tab.

    ``patient`` scopes the registry NAME edit: a rename migrates measurements,
    and without a selected patient that migration would hit EVERY patient —
    so the name field is offered only with a patient in scope (unit/ref-range
    edits touch only the shared reference row and are always allowed).
    """
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
    st.selectbox("DPI", options=dpi_options, format_func=lambda x: f"{x} {'(default)' if x == 200 else ''}", key="pdf_dpi")

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

    # --- Biomarker registry (moved here from the old Mapping tab) ------------
    # Direct edits to the shared reference data, with the same guards as
    # accepted LLM proposals. The DATA migration of a 'name' edit is scoped to
    # the selected patient — without one it would migrate every patient's
    # measurements, so the name field is not offered in that state.
    st.divider()
    st.subheader(t("subheader_registry_edit"))
    st.caption(t("caption_registry_edit"))
    _show_flash("registry_edit")

    # All known canonical names (registry + custom): editing a custom name
    # promotes it to a full registry row.
    all_canonical_options = sorted(set(get_registry_names(session)
                                       + get_custom_biomarkers(session)))

    field_labels = [t("field_name"), t("field_unit"), t("field_ref_range")]
    if patient is None:
        field_labels = [t("field_unit"), t("field_ref_range")]
        st.caption(t("caption_registry_name_needs_patient"))

    col_e1, col_e2, col_e3, col_e4 = st.columns([2, 1, 2, 1])
    with col_e1:
        edit_name = st.selectbox(t("select_registry_biomarker"), all_canonical_options, key="registry_edit_name")
    with col_e2:
        # Patient-scoped key: the option list differs between states (no name
        # option without a patient), so a stale saved value from the other
        # state would be invalid — a fresh widget per scope sidesteps that.
        edit_field_label = st.selectbox(
            t("select_registry_field"),
            field_labels,
            key=f"registry_edit_field_{patient.id if patient is not None else 'app'}",
        )
    field_key = {
        t("field_name"): "name",
        t("field_unit"): "unit",
        t("field_ref_range"): "ref_range",
    }[edit_field_label]
    with col_e3:
        edit_value = st.text_input(
            t("input_registry_value"), key="registry_edit_value",
            placeholder={
                "name": t("placeholder_registry_name"),
                "unit": t("placeholder_registry_unit"),
                "ref_range": t("placeholder_registry_ref_range"),
            }[field_key],
        )
    with col_e4:
        apply_edit = st.button(t("btn_apply_registry_edit"), type="secondary",
                               width='stretch', key="btn_apply_registry_edit")

    if apply_edit:
        if not edit_value.strip():
            st.error(t("error_empty_value"))
        else:
            # patient_id scopes the DATA migration of a 'name' edit to this
            # patient (the registry row itself stays global — see
            # update_biomarker_registry). unit/ref_range edits touch only
            # the shared reference row and never migrate measurements.
            outcome = update_biomarker_registry(session, edit_name, field_key,
                                                edit_value, patient_id=patient.id if patient else None)
            # Queued via _flash: the _rerun_requested below discards this run.
            if outcome["status"] == "applied":
                _flash("registry_edit", "success",
                       t("msg_registry_edit_applied", field=field_key, proposed=outcome["detail"]))
            else:
                _flash("registry_edit", "warning",
                       t("msg_suggestion_skipped", detail=outcome.get("detail", "")))
            st.session_state._rerun_requested = True
