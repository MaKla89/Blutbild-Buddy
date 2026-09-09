"""Patients management tab (moved verbatim from app.py, WP2).

Renders the Patients tab: patient list with report/value counts, rename,
and merge.
"""

import pandas as pd
import streamlit as st
from sqlalchemy import func

from database import Report, DataPoint, rename_patient, merge_patients, get_session
from translations import t, _get_lang


def render_patients_tab(session, patients):
    """Render the Patients management tab."""
    st.subheader(t("subheader_patients_manage"))
    st.caption(t("caption_patients_manage"))

    if not patients:
        st.info(t("msg_no_patients"))
        return

    # Patient list with rename and merge options
    _plural = "" if len(patients) == 1 else ("e" if _get_lang() == "de" else "s")
    st.markdown(f"**{t('heading_patients_list', n=len(patients), plural=_plural)}**")

    patient_options = {p.name: p.id for p in patients}

    # Table-style display of patients.
    # Two grouped queries for all per-patient counts (was N+1: two queries per
    # patient).
    report_counts = dict(
        session.query(Report.patient_id, func.count(Report.id))
        .group_by(Report.patient_id)
        .all()
    )
    dp_counts = dict(
        session.query(Report.patient_id, func.count(DataPoint.id))
        .join(Report)
        .group_by(Report.patient_id)
        .all()
    )
    patient_rows = []
    for p in patients:
        report_count = report_counts.get(p.id, 0)
        dp_count = dp_counts.get(p.id, 0)
        patient_rows.append({
            "name": p.name,
            "reports": report_count,
            "values": dp_count,
            "patient_id": p.id,
        })

    df_patients = pd.DataFrame(patient_rows)
    st.dataframe(
        df_patients.drop(columns=["patient_id"], errors="ignore"),
        column_config={
            "name": st.column_config.TextColumn(t("select_patient")),
            "reports": st.column_config.NumberColumn(t("metric_reports"), format="%d"),
            "values": st.column_config.NumberColumn(t("col_biomarker_count"), format="%d"),
        },
        hide_index=True,
        width='stretch',
    )

    # Rename patient section
    st.divider()
    st.markdown(f"✏️ **{t('rename_patient')}**")

    col_rename1, col_rename2 = st.columns(2)
    with col_rename1:
        selected_for_rename = st.selectbox(
            t("select_patient"),
            options=list(patient_options.keys()),
            key="_rename_patient_select",
        )
    with col_rename2:
        new_name = st.text_input(t("rename_patient"), value=selected_for_rename, key="rename_patient_input")

    if st.button(t("rename_patient"), type="secondary", width='stretch', key="btn_rename_patient_action"):
        if new_name and new_name != selected_for_rename:
            if rename_patient(session, selected_for_rename, new_name):
                st.success(t("msg_renamed", old=selected_for_rename, new=new_name))
                st.session_state._rerun_requested = True
            else:
                st.error(t("msg_rename_error"))
        elif new_name == selected_for_rename:
            st.info(t("warning_same_name"))

    # Merge patients section
    st.divider()
    st.markdown(f"🔗 **{t('subheader_merge_patients')}**")
    st.warning(t("warning_confirm_merge"))

    current_options = list(patient_options.keys())

    col_merge1, col_merge2 = st.columns(2)
    with col_merge1:
        default_source = 0
        if "_merge_source" in st.session_state:
            saved_source = st.session_state._merge_source
            if saved_source in patient_options:
                default_source = current_options.index(saved_source)
        source_name = st.selectbox(
            t("select_source_patient"),
            options=current_options,
            index=default_source,
            key="_merge_source",
        )
    with col_merge2:
        default_target = 1 if len(current_options) > 1 else 0
        if "_merge_target" in st.session_state:
            saved_target = st.session_state._merge_target
            if saved_target in patient_options:
                default_target = current_options.index(saved_target)
            elif default_target >= len(current_options):
                default_target = len(current_options) - 1
        target_name = st.selectbox(
            t("select_target_patient"),
            options=current_options,
            index=default_target,
            key="_merge_target",
        )

    @st.dialog(t("dialog_confirm_merge_title"), width="medium")
    def _confirm_merge():
        st.markdown(
            t(
                "dialog_confirm_merge_text",
                source=source_name,
                target=target_name,
            )
        )
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            if st.button(t("btn_cancel"), key="dlg_merge_cancel", width='stretch'):
                st.rerun()
        with col_c2:
            if st.button(t("btn_confirm_merge"), type="secondary", key="dlg_merge_confirm", width='stretch'):
                # The dialog re-runs independently of main(), so it owns its
                # own DB session (same pattern as the long-op fragments).
                with get_session() as op_session:
                    result = merge_patients(
                        op_session,
                        patient_options[source_name],
                        patient_options[target_name],
                    )
                if "error" in result:
                    st.session_state._flash_merge = ("error", result["error"])
                else:
                    st.session_state._flash_merge = (
                        "success",
                        t("msg_merge_success", reports=result["reports_moved"], dps=result["dps_moved"]),
                    )
                    # Clear merge selections so they reset on rerun
                    for key in ["_merge_source", "_merge_target"]:
                        if key in st.session_state:
                            del st.session_state[key]
                st.rerun()

    if st.button(t("btn_confirm_merge"), type="secondary", width='stretch', key="btn_merge_patients_action"):
        if source_name == target_name:
            st.error(t("error_merge_same_patient"))
        else:
            _confirm_merge()

    # Result toast: rendered on the run AFTER the dialog's st.rerun(), at the
    # bottom of the page. No _rerun_requested needed — the app-scope rerun
    # already reloaded all data.
    if _merge_msg := st.session_state.pop("_flash_merge", None):
        getattr(st, _merge_msg[0])(_merge_msg[1])
