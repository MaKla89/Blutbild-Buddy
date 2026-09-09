"""Reports management tab (moved verbatim from app.py, WP2).

Renders the Reports tab: PDF upload + processing, risk-flag refresh, summary
regeneration, bulk description/interpretation generation, and the report list
with delete. (Reference-range backfill lives in the Biomarker-Einstellungen
tab.)

Since WP6 all long-running operations are submitted as background jobs
(llm_jobs.submit_job) — the buttons no longer block the UI, and progress is
tracked in the "LLM-Jobs" tab.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import func

import config
from database import (
    Report, DataPoint, RiskFlag, get_session,
)
from llm_jobs import submit_job
from translations import t
from ui.helpers import (
    _handle_file_upload,
    _flash, _show_flash,
)


def render_reports_tab(session, patients, reports, dps, gen_summary):
    """Render the Reports management tab."""
    st.subheader(t("subheader_reports_manage"))
    st.caption(t("caption_reports_manage"))

    # File upload section
    st.markdown(f"**{t('btn_upload_and_process')}**")
    uploaded_files = st.file_uploader(
        t("label_upload_pdf"),
        type=["pdf"],
        accept_multiple_files=True,
        key="_uploaded_pdfs",
        on_change=_handle_file_upload,
    )

    # Pending-upload indicator: files are saved to disk on upload but only
    # processed when the user clicks "Process All PDFs" — make that explicit.
    if st.session_state.get("_files_uploaded"):
        st.info(t("info_files_saved"))

    # --- Long-running operations (WP6: background jobs) ----------------------
    # Each operation submits a background job (llm_jobs.py) that runs in a
    # daemon worker thread — a page refresh or close no longer loses progress.
    # The "LLM-Jobs" tab polls the llm_jobs table and shows live status; when a
    # data-affecting job finishes, it triggers a full app rerun so all tabs see
    # the new data. submit_job() returns None when a job of the same kind is
    # already pending/running (kind lock), which we surface as an info message.

    @st.fragment
    def _op_process_pdfs():
        _show_flash("process_pdfs")
        if st.button(t("btn_process_all"), type="primary", width='stretch'):
            patient = st.session_state.get("_selected_patient")
            job = submit_job(
                "process_pdfs", "job_kind_process_pdfs",
                params={"patient_id": patient.id} if patient else None,
            )
            if job is None:
                st.info(t("msg_job_already_running"))
            else:
                _flash("process_pdfs", "info", t("msg_job_submitted"))
                # Files are being processed — clear the pending-upload hint.
                st.session_state._files_uploaded = False
            st.rerun()

    @st.fragment
    def _op_refresh_risks():
        _show_flash("refresh_risks")
        if st.button(t("btn_refresh_risks"), type="primary", width='stretch'):
            job = submit_job("refresh_risks", "job_kind_refresh_risks")
            if job is None:
                st.info(t("msg_job_already_running"))
            else:
                _flash("refresh_risks", "info", t("msg_job_submitted"))
            st.rerun()

    @st.fragment
    def _op_regenerate_summary():
        _show_flash("regen_summary")
        if st.button(t("btn_regenerate_summary"), key="reports_regenerate_summary", width='stretch', type="primary"):
            patient = st.session_state.get("_selected_patient")
            if not patient:
                st.warning(t("error_select_patient"))
                return
            patient_id = patient.id
            job = submit_job(
                "regen_summary", "job_kind_regen_summary",
                params={"patient_id": patient_id},
            )
            if job is None:
                st.info(t("msg_job_already_running"))
            else:
                _flash("regen_summary", "info", t("msg_job_submitted"))
            st.rerun()

    @st.fragment
    def _op_bulk_descriptions():
        _show_flash("bulk_desc")
        if st.button(t("btn_generate_all_descriptions"), type="primary", key="btn_bulk_gen_desc", width='stretch'):
            patient = st.session_state.get("_selected_patient")
            if not patient:
                st.warning(t("error_select_patient"))
                return
            patient_id = patient.id
            job = submit_job(
                "bulk_descriptions", "job_kind_bulk_descriptions",
                params={"patient_id": patient_id},
            )
            if job is None:
                st.info(t("msg_job_already_running"))
            else:
                _flash("bulk_desc", "info", t("msg_job_submitted"))
            st.rerun()

    @st.fragment
    def _op_bulk_interpretations():
        _show_flash("bulk_interp")
        if st.button(t("btn_generate_all_interpretations"), type="primary", key="btn_bulk_gen_interp", width='stretch'):
            patient = st.session_state.get("_selected_patient")
            if not patient:
                st.warning(t("error_select_patient"))
                return
            patient_id = patient.id
            job = submit_job(
                "bulk_interpretations", "job_kind_bulk_interpretations",
                params={"patient_id": patient_id},
            )
            if job is None:
                st.info(t("msg_job_already_running"))
            else:
                _flash("bulk_interp", "info", t("msg_job_submitted"))
            st.rerun()

    @st.fragment
    def _op_reprocess_pdfs():
        """Re-extract existing reports with the current LLM + registry.

        Destructive for the selected scope (report data is replaced), so it
        goes through a confirmation dialog that shows the blast radius:
        report count, manually disabled values that will be restored, and
        reports whose PDF file is no longer on disk.
        """
        _show_flash("reprocess_pdfs")
        if st.button(t("btn_reprocess_pdfs"), type="primary", key="btn_reprocess_pdfs_action", width='stretch'):
            patient = st.session_state.get("_selected_patient")
            patient_id = patient.id if patient else None
            with get_session() as op_session:
                rq = op_session.query(Report)
                if patient_id is not None:
                    rq = rq.filter(Report.patient_id == patient_id)
                filenames = [r.filename for r in rq.all()]
                dq = (
                    op_session.query(DataPoint)
                    .join(Report, DataPoint.report_id == Report.id)
                    .filter(DataPoint.is_disabled == True)  # noqa: E712
                )
                if patient_id is not None:
                    dq = dq.filter(Report.patient_id == patient_id)
                disabled_count = dq.count()
            missing = [fn for fn in filenames if not (Path(config.REPORTS_DIR) / fn).exists()]

            @st.dialog(t("dialog_reprocess_title"), width="medium")
            def _confirm_reprocess():
                st.markdown(t("dialog_reprocess_text", n=len(filenames)))
                if disabled_count:
                    st.warning(t("dialog_reprocess_disabled_note", n=disabled_count))
                if missing:
                    st.warning(t("dialog_reprocess_missing_note", n=len(missing)))
                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    if st.button(t("btn_cancel"), key="dlg_reprocess_cancel", width='stretch'):
                        st.rerun()
                with col_c2:
                    if st.button(t("btn_confirm_reprocess"), type="primary", key="dlg_reprocess_confirm", width='stretch'):
                        job = submit_job(
                            "reprocess_pdfs", "job_kind_reprocess_pdfs",
                            params={"patient_id": patient_id} if patient_id is not None else None,
                        )
                        if job is None:
                            st.session_state._flash_reprocess_pdfs = [
                                ("info", t("msg_job_already_running"))]
                        else:
                            _flash("reprocess_pdfs", "info", t("msg_job_submitted"))
                        st.rerun()

            _confirm_reprocess()

    # Button layout: 3-across row, 2-across bulk row.
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        _op_process_pdfs()
    with col_btn2:
        _op_refresh_risks()
    with col_btn3:
        _op_regenerate_summary()

    # Biomarker Info bulk buttons row
    col_bulk1, col_bulk2 = st.columns(2)
    with col_bulk1:
        _op_bulk_descriptions()
    with col_bulk2:
        _op_bulk_interpretations()

    # Re-extract existing reports (destructive → confirmation dialog)
    _op_reprocess_pdfs()

    # Report list with status and biomarker counts
    st.divider()
    st.markdown(f"**{len(reports)} {t('col_report_filename').lower()}**")
    st.caption(t("caption_stats", n=len(reports), m=len(dps)))

    if not reports:
        st.info(t("info_no_reports"))
    else:
        # Build report data with status and biomarker count.
        # One grouped query for all per-report counts (was N+1: one query per
        # report, twice — once here and again for the delete dialog below).
        dp_counts = dict(
            session.query(DataPoint.report_id, func.count(DataPoint.id))
            .group_by(DataPoint.report_id)
            .all()
        )
        report_data = []
        for r in reports:
            dp_count = dp_counts.get(r.id, 0)
            has_data = dp_count > 0
            patient_name = r.patient.name if r.patient else "?"
            report_data.append({
                "filename": r.filename,
                "report_date": r.report_date.strftime("%d.%m.%Y") if r.report_date else "?",
                "patient": patient_name,
                "status": t("status_processed") if has_data else t("status_not_processed"),
                "biomarker_count": dp_count,
                "report_id": r.id,
            })

        df_reports = pd.DataFrame(report_data)
        st.dataframe(
            df_reports.drop(columns=["report_id"], errors="ignore"),
            column_config={
                "filename": st.column_config.TextColumn(t("col_report_filename")),
                "report_date": st.column_config.TextColumn(t("col_report_date")),
                "patient": st.column_config.TextColumn(t("col_patient")),
                "status": st.column_config.TextColumn(t("col_report_status")),
                "biomarker_count": st.column_config.NumberColumn(
                    t("col_biomarker_count"),
                    format="%d",
                ),
            },
            hide_index=True,
            width='stretch',
        )

        # Delete report section
        st.divider()
        st.markdown(f"🗑️ **{t('btn_delete_report')}**")
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            delete_options = {r.filename: r.id for r in reports}
            selected_filename = st.selectbox(
                t("select_report"),
                options=list(delete_options.keys()),
                key="_delete_report_select",
            )

        # dp_counts (per-report data-point counts) is computed once above and
        # reused here for the confirmation text.

        @st.dialog(t("dialog_confirm_delete_report_title"), width="medium")
        def _confirm_delete_report():
            st.markdown(
                t(
                    "dialog_confirm_delete_report_text",
                    filename=selected_filename,
                    n=dp_counts.get(delete_options.get(selected_filename, -1), 0),
                )
            )
            col_c1, col_c2 = st.columns([1, 1])
            with col_c1:
                if st.button(t("btn_cancel"), key="dlg_del_report_cancel", width='stretch'):
                    st.rerun()
            with col_c2:
                if st.button(t("btn_confirm_delete"), type="secondary", key="dlg_del_report_confirm", width='stretch'):
                    # The dialog re-runs independently of main(), so it owns its
                    # own DB session (same pattern as the long-op fragments).
                    with get_session() as op_session:
                        report = op_session.query(Report).filter(Report.id == delete_options[selected_filename]).first()
                        if report:
                            op_session.query(DataPoint).filter(DataPoint.report_id == report.id).delete()
                            op_session.query(RiskFlag).filter(RiskFlag.report_id == report.id).delete()
                            op_session.delete(report)
                            op_session.commit()
                            st.session_state._flash_delete_report = t("msg_report_deleted", filename=report.filename)
                    st.rerun()

        with col_del2:
            if st.button(t("btn_delete_report"), type="secondary", width='stretch', key="btn_delete_report_action"):
                _confirm_delete_report()

        # Confirmation toast: rendered on the run AFTER the dialog's st.rerun(),
        # at the bottom of the page (same placement as the old inline success).
        # No _rerun_requested needed — the dialog's app-scope st.rerun() already
        # reloaded all data, so this run's report list is fresh.
        if _deleted_msg := st.session_state.pop("_flash_delete_report", None):
            st.success(_deleted_msg)
