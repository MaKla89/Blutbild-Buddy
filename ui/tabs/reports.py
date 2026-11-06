"""Reports management tab.

Renders the Reports tab: PDF upload + processing, re-extraction, and the
report list with delete — pure document management since the UX rework moved
the other long-running operations to the tabs that show their results (risk
refresh → Risiken, summary regeneration → Übersicht, bulk description/
interpretation generation → Trends). Reference-range backfill lives in the
Datenqualität tab.

Since WP6 all long-running operations are submitted as background jobs
(llm_jobs.submit_job) — the buttons no longer block the UI, and progress is
tracked in the "LLM-Jobs" tab.

**Everything here is scoped to the selected patient.** The tab receives the
patient explicitly, and so do all job submissions. Before that, uploads landed
in one shared folder with no owner, the report list showed every patient's
documents (with a "patient" column to tell them apart), and "Alle PDFs
verarbeiten" extracted whatever was in the folder and assigned each report to
whichever name the LLM read off the page — so one patient's click could process
another patient's upload and file it under the wrong person. Now an upload is
bound to the uploading patient in ``pending_uploads``, the list is exactly that
patient's reports plus their own pending files, and extraction assigns to the
patient who uploaded it.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import func

import config
from database import (
    Report, DataPoint, RiskFlag, ReportArtifact, get_session,
    list_pending_uploads,
)
from llm_jobs import submit_job
from translations import t, _get_lang
from ui.helpers import (
    _handle_file_upload,
    _active_job_for, _flash, _job_button, _render_job_state, _show_flash,
    data_point_count,
)


def render_reports_tab(session, patients, patient, gen_summary):
    """Render the Reports management tab for one patient.

    ``patient`` is the selected patient — the scope of the upload queue, the
    report list, the delete list and every job this tab submits. ``None`` (no
    patient exists yet) renders only the hint to create one, because an upload
    without an owner is exactly the leak this tab used to have.
    """
    st.subheader(t("subheader_reports_manage"))
    st.caption(t("caption_reports_manage"))

    if patient is None:
        st.info(t("info_create_patient_first"))
        return

    # File upload section. The uploader is bound to this patient (see
    # _handle_file_upload) — that binding is what scopes both this tab's pending
    # list and the extraction that follows.
    st.markdown(f"**{t('btn_upload_and_process')}**")
    if st.session_state.pop("_upload_needs_patient", False):
        st.warning(t("warning_upload_needs_patient"))
    uploaded_files = st.file_uploader(
        t("label_upload_pdf"),
        type=["pdf"],
        accept_multiple_files=True,
        key="_uploaded_pdfs",
        on_change=lambda: _handle_file_upload(patient.id),
    )

    # Pending-upload indicator: files are saved to disk on upload but only
    # processed when the user clicks "Process All PDFs" — make that explicit.
    if st.session_state.get("_files_uploaded"):
        queued = st.session_state.pop("_pending_queued_count", 0)
        st.info(t("info_files_saved") + (f" ({queued})" if queued else ""))

    # --- Long-running operations (WP6: background jobs) ----------------------
    # Each operation submits a background job (llm_jobs.py) that runs in a
    # daemon worker thread — a page refresh or close no longer loses progress.
    # The "LLM-Jobs" tab polls the llm_jobs table and shows live status; when a
    # data-affecting job finishes, it triggers a full app rerun so all tabs see
    # the new data. submit_job() returns None when a job of the same kind is
    # already pending/running (kind lock), which we surface as an info message.
    #
    # All of these go through ui.helpers._job_button(), which renders the
    # pending/running job as a disabled "⏳ … — running" button + progress bar,
    # toasts on submit and short-circuits the run — the same affordances the
    # Overview summary button has. No @st.fragment needed: the helper's
    # app-scope st.rerun() already replaces the fragment's redraw. The buttons
    # are laid out in the column rows further down.

    @st.fragment
    def _op_reprocess_pdfs():
        """Re-extract THIS patient's existing reports with the current LLM + registry.

        Destructive for the selected scope (report data is replaced), so it
        goes through a confirmation dialog that shows the blast radius:
        report count, manually disabled values that will be restored, and
        reports whose PDF file is no longer on disk.
        """
        _show_flash("reprocess_pdfs")
        # This operation goes through a confirmation dialog, so it cannot use
        # _job_button() directly — the running state is rendered here instead so
        # the button still reflects an in-flight reprocess job. The fragment may
        # re-run on its own (main's session is closed by then), so this check
        # opens its own session like the blast-radius query below.
        with get_session() as _chk_session:
            _reprocess_job = _active_job_for(_chk_session, "reprocess_pdfs",
                                             patient_id=patient.id)
        if _reprocess_job is not None:
            _render_job_state("btn_reprocess_pdfs", _reprocess_job)
        elif st.button(t("btn_reprocess_pdfs"), type="primary",
                       key="btn_reprocess_pdfs_action", width='stretch'):
            # Scoped to this patient — the blast radius the dialog shows and the
            # scope the job re-extracts are the same set of reports.
            patient_id = patient.id
            with get_session() as op_session:
                rq = op_session.query(Report).filter(Report.patient_id == patient_id)
                filenames = [r.filename for r in rq.all()]
                disabled_count = (
                    op_session.query(DataPoint)
                    .join(Report, DataPoint.report_id == Report.id)
                    .filter(DataPoint.is_disabled == True,  # noqa: E712
                            Report.patient_id == patient_id)
                    .count()
                )
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
                            params={"patient_id": patient_id},
                        )
                        if job is None:
                            st.session_state._flash_reprocess_pdfs = [
                                ("info", t("msg_job_already_running"))]
                        else:
                            _flash("reprocess_pdfs", "info", t("msg_job_submitted"))
                        st.rerun()

            _confirm_reprocess()

    # Process uploads. The other long-running operations moved to the tabs that
    # show their results in the UX rework — risk refresh lives with the flags
    # (Risiken), summary regeneration with the summary (Übersicht), bulk
    # description/interpretation generation with the biomarker detail (Trends) —
    # so this tab is pure document management: upload, process, re-extract, delete.
    _show_flash("process_pdfs")
    _job_button(
        session,
        label_key="btn_process_all", kind="process_pdfs",
        job_label_key="job_kind_process_pdfs", key="btn_process_all",
        params={"patient_id": patient.id},
        flash_key="process_pdfs",
        # Files are being processed — clear the pending-upload hint. Must be
        # a callback: the helper's st.rerun() aborts this run, so code after
        # the call never executes.
        on_submit=lambda: st.session_state.__setitem__("_files_uploaded", False),
    )

    # Re-extract existing reports (destructive → confirmation dialog)
    _op_reprocess_pdfs()

    # Report list with status and biomarker counts.
    # The list is the union of:
    #   (a) this patient's reports in the DB — their PDFs may have been removed
    #       from the input folder after processing, and
    #   (b) this patient's own pending uploads (``pending_uploads``) that have no
    #       DB row yet.
    # (b) used to be "every PDF in the shared input folder", which listed other
    # patients' uploads here. The "file present" column shows whether the
    # original PDF is still on disk. Files starting with "_" are skipped,
    # matching process_new_pdfs().
    st.divider()
    pdf_dir = Path(config.REPORTS_DIR)
    db_reports = (
        session.query(Report).filter(Report.patient_id == patient.id)
        .order_by(Report.report_date).all()
    )
    db_filenames = {r.filename for r in db_reports}
    unprocessed = [
        u.filename for u in list_pending_uploads(session, patient.id)
        if u.filename not in db_filenames
    ]

    total_rows = len(db_reports) + len(unprocessed)
    _plural = "" if total_rows == 1 else ("e" if _get_lang() == "de" else "s")
    st.markdown(f"**{t('heading_reports_list', n=total_rows, plural=_plural)}**")
    caption = t("caption_stats", n=len(db_reports), m=data_point_count(session, patient.id))
    if unprocessed:
        caption += t("caption_stats_unprocessed", n=len(unprocessed))
    st.caption(caption)

    if not total_rows:
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
        # Artifact counts (qualitative / narrative findings) per report — shown
        # next to the biomarker count so the user sees at a glance which reports
        # yielded written-text findings. Counts ALL artifacts (like the biomarker
        # column counts all data points), disabled included.
        artifact_counts = dict(
            session.query(ReportArtifact.report_id, func.count(ReportArtifact.id))
            .group_by(ReportArtifact.report_id)
            .all()
        )
        report_data = []
        # Unprocessed uploads first — they are waiting for "Alle PDFs verarbeiten".
        for fn in unprocessed:
            report_data.append({
                "filename": fn,
                "report_date": "—",
                "status": t("status_not_processed"),
                "file_present": "✅" if (pdf_dir / fn).exists() else "❌",
                "biomarker_count": 0,
                "artifact_count": 0,
                "report_id": None,
            })
        for r in db_reports:
            dp_count = dp_counts.get(r.id, 0)
            art_count = artifact_counts.get(r.id, 0)
            # A DB row only exists after successful extraction, so every DB
            # report counts as processed — including pure narrative reports
            # (MRT, endoscopy, OP notes) that yield zero biomarkers but only
            # qualitative artifacts. Only "processed yet the LLM found nothing
            # at all" is flagged, as that hints at a failed extraction.
            status = (
                t("status_processed")
                if dp_count or art_count
                else t("status_processed_empty")
            )
            report_data.append({
                "filename": r.filename,
                "report_date": r.report_date.strftime("%d.%m.%Y") if r.report_date else "?",
                "status": status,
                "file_present": "✅" if (pdf_dir / r.filename).exists() else "❌",
                "biomarker_count": dp_count,
                "artifact_count": art_count,
                "report_id": r.id,
            })

        df_reports = pd.DataFrame(report_data)
        st.dataframe(
            df_reports.drop(columns=["report_id"], errors="ignore"),
            column_config={
                "filename": st.column_config.TextColumn(t("col_report_filename")),
                "report_date": st.column_config.TextColumn(t("col_report_date")),
                "status": st.column_config.TextColumn(t("col_report_status")),
                "file_present": st.column_config.TextColumn(
                    t("col_file_present"),
                    width="small",
                    help=t("help_col_file_present"),
                ),
                "biomarker_count": st.column_config.NumberColumn(
                    t("col_biomarker_count"),
                    format="%d",
                ),
                "artifact_count": st.column_config.NumberColumn(
                    t("col_artifact_count"),
                    format="%d",
                    help=t("help_col_artifact_count"),
                ),
            },
            hide_index=True,
            width='stretch',
        )

        # Delete report section (only DB reports can be deleted — unprocessed
        # uploads have no DB row; remove those from the input folder directly).
        st.divider()
        st.markdown(f"**{t('btn_delete_report')}**")
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            delete_options = {r.filename: r.id for r in db_reports}
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
                        # Scoped by patient as well as id: a filename is not
                        # unique across patients, and this dialog only ever
                        # offers this patient's own reports.
                        report = op_session.query(Report).filter(
                            Report.id == delete_options[selected_filename],
                            Report.patient_id == patient.id,
                        ).first()
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
