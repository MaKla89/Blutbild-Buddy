import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Load .env file into os.environ before config reads it

import config

from database import (
    init_db, get_session, Patient, Report, DataPoint, RiskFlag,
    get_latest_patient_summary, mark_stale_llm_jobs_interrupted,
)
from sqlalchemy import func
from llm_client import generate_patient_summary
from llm_jobs import submit_job, active_job_ids
from translations import t, set_lang, _get_lang

# Shared helpers moved to ui/helpers.py (WP1). Re-exported here so existing
# imports and the test suite (which does `import app` and uses these names) keep working.
from ui.helpers import (  # noqa: F401
    _handle_file_upload,
    load_data,
    _collect_historical_data,
    _HISTORICAL_MAX_POINTS_PER_BIOMARKER,
    _run_bulk_generation,
    _flash,
    _show_flash,
)

# Management tabs moved to ui/tabs/ (WP2). Re-exported under their old names so
# existing imports and the test suite keep working.
from ui.tabs.reports import render_reports_tab as _render_reports_tab  # noqa: F401
from ui.tabs.patients import render_patients_tab as _render_patients_tab  # noqa: F401
from ui.tabs.llm_settings import render_llm_settings_tab as _render_llm_settings_tab  # noqa: F401
from ui.tabs.overview import render_overview_tab  # noqa: F401
from ui.tabs.detail import render_detail_tab  # noqa: F401
from ui.tabs.trends import render_trends_tab  # noqa: F401
from ui.tabs.risks import render_risks_tab  # noqa: F401
from ui.tabs.all_values import render_all_values_tab  # noqa: F401
from ui.tabs.mapping import render_mapping_tab  # noqa: F401
from ui.tabs.llm_jobs import render_llm_jobs_tab  # noqa: F401
from ui.cards import inject_card_css

st.set_page_config(page_title="Blutbild-Buddy", page_icon="🩸", layout="wide")

init_db()

# Jobs that were pending/running when the app last shut down (or crashed) can
# never finish — their worker threads are gone. Mark them interrupted so the
# LLM-Jobs tab shows a truthful state and the kind locks start clean.
# NOTE: Streamlit re-executes this script on EVERY rerun, so this also runs
# while jobs are legitimately running in this process — those ids are
# excluded (llm_jobs.active_job_ids()). Best-effort only: a DB lock must
# never take down the whole app view.
try:
    with get_session() as _startup_session:
        mark_stale_llm_jobs_interrupted(_startup_session, exclude_ids=active_job_ids())
except Exception:
    pass  # cosmetic state — retry on next rerun

# Initialize LLM config defaults in session state
if "llm_base_url" not in st.session_state:
    st.session_state.llm_base_url = config.get_unsloth_base_url()
if "llm_model" not in st.session_state:
    st.session_state.llm_model = config.get_unsloth_model()
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = config.get_unsloth_api_key()
if "llm_timeout" not in st.session_state:
    st.session_state.llm_timeout = config.get_llm_timeout()
if "pdf_dpi" not in st.session_state:
    st.session_state.pdf_dpi = 150
if "pdf_batch_size" not in st.session_state:
    st.session_state.pdf_batch_size = 1


def _generate_summary_for_patient(session, patient, llm):
    """Gather data and call LLM to generate a patient health summary."""
    from datetime import datetime, timedelta
    # Use naive datetime to match naive Report.report_date storage
    cutoff = datetime.now() - timedelta(days=2 * 365)

    # Recent data points (last 2 years)
    recent_dps = (
        session.query(DataPoint, Report.report_date)
        .join(Report)
        .filter(
            Report.patient_id == patient.id,
            DataPoint.value.isnot(None),
            DataPoint.is_disabled == False,  # noqa: E712
            Report.report_date >= cutoff,
        )
        .order_by(Report.report_date)
        .all()
    )
    recent_data = [
        (rd, dp.canonical_name, dp.value, dp.unit, dp.ref_low, dp.ref_high, dp.is_abnormal)
        for dp, rd in recent_dps
    ]

    # Historical data for chronically abnormal biomarkers (R3-11: bounded to
    # pre-cutoff values of markers that are also abnormal recently — no
    # overlap with the recent window, capped per biomarker).
    historical_data = _collect_historical_data(session, patient.id, cutoff)

    # Risk flags
    risk_flags = (
        session.query(RiskFlag.category, RiskFlag.description)
        .join(Report)
        .filter(Report.patient_id == patient.id)
        .all()
    )
    risk_data = [(rf.category, rf.description) for rf in risk_flags]

    lang = _get_lang()
    try:
        return generate_patient_summary(llm, patient.name, recent_data, historical_data, risk_data, lang)
    except Exception:
        return None


def _render_health_summary(session, patient, t):
    """Render the patient health summary section in the overview tab.

    Since WP6 the regenerate button submits a background job (llm_jobs)
    instead of blocking the UI — progress is visible in the "LLM-Jobs" tab,
    and the LLM-Jobs tab triggers a full rerun when the job finishes so the
    new summary shows up here.
    """
    st.subheader(t("subheader_health_summary"))

    # Result of the previous click (queued via _flash, shown on this run).
    _show_flash("overview_regenerate_summary")

    if st.button(t("btn_regenerate_summary"), key="overview_regenerate_summary", type="primary"):
        job = submit_job(
            "regen_summary", "job_kind_regen_summary",
            params={"patient_id": patient.id},
        )
        if job is None:
            st.info(t("msg_job_already_running"))
        else:
            _flash("overview_regenerate_summary", "info", t("msg_job_submitted"))

    summary = get_latest_patient_summary(session, patient.id)

    if summary:
        st.markdown(f"<div class='bb-card bb-card--summary bb-card--info'>{summary.summary_text}</div>", unsafe_allow_html=True)
        date_str = summary.created_at.strftime("%d.%m.%Y %H:%M") if summary.created_at else "?"
        st.caption(f"{t('last_updated')}: {date_str}")
    else:
        st.info(t("msg_no_summary_auto"))

    st.caption(t("summary_disclaimer"))


def _patient_report_stats(session):
    """One aggregate query for report count + latest date per patient."""
    return {
        pid: (count, latest)
        for pid, count, latest in session.query(
            Report.patient_id, func.count(Report.id), func.max(Report.report_date)
        )
        .filter(Report.patient_id.isnot(None))
        .group_by(Report.patient_id)
        .all()
    }


def _patient_label(stats, patient):
    """Selectbox label for a patient: "Name — N Berichte, letzter DD.MM.YYYY"
    (DE) / "Name — N reports, last DD.MM.YYYY" (EN); patients without reports
    get just their name. Used as the selectbox format_func."""
    count, latest = stats.get(patient.id, (0, None))
    if count and latest is not None:
        # Language-aware plural suffix: "Bericht(e)" in DE, "report(s)" in EN.
        plural = "" if count == 1 else ("e" if _get_lang() == "de" else "s")
        return t(
            "patient_option_with_reports",
            name=patient.name,
            n=count,
            plural=plural,
            date=latest.strftime("%d.%m.%Y"),
        )
    return patient.name


def main():
    lang = _get_lang()

    # Shared card CSS (health summary box, risk cards) — injected once per
    # run before any tab renders; re-injection on rerun is harmless.
    inject_card_css()

    # Language selector at top
    col_title, col_lang = st.columns([3, 1])
    with col_title:
        st.title(t("app_title"))
    with col_lang:
        st.selectbox(
            "Language",
            options=["de", "en"],
            index=0 if lang == "de" else 1,
            label_visibility="collapsed",
            key="_lang_selector",
            format_func=lambda x: f"🌐 {x.upper()}",
            on_change=lambda: set_lang(st.session_state._lang_selector),
        )

    session = get_session()
    patients, reports, dps, flags = load_data(session)

    # Patient selector at top (not in sidebar) — labels carry report count
    # and latest report date so the choice is informative at a glance.
    # The options are the Patient objects themselves (format_func renders the
    # label), so st.session_state._selected_patient holds the actual patient —
    # consumers must never treat it as a name string.
    patient_stats = _patient_report_stats(session)
    selected_patient = st.selectbox(
        t("select_patient"),
        patients,
        format_func=lambda p: _patient_label(patient_stats, p),
        key="_selected_patient",
        disabled=not patients,
    )

    # Determine which tabs to show
    has_patient = bool(patients) and selected_patient is not None
    
    # Build tab labels — result-based tabs first (left), settings-related tabs second (right)
    data_tabs = [
        t("tab_overview"),
        t("tab_detail"),
        t("tab_trends"),
        t("tab_risks"),
        t("tab_all_values"),
    ]
    
    management_tabs = [
        t("tab_reports"),
        t("tab_patients"),
        t("tab_llm_settings"),
        t("tab_llm_jobs"),
        t("tab_mapping"),
    ]

    # With a patient selected the management tabs are grouped under one
    # top-level "Verwaltung" tab (nested st.tabs inside); without a patient
    # they are shown directly as the only tabs. The inner st.tabs() call must
    # happen INSIDE the parent tab's context so Streamlit renders the sub-tab
    # bar within it, not as a second top-level row.
    if has_patient:
        top_tabs = st.tabs(data_tabs + [t("tab_management")])
        (data_tab_overview, data_tab_detail, data_tab_trends,
         data_tab_risks, data_tab_all_values, mgmt_tab) = top_tabs
        with mgmt_tab:
            mgmt_subtabs = st.tabs(management_tabs)
    else:
        mgmt_subtabs = st.tabs(management_tabs)

    # --- TAB: Reports ---
    with mgmt_subtabs[0]:
        _render_reports_tab(session, patients, reports, dps, _generate_summary_for_patient)

    # --- TAB: Patients ---
    with mgmt_subtabs[1]:
        _render_patients_tab(session, patients)

    # --- TAB: LLM Settings ---
    with mgmt_subtabs[2]:
        _render_llm_settings_tab(session)

    # --- TAB: LLM Jobs ---
    with mgmt_subtabs[3]:
        render_llm_jobs_tab(session)

    # First rerun check: runs after the management sub-tabs (Reports, Patients,
    # LLM Settings, LLM Jobs) have rendered. Nested tabs execute all of their
    # content on every rerun — exactly like the old flat tab bar — so a flag
    # set by any of them is picked up here.
    if st.session_state.get("_rerun_requested", False):
        st.session_state._rerun_requested = False
        st.rerun()

    # If no patient selected, stop here
    if not has_patient:
        st.info(t("info_select_patient_first"))
        session.close()
        return

    patient = selected_patient
    patient_reports = (
        session.query(Report)
        .filter(Report.patient_id == patient.id)
        .order_by(Report.report_date)
        .all()
    )

    # Compute time range for chart filters
    date_range_results = (
        session.query(func.min(Report.report_date), func.max(Report.report_date))
        .filter(Report.patient_id == patient.id)
        .filter(Report.report_date.isnot(None))
        .first()
    )
    min_date = date_range_results[0] if date_range_results else None
    max_date = date_range_results[1] if date_range_results else None

    time_range_key = f"time_range_{patient.id}"
    if time_range_key not in st.session_state:
        st.session_state[time_range_key] = [min_date, max_date] if min_date and max_date else None

    # --- TAB: Overview ---
    with data_tab_overview:
        render_overview_tab(
            session, patient, patient_reports, min_date, max_date, time_range_key,
            _render_health_summary,
        )

    # --- TAB: Detailansicht ---
    with data_tab_detail:
        render_detail_tab(session, patient, patient_reports)

    # --- TAB: Trends ---
    with data_tab_trends:
        render_trends_tab(
            session, patient, patient_reports, min_date, max_date, time_range_key, lang,
        )

    # --- TAB: Risks ---
    with data_tab_risks:
        render_risks_tab(session, patient, patient_reports, lang)

    # --- TAB: Alle Werte ---
    with data_tab_all_values:
        render_all_values_tab(session, patient, patient_reports)

    # --- TAB: Biomarker Mapping (management sub-tab) ---
    with mgmt_subtabs[4]:
        render_mapping_tab(session, patient, patient_reports)

    # Second rerun check: patient-specific tabs (Detail, Trends, Risks,
    # All-Values, Mapping) render AFTER the check above, so flags they set
    # (e.g. the per-value disable toggle) would otherwise only take effect
    # on the next unrelated interaction.
    if st.session_state.get("_rerun_requested", False):
        st.session_state._rerun_requested = False
        session.close()
        st.rerun()

    session.close()


if __name__ == "__main__":
    main()
