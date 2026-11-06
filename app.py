import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Load .env file into os.environ before config reads it

import config

from database import (
    init_db, get_session, Patient, Report, DataPoint, RiskFlag,
    get_latest_patient_summary, patient_age_years, collect_report_artifacts,
    mark_stale_llm_jobs_interrupted, get_active_llm_jobs,
)
from llm_client import generate_patient_summary
from llm_jobs import active_job_ids, abort_job
from translations import t, set_lang, _get_lang

# Shared helpers moved to ui/helpers.py (WP1). Re-exported here so existing
# imports and the test suite (which does `import app` and uses these names) keep working.
from ui.helpers import (  # noqa: F401
    _handle_file_upload,
    load_data,
    data_point_count,
    _collect_historical_data,
    _HISTORICAL_MAX_POINTS_PER_BIOMARKER,
    _run_bulk_generation,
    _flash,
    _job_button,
    _show_flash,
)

# Management tabs moved to ui/tabs/ (WP2). Re-exported under their old names so
# existing imports and the test suite keep working.
from ui.tabs.reports import render_reports_tab as _render_reports_tab  # noqa: F401
from ui.tabs.patients import render_patients_tab as _render_patients_tab  # noqa: F401
from ui.tabs.llm_settings import render_llm_settings_tab as _render_llm_settings_tab  # noqa: F401
from ui import queries
from ui.tabs.overview import render_overview_tab  # noqa: F401
from ui.tabs.detail import render_detail_tab  # noqa: F401
from ui.tabs.trends import render_trends_tab  # noqa: F401
from ui.tabs.risks import render_risks_tab  # noqa: F401
from ui.tabs.all_values import render_all_values_tab  # noqa: F401
from ui.tabs.chat import render_chat_tab  # noqa: F401
from ui.tabs.mapping import render_mapping_tab  # noqa: F401
from ui.tabs.llm_jobs import render_llm_jobs_tab  # noqa: F401
from ui.cards import inject_card_css, render_markdown_card

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
    # 200 DPI (not the old 150): a sharper render is the cheapest defence
    # against column mix-ups — thin table rules and adjacent reference/result
    # columns blur together at lower DPI, which is what makes the model read a
    # value from the wrong column in the first place.
    st.session_state.pdf_dpi = 200
if "pdf_batch_size" not in st.session_state:
    st.session_state.pdf_batch_size = 0


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
    artifacts = collect_report_artifacts(session, patient.id, start_date=cutoff)

    # Risk flags — include urgency + LLM explanation so the summary agrees
    # with the Risks tab (same findings, same urgency) instead of re-guessing.
    risk_flags = (
        session.query(RiskFlag.category, RiskFlag.description,
                      RiskFlag.severity, RiskFlag.explanation)
        .join(Report)
        .filter(Report.patient_id == patient.id)
        .all()
    )
    _urgency_by_severity = {3: "critical", 2: "long_term"}
    risk_data = [
        (rf.category, rf.description, _urgency_by_severity.get(rf.severity), rf.explanation)
        for rf in risk_flags
    ]

    profile = {
        "age": patient_age_years(patient),
        "sex": patient.sex,
        "clinical_context": patient.clinical_context,
    }

    lang = _get_lang()
    try:
        return generate_patient_summary(
            llm, patient.name, recent_data, historical_data, risk_data, lang,
            profile=profile, artifacts=artifacts,
        )
    except Exception:
        return None


def _render_health_summary(session, patient, t):
    """Render the patient health summary section in the overview tab.

    Since WP6 the regenerate button submits a background job (llm_jobs)
    instead of blocking the UI — progress is visible in the "LLM-Jobs" tab,
    and the LLM-Jobs poll fragment fires an app-scope st.rerun() when the
    job finishes, so the new summary shows up here without a manual reload.
    """
    st.subheader(t("subheader_health_summary"))

    # Result of the previous click (queued via _flash, shown on this run).
    _show_flash("overview_regenerate_summary")

    # A regen_summary job for THIS patient may already be pending/running — in
    # this process or started before a page reload. _job_button() detects it in
    # the DB and renders a disabled "⏳ running" button + progress bar instead of
    # a live button, so the click registers visibly and cannot double-submit
    # (the _flash info alone is easy to miss above the fold).
    _job_button(
        session,
        label_key="btn_regenerate_summary", kind="regen_summary",
        job_label_key="job_kind_regen_summary", key="overview_regenerate_summary",
        params={"patient_id": patient.id},
        flash_key="overview_regenerate_summary",
    )

    summary = get_latest_patient_summary(session, patient.id)

    if summary:
        st.markdown(render_markdown_card(summary.summary_text, "info", summary=True),
                    unsafe_allow_html=True)
        date_str = summary.created_at.strftime("%d.%m.%Y %H:%M") if summary.created_at else "?"
        st.caption(f"{t('last_updated')}: {date_str}")
    else:
        st.info(t("msg_no_summary_auto"))

    st.caption(t("summary_disclaimer"))


def _patient_report_stats(session):
    """Report count + latest date per patient (cached — see ui.queries)."""
    return queries.report_stats()


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


def _render_job_strip(patient_id):
    """One compact row per active job: label, live progress, abort button.

    Called from the global running-jobs strip (see main()). It re-queries the
    DB on every 2 s fragment tick — by then main's session is closed — so it
    opens its own session. The abort buttons use their OWN keys
    (strip_abort_job_*): the LLM-Jobs tab renders an abort button for the same
    job in the SAME rerun with key abort_job_{id}, and a duplicate element key
    would raise StreamlitDuplicateElementKey.
    """
    with get_session() as s:
        jobs = get_active_llm_jobs(s, patient_id=patient_id)
    if not jobs:
        return
    st.markdown(f"**{t('strip_running_jobs')}**")
    for job in jobs:
        c_label, c_abort = st.columns([3, 1])
        with c_label:
            icon = "⏳" if job.status == "pending" else "🔄"
            st.markdown(f"{icon} **{job.label}**")
            pct = (job.progress_current or 0) / 100.0
            st.progress(pct, text=job.progress_message or "")
        with c_abort:
            if st.button(t("btn_abort_job"), key=f"strip_abort_job_{job.id}"):
                abort_job(job.id)


def main():
    lang = _get_lang()

    # Shared card CSS (health summary box, risk cards) — injected once per
    # run before any tab renders; re-injection on rerun is harmless.
    inject_card_css()

    # Language selector at top — visible label (the collapsed label made the
    # widget look like a decorative dropdown, and "Language" was hardcoded while
    # everything else around it was translated).
    col_title, col_lang = st.columns([3, 1])
    with col_title:
        st.title(t("app_title"))
    with col_lang:
        st.selectbox(
            t("label_language"),
            options=["de", "en"],
            index=0 if lang == "de" else 1,
            key="_lang_selector",
            format_func=lambda x: f"🌐 {x.upper()}",
            on_change=lambda: set_lang(st.session_state._lang_selector),
        )

    session = get_session()
    # The patient picker is the identity switch (there is no auth), so it must
    # list every patient. Everything downstream is scoped to the selection: the
    # Reports tab loads its own report list filtered to the selected patient,
    # which is what keeps that tab's list, stats and delete picker from showing
    # another patient's documents.
    patients, _ = load_data(session)

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

    # --- Running-jobs strip (global, below the patient picker) ---------------
    # A compact live view of every pending/running job for this scope — progress
    # without having to navigate to LLM-Jobs. Informational + abort only: the
    # transition detection that flashes results and triggers the full app rerun
    # stays exclusively in the jobs-tab poll (two fragments racing st.rerun() on
    # the same transition would double-fire). Defined with run_every=2 ONLY
    # while a job is actually pending/running — an unconditional fragment would
    # re-execute every 2 s forever, and every full rerun would register another
    # browser interval (same rule as the jobs-tab poll, ui/tabs/llm_jobs.py).
    #
    # The strip renders into an ALWAYS-created container, even when it stays
    # empty. Streamlit identifies a block by its delta path (its position among
    # its siblings), so a block that appears when a job starts and vanishes when
    # it finishes shifts EVERY block below it by one slot — which remounts the
    # tab bar and resets the selected tab to the first one (the Overview
    # "dashboard"). A constant number of blocks above the tab bar keeps those
    # paths stable. (The tab bars are additionally keyed, see below.)
    _job_strip_slot = st.container()
    strip_patient_id = selected_patient.id if selected_patient is not None else None
    with get_session() as s:
        _strip_job_active = bool(get_active_llm_jobs(s, patient_id=strip_patient_id))

    if _strip_job_active:
        @st.fragment(run_every=2)
        def _job_strip_poll():
            _render_job_strip(strip_patient_id)

        with _job_strip_slot:
            _job_strip_poll()

    # Determine which tabs to show
    has_patient = bool(patients) and selected_patient is not None

    # ONE tab bar in both states — with AND without a patient. The old two-state
    # layout (data tabs only when a patient existed, management tabs only when
    # not) hid the workflow entry points: "Berichte" sat two levels deep, and
    # the no-patient state ended on a dead empty tab that was created but never
    # rendered into. Now the top row is always data tabs + Berichte +
    # Verwaltung{Patienten, Einstellungen, LLM-Jobs, Datenqualität}; without a
    # patient the data tabs (and the per-patient Datenqualität sub-tab) render
    # a one-line placeholder instead of vanishing, so the layout never changes
    # shape when the first patient is created. The inner st.tabs() call must
    # happen INSIDE the parent tab's context so Streamlit renders the sub-tab
    # bar within it, not as a second top-level row.
    data_tabs = [
        t("tab_overview"),
        t("tab_detail"),
        t("tab_trends"),
        t("tab_risks"),
        t("tab_all_values"),
        t("tab_chat"),
    ]

    management_tabs = [
        t("tab_patients"),
        t("tab_llm_settings"),
        t("tab_llm_jobs"),
        t("tab_mapping"),
    ]

    # Both tab bars MUST keep a stable user key (regression: the app jumped back
    # to the Overview "dashboard" whenever a job started or finished).
    # Unkeyed st.tabs() stores the selected tab ONLY in React component state,
    # keyed by the block's position (delta path) in the main container — the
    # backend re-sends default_tab_index 0 on every run. The running-jobs strip
    # above renders a variable number of blocks, so when a job starts (strip
    # appears) or finishes (strip disappears) the tab bar shifts one slot, React
    # remounts it, and the selection resets to the first tab. A user key gives
    # the block a stable id, which makes the frontend persist and restore the
    # active tab label in the widget manager's element state — immune to the shift.
    top_tabs = st.tabs(data_tabs + [t("tab_reports"), t("tab_management")],
                       key="_main_tab_bar")
    (data_tab_overview, data_tab_detail, data_tab_trends,
     data_tab_risks, data_tab_all_values, data_tab_chat,
     reports_tab, mgmt_tab) = top_tabs
    with mgmt_tab:
        mgmt_subtabs = st.tabs(management_tabs, key="_mgmt_tab_bar")

    # --- TAB: Reports (document management — top level since the UX rework) ---
    # The selected patient is passed explicitly rather than read from session
    # state inside the tab: it is the scope of everything this tab lists and of
    # every job it submits.
    with reports_tab:
        _render_reports_tab(session, patients, selected_patient,
                            _generate_summary_for_patient)

    # --- TAB: Patients ---
    with mgmt_subtabs[0]:
        _render_patients_tab(session, patients)

    # --- TAB: Settings (LLM connection + PDF rendering + biomarker registry) ---
    with mgmt_subtabs[1]:
        _render_llm_settings_tab(session, selected_patient)

    # --- TAB: LLM Jobs ---
    with mgmt_subtabs[2]:
        render_llm_jobs_tab(session, selected_patient)

    # First rerun check: runs after the management tabs (Reports, Patients,
    # Settings, LLM Jobs) have rendered. Nested tabs execute all of their
    # content on every rerun — exactly like the old flat tab bar — so a flag
    # set by any of them is picked up here.
    if st.session_state.get("_rerun_requested", False):
        st.session_state._rerun_requested = False
        st.rerun()

    # If no patient selected, stop here — but render the SAME tab bar with a
    # one-line placeholder in each data tab (and the per-patient Datenqualität
    # sub-tab), so creating the first patient never reshuffles the layout.
    if not has_patient:
        for _tab in (data_tab_overview, data_tab_detail, data_tab_trends,
                     data_tab_risks, data_tab_all_values, data_tab_chat):
            with _tab:
                st.info(t("info_no_patient_yet"))
        with mgmt_subtabs[3]:
            st.info(t("info_no_patient_yet"))
        session.close()
        return

    patient = selected_patient
    # Cached read-only queries (ui.queries): these ran on every rerun before,
    # including the 2 s fragment polls. They return plain namedtuples, so the
    # tabs must not treat them as ORM objects.
    patient_reports = queries.patient_reports(patient.id)
    min_date, max_date = queries.patient_date_range(patient.id)

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

    # --- TAB: Chat mit Daten ("chat with your data") ---
    with data_tab_chat:
        render_chat_tab(session, patient, lang, min_date, max_date,
                        f"chat_time_range_{patient.id}")

    # --- TAB: Datenqualität (per-patient measurements review; management sub-tab) ---
    with mgmt_subtabs[3]:
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
