"""Shared Streamlit helpers for Blutbild-Buddy.

Moved verbatim from app.py (WP1 of the app.py split). These are the cross-tab
utilities: file-upload handling, the "is a background job running?" guard,
data loading, historical-data collection for summaries, and the per-item
error-isolated bulk LLM generation loop.

Imported by app.py (which re-exports them) and by the ui.tabs.* modules.
This module must NOT import from app or ui.tabs to avoid circular imports.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from database import (
    Patient, Report, DataPoint, RiskFlag, upsert_biomarker_info,
    get_active_llm_jobs,
)
from llm_jobs import submit_job
from translations import t


# Maximum allowed file size: 50 MB
_MAX_FILE_SIZE = 50 * 1024 * 1024


def _handle_file_upload():
    """Save uploaded files to reports_input/ and set a flag to trigger processing."""
    uploaded_files = st.session_state.get("_uploaded_pdfs")
    if not uploaded_files:
        return
    reports_dir = Path(config.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in uploaded_files:
        # Sanitize filename to prevent path traversal
        safe_name = Path(f.name).name
        # Validate file extension
        if safe_name.lower().endswith(".pdf") is False:
            st.warning(f"{t('warning_invalid_file')}: {f.name}")
            continue
        # Enforce maximum file size
        if f.size > _MAX_FILE_SIZE:
            st.warning(f"{t('warning_file_too_large')}: {f.name} ({f.size / (1024*1024):.1f} MB > 50 MB)")
            continue
        dest = reports_dir / safe_name
        with open(dest, "wb") as dst:
            dst.write(f.read())
        saved.append(safe_name)
    if saved:
        # Only flag when something actually landed on disk — the Reports tab
        # shows a "click Process All PDFs" hint based on this.
        st.session_state._files_uploaded = True


def _flash(key: str, level: str, text: str):
    """Queue a transient message to be shown on the NEXT completed run.

    Long-op fragments end with st.rerun(), which aborts the current run and
    discards everything rendered in it — including any st.success/st.error
    called just before. Queueing the result here and rendering it via
    _show_flash() on the following run makes operation results visible.
    """
    st.session_state.setdefault(f"_flash_{key}", []).append((level, text))


def _show_flash(key: str):
    """Pop and render all messages queued by _flash(key, ...) since last run."""
    for level, text in st.session_state.pop(f"_flash_{key}", []):
        getattr(st, level)(text)


def _active_job_for(session, kind: str, params: dict | None = None):
    """The pending/running LlmJob matching ``kind`` and (subset-matched) ``params``.

    Subset matching means a patient-scoped button only sees ITS OWN job: a
    regen_summary job for another patient must not disable this patient's
    button. Returns None when nothing is running.
    """
    for j in get_active_llm_jobs(session):
        if j.kind != kind:
            continue
        try:
            jp = json.loads(j.params_json) if j.params_json else {}
        except Exception:
            continue
        if all(jp.get(k) == v for k, v in (params or {}).items()):
            return j
    return None


def _render_job_state(label_key: str, job, button_type: str = "secondary",
                      width: str | None = None, state_key_prefix: str | None = None):
    """Render the "this job is already running" state: disabled labelled button
    plus a progress bar when the worker reports one.

    This is the click feedback for background jobs — without it the button
    looks idle while a worker thread is running, and the user re-clicks.

    ``state_key_prefix`` scopes the widget key to the call site: every tab
    renders on every rerun, so the same running job surfaced from two tabs
    (e.g. regen_summary on Overview AND Reports) would otherwise register the
    same key twice and raise StreamlitDuplicateElementKey.
    """
    kwargs = {} if width is None else {"width": width}
    st.button(
        f"⏳ {t(label_key)} — {t(f'job_status_{job.status}')}",
        key=f"_job_state_{state_key_prefix or 'shared'}_{job.id}",
        type=button_type, disabled=True, **kwargs,
    )
    if job.progress_message:
        st.progress(
            (job.progress_current or 0) / 100.0,
            # The worker's progress message is already a full sentence — don't
            # prefix it again.
            text=job.progress_message,
        )


def _job_button(session, *, label_key: str, kind: str, job_label_key: str,
                key: str, params: dict | None = None, flash_key: str | None = None,
                button_type: str = "primary", width: str | None = "stretch",
                help_key: str | None = None, preconditions: tuple = (),
                on_submit=None):
    """One consistent background-job button for every long-running operation.

    Behaviour (identical across the Reports, Mapping, Trends and Overview tabs):

      * while a matching job is pending/running, a disabled "⏳ label — running"
        button + progress bar is shown instead of the live button, so the click
        registers visibly and cannot double-submit;
      * on click, ``submit_job`` spawns a worker and returns immediately — the
        UI never blocks;
      * a toast (survives the rerun) plus a ``_flash`` message confirm the
        submission, and ``st.rerun()`` short-circuits the rest of the script run
        so the disabled state appears right away instead of after every tab has
        re-rendered;
      * ``preconditions`` is a tuple of (ok: bool, message: str) checks — the
        first failing one is shown and the job is not submitted;
      * ``on_submit`` runs just before the rerun. It must be a callback rather
        than code after the call: ``st.rerun()`` raises, so anything written
        after ``_job_button(...)`` in the caller never executes.

    Returns False when nothing was submitted (job already running, button not
    clicked, or a precondition failed). When a job IS submitted, ``st.rerun()``
    raises and the caller's code after this call never runs — which is why
    post-submit side effects go through ``on_submit``.
    """
    job = _active_job_for(session, kind, params)
    if job is not None:
        # The button's own key doubles as the call-site scope for the disabled
        # state key, keeping both variants of one job kind unique per tab.
        _render_job_state(label_key, job, button_type=button_type, width=width,
                          state_key_prefix=key)
        return False

    kwargs = {} if width is None else {"width": width}
    if help_key:
        kwargs["help"] = t(help_key)
    clicked = st.button(t(label_key), key=key, type=button_type, **kwargs)
    if not clicked:
        return False

    for ok, message in preconditions:
        if not ok:
            st.warning(message)
            return False

    submitted = submit_job(kind, job_label_key, params=params)
    if submitted is None:
        # Lost a race against another tab/session — the kind lock held.
        st.info(t("msg_job_already_running"))
        return False
    st.toast(t("msg_job_submitted"))
    if flash_key:
        _flash(flash_key, "info", t("msg_job_submitted"))
    if on_submit is not None:
        on_submit()
    # st.rerun() raises, so main()'s normal end-of-run session.close() never
    # runs on this path — release the DB connection here instead of leaking it
    # per click.
    session.close()
    st.rerun()


def _time_range_slider(min_date, max_date, value_key, slider_key, label):
    """Robust date-range slider shared by the Overview and Trends tabs.

    Returns a (start_date, end_date) tuple, or (None, None) when no slider can
    be shown. Guards against two StreamlitAPIException cases:
      - min_date == max_date (patient with exactly one report): st.slider
        requires min_value < max_value, so we skip the slider and use the
        single date as both bounds.
      - stale session state from a previous run (e.g. reports deleted since):
        the stored value is clamped into [min_date, max_date].
    """
    if not min_date or not max_date:
        return None, None

    if min_date >= max_date:
        # Single report (or all reports on one day): nothing to slide.
        return min_date, max_date

    value = st.session_state.get(value_key)
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        value = [min_date, max_date]
    else:
        start, end = value
        # Clamp into the current range so deleted reports can't break the slider.
        start = min(max(start, min_date), max_date)
        end = min(max(end, min_date), max_date)
        if start > end:
            start, end = end, start
        value = [start, end]

    time_range = st.slider(
        label,
        min_value=min_date,
        max_value=max_date,
        value=value,
        format="DD.MM.YYYY",
        key=slider_key,
    )
    st.session_state[value_key] = time_range
    return time_range[0], time_range[1]

def load_data(session):
    patients = session.query(Patient).all()
    reports = session.query(Report).order_by(Report.report_date).all()
    dps = session.query(DataPoint).filter(DataPoint.is_disabled == False).all()  # noqa: E712
    flags = session.query(RiskFlag).all()
    return patients, reports, dps, flags


# R3-11: cap on historical data points per biomarker in the summary prompt.
_HISTORICAL_MAX_POINTS_PER_BIOMARKER = 12


def _collect_historical_data(session, patient_id, cutoff):
    """Collect chronically abnormal biomarkers from BEFORE the recent window (R3-11).

    The old query pulled ALL-TIME abnormal values, which both overlapped the
    recent-2-years window (duplicated context) and grew without bound for
    long-term patients. Now:
      - only biomarkers that are ALSO abnormal in the recent window qualify
        ("chronic" — a one-off old abnormality is not a long-term trend),
      - only data points strictly before `cutoff` are included (no overlap),
      - each biomarker is capped at the N most recent historical values.

    Returns the same tuple format as the recent data:
    (report_date, canonical_name, value, unit, ref_low, ref_high, is_abnormal)
    """
    chronic = {
        (cname, unit)
        for cname, unit in session.query(DataPoint.canonical_name, DataPoint.unit)
        .join(Report)
        .filter(
            Report.patient_id == patient_id,
            DataPoint.value.isnot(None),
            DataPoint.is_abnormal == True,  # noqa: E712
            DataPoint.is_disabled == False,  # noqa: E712
            Report.report_date >= cutoff,
        )
        .distinct()
        .all()
    }
    if not chronic:
        return []

    old_dps = (
        session.query(DataPoint, Report.report_date)
        .join(Report)
        .filter(
            Report.patient_id == patient_id,
            DataPoint.value.isnot(None),
            DataPoint.is_abnormal == True,  # noqa: E712
            DataPoint.is_disabled == False,  # noqa: E712
            Report.report_date < cutoff,
        )
        .order_by(Report.report_date.desc())
        .all()
    )

    per_biomarker = {}
    for dp, rd in old_dps:
        key = (dp.canonical_name, dp.unit)
        if key not in chronic:
            continue
        bucket = per_biomarker.setdefault(key, [])
        if len(bucket) < _HISTORICAL_MAX_POINTS_PER_BIOMARKER:
            bucket.append((rd, dp.canonical_name, dp.value, dp.unit, dp.ref_low, dp.ref_high, dp.is_abnormal))

    # Flatten, oldest first, for readability in the prompt.
    result = []
    for key in sorted(per_biomarker):
        result.extend(sorted(per_biomarker[key], key=lambda x: x[0] or datetime.min))
    return result


def _run_bulk_generation(session, p, llm, items, generate_one, upsert_field, label_key, success_key, flash_key=None):
    """Per-biomarker LLM generation loop with per-item error isolation (R3-13).

    Previously the whole loop sat in a single try/except: one hard failure
    (e.g. an LLM timeout on item 7 of 40) aborted all remaining items even
    though earlier per-item upserts had already committed. Now each item is
    isolated — failures are collected and reported, the rest still run.

    Args:
        items: list of (canonical_name, unit) tuples
        generate_one: callable(llm, item) -> str | None
        upsert_field: 'description' or 'interpretation'
        flash_key: when set, the final result message is queued via _flash()
            for the caller's fragment (whose st.rerun() would discard anything
            rendered in this run) instead of being rendered inline.

    Returns:
        (ok_count, errors) where errors is a list of "name: message" strings.
    """
    total = len(items)
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    ok_count = 0
    errors = []
    for idx, item in enumerate(items):
        cname = item[0]
        try:
            status_placeholder.info(f"{t(label_key)} ({idx + 1}/{total}): {cname}")
            result = generate_one(llm, item)
            if result:
                upsert_biomarker_info(session, p.id, cname, **{upsert_field: result})
                ok_count += 1
        except Exception as e:
            errors.append(f"{cname}: {e}")
        progress_bar.progress((idx + 1) / total)
    status_placeholder.empty()
    progress_bar.empty()
    if flash_key is not None:
        # The caller's fragment ends with st.rerun(), which discards everything
        # rendered in this run — queue the result for the next run instead.
        if not errors:
            _flash(flash_key, "success", t(success_key))
        else:
            _flash(flash_key, "warning", t("msg_bulk_partial", ok=ok_count, total=total, failed=len(errors)))
            for err in errors[:5]:
                _flash(flash_key, "info", f"⚠️ {err}")
    elif not errors:
        st.success(t(success_key))
    else:
        st.warning(t("msg_bulk_partial", ok=ok_count, total=total, failed=len(errors)))
        for err in errors[:5]:
            st.caption(f"⚠️ {err}")
    return ok_count, errors
