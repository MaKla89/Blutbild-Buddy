"""LLM-Jobs status tab (WP6).

Shows recent and running background LLM jobs with live progress. The list is
rendered inside a ``@st.fragment(run_every=2)`` so it polls the ``llm_jobs``
table every 2 seconds WITHOUT re-running the whole app — cheap, and it keeps
working while the user sits on any other tab.

When a data-affecting job (see llm_jobs.DATA_AFFECTING_KINDS) transitions
from pending/running to done/error/interrupted, this fragment triggers a
full app rerun (``st.rerun()``) so main() reloads the data and the new
results show up in the other tabs — the user never has to reload by hand.

Note the deliberate use of ``st.rerun()`` instead of the
``st.session_state._rerun_requested`` flag used elsewhere: a ``run_every``
poll re-executes ONLY this fragment, never the rest of the script, so a
flag set here would sit unconsumed in session state until the user happened
to click something. An app-scoped ``st.rerun()`` raises out of the fragment
and is what actually starts the full rerun.

The fragment runs on the main script thread (not in a worker), so reading
``st.session_state`` here is safe — the "no session state in workers" rule
applies to the job worker threads only.
"""

import streamlit as st

from database import (
    clear_finished_llm_jobs, get_active_llm_jobs, get_session, list_llm_jobs,
)
from llm_jobs import DATA_AFFECTING_KINDS, abort_job
from translations import t
from ui.helpers import _flash

# Kinds whose result should be flashed into the originating tab when the job
# finishes, on top of DATA_AFFECTING_KINDS. test_connection changes no data, so
# it is deliberately not in DATA_AFFECTING_KINDS — but the user still needs the
# verdict in the LLM Settings tab, and the flash only renders after the rerun.
_EXTRA_FLASH_KINDS = {"test_connection"}

# Which _flash() keys receive the result message when a job kind finishes.
# A kind can be started from several tabs, and every tab renders on every
# rerun, so each entry gets the message. Keys match the flash_key each tab
# passes to ui.helpers._job_button().
_FINISH_FLASH_KEYS = {
    "regen_summary": ("overview_regenerate_summary", "regen_summary"),
    "gen_description": ("gen_desc",),
    "gen_interpretation": ("gen_interp",),
    "process_pdfs": ("process_pdfs",),
    "reprocess_pdfs": ("reprocess_pdfs",),
    "refresh_risks": ("refresh_risks",),
    "bulk_descriptions": ("bulk_desc",),
    "bulk_interpretations": ("bulk_interp",),
    "backfill": ("backfill_mapping",),
    "data_quality_check": ("data_quality_check",),
    "test_connection": ("llm_settings",),
}


def render_llm_jobs_tab(session):
    st.subheader(t("subheader_llm_jobs"))
    st.caption(t("caption_llm_jobs"))

    if st.button(t("btn_clear_finished_jobs"), key="clear_finished_jobs"):
        with get_session() as s:
            n = clear_finished_llm_jobs(s)
        st.toast(t("msg_jobs_cleared", n=n))

    # Auto-poll only while a job is actually pending/running. An unconditional
    # run_every=2 kept re-executing this fragment every 2 seconds forever — even
    # hours after the last job finished — and every full app rerun registered
    # another interval in the browser, so an idle screen got progressively more
    # busy. The fragment is therefore only *defined* while something runs: once
    # the last job finishes, the poll triggers an app rerun, that rerun sees no
    # active job, and the polling fragment is never registered again (a leftover
    # browser interval then resolves to an unknown fragment id and is dropped).
    # run_every must be decided at definition time, hence the two branches.
    with get_session() as s:
        jobs_active = bool(get_active_llm_jobs(s))

    if jobs_active:
        @st.fragment(run_every=2)
        def _jobs_poll():
            _render_jobs()

        _jobs_poll()
    else:
        _render_jobs()


def _render_jobs():
    """The job list itself, with transition detection (see render_llm_jobs_tab)."""
    with get_session() as s:
        jobs = list_llm_jobs(s, limit=20)

    if not jobs:
        st.info(t("msg_no_jobs_yet"))
        return

    # Transition detection: remember the last-seen status per job id and
    # refresh the app when a data-affecting job finishes.
    prev = dict(st.session_state.get("_llm_job_statuses", {}))
    now = {j.id: j.status for j in jobs}
    just_finished = []
    for jid, old_status in list(prev.items()):
        new_status = now.get(jid)
        if new_status is None:  # job scrolled out of the list — drop it
            del prev[jid]
            continue
        if (new_status != old_status
                and old_status in ("pending", "running")
                and new_status in ("done", "error", "interrupted")):
            job = next(j for j in jobs if j.id == jid)
            if job.kind in DATA_AFFECTING_KINDS or job.kind in _EXTRA_FLASH_KINDS:
                just_finished.append(job)
    # Persist the new statuses BEFORE rerunning: otherwise the rerun's
    # first poll would see the same transition again and rerun forever.
    st.session_state._llm_job_statuses = now

    if just_finished:
        # Tell the user the job is over (toasts live in the notification
        # layer, so they survive the rerun below), and queue a result
        # message for the tab that started the job — the rerun discards
        # anything rendered in this run, hence _flash().
        for job in just_finished:
            st.toast(f"{job.label} — {t(f'job_status_{job.status}')}")
            if job.status == "done":
                # info, not success: the worker's result text already
                # carries its own nuance (partial failures included) and the
                # toast above shows the ✅/❌ status.
                for flash_key in _FINISH_FLASH_KEYS.get(job.kind, ()):
                    _flash(flash_key, "info", job.result_text or job.label)
            elif job.kind in _EXTRA_FLASH_KINDS:
                # A failed connection test must be visible where the user
                # clicked, not only in the LLM-Jobs tab.
                for flash_key in _FINISH_FLASH_KEYS.get(job.kind, ()):
                    _flash(flash_key, "error", job.error or job.label)
        st.rerun()

    for i, job in enumerate(jobs):
        _render_job_row(job)
        if i < len(jobs) - 1:
            st.divider()

    # Only explain the "interrupted" status when one is actually visible.
    if any(j.status == "interrupted" for j in jobs):
        st.caption(t("job_interrupted_note"))


def _render_job_row(job):
    """Render one job as a compact row: label + status, progress bar while
    running (with an abort button), result/error text once finished."""
    c_label, c_status = st.columns([3, 1])
    c_label.markdown(f"**{job.label}**")
    if job.status in ("pending", "running"):
        with c_status:
            st.markdown(t(f"job_status_{job.status}"))
            if st.button(t("btn_abort_job"), key=f"abort_job_{job.id}"):
                abort_job(job.id)
    else:
        c_status.markdown(t(f"job_status_{job.status}"))

    if job.status in ("pending", "running"):
        pct = (job.progress_current or 0) / 100.0
        st.progress(pct, text=job.progress_message or "")
    elif job.status == "done":
        if job.result_text:
            st.markdown(job.result_text)
    elif job.status == "error":
        st.error(job.error or "?")
    elif job.status == "interrupted" and job.result_text:
        # User-aborted jobs carry a short note in result_text.
        st.caption(job.result_text)

    created = job.created_at.strftime("%d.%m.%Y %H:%M") if job.created_at else "?"
    st.caption(f"{t('col_job_created')}: {created}")
