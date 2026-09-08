"""LLM-Jobs status tab (WP6).

Shows recent and running background LLM jobs with live progress. The list is
rendered inside a ``@st.fragment(run_every=2)`` so it polls the ``llm_jobs``
table every 2 seconds WITHOUT re-running the whole app — cheap, and it keeps
working while the user sits on any other tab.

When a data-affecting job (see llm_jobs.DATA_AFFECTING_KINDS) transitions
from pending/running to done/error/interrupted, this tab sets
``st.session_state._rerun_requested = True`` so main() reloads the data and
the new results show up in the other tabs.

The fragment runs on the main script thread (not in a worker), so reading
``st.session_state`` here is safe — the "no session state in workers" rule
applies to the job worker threads only.
"""

import streamlit as st

from database import clear_finished_llm_jobs, get_session, list_llm_jobs
from llm_jobs import DATA_AFFECTING_KINDS, abort_job
from translations import t


def render_llm_jobs_tab(session):
    st.subheader(t("subheader_llm_jobs"))
    st.caption(t("caption_llm_jobs"))

    if st.button(t("btn_clear_finished_jobs"), key="clear_finished_jobs"):
        with get_session() as s:
            n = clear_finished_llm_jobs(s)
        st.toast(t("msg_jobs_cleared", n=n))

    @st.fragment(run_every=2)
    def _jobs_poll():
        with get_session() as s:
            jobs = list_llm_jobs(s, limit=20)

        if not jobs:
            st.info(t("msg_no_jobs_yet"))
            return

        # Transition detection: remember the last-seen status per job id and
        # request a full app rerun when a data-affecting job finishes.
        prev = dict(st.session_state.get("_llm_job_statuses", {}))
        now = {j.id: j.status for j in jobs}
        for jid, old_status in list(prev.items()):
            new_status = now.get(jid)
            if new_status is None:  # job scrolled out of the list — drop it
                del prev[jid]
                continue
            if (new_status != old_status
                    and old_status in ("pending", "running")
                    and new_status in ("done", "error", "interrupted")):
                job = next(j for j in jobs if j.id == jid)
                if job.kind in DATA_AFFECTING_KINDS:
                    st.session_state._rerun_requested = True
        st.session_state._llm_job_statuses = now

        for i, job in enumerate(jobs):
            _render_job_row(job)
            if i < len(jobs) - 1:
                st.divider()

        # Only explain the "interrupted" status when one is actually visible.
        if any(j.status == "interrupted" for j in jobs):
            st.caption(t("job_interrupted_note"))

    _jobs_poll()


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
