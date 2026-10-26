"""Background LLM job queue (WP6).

Long-running LLM operations (PDF processing, risk refresh, summary
regeneration, bulk description/interpretation generation, reference-range
backfill, registry validation) run in daemon worker threads instead of
blocking the Streamlit script. This means:

  * a page refresh or browser close does NOT lose progress — the worker
    keeps running and writes its state to the ``llm_jobs`` table,
  * the "LLM-Jobs" tab polls that table (``list_llm_jobs``) to show recent
    and running jobs with live progress,
  * at most one job per kind runs at a time (kind lock), so two clicks can
    never double-submit the same operation.

Worker-thread rules (important):
  * workers must NEVER touch ``st.session_state`` — all settings are
    captured as a snapshot at submit time and stored in
    ``LlmJob.config_json``, applied via ``llm_client.set_job_config()``
    (thread-local) so every nested LLM call resolves the right config,
  * translations use ``t(key, lang=...)`` with the language captured at
    submit time — never the session-state-based default,
  * each worker owns its own DB session (SQLite WAL + busy_timeout make
    concurrent access with the UI safe).
"""

import json
import logging
import threading
from datetime import datetime, timedelta

_log = logging.getLogger(__name__)

import streamlit as st

import config
from openai import APIConnectionError, APITimeoutError
from sqlalchemy import func
from database import (
    Patient, Report, DataPoint, RiskFlag, PatientSummary, BiomarkerInfo,
    get_session, create_llm_job, get_llm_job,
    upsert_biomarker_info, backfill_reference_ranges, create_patient_summary,
    get_registry, patient_age_years, collect_report_artifacts,
    DataQualityFinding, clear_patient_data_quality_findings,
    find_variant_candidates,
    get_chat_conversation, ReportArtifact,
)
from llm_client import (
    _build_client, set_job_config,
    generate_patient_summary, generate_biomarker_description,
    generate_biomarker_interpretation,
    run_biomarker_quality_check, run_duplicate_check,
    is_job_aborted, request_job_abort, set_current_job_id, clear_job_abort,
    chat_with_data,
)
from analyzer import refresh_risk_flags
from extractor import process_new_pdfs
from translations import t, _get_lang

# Kinds that mutate data the main tabs render — when one of these finishes,
# the UI requests a full app rerun so the new data is picked up.
DATA_AFFECTING_KINDS = {
    "process_pdfs", "reprocess_pdfs", "refresh_risks", "regen_summary",
    "bulk_descriptions", "bulk_interpretations", "backfill",
    "data_quality_check", "gen_description", "gen_interpretation",
}

# One worker per kind: prevents duplicate submissions of the same operation.
_active_kinds: set[str] = set()
# Job ids with a live worker thread in THIS process — startup cleanup must
# never mark these interrupted (they are still running).
_active_job_ids: set[int] = set()
_kind_lock = threading.Lock()


def active_job_ids() -> frozenset[int]:
    """Job ids whose worker threads are alive in this process."""
    with _kind_lock:
        return frozenset(_active_job_ids)

# Job id of the currently running worker (set in _run_job) so nested loops
# can update progress without threading it through every call signature.
_current_job_id_var = threading.local()


def _capture_config_snapshot() -> dict:
    """Capture LLM/render settings + language from session state.

    Called on the main script thread at submit time only. The snapshot is
    stored in ``LlmJob.config_json`` and applied in the worker via
    ``set_job_config()`` — workers never read session state themselves.
    """
    return {
        "base_url": st.session_state.get("llm_base_url"),
        "api_key": st.session_state.get("llm_api_key"),
        "model": st.session_state.get("llm_model"),
        "timeout": st.session_state.get("llm_timeout"),
        "dpi": st.session_state.get("pdf_dpi", 150),
        "batch_size": st.session_state.get("pdf_batch_size", 0),
        "lang": _get_lang(),
    }


def submit_job(kind: str, label_key: str, params: dict | None = None, runner=None):
    """Submit a background job; returns the LlmJob row, or None if a job of
    the same kind is already pending/running (UI shows "already running").

    ``runner`` is a test hook: a callable(job_id) that executes the worker
    synchronously instead of spawning a daemon thread.
    """
    with _kind_lock:
        if kind in _active_kinds:
            return None
        _active_kinds.add(kind)

    lang = _get_lang()
    snapshot = _capture_config_snapshot()
    session = get_session()
    try:
        job = create_llm_job(session, kind, t(label_key, lang=lang),
                             config=snapshot, params=params)
    except Exception:
        # Job creation failed (e.g. transient DB lock): release the kind lock
        # so this kind isn't permanently blocked in this process.
        with _kind_lock:
            _active_kinds.discard(kind)
        raise
    finally:
        session.close()

    with _kind_lock:
        _active_job_ids.add(job.id)

    if runner is not None:
        try:
            runner(job.id)
        finally:
            with _kind_lock:
                _active_job_ids.discard(job.id)
    else:
        try:
            threading.Thread(target=_run_job, args=(job.id, kind), daemon=True).start()
        except Exception as e:
            # Thread creation failed (e.g. OS thread limit): without this the
            # kind would stay in _active_kinds forever and the row would sit
            # 'pending' — the kind blocked for the whole process lifetime.
            with _kind_lock:
                _active_kinds.discard(kind)
                _active_job_ids.discard(job.id)
            try:
                s = get_session()
                try:
                    j = get_llm_job(s, job.id)
                    if j is not None and j.status == "pending":
                        j.status = "error"
                        j.error = f"worker could not start: {e}"
                        j.finished_at = datetime.utcnow()
                        s.commit()
                finally:
                    s.close()
            except Exception:
                _log.exception("could not mark job %s errored after thread start failure", job.id)
            raise
    return job


def abort_job(job_id: int) -> bool:
    """Abort a pending/running job (user action from the LLM-Jobs tab).

    Marks the DB row and sets the worker's abort flag. The worker checks the
    flag between steps (and cuts its in-flight LLM call short via a 1s
    timeout), then records the job as 'interrupted'. Returns True if the job
    was still active.
    """
    session = get_session()
    try:
        job = get_llm_job(session, job_id)
        if job is None or job.status not in ("pending", "running"):
            return False
        job.aborted = True
        session.commit()
    finally:
        session.close()
    request_job_abort(job_id)
    return True


def _run_job(job_id: int, kind_hint: str = ""):
    """Worker entry point: run the job's work function and record the outcome.

    ``kind_hint`` is the kind passed by submit_job(): the finally block must
    release the kind lock even when the DB row vanished before the worker
    started (job is None → kind never read from the row). Test runners call
    this as ``runner(job_id)``, so the hint stays optional.
    """
    session = get_session()
    kind = kind_hint
    lang = "de"  # default; overwritten from the config snapshot below
    try:
        job = get_llm_job(session, job_id)
        if job is None:
            return
        kind = job.kind
        cfg = json.loads(job.config_json) if job.config_json else {}
        params = json.loads(job.params_json) if job.params_json else {}
        set_job_config(cfg)
        _current_job_id_var.job_id = job_id
        set_current_job_id(job_id)
        # A SQLite rowid can be reused by a new job after the old row was
        # deleted. The DB row's aborted flag is the source of truth for THIS
        # rowid instance: if it is not aborted, any id lingering in the
        # in-memory set is stale from a deleted prior job → drop it. If it IS
        # aborted (a pending job aborted before its worker started), keep the
        # flag so the worker still sees the abort.
        if not job.aborted:
            clear_job_abort(job_id)
        lang = cfg.get("lang") or "de"

        job.status = "running"
        job.started_at = datetime.utcnow()
        session.commit()

        work_fn = _WORK_FNS.get(kind)
        if work_fn is None:
            raise ValueError(f"Unknown job kind: {kind}")
        level, text = work_fn(session, cfg, lang, params)

        job = get_llm_job(session, job_id)  # re-fetch (progress commits refreshed it)
        if is_job_aborted():
            # User aborted: partial results are kept, but the job is not "done".
            job.status = "interrupted"
            job.result_text = t("job_aborted_note", lang=lang)
        else:
            job.status = "done"
            job.result_text = text
        job.finished_at = datetime.utcnow()
        session.commit()
    except Exception as e:
        try:
            j = get_llm_job(session, job_id)
            if j is not None:
                if is_job_aborted():
                    # The exception is (most likely) the fast-fail timeout of an
                    # aborted LLM call — surface it as interrupted, not error.
                    j.status = "interrupted"
                    j.result_text = t("job_aborted_note", lang=lang)
                else:
                    j.status = "error"
                    j.error = str(e)
                j.finished_at = datetime.utcnow()
                session.commit()
        except Exception:
            # Status write failed (e.g. SQLite lock): the row may stay
            # 'running' until the next startup stale-marking — log it so the
            # failure is at least visible in the server output.
            _log.exception("could not record final status for job %s", job_id)
    finally:
        set_job_config(None)
        _current_job_id_var.job_id = None
        set_current_job_id(None)
        clear_job_abort(job_id)
        session.close()
        with _kind_lock:
            # Unconditional: submit_job() registered the kind BEFORE spawning
            # this worker, so it must be released even when the job row
            # vanished and `kind` was never read from it (kind_hint covers).
            if kind:
                _active_kinds.discard(kind)
            _active_job_ids.discard(job_id)


def _update_progress(session, job_id: int, msg: str, pct: float):
    """Persist progress for the status tab. Commits are cheap (single row)."""
    if job_id is None:
        return
    try:
        j = get_llm_job(session, job_id)
        if j is None or j.status not in ("pending", "running"):
            return
        total = int(round(pct * 100))
        if j.progress_total == total and j.progress_message == msg:
            return
        j.progress_current = total
        j.progress_total = total
        j.progress_message = msg
        session.commit()
    except Exception:
        # Progress is best-effort — never let it kill the job.
        try:
            session.rollback()
        except Exception:
            pass


# ── Job work functions ───────────────────────────────────────────────────────
# Each returns (level, text) for the job's result row. level is one of
# "success" | "warning" | "info" | "error". All LLM calls resolve their config
# from the thread-local snapshot set in _run_job — no session state involved.

def _save_patient_summary(session, patient_id: int, text: str):
    """Store a generated summary against the patient's latest report."""
    latest_report = (
        session.query(Report)
        .filter(Report.patient_id == patient_id)
        .order_by(Report.report_date.desc())
        .first()
    )
    create_patient_summary(session, patient_id, latest_report.id if latest_report else None, text)


def _gather_summary_data(session, patient):
    """Collect recent (2y) + historical data points, risk flags and the patient
    profile for the summary prompt. Same logic as app._generate_summary_for_patient."""
    cutoff = datetime.now() - timedelta(days=2 * 365)
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

    from ui.helpers import _collect_historical_data
    historical_data = _collect_historical_data(session, patient.id, cutoff)

    rf_rows = (
        session.query(RiskFlag.category, RiskFlag.description,
                      RiskFlag.severity, RiskFlag.explanation)
        .join(Report)
        .filter(Report.patient_id == patient.id)
        .all()
    )
    _urgency_by_severity = {3: "critical", 2: "long_term"}
    risk_data = [
        (rf.category, rf.description, _urgency_by_severity.get(rf.severity), rf.explanation)
        for rf in rf_rows
    ]
    profile = {
        "age": patient_age_years(patient),
        "sex": patient.sex,
        "clinical_context": patient.clinical_context,
    }
    # Enabled report artifacts within the summary's recent window (same 2-year
    # cutoff as the recent data points). Disabled artifacts are excluded.
    artifacts = collect_report_artifacts(session, patient.id, start_date=cutoff)
    return recent_data, historical_data, risk_data, profile, artifacts


def _job_process_pdfs(session, cfg, lang, params):
    def progress(msg, pct):
        _update_progress(session, getattr(_current_job_id_var, "job_id", None), msg, pct)

    processed, errors = process_new_pdfs(
        session, progress_callback=progress,
        dpi=cfg.get("dpi"), batch_size=cfg.get("batch_size"),
    )
    if is_job_aborted():
        # Partial results are already committed per PDF; skip the follow-up
        # LLM steps (risk refresh, summary) — _run_job marks it interrupted.
        return "info", t("job_aborted_note", lang=lang)
    n_risks = 0
    if processed:
        try:
            n_risks = refresh_risk_flags(session, _build_client(), lang=lang)
        except Exception as e:
            errors.append(f"{t('error_risk_flags', lang=lang)}: {e}")

    # Auto-generate the health summary for the patient selected at submit time.
    patient_id = params.get("patient_id")
    if processed and patient_id is not None:
        p = session.query(Patient).filter(Patient.id == patient_id).first()
        if p:
            try:
                llm = _build_client()
                recent, historical, risks, profile, artifacts = _gather_summary_data(session, p)
                text = generate_patient_summary(
                    llm, p.name, recent, historical, risks, lang, profile=profile,
                    artifacts=artifacts,
                )
                if text:
                    _save_patient_summary(session, p.id, text)
            except Exception as e:
                errors.append(f"{t('warning_summary_failed', lang=lang)}: {e}")

    if processed and not errors:
        return "success", t("msg_processed", lang=lang, file_list=", ".join(processed), n=n_risks)
    if processed:
        return "warning", t("msg_partial_processing", lang=lang, count=len(processed), errors=len(errors))
    return "info", t("msg_all_processed", lang=lang)


# ── Reprocess (re-extract existing reports with the current LLM/registry) ────

def _snapshot_reprocess(session, patient_id: int | None) -> dict:
    """Snapshot the data a reprocess is about to replace.

    Returns {"reports": {filename: {...}}, "disabled": {(filename, original_name, unit): bool}}.
    The report snapshot feeds the before/after diff AND is the restore payload
    used by ``_restore_reports`` when a re-extraction fails — a failed
    reprocess must never cost the user data they already had, so every field a
    report row carries is captured here (not just the diff-relevant ones).
    The disabled set preserves the user's manual value disabling across the
    cascade delete (matched by filename + original name + unit — the stable
    identity of a measurement).
    """
    q = session.query(Report).filter(Report.file_hash.isnot(None))
    if patient_id is not None:
        q = q.filter(Report.patient_id == patient_id)
    reports = {}
    for r in q.all():
        dps = (
            session.query(DataPoint)
            .filter(DataPoint.report_id == r.id)
            .all()
        )
        dp_key_by_id = {
            dp.id: (dp.original_name or "", dp.unit or "") for dp in dps
        }
        reports[r.filename] = {
            "patient_id": r.patient_id,
            "report_date": r.report_date,
            "panel_type": r.panel_type,
            "file_hash": r.file_hash,
            "raw_text": r.raw_text,
            "dps": [
                (dp.original_name or "", dp.unit or "", dp.canonical_name,
                 dp.value, dp.ref_low, dp.ref_high)
                for dp in dps
            ],
            # Restore payload: full row fidelity, including the abnormality flag
            # and the user's manual disabling.
            "dp_rows": [
                {"original_name": dp.original_name, "canonical_name": dp.canonical_name,
                 "value": dp.value, "unit": dp.unit, "ref_low": dp.ref_low,
                 "ref_high": dp.ref_high, "is_abnormal": dp.is_abnormal,
                 "is_disabled": dp.is_disabled}
                for dp in dps
            ],
            "artifacts": [
                {"category": art.category, "text": art.text,
                 "source_excerpt": art.source_excerpt, "source_page": art.source_page,
                 "is_disabled": art.is_disabled}
                for art in session.query(ReportArtifact)
                .filter(ReportArtifact.report_id == r.id).all()
            ],
            # Risk flags are derived data, but recomputing them needs a working
            # LLM — so a failed reprocess restores them verbatim instead.
            # ``dp_key`` re-links a flag to its data point by identity after the
            # restore (the restored rows have fresh rowids).
            "risk_flags": [
                {"category": rf.category, "severity": rf.severity,
                 "rank": rf.rank, "description": rf.description,
                 "explanation": rf.explanation,
                 "dp_key": dp_key_by_id.get(rf.data_point_id)}
                for rf in session.query(RiskFlag)
                .filter(RiskFlag.report_id == r.id).all()
            ],
            "summaries": [
                {"summary_text": ps.summary_text, "created_at": ps.created_at}
                for ps in session.query(PatientSummary)
                .filter(PatientSummary.report_id == r.id).all()
            ],
        }
    dq = session.query(DataPoint, Report.filename).join(
        Report, DataPoint.report_id == Report.id,
    ).filter(DataPoint.is_disabled == True)  # noqa: E712
    if patient_id is not None:
        dq = dq.filter(Report.patient_id == patient_id)
    disabled = {
        (filename, dp.original_name or "", dp.unit or ""): True
        for dp, filename in dq.all()
    }
    # Disabled artifacts, matched back after re-extraction by
    # (filename, category, text) — the stable identity of a written-text finding.
    aq = session.query(ReportArtifact, Report.filename).join(
        Report, ReportArtifact.report_id == Report.id,
    ).filter(ReportArtifact.is_disabled == True)  # noqa: E712
    if patient_id is not None:
        aq = aq.filter(Report.patient_id == patient_id)
    disabled_artifacts = {
        (filename, art.category or "", art.text or ""): True
        for art, filename in aq.all()
    }
    return {"reports": reports, "disabled": disabled, "disabled_artifacts": disabled_artifacts}


def _restore_reports(session, before: dict, filenames: list[str]) -> int:
    """Re-insert the reports a failed re-extraction had deleted.

    ``_job_reprocess_pdfs`` must delete the old rows before extracting (the
    ``file_hash`` unique constraint rejects a second row for the same PDF), so
    a PDF whose extraction throws leaves a report that simply no longer exists.
    Restoring the snapshot makes that failure non-destructive: the user gets
    their previous data back and the job reports what could not be re-read.

    Returns the number of reports restored.
    """
    if not filenames:
        return 0
    restored = 0
    for fn in filenames:
        info = before["reports"].get(fn)
        if not info:
            continue
        r = Report(
            patient_id=info["patient_id"], filename=fn,
            file_hash=info["file_hash"], report_date=info["report_date"],
            panel_type=info["panel_type"], raw_text=info["raw_text"],
        )
        session.add(r)
        session.flush()
        dp_id_by_key: dict[tuple[str, str | None], int] = {}
        for dp in info["dp_rows"]:
            new_dp = DataPoint(
                report_id=r.id, canonical_name=dp["canonical_name"],
                original_name=dp["original_name"], value=dp["value"],
                unit=dp["unit"], ref_low=dp["ref_low"], ref_high=dp["ref_high"],
                is_abnormal=dp["is_abnormal"], is_disabled=dp["is_disabled"],
            )
            session.add(new_dp)
            session.flush()
            dp_id_by_key[(dp["original_name"] or "", dp["unit"] or "")] = new_dp.id
        for art in info["artifacts"]:
            session.add(ReportArtifact(
                report_id=r.id, category=art["category"], text=art["text"],
                source_excerpt=art["source_excerpt"], source_page=art["source_page"],
                is_disabled=art["is_disabled"],
            ))
        for rf in info["risk_flags"]:
            key = rf["dp_key"]
            session.add(RiskFlag(
                report_id=r.id,
                data_point_id=dp_id_by_key.get(key) if key else None,
                category=rf["category"], severity=rf["severity"], rank=rf["rank"],
                description=rf["description"], explanation=rf["explanation"],
            ))
        for ps in info["summaries"]:
            session.add(PatientSummary(
                patient_id=info["patient_id"], report_id=r.id,
                summary_text=ps["summary_text"], created_at=ps["created_at"],
            ))
        restored += 1
    session.commit()
    return restored


def _diff_reprocess(before: dict, session, filenames: list[str]) -> dict:
    """Compare the pre-reprocess snapshot with the freshly extracted data.

    Data points are matched by (original_name, unit) — the LLM's raw output is
    stable across runs even when canonicalization changes; canonical name and
    reference ranges are what we want to see move.
    """
    after = {}
    for fn in filenames:
        r = session.query(Report).filter(Report.filename == fn).first()
        if r is None:
            continue
        dps = session.query(DataPoint).filter(DataPoint.report_id == r.id).all()
        after[fn] = {
            "patient_id": r.patient_id,
            "report_date": r.report_date,
            "panel_type": r.panel_type,
            "dps": [
                (dp.original_name or "", dp.unit or "", dp.canonical_name,
                 dp.value, dp.ref_low, dp.ref_high)
                for dp in dps
            ],
        }

    name_changes = 0
    ref_changes = 0
    value_changes = 0
    patient_moves = 0
    missing = []
    for fn, old in before["reports"].items():
        new = after.get(fn)
        if new is None:
            missing.append(fn)
            continue
        if old["patient_id"] != new["patient_id"]:
            patient_moves += 1
        old_by_key = {(o[0], o[1]): o for o in old["dps"]}
        new_by_key = {(n[0], n[1]): n for n in new["dps"]}
        for key, o in old_by_key.items():
            n = new_by_key.get(key)
            if n is None:
                continue  # dropped by the new extraction — not a "change"
            if o[2] != n[2]:
                name_changes += 1
            if (o[4], o[5]) != (n[4], n[5]):
                ref_changes += 1
            if o[3] != n[3]:
                value_changes += 1
    return {
        "name_changes": name_changes,
        "ref_changes": ref_changes,
        "value_changes": value_changes,
        "patient_moves": patient_moves,
        "missing": missing,
    }


def _restore_disabled(session, disabled: dict) -> int:
    """Re-apply the user's manual disabling after a reprocess.

    Matches (filename, original_name, unit); when several data points share
    the key (e.g. duplicate rows from one extraction), all are disabled —
    consistent with the pre-reprocess state where the same set was disabled.
    """
    if not disabled:
        return 0
    restored = 0
    for filename, original_name, unit in disabled:
        dps = (
            session.query(DataPoint)
            .join(Report, DataPoint.report_id == Report.id)
            .filter(
                Report.filename == filename,
                DataPoint.original_name == original_name,
                DataPoint.unit == unit,
            )
            .all()
        )
        for dp in dps:
            if not dp.is_disabled:
                dp.is_disabled = True
                restored += 1
    session.commit()
    return restored


def _restore_disabled_artifacts(session, disabled: dict) -> int:
    """Re-apply the user's manual artifact disabling after a reprocess.

    Matches (filename, category, text); a re-extraction that reproduces the
    same finding keeps the user's disabled state. Findings the new extraction
    no longer produces simply disappear (nothing to restore).
    """
    if not disabled:
        return 0
    restored = 0
    for filename, category, text in disabled:
        arts = (
            session.query(ReportArtifact)
            .join(Report, ReportArtifact.report_id == Report.id)
            .filter(
                Report.filename == filename,
                ReportArtifact.category == category,
                ReportArtifact.text == text,
            )
            .all()
        )
        for art in arts:
            if not art.is_disabled:
                art.is_disabled = True
                restored += 1
    session.commit()
    return restored


def _job_reprocess_pdfs(session, cfg, lang, params):
    """Re-extract existing reports with the current LLM + registry.

    Flow: snapshot old data → delete the Report rows whose PDF is on disk
    (cascades to data points and risk flags; patient identity is pinned by
    re-attributing each PDF to its previous patient) → run
    process_new_pdfs(force=True, only_files=targets) → restore any report we
    deleted but did not get back → refresh risk flags → restore manually
    disabled values → diff old vs. new for the result text.

    The delete-then-extract order is forced by the ``file_hash`` unique
    constraint (a second row for the same PDF would be rejected), which makes
    the job destructive by construction — so the snapshot doubles as a restore
    payload and EVERY path that does not produce a replacement row (per-PDF
    extraction error, a crash before the per-PDF loop, an abort, a PDF missing
    from disk) must leave the previous data intact. A failed reprocess may
    waste time, it may never lose data.

    The registry is only READ here — reprocessing consumes improved mappings,
    it never writes them back (that stays an explicit user action).
    """
    from pdf_processor import pdf_page_count
    from pathlib import Path as _Path

    patient_id = params.get("patient_id")
    before = _snapshot_reprocess(session, patient_id)
    if not before["reports"]:
        return "info", t("msg_reprocess_nothing", lang=lang)

    # Pin each PDF to its previous patient so a different LLM answer for the
    # name can't fork a duplicate patient.
    pinned = {fn: info["patient_id"] for fn, info in before["reports"].items()}

    # A report whose PDF file is gone can never be re-extracted, so deleting it
    # would be pure loss with no chance of replacement — keep those rows.
    pdf_dir = _Path(config.REPORTS_DIR)
    missing_files = [fn for fn in before["reports"] if not (pdf_dir / fn).exists()]
    targets = [fn for fn in before["reports"] if (pdf_dir / fn).exists()]

    # Delete old rows and everything that referenced them. Bulk Query.delete()
    # does NOT apply ORM cascades (and SQLite doesn't enforce FKs), so the
    # children must be removed explicitly — otherwise orphaned data points
    # would survive the reprocess and pollute the diff.
    if targets:
        report_ids = [r.id for r in session.query(Report).filter(
            Report.filename.in_(targets)
        ).all()]
        if report_ids:
            from database import PatientSummary, RiskFlag
            session.query(RiskFlag).filter(RiskFlag.report_id.in_(report_ids)).delete()
            session.query(DataPoint).filter(DataPoint.report_id.in_(report_ids)).delete()
            session.query(ReportArtifact).filter(ReportArtifact.report_id.in_(report_ids)).delete()
            session.query(PatientSummary).filter(PatientSummary.report_id.in_(report_ids)).delete()
            session.query(Report).filter(Report.id.in_(report_ids)).delete()
            session.commit()

    # Estimate pages for the progress bar (best-effort — missing files are
    # simply skipped by process_new_pdfs).
    total_pages = 0
    for fn in targets:
        p = pdf_dir / fn
        if p.exists():
            try:
                total_pages += pdf_page_count(p)
            except Exception:
                pass

    def progress(msg, pct):
        _update_progress(session, getattr(_current_job_id_var, "job_id", None), msg, pct)

    processed: list[str] = []
    errors: list[str] = []
    try:
        processed, errors = process_new_pdfs(
            session, progress_callback=progress,
            dpi=cfg.get("dpi"), batch_size=cfg.get("batch_size"),
            force=True, only_files=set(targets),
        )
    except Exception as exc:
        # process_new_pdfs builds the LLM client and the registry BEFORE its
        # per-PDF try/except. A failure there (bad config, client build error)
        # would otherwise propagate with every snapshotted report already
        # deleted — total data loss reported as a mere job error.
        session.rollback()
        errors.append(str(exc))

    # Anything we deleted but did not get back is restored from the snapshot:
    # a per-PDF extraction failure, an abort mid-run, or a crash before the
    # loop must never cost the user reports they already had.
    failed = [fn for fn in targets if fn not in processed]
    restored_reports = _restore_reports(session, before, failed) if failed else 0

    # Re-attribute any report whose LLM-extracted patient name resolved to a
    # different (or new) patient — the PDF's previous attribution wins.
    moved_to: set[int] = set()
    for fn in processed:
        expected = pinned.get(fn)
        if expected is None:
            continue
        r = session.query(Report).filter(Report.filename == fn).first()
        if r is not None and r.patient_id != expected:
            moved_to.add(r.patient_id)
            r.patient_id = expected
    if processed:
        session.commit()

    # A divergent LLM name answer may have created a stray patient row that
    # is now empty again — remove it (and its orphaned summaries/infos) so the
    # patient list stays clean.
    for pid in moved_to:
        p = session.get(Patient, pid)
        if p is not None and not (
            session.query(Report).filter(Report.patient_id == pid).first()
        ):
            session.query(PatientSummary).filter(PatientSummary.patient_id == pid).delete()
            session.query(BiomarkerInfo).filter(BiomarkerInfo.patient_id == pid).delete()
            session.delete(p)  # cascade removes its (now empty) reports
    if moved_to:
        session.commit()

    # Restore the user's manual disabling (matched by filename+name+unit).
    disabled_restored = _restore_disabled(session, before["disabled"])
    artifact_restored = _restore_disabled_artifacts(session, before.get("disabled_artifacts", {}))

    if is_job_aborted():
        # Partial results are kept; the reports we had deleted but not yet
        # re-extracted were put back above, so an abort is non-destructive too.
        note = t("job_aborted_note", lang=lang)
        if restored_reports:
            note += f"\n{t('reprocess_restored', lang=lang, n=restored_reports)}"
        return "info", note

    n_risks = 0
    if processed:
        try:
            n_risks = refresh_risk_flags(session, _build_client(), lang=lang)
        except Exception as e:
            errors.append(f"{t('error_risk_flags', lang=lang)}: {e}")

    # Diff over every snapshotted PDF that we touched (processed OR restored),
    # so a restored report is compared against itself (no changes) instead of
    # being listed as "missing". Reports skipped for a missing PDF file are
    # reported separately below.
    diff = _diff_reprocess(before, session, targets)

    if not processed:
        # Nothing could be re-extracted. The snapshot restore above means the
        # DB is exactly as it was — say so explicitly, because this is the
        # case that previously looked like "the LLM is unreachable" while the
        # real damage (all reports deleted) had already happened.
        # Raise rather than return ("error", …): the level is not part of the
        # job status, so a returned error would still render as a green
        # "finished" row. Raising lands in LlmJob.error, which the LLM-Jobs tab
        # shows in red.
        parts = [t("reprocess_failed_all", lang=lang, n=len(before["reports"]))]
        if restored_reports:
            parts.append(t("reprocess_restored", lang=lang, n=restored_reports))
        for fn in missing_files[:5]:
            parts.append(f"⚠️ {t('reprocess_missing_file', lang=lang, filename=fn)}")
        for err in errors[:5]:
            parts.append(f"⚠️ {err}")
        raise RuntimeError("\n".join(parts))

    parts = [t("msg_reprocess_done", lang=lang, n=len(processed), risks=n_risks)]
    if restored_reports:
        parts.append(t("reprocess_restored", lang=lang, n=restored_reports))
    if diff["name_changes"]:
        parts.append(t("reprocess_diff_names", lang=lang, n=diff["name_changes"]))
    if diff["ref_changes"]:
        parts.append(t("reprocess_diff_refs", lang=lang, n=diff["ref_changes"]))
    if diff["value_changes"]:
        parts.append(t("reprocess_diff_values", lang=lang, n=diff["value_changes"]))
    if not (diff["name_changes"] or diff["ref_changes"] or diff["value_changes"]):
        parts.append(t("reprocess_diff_none", lang=lang))
    if disabled_restored:
        parts.append(t("reprocess_disabled_restored", lang=lang, n=disabled_restored))
    if artifact_restored:
        parts.append(t("reprocess_artifacts_restored", lang=lang, n=artifact_restored))
    if diff["patient_moves"]:
        parts.append(t("reprocess_patient_pinned", lang=lang, n=diff["patient_moves"]))
    for fn in missing_files[:5]:
        parts.append(f"⚠️ {t('reprocess_missing_file', lang=lang, filename=fn)}")
    # The diff also flags the skipped reports as missing — they are already
    # reported above with the explicit "PDF file missing" message.
    for fn in [f for f in diff["missing"] if f not in missing_files][:5]:
        parts.append(f"⚠️ {fn}")
    for err in errors[:5]:
        parts.append(f"⚠️ {err}")

    if processed and not errors and not diff["missing"] and not restored_reports and not missing_files:
        return "success", "\n".join(parts)
    if processed:
        return "warning", "\n".join(parts)
    return "error", "\n".join(parts)


def _job_refresh_risks(session, cfg, lang, params):
    n = refresh_risk_flags(session, _build_client(), lang=lang)
    return "success", t("msg_risks_refreshed_all", lang=lang, n=n)


def _job_chat_answer(session, cfg, lang, params):
    """Answer one chat question in the background (see ui/tabs/chat.py).

    The question, the conversation id and the data context are passed in
    ``params`` (captured at submit time on the UI thread), so the worker never
    needs Streamlit state. The answer goes into ``result_text`` — the chat tab
    adopts it into a ChatMessage row once the job is finished, which is what
    lets an in-flight answer survive a tab switch or a page refresh.

    Deliberately NOT in DATA_AFFECTING_KINDS: the tab polls this job itself, and
    an app-wide rerun would throw the user off whatever tab they navigated to.
    """
    conversation_id = params.get("conversation_id")
    messages = params.get("messages") or []
    data_context = params.get("data_context") or ""
    if not messages:
        return "error", t("error_chat_no_question", lang=lang)
    _update_progress(session, getattr(_current_job_id_var, "job_id", None),
                     t("chat_thinking", lang=lang), 0.3)
    answer = chat_with_data(_build_client(), data_context, messages, lang=lang)
    answer = (answer or "").strip()
    if not answer:
        return "error", t("error_chat_empty_answer", lang=lang)
    # Touch the conversation so it bubbles to the top of the history list even
    # if the user never looked at the tab while the answer was generating.
    conv = get_chat_conversation(session, conversation_id) if conversation_id else None
    if conv is not None:
        conv.updated_at = datetime.utcnow()
        session.commit()
    return "success", answer


def _job_regen_summary(session, cfg, lang, params):
    patient_id = params.get("patient_id")
    p = session.query(Patient).get(patient_id)
    if p is None:
        return "error", t("error_select_patient", lang=lang)
    # Early progress message: the Overview button shows this while the (single,
    # slow) summary LLM call is in flight — without it the job row stays blank.
    _update_progress(session, getattr(_current_job_id_var, "job_id", None),
                     t("msg_generating_summary", lang=lang), 0.1)
    llm = _build_client()
    recent, historical, risks, profile, artifacts = _gather_summary_data(session, p)
    text = generate_patient_summary(
        llm, p.name, recent, historical, risks, lang, profile=profile,
        artifacts=artifacts,
    )
    if not text:
        return "error", t("error_summary_generation", lang=lang)
    _save_patient_summary(session, p.id, text)
    return "success", t("msg_summary_regenerated", lang=lang)


def _patient_biomarkers(session, patient_id: int):
    """Distinct (canonical_name, unit) pairs for a patient's active data points."""
    rows = (
        session.query(DataPoint.canonical_name, DataPoint.unit)
        .join(Report)
        .filter(
            Report.patient_id == patient_id,
            DataPoint.is_disabled == False,  # noqa: E712
        )
        .distinct()
        .order_by(DataPoint.canonical_name)
        .all()
    )
    return [tuple(r) for r in rows]


def _run_bulk(session, patient_id, llm, items, upsert_field, generate_one):
    """Per-item LLM generation loop with per-item error isolation (R3-13).

    Worker-thread version of ui.helpers._run_bulk_generation: progress goes to
    the job row instead of st.progress(), and no session state is touched.
    """
    total = len(items)
    ok = 0
    errors = []
    for idx, item in enumerate(items):
        if is_job_aborted():
            # User aborted: stop before the next LLM call; partial results kept.
            break
        cname = item[0]
        try:
            text = generate_one(llm, item)
            if text:
                upsert_biomarker_info(session, patient_id, cname, **{upsert_field: text})
                ok += 1
        except Exception as e:
            errors.append(f"{cname}: {e}")
        _update_progress(session, getattr(_current_job_id_var, "job_id", None),
                         f"{cname} ({idx + 1}/{total})", (idx + 1) / total)
    return ok, errors


def _bulk_result(ok: int, total: int, errors: list[str], success_key: str, lang: str):
    if not errors:
        return "success", t(success_key, lang=lang)
    level = "warning" if ok else "error"
    text = t("msg_bulk_partial", lang=lang, ok=ok, total=total, failed=len(errors))
    for err in errors[:5]:
        text += f"\n⚠️ {err}"
    return level, text


def _job_bulk_descriptions(session, cfg, lang, params):
    patient_id = params.get("patient_id")
    items = _patient_biomarkers(session, patient_id)
    if not items:
        return "info", t("info_no_biomarker_data", lang=lang)
    llm = _build_client()
    # Active registry (honors accepted proposals) for the per-biomarker
    # unit/ref-range context line — the former full canonical list is gone.
    registry = get_registry(session)
    ok, errors = _run_bulk(
        session, patient_id, llm, items, "description",
        generate_one=lambda llm_, item: generate_biomarker_description(
            llm_, item[0], item[1], lang, registry=registry,
        ),
    )
    return _bulk_result(ok, len(items), errors, "msg_all_descriptions_generated", lang)


def _related_context_lines(session, patient_id: int, cname: str) -> list[str]:
    """Latest values of clinically related markers, for the interpretation prompt.

    config.RELATED_BIOMARKERS lists which markers belong to the same panel;
    a single marker interpreted in isolation cannot be judged correctly
    (low MCV with low ferritin vs. normal ferritin are different pictures).
    Returns [] when the marker has no configured relations or the patient has
    no matching measurements.
    """
    related = config.get_related_biomarkers(cname)
    if not related:
        return []
    rows = (
        session.query(
            DataPoint.canonical_name,
            DataPoint.value,
            DataPoint.unit,
            DataPoint.ref_low,
            DataPoint.ref_high,
            Report.report_date,
        )
        .join(Report, DataPoint.report_id == Report.id)
        .filter(
            Report.patient_id == patient_id,
            func.lower(DataPoint.canonical_name).in_([r.lower() for r in related]),
            DataPoint.value.isnot(None),
            DataPoint.is_disabled == False,  # noqa: E712
        )
        .order_by(Report.report_date.desc())
        .all()
    )
    latest: dict[str, tuple] = {}
    for name, value, unit, ref_low, ref_high, rdate in rows:
        key = name.lower()
        if key not in latest:  # order_by desc → first hit is the latest
            latest[key] = (name, value, unit, ref_low, ref_high, rdate)
    lines = []
    for r in related:
        hit = latest.get(r.lower())
        if hit is None:
            continue
        _name, value, unit, ref_low, ref_high, rdate = hit
        ref_str = ""
        if ref_low is not None and ref_high is not None:
            ref_str = f" (Ref {ref_low}-{ref_high})"
        elif ref_high is not None:
            ref_str = f" (Ref < {ref_high})"
        elif ref_low is not None:
            ref_str = f" (Ref > {ref_low})"
        abnormal = (
            (ref_low is not None and value < ref_low)
            or (ref_high is not None and value > ref_high)
        )
        marker = " [ABNORMAL]" if abnormal else ""
        date_str = rdate.strftime("%d.%m.%Y") if rdate else "?"
        lines.append(f"{r} = {value} {unit or ''}{ref_str}{marker} ({date_str})")
    return lines


def _job_bulk_interpretations(session, cfg, lang, params):
    patient_id = params.get("patient_id")
    items = _patient_biomarkers(session, patient_id)
    if not items:
        return "info", t("info_no_biomarker_data", lang=lang)
    llm = _build_client()

    def gen_one(llm_, item):
        cname, unit = item
        dp_rows = (
            session.query(DataPoint, Report.report_date)
            .join(Report)
            .filter(
                DataPoint.canonical_name == cname,
                DataPoint.unit == unit,
                DataPoint.value.isnot(None),
                DataPoint.is_disabled == False,  # noqa: E712
                Report.patient_id == patient_id,
            )
            .order_by(Report.report_date)
            .all()
        )
        dp_tuples = [(rd, dp.value, dp.ref_low, dp.ref_high) for dp, rd in dp_rows]
        context_lines = _related_context_lines(session, patient_id, cname)
        return generate_biomarker_interpretation(
            llm_, cname, unit, dp_tuples, lang, context_lines=context_lines
        )

    ok, errors = _run_bulk(
        session, patient_id, llm, items, "interpretation",
        generate_one=gen_one,
    )
    return _bulk_result(ok, len(items), errors, "msg_all_interpretations_generated", lang)


def _job_gen_description(session, cfg, lang, params):
    """Single-biomarker explanation (Trends tab button).

    Same LLM call as the bulk job, but scoped to one (name, unit) pair so the
    user gets progress + abort for the one value they are looking at instead of
    a blocking st.spinner() in the fragment.
    """
    patient_id = params.get("patient_id")
    cname = params.get("canonical_name")
    unit = params.get("unit")
    if patient_id is None or not cname:
        return "error", t("error_select_patient", lang=lang)
    _update_progress(session, getattr(_current_job_id_var, "job_id", None),
                     t("msg_generating_description", lang=lang), 0.1)
    llm = _build_client()
    text = generate_biomarker_description(
        llm, cname, unit, lang, registry=get_registry(session),
    )
    if not text:
        return "error", t("error_summary_generation", lang=lang)
    upsert_biomarker_info(session, patient_id, cname, description=text)
    return "success", t("msg_description_generated", lang=lang)


def _job_gen_interpretation(session, cfg, lang, params):
    """Single-biomarker interpretation (Trends tab button).

    The data-point window is part of the params (ISO date strings — params_json
    is plain JSON) so the worker reproduces exactly the range the user had
    selected on the Trends tab when they clicked.
    """
    patient_id = params.get("patient_id")
    cname = params.get("canonical_name")
    unit = params.get("unit")
    if patient_id is None or not cname:
        return "error", t("error_select_patient", lang=lang)
    _update_progress(session, getattr(_current_job_id_var, "job_id", None),
                     t("msg_generating_interpretation", lang=lang), 0.1)

    q = (
        session.query(DataPoint, Report.report_date)
        .join(Report)
        .filter(
            DataPoint.canonical_name == cname,
            DataPoint.value.isnot(None),
            DataPoint.unit == unit,
            DataPoint.is_disabled == False,  # noqa: E712
            Report.patient_id == patient_id,
        )
    )
    start = _parse_param_date(params.get("start_date"))
    end = _parse_param_date(params.get("end_date"))
    if start is not None:
        q = q.filter(Report.report_date >= start)
    if end is not None:
        q = q.filter(Report.report_date <= end)
    rows = q.order_by(Report.report_date).all()
    dp_tuples = [(rd, dp.value, dp.ref_low, dp.ref_high) for dp, rd in rows]

    llm = _build_client()
    text = generate_biomarker_interpretation(
        llm, cname, unit, dp_tuples, lang,
        context_lines=_related_context_lines(session, patient_id, cname),
    )
    if not text:
        return "error", t("error_summary_generation", lang=lang)
    upsert_biomarker_info(session, patient_id, cname, interpretation=text)
    return "success", t("msg_interpretation_generated", lang=lang)


def _parse_param_date(value):
    """params_json holds plain JSON, so dates arrive as ISO strings."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# Upper bound for the connection-test ping. The configured LLM timeout can be
# minutes (a big report extraction legitimately needs that), but a 1-token ping
# should answer in seconds — a cold local model load is the only slow case.
# Without a cap, an unreachable base URL leaves the test "running" for the full
# configured timeout, which is exactly the unresponsiveness it is meant to
# diagnose.
_TEST_CONNECTION_MAX_TIMEOUT = 120


def _job_test_connection(session, cfg, lang, params):
    """Ping the configured LLM endpoint (LLM Settings → "Test Connection").

    Runs as a job rather than inline in the render path: an inline call blocks
    the whole Streamlit script — the app is frozen and unresponsive for the full
    configured timeout (up to 2000s) when the endpoint is unreachable.
    """
    base_url = cfg.get("base_url") or config.get_unsloth_base_url()
    model = cfg.get("model") or config.get_unsloth_model()
    timeout = int(cfg.get("timeout") or config.get_llm_timeout())
    timeout = max(1, min(timeout, _TEST_CONNECTION_MAX_TIMEOUT))

    llm = _build_client(cfg)
    try:
        llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "OK"}],
            max_tokens=1,
            timeout=timeout,
        )
    except (APIConnectionError, APITimeoutError) as e:
        # Raise so the job row lands in status "error" and the LLM-Jobs tab
        # renders the diagnostic in red, instead of a green "done" row.
        raise RuntimeError(
            f"{t('msg_llm_disconnected', lang=lang)} {e} — "
            f"{t('error_llm_connection_hint', lang=lang, base_url=base_url)}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"{t('msg_llm_error', lang=lang)} {type(e).__name__}: {e}") from e
    note = ""
    if timeout < int(cfg.get("timeout") or config.get_llm_timeout()):
        note = f"\n{t('note_test_timeout_capped', lang=lang, timeout=timeout)}"
    return "success", f"{t('msg_llm_connected', lang=lang)}{note}"


def _job_backfill(session, cfg, lang, params):
    result = backfill_reference_ranges(session)
    if result["updated"] > 0:
        return "success", t("msg_backfill_complete", lang=lang, n=result["updated"])
    if (result["skipped_unknown"] or result["skipped_no_range"]
            or result["skipped_unconvertible"]):
        return "info", t(
            "msg_backfill_skipped", lang=lang,
            unknown=result["skipped_unknown"],
            unconvertible=result["skipped_unconvertible"],
            no_range=result["skipped_no_range"],
        )
    return "info", t("msg_no_backfill_needed", lang=lang)


def _job_data_quality_check(session, cfg, lang, params):
    """Run the data-quality check for one patient.

    Phase 1: one LLM call per distinct biomarker (carrying all of its
    measurements), so within-biomarker context — e.g. unit consistency across
    visits — is preserved. Phase 2: a single duplicate check over all of the
    patient's biomarkers.

    All LLM calls happen FIRST (no DB write lock held across them). Findings
    are then persisted to the data_quality_findings table — replacing any
    findings from a previous run for this patient so stale results don't linger.
    """
    patient_id = params.get("patient_id")
    if patient_id is None:
        raise ValueError("data_quality_check requires a patient_id")

    # Build the ordered measurement list. Disabled values are included so the
    # user can review them.
    dps = (
        session.query(DataPoint, Report.report_date)
        .join(Report, DataPoint.report_id == Report.id)
        .filter(Report.patient_id == patient_id)
        .order_by(Report.report_date.asc(), DataPoint.id.asc())
        .all()
    )
    dp_dicts = [
        {
            "id": dp.id,
            "name": dp.canonical_name,
            "value": dp.value,
            "unit": dp.unit,
            "ref_low": dp.ref_low,
            "ref_high": dp.ref_high,
            "report_date": rd.strftime("%Y-%m-%d") if rd else "",
        }
        for dp, rd in dps
    ]

    # Group by canonical name (order-preserving) — one LLM call per biomarker.
    groups: dict[str, list[dict]] = {}
    for d in dp_dicts:
        groups.setdefault(d["name"], []).append(d)

    llm = _build_client()
    registry = get_registry(session)
    findings: list[dict] = []
    errors: list[str] = []
    total = len(groups)

    # Phase 1: per-biomarker quality check (sequential, abort-aware).
    for idx, (name, group) in enumerate(groups.items()):
        if is_job_aborted():
            break
        try:
            findings.extend(run_biomarker_quality_check(llm, registry, name, group, lang)["findings"])
        except Exception as e:
            errors.append(f"{name}: {e}")
        _update_progress(session, getattr(_current_job_id_var, "job_id", None),
                         f"{name} ({idx + 1}/{total})", (idx + 1) / total)

    # Phase 2: duplicate check over all of the patient's biomarkers.
    #
    # Typo detection used to be LLM-only: the model had to spot that
    # 'Trigylceride' and 'Tryglyceride' are the same marker as 'Triglyceride'
    # while reading ~94 canonical names, and small quantised models routinely
    # return {"findings": []} — the split then survives every check. So the
    # pairs are found deterministically first (find_variant_candidates, pure
    # string similarity) and the LLM is asked to CONFIRM that short list. Any
    # candidate the LLM did not confirm is still emitted as a merge finding, so
    # a silent/failed duplicate check can no longer hide a duplicate.
    if not is_job_aborted():
        try:
            entries = [(d["name"], d["unit"]) for d in dp_dicts]
            candidates = find_variant_candidates(session, patient_id, registry)
            dup_findings = run_duplicate_check(
                llm, entries, registry, lang, candidates=candidates)["findings"]
        except Exception as e:
            dup_findings = []
            errors.append(f"{t('label_duplicate_check', lang=lang)}: {e}")
        confirmed = {
            (str(f.get("biomarker", "")).strip().lower(),
             str((f.get("params") or {}).get("target", "")).strip().lower())
            for f in dup_findings if f.get("action") == "merge"
        }
        findings.extend(dup_findings)
        for cand in candidates:
            source, target = cand["source"], cand["target"]
            if (source.lower(), target.lower()) in confirmed:
                continue
            src_units = sorted({(u or "").strip() for u in cand.get("source_units") or [] if (u or "").strip()})
            tgt_units = sorted({(u or "").strip() for u in cand.get("target_units") or [] if (u or "").strip()})
            # Units are NOT part of the similarity key, so a name typo can pair
            # with a unit difference. Renaming never converts a value, so the
            # user must see the units before accepting — otherwise mg/dl and
            # mmol/l measurements of one marker end up on a single trend line.
            unit_note = ""
            if src_units and tgt_units and {u.lower() for u in src_units} != {u.lower() for u in tgt_units}:
                unit_note = t(
                    "dq_candidate_merge_unit_warning", lang=lang,
                    source_units=", ".join(src_units), target_units=", ".join(tgt_units))
            findings.append({
                "category": "merge",
                "target_type": "registry",
                "biomarker": source,
                "description": t(
                    "dq_candidate_merge_description", lang=lang,
                    source=source, target=target, ratio=cand["ratio"]) + unit_note,
                "action": "merge",
                "params": {"target": target},
                "data_points": None,
            })

    # Map each data_point finding's 1-based "#n" indices (per biomarker group)
    # to real DataPoint DB ids — the apply step resolves them via session.get().
    mapped: list[dict] = []
    for f in findings:
        if f.get("target_type") != "data_point" or not f.get("data_points"):
            mapped.append(f)
            continue
        group = groups.get(f["biomarker"], [])
        ids = [group[i - 1]["id"] for i in f["data_points"] if 1 <= i <= len(group)]
        if not ids:
            continue  # drop a finding whose indices resolve to nothing
        f["data_points"] = ids
        mapped.append(f)
    findings = mapped

    # Replace previous findings for this patient (a re-run supersedes the old).
    clear_patient_data_quality_findings(session, patient_id)
    now = datetime.utcnow()
    for f in findings:
        session.add(DataQualityFinding(
            patient_id=patient_id,
            category=f["category"],
            target_type=f["target_type"],
            biomarker=f["biomarker"],
            report_id=None,
            data_point_ids=json.dumps(f["data_points"]) if f.get("data_points") else None,
            description=f["description"],
            action=f["action"],
            params_json=json.dumps(f["params"]),
            status="pending",
            created_at=now,
        ))
    session.commit()

    if not errors:
        return "success", t("msg_data_quality_checked", lang=lang, n=len(dp_dicts), s=len(findings))
    level = "warning" if findings else "error"
    text = t("msg_data_quality_partial", lang=lang, n=total, failed=len(errors), s=len(findings))
    for err in errors[:5]:
        text += f"\n⚠️ {err}"
    return level, text


_WORK_FNS = {
    "process_pdfs": _job_process_pdfs,
    "reprocess_pdfs": _job_reprocess_pdfs,
    "refresh_risks": _job_refresh_risks,
    "regen_summary": _job_regen_summary,
    "bulk_descriptions": _job_bulk_descriptions,
    "bulk_interpretations": _job_bulk_interpretations,
    "gen_description": _job_gen_description,
    "gen_interpretation": _job_gen_interpretation,
    "backfill": _job_backfill,
    "data_quality_check": _job_data_quality_check,
    "chat_answer": _job_chat_answer,
    "test_connection": _job_test_connection,
}
