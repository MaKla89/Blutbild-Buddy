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
import threading
from datetime import datetime, timedelta

import streamlit as st

import config
from database import (
    Patient, Report, DataPoint, RiskFlag, PatientSummary, BiomarkerInfo,
    get_session, create_llm_job, get_llm_job,
    upsert_biomarker_info, backfill_reference_ranges, create_patient_summary,
    get_registry,
    DataQualityFinding, clear_patient_data_quality_findings,
)
from llm_client import (
    _build_client, set_job_config,
    generate_patient_summary, generate_biomarker_description,
    generate_biomarker_interpretation,
    run_biomarker_quality_check, run_duplicate_check,
    is_job_aborted, request_job_abort, set_current_job_id,
)
from analyzer import refresh_risk_flags
from extractor import process_new_pdfs
from translations import t, _get_lang

# Kinds that mutate data the main tabs render — when one of these finishes,
# the UI requests a full app rerun so the new data is picked up.
DATA_AFFECTING_KINDS = {
    "process_pdfs", "reprocess_pdfs", "refresh_risks", "regen_summary",
    "bulk_descriptions", "bulk_interpretations", "backfill",
    "data_quality_check",
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
        "batch_size": st.session_state.get("pdf_batch_size", 1),
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
        threading.Thread(target=_run_job, args=(job.id,), daemon=True).start()
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


def _run_job(job_id: int):
    """Worker entry point: run the job's work function and record the outcome."""
    session = get_session()
    kind = ""
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
            pass
    finally:
        set_job_config(None)
        _current_job_id_var.job_id = None
        set_current_job_id(None)
        session.close()
        with _kind_lock:
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
    """Collect recent (2y) + historical data points and risk flags for the
    summary prompt. Same logic as app._generate_summary_for_patient."""
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
        session.query(RiskFlag.category, RiskFlag.description)
        .join(Report)
        .filter(Report.patient_id == patient.id)
        .all()
    )
    risk_data = [(rf.category, rf.description) for rf in rf_rows]
    return recent_data, historical_data, risk_data


def _job_process_pdfs(session, cfg, lang, params):
    processed, errors = process_new_pdfs(
        session, progress_callback=None,
        dpi=cfg.get("dpi"), batch_size=cfg.get("batch_size"),
    )
    if is_job_aborted():
        # Partial results are already committed per PDF; skip the follow-up
        # LLM steps (risk refresh, summary) — _run_job marks it interrupted.
        return "info", t("job_aborted_note", lang=lang)
    n_risks = 0
    if processed:
        try:
            n_risks = refresh_risk_flags(session, _build_client())
        except Exception as e:
            errors.append(f"{t('error_risk_flags', lang=lang)}: {e}")

    # Auto-generate the health summary for the patient selected at submit time.
    patient_name = params.get("patient")
    if processed and patient_name:
        p = session.query(Patient).filter(Patient.name == patient_name).first()
        if p:
            try:
                llm = _build_client()
                recent, historical, risks = _gather_summary_data(session, p)
                text = generate_patient_summary(llm, p.name, recent, historical, risks, lang)
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
    The report snapshot feeds the before/after diff; the disabled set preserves
    the user's manual value disabling across the cascade delete (matched by
    filename + original name + unit — the stable identity of a measurement).
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
        reports[r.filename] = {
            "patient_id": r.patient_id,
            "report_date": r.report_date,
            "panel_type": r.panel_type,
            "dps": [
                (dp.original_name or "", dp.unit or "", dp.canonical_name,
                 dp.value, dp.ref_low, dp.ref_high)
                for dp in dps
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
    return {"reports": reports, "disabled": disabled}


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


def _job_reprocess_pdfs(session, cfg, lang, params):
    """Re-extract existing reports with the current LLM + registry.

    Flow: snapshot old data → delete Report rows (cascades to data points and
    risk flags; patient identity is pinned by re-attributing each PDF to its
    previous patient) → run process_new_pdfs(force=True) → refresh risk flags
    → restore manually disabled values → diff old vs. new for the result text.

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

    # Delete old rows and everything that referenced them. Bulk Query.delete()
    # does NOT apply ORM cascades (and SQLite doesn't enforce FKs), so the
    # children must be removed explicitly — otherwise orphaned data points
    # would survive the reprocess and pollute the diff.
    report_ids = [r.id for r in session.query(Report).filter(
        Report.filename.in_(list(before["reports"].keys()))
    ).all()]
    if report_ids:
        from database import PatientSummary, RiskFlag
        session.query(RiskFlag).filter(RiskFlag.report_id.in_(report_ids)).delete()
        session.query(DataPoint).filter(DataPoint.report_id.in_(report_ids)).delete()
        session.query(PatientSummary).filter(PatientSummary.report_id.in_(report_ids)).delete()
        session.query(Report).filter(Report.id.in_(report_ids)).delete()
        session.commit()

    # Estimate pages for the progress bar (best-effort — missing files are
    # simply skipped by process_new_pdfs).
    pdf_dir = _Path(config.REPORTS_DIR)
    total_pages = 0
    for fn in before["reports"]:
        p = pdf_dir / fn
        if p.exists():
            try:
                total_pages += pdf_page_count(p)
            except Exception:
                pass

    def progress(msg, pct):
        _update_progress(session, getattr(_current_job_id_var, "job_id", None), msg, pct)

    processed, errors = process_new_pdfs(
        session, progress_callback=progress,
        dpi=cfg.get("dpi"), batch_size=cfg.get("batch_size"), force=True,
    )

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

    if is_job_aborted():
        return "info", t("job_aborted_note", lang=lang)

    n_risks = 0
    if processed:
        try:
            n_risks = refresh_risk_flags(session, _build_client())
        except Exception as e:
            errors.append(f"{t('error_risk_flags', lang=lang)}: {e}")

    diff = _diff_reprocess(before, session, processed)

    parts = [t("msg_reprocess_done", lang=lang, n=len(processed), risks=n_risks)]
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
    if diff["patient_moves"]:
        parts.append(t("reprocess_patient_pinned", lang=lang, n=diff["patient_moves"]))
    for fn in diff["missing"][:5]:
        parts.append(f"⚠️ {fn}")
    for err in errors[:5]:
        parts.append(f"⚠️ {err}")

    if processed and not errors and not diff["missing"]:
        return "success", "\n".join(parts)
    if processed:
        return "warning", "\n".join(parts)
    return "error", "\n".join(parts)


def _job_refresh_risks(session, cfg, lang, params):
    n = refresh_risk_flags(session, _build_client())
    return "success", t("msg_risks_refreshed_all", lang=lang, n=n)


def _job_regen_summary(session, cfg, lang, params):
    patient_id = params.get("patient_id")
    p = session.query(Patient).get(patient_id)
    if p is None:
        return "error", t("error_select_patient", lang=lang)
    llm = _build_client()
    recent, historical, risks = _gather_summary_data(session, p)
    text = generate_patient_summary(llm, p.name, recent, historical, risks, lang)
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
    ok, errors = _run_bulk(
        session, patient_id, llm, items, "description",
        generate_one=lambda llm_, item: generate_biomarker_description(
            llm_, item[0], item[1], config.CANONICAL_BIOMARKERS, lang,
        ),
    )
    return _bulk_result(ok, len(items), errors, "msg_all_descriptions_generated", lang)


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
        return generate_biomarker_interpretation(llm_, cname, unit, dp_tuples, lang)

    ok, errors = _run_bulk(
        session, patient_id, llm, items, "interpretation",
        generate_one=gen_one,
    )
    return _bulk_result(ok, len(items), errors, "msg_all_interpretations_generated", lang)


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
    if not is_job_aborted():
        try:
            entries = [(d["name"], d["unit"]) for d in dp_dicts]
            findings.extend(run_duplicate_check(llm, entries, registry, lang)["findings"])
        except Exception as e:
            errors.append(f"{t('label_duplicate_check', lang=lang)}: {e}")

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
    "backfill": _job_backfill,
    "data_quality_check": _job_data_quality_check,
}
