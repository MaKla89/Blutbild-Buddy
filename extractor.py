from datetime import datetime
from pathlib import Path

import streamlit as st
from sqlalchemy.orm import Session as SASession

from config import REPORTS_DIR, resolve_canonical, get_default_ref_range
from database import Report, DataPoint, Patient, get_or_create_patient, _canonical_base, get_registry
from pdf_processor import pdf_to_base64_jpeg, file_hash
from llm_client import _build_client, extract_page, extract_pages_batch, is_job_aborted, normalize_names


def _normalize_name(name: str, unit: str | None, mappings: list[dict], canonical_list: list[str], registry: dict | None = None) -> str:
    # 1) LLM mapping (if available from batch call). A mapping applies when the
    # name matches case-insensitively AND the units are compatible: a mapping
    # with null/missing unit applies to any unit, otherwise both units must be
    # equal case-insensitively. An exact-unit match wins over a unit-less one.
    name_lower = name.strip().lower()
    unit_lower = unit.strip().lower() if isinstance(unit, str) and unit.strip() else None
    fallback_mapping = None
    for m in mappings:
        if m["original"].strip().lower() != name_lower:
            continue
        m_unit = m.get("unit")
        m_unit_lower = m_unit.strip().lower() if isinstance(m_unit, str) and m_unit.strip() else None
        if m_unit_lower is None:
            # Unit-less mapping: applies to any unit (kept as fallback only)
            if fallback_mapping is None:
                fallback_mapping = m
        elif m_unit_lower == unit_lower:
            return m["canonical"]  # exact-unit match — best possible
    if fallback_mapping is not None:
        return fallback_mapping["canonical"]

    # 2) Deterministic alias lookup — no LLM needed
    resolved = resolve_canonical(name, unit, registry=registry)
    if resolved:
        return resolved

    # 3) Fallback: try exact canonical name match (case-insensitive)
    name_lower = name.strip().lower()
    for canonical in canonical_list:
        if _canonical_base(canonical).lower() == name_lower:
            return canonical

    # 4) Nothing matched — return original name as-is
    return name


def _parse_reference_range(ref_str: str | None) -> tuple[float | None, float | None]:
    if not ref_str:
        return None, None
    ref_str = ref_str.strip().replace(",", ".")
    # pattern: "< 5.2" → ref_high = 5.2
    if ref_str.startswith("<"):
        try:
            return None, float(ref_str.lstrip("<").strip())
        except ValueError:
            return None, None
    # pattern: "> 120" → ref_low = 120
    if ref_str.startswith(">"):
        try:
            return float(ref_str.lstrip(">").strip()), None
        except ValueError:
            return None, None
    # pattern: "4.2-6.1" or "4.2 – 6.1" or "4,2-6,1"
    for sep in ["–", "-", "—", "−"]:
        if sep in ref_str:
            parts = ref_str.split(sep, 1)
            try:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return low, high
            except ValueError:
                pass
    return None, None


def _is_value_abnormal(value: float | None, ref_low: float | None, ref_high: float | None) -> bool:
    """Return True if value is outside the reference range.

    One-sided bounds count too (R3-08): "< 5.2" → high only, "> 120" → low
    only. A value beyond the single bound is abnormal. With no usable bound
    (or a non-numeric value) nothing can be flagged.
    """
    if value is None:
        return False
    if ref_low is not None and value < ref_low:
        return True
    if ref_high is not None and value > ref_high:
        return True
    return False




_NORMALIZE_CHUNK = 40  # deduplicated name entries per normalization LLM call


def _extract_report_pages(llm, b64_pages: list[str], batch_size: int) -> list:
    """Extract every page of ONE report.

    ``batch_size`` <= 0 → ONE LLM call carrying all pages of the report
    ("one report = one call"). Measured on the local endpoint: 4 pages in a
    single request ≈ 60 s, the same 4 pages as 4 separate calls ≈ 200 s —
    batching is both cheaper and faster, so it is the default.
    ``batch_size`` >= 1 → that many pages per call, for models that choke on
    many images at once. A one-page report always uses the single-page call.
    """
    if len(b64_pages) == 1:
        return [extract_page(llm, b64_pages[0])]
    if batch_size <= 0 or batch_size >= len(b64_pages):
        return extract_pages_batch(llm, b64_pages)
    results: list = []
    for i in range(0, len(b64_pages), batch_size):
        results.extend(extract_pages_batch(llm, b64_pages[i:i + batch_size]))
    return results


def _apply_global_normalization(
    session: SASession, llm, entries: list[dict], report_ids: list[int],
    registry: dict, canonical_names: list[str], progress_callback=None,
) -> int:
    """Map extracted names to canonical names in bounded GLOBAL LLM calls.

    Runs AFTER every report has been extracted and committed, so no single
    request ever carries the whole batch: the deduplicated (name, unit) entries
    of all newly processed reports are split into chunks of
    ``_NORMALIZE_CHUNK`` and each chunk gets its own call (the old code sent
    every entry of every report in one request, which grows unbounded with the
    number of PDFs and times out). Data points whose canonical name changes get
    a reference range that came from the registry default re-derived, and their
    abnormality flag recomputed. Returns the number of updated data points.
    """
    if not entries or not report_ids:
        return 0
    mappings: list[dict] = []
    n_chunks = (len(entries) + _NORMALIZE_CHUNK - 1) // _NORMALIZE_CHUNK
    for ci in range(0, len(entries), _NORMALIZE_CHUNK):
        if is_job_aborted():
            break
        chunk_no = ci // _NORMALIZE_CHUNK + 1
        if progress_callback:
            progress_callback(
                f"Normalizing biomarker names ({chunk_no}/{n_chunks})",
                0.85 + 0.15 * chunk_no / n_chunks,
            )
        try:
            norm = normalize_names(
                llm, entries[ci:ci + _NORMALIZE_CHUNK], canonical_names, registry=registry
            )
            mappings.extend(norm.mappings)
        except Exception:
            # A failed chunk keeps that chunk's deterministic resolution —
            # strictly better than losing the whole run.
            continue
    if not mappings:
        return 0

    changed = 0
    dps = session.query(DataPoint).filter(DataPoint.report_id.in_(report_ids)).all()
    for dp in dps:
        if not dp.original_name:
            continue
        new_canon = _normalize_name(dp.original_name, dp.unit, mappings, canonical_names, registry=registry)
        if new_canon == dp.canonical_name:
            continue
        old_canon = dp.canonical_name
        dp.canonical_name = new_canon
        # Only a range that was filled from the registry default (not read off
        # the PDF) follows the new canonical name; a PDF-provided range stays.
        d_low, d_high, _du = get_default_ref_range(old_canon, dp.unit, registry=registry)
        if (dp.ref_low, dp.ref_high) == (d_low, d_high):
            n_low, n_high, _nu = get_default_ref_range(new_canon, dp.unit, registry=registry)
            dp.ref_low, dp.ref_high = n_low, n_high
        dp.is_abnormal = _is_value_abnormal(dp.value, dp.ref_low, dp.ref_high)
        changed += 1
    if changed:
        session.commit()
    return changed


def process_new_pdfs(session: SASession, progress_callback=None, dpi: int | None = None, batch_size: int | None = None, force: bool = False) -> tuple[list[str], list[str]]:
    """Process all unprocessed PDFs in REPORTS_DIR.

    Two phases, so no single LLM request grows with the number of PDFs:
      Phase 1 — extraction: ONE LLM call per report (all of its pages in one
        request; ``batch_size`` > 0 splits big reports into page batches).
        Canonical names are resolved deterministically from the registry and
        each report is committed immediately, so an abort or restart keeps
        everything processed so far.
      Phase 2 — normalization: a global pass over the deduplicated names of all
        newly processed reports, split into bounded LLM calls, updating the
        stored data points. Risk assessment and the health summary run
        afterwards as their own calls (llm_jobs._job_process_pdfs).

    Args:
        session: DB session (commits per PDF; rolls back on failure).
        progress_callback: optional callable(msg, pct) for UI/job progress.
        dpi: PDF render DPI; falls back to the app setting (session state).
        batch_size: pages per extraction call; 0 = all pages of a report in one
            call. Falls back to the app setting.
        force: re-extract even when the file hash already exists in the DB.
            The caller must have deleted the old Report rows first (the
            ``file_hash`` unique constraint would otherwise reject the insert);
            used by the "reprocess" job after a registry/LLM improvement.

    Returns:
        (processed, errors) — processed is a list of filenames, errors a list
        of "filename: message" strings. Per-PDF failures never abort the rest.
    """
    llm = _build_client()
    processed = []
    errors = []
    # Active registry from the DB (honors accepted LLM proposals); falls back
    # to the config baseline for names not yet seeded.
    registry = get_registry(session)
    canonical_names = sorted(registry.keys())
    # Resolve render settings once: explicit params (background jobs pass a
    # snapshot captured at submit time) win over the live app settings.
    if dpi is None:
        dpi = st.session_state.get("pdf_dpi", 100)
    if batch_size is None:
        batch_size = int(st.session_state.get("pdf_batch_size", 0))
    batch_size = max(0, batch_size)

    pdf_dir = Path(REPORTS_DIR)
    pdf_files = sorted(p for p in pdf_dir.glob("*.pdf") if not p.name.startswith("_"))

    total = len(pdf_files)
    # Phase 1 collects the deduplicated (name, unit) entries and the new report
    # ids so Phase 2 can normalize them in bounded global calls.
    pending_entries: dict[tuple[str, str], dict] = {}
    new_report_ids: list[int] = []

    for idx, pdf_path in enumerate(pdf_files):
        if is_job_aborted():
            # User aborted the job: stop before starting the next PDF. Partial
            # results so far are kept; the worker marks the job interrupted.
            break
        fhash = file_hash(pdf_path)
        if not force:
            existing = session.query(Report).filter(Report.file_hash == fhash).first()
            if existing:
                if progress_callback:
                    progress_callback(f"Skipping {pdf_path.name} (duplicate)", (idx + 1) / total * 0.85)
                continue

        try:
            if progress_callback:
                progress_callback(f"Processing {pdf_path.name} ({idx + 1}/{total})", (idx + 1) / total * 0.85)

            b64_pages = pdf_to_base64_jpeg(pdf_path, dpi=dpi)

            all_entries = []
            patient_name = None
            report_date = None
            panel_type = None

            for page_data in _extract_report_pages(llm, b64_pages, batch_size):
                if page_data.patient_name and not patient_name:
                    patient_name = page_data.patient_name
                if page_data.report_date and not report_date:
                    report_date = page_data.report_date
                if page_data.panel_type and not panel_type:
                    panel_type = page_data.panel_type
                all_entries.extend(page_data.entries)

            if not patient_name:
                parts = pdf_path.stem.split("-")
                patient_name = parts[1].strip() if len(parts) > 1 else pdf_path.stem

            if not report_date:
                year_str = pdf_path.stem[:4]
                report_date = f"{year_str}-01-01" if year_str.isdigit() else None

            parsed_date = None
            if report_date:
                for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%d/%m/%Y", "%d %B %Y", "%d.%m.%y"):
                    try:
                        parsed_date = datetime.strptime(report_date, fmt)
                        break
                    except ValueError:
                        continue

            patient = get_or_create_patient(session, patient_name or "Unknown")

            report = Report(
                patient_id=patient.id,
                filename=pdf_path.name,
                file_hash=fhash,
                report_date=parsed_date,
                panel_type=panel_type,
                raw_text=str([e.model_dump() for e in all_entries]),
            )
            session.add(report)
            session.flush()

            for entry in all_entries:
                if not entry.name:
                    continue
                # Deterministic registry resolution only — the LLM name mapping
                # happens once for ALL reports in Phase 2, not once per PDF.
                canonical = _normalize_name(entry.name, entry.unit, [], canonical_names, registry=registry)
                ref_low, ref_high = _parse_reference_range(entry.reference_range)

                # Fallback to default reference ranges (active DB registry) when PDF omits them
                if ref_low is None and ref_high is None:
                    def_ref_low, def_ref_high, def_unit = get_default_ref_range(canonical, entry.unit, registry=registry)
                    if def_ref_low is not None or def_ref_high is not None:
                        ref_low = def_ref_low
                        ref_high = def_ref_high

                # One-sided bounds count too (R3-08): "< 5.2" → high only,
                # "> 120" → low only. A value beyond the single bound is abnormal.
                is_abnormal = _is_value_abnormal(entry.value, ref_low, ref_high)

                dp = DataPoint(
                    report_id=report.id,
                    canonical_name=canonical,
                    original_name=entry.name,
                    value=entry.value,
                    unit=entry.unit,
                    ref_low=ref_low,
                    ref_high=ref_high,
                    is_abnormal=is_abnormal,
                )
                session.add(dp)

                # Deduplicate by (name case-insensitive, unit or "") so the LLM
                # sees units — needed to disambiguate unit-qualified variants.
                key = (entry.name.strip().lower(), (entry.unit or "").strip())
                if key not in pending_entries:
                    pending_entries[key] = {"name": entry.name, "unit": entry.unit}

            session.commit()
            new_report_ids.append(report.id)
            processed.append(pdf_path.name)

        except Exception as exc:
            session.rollback()
            error_msg = f"{pdf_path.name}: {exc}"
            errors.append(error_msg)
            if progress_callback:
                progress_callback(f"Error processing {pdf_path.name}", (idx + 1) / total * 0.85)

    # Phase 2 — global name normalization in bounded LLM calls (see helper).
    if new_report_ids and not is_job_aborted():
        try:
            _apply_global_normalization(
                session, llm, list(pending_entries.values()), new_report_ids,
                registry, canonical_names, progress_callback=progress_callback,
            )
        except Exception as exc:
            # Normalization is a refinement on top of already-committed data —
            # never fail the whole run for it.
            session.rollback()
            errors.append(f"Normalization: {exc}")

    return processed, errors
