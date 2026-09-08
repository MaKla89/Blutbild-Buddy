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




def process_new_pdfs(session: SASession, progress_callback=None, dpi: int | None = None, batch_size: int | None = None, force: bool = False) -> tuple[list[str], list[str]]:
    """Process all unprocessed PDFs in REPORTS_DIR.

    Args:
        session: DB session (commits per PDF; rolls back on failure).
        progress_callback: optional callable(msg, pct) for UI/job progress.
        dpi: PDF render DPI; falls back to the app setting (session state).
        batch_size: pages per LLM call; falls back to the app setting.
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
        batch_size = max(1, st.session_state.get("pdf_batch_size", 3))

    pdf_dir = Path(REPORTS_DIR)
    pdf_files = sorted(p for p in pdf_dir.glob("*.pdf") if not p.name.startswith("_"))

    total = len(pdf_files)
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
                    progress_callback(f"Skipping {pdf_path.name} (duplicate)", (idx + 1) / total)
                continue

        try:
            if progress_callback:
                progress_callback(f"Processing {pdf_path.name} ({idx + 1}/{total})", (idx + 1) / total)

            b64_pages = pdf_to_base64_jpeg(pdf_path, dpi=dpi)

            all_entries = []
            patient_name = None
            report_date = None
            panel_type = None

            if batch_size <= 1 or len(b64_pages) <= 1:
                for b64 in b64_pages:
                    page_data = extract_page(llm, b64)
                    if page_data.patient_name and not patient_name:
                        patient_name = page_data.patient_name
                    if page_data.report_date and not report_date:
                        report_date = page_data.report_date
                    if page_data.panel_type and not panel_type:
                        panel_type = page_data.panel_type
                    all_entries.extend(page_data.entries)
            else:
                for i in range(0, len(b64_pages), batch_size):
                    batch = b64_pages[i:i + batch_size]
                    page_results = extract_pages_batch(llm, batch)
                    for page_data in page_results:
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

            # Deduplicate by (name case-insensitive, unit or "") so the LLM
            # sees units — needed to disambiguate unit-qualified variants.
            seen: set[tuple[str, str]] = set()
            raw_entries: list[dict] = []
            for e in all_entries:
                if not e.name:
                    continue
                key = (e.name.strip().lower(), (e.unit or "").strip())
                if key in seen:
                    continue
                seen.add(key)
                raw_entries.append({"name": e.name, "unit": e.unit})
            norm_mapping = normalize_names(llm, raw_entries, canonical_names, registry=registry)

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
                canonical = _normalize_name(entry.name, entry.unit, norm_mapping.mappings, canonical_names, registry=registry)
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

            session.commit()
            processed.append(pdf_path.name)

        except Exception as exc:
            session.rollback()
            error_msg = f"{pdf_path.name}: {exc}"
            errors.append(error_msg)
            if progress_callback:
                progress_callback(f"Error processing {pdf_path.name}", (idx + 1) / total)

    return processed, errors
