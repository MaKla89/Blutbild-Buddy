from datetime import datetime
import json
from pathlib import Path

import streamlit as st
from sqlalchemy.orm import Session as SASession

from config import REPORTS_DIR, resolve_canonical, get_default_ref_range
from database import (
    Report, DataPoint, Patient, ReportArtifact, DataQualityFinding,
    get_or_create_patient, _canonical_base, get_registry, clean_unit,
)
from pdf_processor import pdf_to_base64_jpeg, file_hash
from llm_client import _build_client, extract_page, extract_pages_batch, is_job_aborted, normalize_names
from schemas import bare_number, looks_like_range
from translations import t, _get_lang


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


# ── Table layout / column-confusion guards ───────────────────────────────────
# A lab report table has one name column, ONE reference column, and often
# SEVERAL result columns (one per draw date, or a lying/standing pair). The two
# failure modes this guards:
#   (a) the reference cell is read as the measurement (or vice versa) — the
#       reference and result columns are swapped;
#   (b) a multi-column report is collapsed into one series, silently keeping
#       only one of its measurements.
# The extraction prompt now forces the model to declare the layout first
# ("columns", each with a role) and to tag every entry with its result column
# ("column_index"). Everything below validates that declaration deterministically
# and turns a violation into a DataQualityFinding the user confirms — a detected
# swap is NEVER applied silently, because "which cell was right" cannot always be
# decided from the data alone.

_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y", "%Y/%m/%d", "%d/%m/%Y", "%d %b %Y", "%d %B %Y")

# Thresholds for check (7) — the systematic ``value == reference bound`` pattern.
# A single value landing exactly on a limit is ordinary; a whole page where a
# third of the rows do is a column misread (the model grabbed the reference
# cell's first/second number as the measurement). Both a minimum count and a
# minimum fraction are required so a short page with one coincidence never fires.
_MIN_BOUND_MATCHES = 4
_MIN_BOUND_FRACTION = 0.3


def _parse_result_date(text: str | None) -> datetime | None:
    """Parse a result column's header into a date, when it is one.

    Only numeric/ISO shapes are accepted — a header like "lying"/"standing" or
    "Vor Behandlung" is a series label, not a date, and stays in
    ``DataPoint.column_label`` instead.
    """
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _page_layout(page) -> dict:
    """Normalise the layout the LLM declared for one page.

    Returns ``{"declared": bool, "result_cols": {index: (result_date, label)},
    "ref_indexes": [int], "has_ref": bool}``. ``declared`` is False when the
    model returned no layout at all — every check below then degrades to the old
    flat behaviour instead of failing, so a weaker model still produces data.

    Result columns are keyed by their declared index; a missing/duplicated index
    is filled positionally so a sloppy layout declaration still yields usable
    series identity.
    """
    cols = list(getattr(page, "columns", None) or [])
    if not cols:
        return {"declared": False, "result_cols": {}, "ref_indexes": [], "has_ref": False}

    result_cols: dict[int, tuple[datetime | None, str | None]] = {}
    ref_indexes: list[int] = []
    for pos, col in enumerate(cols):
        idx = col.index if isinstance(col.index, int) else pos
        header = (col.header or "").strip() or None
        if col.role == "result":
            rdate = _parse_result_date(col.result_date) or _parse_result_date(header)
            result_cols[idx] = (rdate, header)
        elif col.role == "reference":
            ref_indexes.append(idx)
    return {
        "declared": True,
        "result_cols": result_cols,
        "ref_indexes": ref_indexes,
        "has_ref": bool(ref_indexes),
    }


def _entry_series(layout: dict, column_index: int | None) -> tuple[datetime | None, str | None]:
    """The (result_date, column_label) a page's layout assigns to an entry.

    Single-result-column pages get the series identity even when the entry does
    not echo ``column_index`` — there is only one candidate, so it is
    unambiguous. With several result columns an entry without ``column_index``
    cannot be placed and stays unlabelled (the layout check flags it instead).
    """
    result_cols = layout.get("result_cols") or {}
    if not result_cols:
        return None, None
    if column_index is not None and column_index in result_cols:
        return result_cols[column_index]
    if len(result_cols) == 1:
        return next(iter(result_cols.values()))
    return None, None


def _column_findings(
    page, layout: dict, entries: list, canonicals: list[str | None],
    units: list[str | None], ref_pairs: list[tuple[float | None, float | None]],
    from_pdf_ref: list[bool], registry: dict, lang: str,
) -> list[dict]:
    """Deterministic column-confusion checks for ONE page.

    Returns finding dicts (category/description/action/params + the 0-based
    positions of the affected entries on this page). Deliberately narrow: every
    rule fires on a shape that cannot occur in a correctly read table, because a
    false positive costs the user a review click on every report.
    """
    findings: list[dict] = []
    n_result = len(layout.get("result_cols") or {})

    # (6) A table has at most one reference column. Two of them means the model
    # mislabelled a result column as the reference — every range on the page is
    # then suspect.
    if len(layout.get("ref_indexes") or []) > 1:
        findings.append({
            "category": "column_layout",
            "entry_positions": [],
            "description": t("dq_multiple_reference_columns", lang=lang,
                             indexes=", ".join(str(i) for i in layout["ref_indexes"])),
            "action": "acknowledge", "params": {},
        })

    for pos, entry in enumerate(entries):
        name = (entry.name or "").strip()
        if not name:
            continue
        canonical = canonicals[pos]
        ref_text = (entry.reference_range or "").strip()
        low, high = ref_pairs[pos]

        # (1) The value field holds a range ("4.2-6.1"): _parse_value coerced it
        # to None and kept the text in value_raw. A result cell is never a range,
        # so the row was read one column off. The fix is fully determined: the
        # range text becomes the reference range, and if the reference field
        # holds a bare number that number is the measurement that was displaced.
        if entry.value is None and looks_like_range(entry.value_raw or ""):
            new_low, new_high = _parse_reference_range(entry.value_raw)
            new_value = bare_number(ref_text)
            findings.append({
                "category": "column_swap",
                "entry_positions": [pos],
                "description": t("dq_swap_value_is_range", lang=lang,
                                 name=name, raw=entry.value_raw),
                "action": "swap_value_ref_range",
                "params": {"value": new_value, "low": new_low, "high": new_high,
                           "unit": units[pos]},
            })
            continue

        # (2) The reference field holds a single bare number. Reference cells are
        # ranges, one-sided bounds or qualitative words — a lone number there is
        # almost always a measurement read from the wrong column (typically the
        # second result column of a multi-column report).
        stray = bare_number(ref_text)
        if stray is not None and layout.get("has_ref"):
            d_low, d_high, _du = get_default_ref_range(canonical, units[pos], registry=registry)
            fix = ({"action": "set_ref_range",
                    "params": {"low": d_low, "high": d_high, "unit": units[pos]}}
                   if (d_low is not None or d_high is not None)
                   else {"action": "clear_ref_range", "params": {}})
            findings.append({
                "category": "column_swap",
                "entry_positions": [pos],
                "description": t("dq_swap_ref_holds_result", lang=lang,
                                 name=name, ref=ref_text),
                **fix,
            })

        # (3) ref_low == ref_high: one cell was copied into both bounds. A real
        # reference range is never a single point.
        if low is not None and high is not None and low == high and from_pdf_ref[pos]:
            d_low, d_high, _du = get_default_ref_range(canonical, units[pos], registry=registry)
            if (d_low is not None or d_high is not None) and (d_low, d_high) != (low, high):
                fix = {"action": "set_ref_range",
                       "params": {"low": d_low, "high": d_high, "unit": units[pos]}}
            else:
                fix = {"action": "clear_ref_range", "params": {}}
            findings.append({
                "category": "column_swap",
                "entry_positions": [pos],
                "description": t("dq_degenerate_ref_range", lang=lang, name=name, value=low),
                **fix,
            })

        # (4) The entry names a column that is not a declared result column —
        # e.g. it points at the reference column, so the stored "measurement" is
        # a reference value.
        if layout.get("declared") and n_result and entry.column_index is not None \
                and entry.column_index not in (layout.get("result_cols") or {}):
            findings.append({
                "category": "column_swap",
                "entry_positions": [pos],
                "description": t("dq_column_index_invalid", lang=lang,
                                 name=name, column_index=entry.column_index,
                                 valid=", ".join(str(i) for i in sorted(layout["result_cols"])) or "—"),
                "action": "disable_value", "params": {},
            })

    # (5) With several result columns, every row should contribute one entry per
    # column. A name appearing fewer times than that means a result column was
    # dropped for that row — the series then looks complete but is not.
    if n_result >= 2:
        by_name: dict[str, list[int]] = {}
        for pos, entry in enumerate(entries):
            nm = (entry.name or "").strip().lower()
            if nm:
                by_name.setdefault(nm, []).append(pos)
        for nm, positions in by_name.items():
            if 0 < len(positions) < n_result:
                findings.append({
                    "category": "column_layout",
                    "entry_positions": positions,
                    "description": t("dq_missing_result_column", lang=lang,
                                     name=entries[positions[0]].name,
                                     found=len(positions), expected=n_result),
                    "action": "acknowledge", "params": {},
                })

    # (7) Systematic value == reference bound. The signature failure of a
    # low-resolution render: the model reads the REFERENCE cell's first number as
    # the measurement, so ``value`` equals the lower bound of its own stored range
    # (or the upper bound, when it grabbed the second number). A single value
    # sitting exactly on a limit is normal — a whole page where a third of the
    # rows do is a column misread. This is the shape checks (1)-(4) cannot see:
    # the value is a bare number, the ref is a valid range, the column_index is
    # correct — only the AGGREGATE is wrong. It is also the only detector for
    # this failure that works without the LLM verify pass, so it fires on ingest.
    # Positions already flagged above are excluded to avoid duplicate findings.
    flagged = {p for f in findings for p in f["entry_positions"]}
    bound_hits: list[int] = []
    ref_rows = 0
    for pos, entry in enumerate(entries):
        if pos in flagged or not (entry.name or "").strip():
            continue
        low, high = ref_pairs[pos]
        if entry.value is None or not from_pdf_ref[pos]:
            continue  # a default ref range matching the value is not a misread
        ref_rows += 1
        if (low is not None and abs(entry.value - low) <= 1e-9) or \
                (high is not None and abs(entry.value - high) <= 1e-9):
            bound_hits.append(pos)
    if len(bound_hits) >= _MIN_BOUND_MATCHES and ref_rows and \
            len(bound_hits) / ref_rows >= _MIN_BOUND_FRACTION:
        sample = ", ".join((entries[p].name or "").strip() for p in bound_hits[:5])
        findings.append({
            "category": "column_swap",
            "entry_positions": bound_hits,
            "description": t("dq_value_matches_ref_bound", lang=lang,
                             count=len(bound_hits), total=ref_rows, names=sample),
            "action": "acknowledge", "params": {},
        })

    return findings





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


def process_new_pdfs(session: SASession, progress_callback=None, dpi: int | None = None, batch_size: int | None = None, force: bool = False, only_files: set[str] | None = None, lang: str | None = None) -> tuple[list[str], list[str]]:
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
        only_files: restrict processing to this set of PDF filenames. The
            reprocess job passes exactly the reports it snapshotted — without
            it, ``force=True`` would also re-extract PDFs whose rows are still
            in the DB (other patients' reports, or unprocessed uploads), and
            every one of those inserts fails on the ``file_hash`` unique
            constraint and shows up as a spurious error.
        lang: language for the column-confusion findings this run creates.
            Background workers MUST pass it (from their config snapshot) —
            they run outside the Streamlit script context, where reading
            session state is not safe.

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
    if lang is None:
        lang = _get_lang()
    # Resolve render settings once: explicit params (background jobs pass a
    # snapshot captured at submit time) win over the live app settings.
    if dpi is None:
        dpi = st.session_state.get("pdf_dpi", 100)
    if batch_size is None:
        batch_size = int(st.session_state.get("pdf_batch_size", 0))
    batch_size = max(0, batch_size)

    pdf_dir = Path(REPORTS_DIR)
    pdf_files = sorted(p for p in pdf_dir.glob("*.pdf") if not p.name.startswith("_"))
    if only_files is not None:
        pdf_files = [p for p in pdf_files if p.name in only_files]

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
        try:
            fhash = file_hash(pdf_path)
            if not force:
                existing = session.query(Report).filter(Report.file_hash == fhash).first()
                if existing:
                    if progress_callback:
                        progress_callback(f"Skipping {pdf_path.name} (duplicate)", (idx + 1) / total * 0.85)
                    continue

            if progress_callback:
                progress_callback(f"Processing {pdf_path.name} ({idx + 1}/{total})", (idx + 1) / total * 0.85)

            b64_pages = pdf_to_base64_jpeg(pdf_path, dpi=dpi)

            # Per-page grouping is needed because the column checks are per page
            # (each page declares its own layout) and their findings address
            # entries by position within that page. ``page_columns`` keeps the
            # raw declared layout for raw_text, so a later audit pass can be run
            # without re-extracting.
            pages_entries: list[list] = []
            page_layouts: list[dict] = []
            page_columns: list[list[dict]] = []
            patient_name = None
            report_date = None
            panel_type = None
            # Non-numeric findings from the written text, deduplicated per
            # report by (category, text) — the same note often repeats across
            # pages of one report, and we want ONE artifact carrying the first
            # page it appeared on. Key → (text, source_excerpt, page).
            artifact_by_key: dict[tuple[str, str], tuple[str, str | None, int]] = {}

            for page_idx, page_data in enumerate(_extract_report_pages(llm, b64_pages, batch_size)):
                if page_data.patient_name and not patient_name:
                    patient_name = page_data.patient_name
                if page_data.report_date and not report_date:
                    report_date = page_data.report_date
                if page_data.panel_type and not panel_type:
                    panel_type = page_data.panel_type
                pages_entries.append(list(page_data.entries))
                page_layouts.append(_page_layout(page_data))
                page_columns.append([c.model_dump() for c in (page_data.columns or [])])
                page_no = page_idx + 1  # 1-based, matches how pages are shown to the user
                for art in getattr(page_data, "artifacts", []) or []:
                    text = (art.text or "").strip()
                    if not text:
                        continue
                    category = (art.category or "general_insight").strip() or "general_insight"
                    key = (category.lower(), text.lower())
                    if key not in artifact_by_key:
                        excerpt = (art.source_excerpt or "").strip() or None
                        artifact_by_key[key] = (category, text, excerpt, page_no)

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

            # Extra measurement series this report carries: the result columns
            # whose header was a date. A single-column report has none of these,
            # so result_dates stays null and nothing downstream changes.
            extra_series: list[str] = []
            for layout in page_layouts:
                for rdate, label in (layout.get("result_cols") or {}).values():
                    if rdate is not None:
                        iso = rdate.strftime("%Y-%m-%d")
                        if iso != (parsed_date.strftime("%Y-%m-%d") if parsed_date else None) \
                                and iso not in extra_series:
                            extra_series.append(iso)

            # raw_text is stored as JSON, grouped per page (each page declares
            # its own column layout and its entries are addressed by position
            # within that page). Nothing parses it for display — it exists so a
            # later audit pass (the verify job) can re-send the exact layout +
            # entries the model produced, without re-extracting.
            report = Report(
                patient_id=patient.id,
                filename=pdf_path.name,
                file_hash=fhash,
                report_date=parsed_date,
                panel_type=panel_type,
                result_dates=json.dumps(extra_series) if extra_series else None,
                raw_text=json.dumps(
                    {"pages": [
                        {"columns": cols,
                         "entries": [e.model_dump(mode="json") for e in ents]}
                        for cols, ents in zip(page_columns, pages_entries)
                    ]},
                    default=str,
                ),
            )
            session.add(report)
            session.flush()

            # Insert per PAGE so a finding's entry positions map to the right
            # data point ids, and so each entry picks up its page's layout.
            for page_idx, entries in enumerate(pages_entries):
                layout = page_layouts[page_idx]
                canonicals: list[str | None] = []
                units: list[str | None] = []
                ref_pairs: list[tuple[float | None, float | None]] = []
                from_pdf_ref: list[bool] = []
                dp_by_pos: dict[int, DataPoint] = {}

                for pos, entry in enumerate(entries):
                    if not entry.name:
                        canonicals.append(None)
                        units.append(None)
                        ref_pairs.append((None, None))
                        from_pdf_ref.append(False)
                        continue
                    # Collapse stray internal whitespace in the unit ('ml/ min' →
                    # 'ml/min') BEFORE anything uses it. OCR/LLM extraction emits
                    # these, and because merge grouping keys on the unit a stray
                    # space splits one biomarker into two entries that never merge.
                    # Cleaning at ingestion keeps every downstream consumer
                    # (registry lookup, ref-range default, dedup, storage) aligned.
                    unit = clean_unit(entry.unit)
                    # Deterministic registry resolution only — the LLM name mapping
                    # happens once for ALL reports in Phase 2, not once per PDF.
                    canonical = _normalize_name(entry.name, unit, [], canonical_names, registry=registry)
                    ref_low, ref_high = _parse_reference_range(entry.reference_range)
                    had_pdf_ref = ref_low is not None or ref_high is not None

                    # Fallback to default reference ranges (active DB registry) when PDF omits them
                    if ref_low is None and ref_high is None:
                        def_ref_low, def_ref_high, def_unit = get_default_ref_range(canonical, unit, registry=registry)
                        if def_ref_low is not None or def_ref_high is not None:
                            ref_low = def_ref_low
                            ref_high = def_ref_high

                    # One-sided bounds count too (R3-08): "< 5.2" → high only,
                    # "> 120" → low only. A value beyond the single bound is abnormal.
                    is_abnormal = _is_value_abnormal(entry.value, ref_low, ref_high)

                    # Which measurement series this value belongs to, from the
                    # page's declared layout. (None, None) for the ordinary
                    # single-column report, which keeps every existing consumer
                    # on the (marker, unit, report_date) key.
                    series_date, series_label = _entry_series(layout, entry.column_index)

                    dp = DataPoint(
                        report_id=report.id,
                        canonical_name=canonical,
                        original_name=entry.name,
                        value=entry.value,
                        unit=unit,
                        ref_low=ref_low,
                        ref_high=ref_high,
                        is_abnormal=is_abnormal,
                        result_date=series_date,
                        column_label=series_label,
                    )
                    session.add(dp)
                    dp_by_pos[pos] = dp

                    canonicals.append(canonical)
                    units.append(unit)
                    ref_pairs.append((ref_low, ref_high))
                    from_pdf_ref.append(had_pdf_ref)

                    # Deduplicate by (name case-insensitive, cleaned unit or "") so
                    # the LLM sees units — needed to disambiguate unit-qualified
                    # variants. The cleaned unit also collapses 'ml/min' vs 'ml/ min'
                    # into one pending entry.
                    key = (entry.name.strip().lower(), unit or "")
                    if key not in pending_entries:
                        pending_entries[key] = {"name": entry.name, "unit": unit}

                # Flush so every DataPoint on this page has a real id BEFORE the
                # findings below reference ``dp_by_pos[p].id``. Without this the
                # ids are still None, the finding's data_point_ids becomes
                # "[null]", and an accepted swap could never be applied.
                session.flush()

                # Column-confusion checks for this page. Findings are stored as
                # pending DataQualityFindings — a detected swap is NEVER applied
                # silently, because which cell was correct cannot always be
                # decided from the stored data alone; the user confirms in the
                # Mapping tab.
                try:
                    page_findings = _column_findings(
                        pages_entries[page_idx], layout, entries, canonicals, units,
                        ref_pairs, from_pdf_ref, registry, lang,
                    )
                except Exception:
                    page_findings = []  # a broken check must never lose a report
                for fdef in page_findings:
                    ids = [dp_by_pos[p].id for p in fdef["entry_positions"]
                           if p in dp_by_pos and dp_by_pos[p].id is not None]
                    session.add(DataQualityFinding(
                        patient_id=patient.id,
                        category=fdef["category"],
                        target_type="data_point" if ids else "report",
                        biomarker=(entries[fdef["entry_positions"][0]].name if fdef["entry_positions"]
                                   else (panel_type or pdf_path.name)),
                        report_id=report.id,
                        data_point_ids=json.dumps(ids) if ids else None,
                        description=fdef["description"],
                        action=fdef["action"],
                        params_json=json.dumps(fdef["params"]),
                        source="extraction",
                        status="pending",
                    ))

            # Store the deduplicated written-text findings for this report.
            for category, text_val, excerpt, page_no in artifact_by_key.values():
                session.add(ReportArtifact(
                    report_id=report.id,
                    category=category,
                    text=text_val,
                    source_excerpt=excerpt,
                    source_page=page_no,
                ))

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
