import json
import threading
from datetime import datetime

from openai import OpenAI
from PIL import Image

from config import get_unsloth_base_url, get_unsloth_model, get_unsloth_api_key, get_llm_timeout, _CANONICAL_BIOMARKERS
from pdf_processor import image_to_base64_jpeg
from schemas import ExtractedPage, NormalizationMapping
from translations import _get_lang

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Limits concurrent LLM calls to prevent flooding the endpoint (e.g. local Unsloth).
# A semaphore ensures at most N LLM requests run simultaneously.
_MAX_CONCURRENT_LLM_CALLS = 2
_llm_semaphore = threading.Semaphore(_MAX_CONCURRENT_LLM_CALLS)


class LLMResponseError(ValueError):
    """Raised when the LLM returns malformed or unexpected JSON."""
    pass


# ── Job abort (user-initiated cancellation from the LLM-Jobs tab) ────────────
# Per-job abort set: the UI calls request_job_abort(job_id); worker threads
# check is_job_aborted() between steps, and _abort_timeout() cuts an in-flight
# HTTP request of an aborted job short (1s → APITimeoutError, caught by the
# worker and surfaced as "interrupted"). The current job id is thread-local
# (set by llm_jobs._run_job) because LLM call sites don't take a job id.
# LlmJob.id is a reusable SQLite rowid, so _run_job's finally block calls
# clear_job_abort() to drop the id when the worker exits — otherwise a later
# job could inherit a stale id and start already-aborted.
_aborted_job_ids: set[int] = set()
_abort_lock = threading.Lock()
_current_job_id_local = threading.local()


def request_job_abort(job_id: int):
    """Mark one job as aborted; its worker stops after the current step."""
    with _abort_lock:
        _aborted_job_ids.add(job_id)


def clear_job_abort(job_id: int):
    """Discard a job id from the aborted set once its worker has finished.

    LlmJob.id is a SQLite rowid alias (INTEGER PRIMARY KEY, no AUTOINCREMENT)
    and clear_finished_llm_jobs() deletes finished rows, so SQLite reuses freed
    rowids for later jobs. Without this, a new job could inherit a stale id and
    start already-aborted. Called from llm_jobs._run_job's finally block.
    """
    with _abort_lock:
        _aborted_job_ids.discard(job_id)


def is_job_aborted(job_id: int | None = None) -> bool:
    """True if the given (or this thread's current) job was aborted."""
    if job_id is None:
        job_id = getattr(_current_job_id_local, "job_id", None)
    if job_id is None:
        return False
    with _abort_lock:
        return job_id in _aborted_job_ids


def set_current_job_id(job_id: int | None):
    """Set the thread-local current job id (called by llm_jobs._run_job)."""
    _current_job_id_local.job_id = job_id


def _abort_timeout() -> int | None:
    """Timeout for the next LLM call: 1s while the current job is aborted (so
    an in-flight request fails fast), otherwise the configured default."""
    return 1 if is_job_aborted() else None


def _strip_markdown_json(text: str) -> str:
    """Remove markdown code-fence wrappers that LLMs sometimes add around JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # skip the opening ``json` or `` line
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text

EXTRACTION_SYSTEM_PROMPT = """You are a medical data extraction assistant. Your task is to extract structured data from lab report PDF page images.

Look at the image carefully and extract ALL test results visible. For each test, provide:
- name (the test name exactly as printed)
- value (numeric value — if the value is a non-numeric result like "negativ" or "nicht erhöht" set value to null)
- unit (measurement unit, e.g. g/dl, U/l, mmol/l)
- reference_range (the reference range exactly as printed, e.g. "4.2-6.1" or "< 5.2" or "negativ")

Also extract:
- patient_name (full name of the patient from the report header)
- report_date (the date on the report in ISO format YYYY-MM-DD, e.g. "2025-08-12", converted from whatever format is printed; null if no date can be determined)
- panel_type (what kind of panel this is, like "Blutbild", "Leberwerte", etc.)

Return ONLY valid JSON matching this structure, no markdown or explanation:
{"patient_name": "... or null", "report_date": "... or null", "panel_type": "... or null", "entries": [{"name": "...", "value": number or null, "unit": "... or null", "reference_range": "... or null"}]}"""

EXTRACTION_SYSTEM_PROMPT_BATCH = """You are a medical data extraction assistant. Your task is to extract structured data from lab report PDF page images. You will receive multiple page images in a single request.

For EACH page image, extract ALL test results visible. For each test, provide:
- name (the test name exactly as printed)
- value (numeric value — if the value is a non-numeric result like "negativ" or "nicht erhöht" set value to null)
- unit (measurement unit, e.g. g/dl, U/l, mmol/l)
- reference_range (the reference range exactly as printed, e.g. "4.2-6.1" or "< 5.2" or "negativ")

Also extract for each page:
- patient_name (full name of the patient from the report header)
- report_date (the date on the report in ISO format YYYY-MM-DD, e.g. "2025-08-12", converted from whatever format is printed; null if no date can be determined)
- panel_type (what kind of panel this is, like "Blutbild", "Leberwerte", etc.)
- page_index (the 0-based position of that page image in the order you received them: first image = 0, second = 1, ...)

Return ONLY valid JSON matching this structure, no markdown or explanation:
{"pages": [{"page_index": 0, "patient_name": "... or null", "report_date": "... or null", "panel_type": "... or null", "entries": [{"name": "...", "value": number or null, "unit": "... or null", "reference_range": "... or null"}]}]}"""

NORMALIZATION_SYSTEM_PROMPT = """You are a medical terminology normalization assistant. You will be given a list of extracted biomarker entries (name plus measurement unit where known) from lab reports and a canonical registry of standard biomarkers.

Each canonical biomarker has:
- A canonical name (the exact name to use)
- Expected units (e.g. g/dl, nmol/l)
- Aliases (alternative names the lab report might use)

Your task is to map each extracted entry to the closest matching canonical name from the provided registry. Consider:
- German abbreviations (e.g. "Hb" → "Hämoglobin (Hb)")
- Slight variations in spelling or punctuation
- Truncated names
- Multi-word names that might be split differently
- Unit-qualified variants: some biomarkers have multiple canonical entries distinguished by unit (e.g. "HbA1c (%)" vs "HbA1c (mmol/mol)"). When this applies, map to the variant that matches the extracted unit. If two extracted entries share the same name but differ in unit, they need SEPARATE mappings — one per (name, unit) pair.
- Case sensitivity: always return the canonical name EXACTLY as it appears in the canonical list, preserving its original capitalization. Do NOT change the case of canonical names.

Return ONLY valid JSON, no markdown or explanation:
{"mappings": [{"original": "extracted name exactly", "unit": "unit of that extracted entry, or null", "canonical": "matched canonical name"}]}

The "unit" field in each mapping echoes the unit of the extracted entry it maps (null when the entry had no unit).

If no good match exists in the canonical registry, use the original name as-is."""


# ── LLM config resolution ─────────────────────────────────────────────────────
# Config is resolved in this order:
#   1. an explicit `config` dict passed to the function (background jobs pass
#      a snapshot captured at submit time — see llm_jobs.py),
#   2. a thread-local job config set by llm_jobs._run_job (so nested calls
#      inside a worker don't need the dict threaded through every signature),
#   3. Streamlit session state (the normal in-app path),
#   4. config.py env-var defaults.

_job_config_var = threading.local()


def set_job_config(cfg: dict | None):
    """Set the thread-local LLM config for the current worker thread."""
    _job_config_var.config = cfg


def _resolve_config(explicit: dict | None) -> dict:
    if explicit:
        return explicit
    cfg = getattr(_job_config_var, "config", None)
    if cfg is not None:
        # A worker's config snapshot (possibly an empty dict → fall back to
        # env-var defaults downstream in _build_client). `is not None` matters:
        # a falsy empty dict must NOT fall through to session state.
        return cfg
    import streamlit as st
    if hasattr(st, "session_state"):
        return {
            "base_url": st.session_state.get("llm_base_url"),
            "api_key": st.session_state.get("llm_api_key"),
            "model": st.session_state.get("llm_model"),
            "timeout": st.session_state.get("llm_timeout"),
        }
    return {}


def _build_client(config: dict | None = None) -> OpenAI:
    cfg = _resolve_config(config)
    base_url = cfg.get("base_url") or get_unsloth_base_url()
    api_key = cfg.get("api_key") or get_unsloth_api_key()
    # OpenAI SDK rejects empty API keys; use a placeholder for local servers
    if not api_key:
        api_key = "not-needed"
    return OpenAI(base_url=base_url, api_key=api_key)


def _get_model(config: dict | None = None) -> str:
    cfg = _resolve_config(config)
    return cfg.get("model") or get_unsloth_model()


def _get_timeout(config: dict | None = None) -> int:
    cfg = _resolve_config(config)
    return cfg.get("timeout") or get_llm_timeout()


def extract_pages_batch(llm: OpenAI, page_images: list[Image.Image] | list[str], dpi: int = 100, quality: int = 75) -> list[ExtractedPage]:
    """Extract data from multiple pages in a single LLM call.

    Accepts either PIL Images or pre-encoded base64 strings. When base64 strings
    are passed (from pdf_to_base64_jpeg), skips the JPEG encoding step entirely.

    Raises:
        LLMResponseError: if the response is unparseable, OR if the returned
            ``page_index`` set does not exactly cover 0..len(page_images)-1
            (R3-12: a short response must surface as an error instead of
            silently dropping pages).
    """
    import threading
    content_parts = [{"type": "text", "text": "Extract all lab data from each page."}]

    # Convert images to base64 only if not already encoded
    b64_list = []
    for item in page_images:
        if isinstance(item, str):
            b64_list.append(item)
        else:
            b64_list.append(image_to_base64_jpeg(item, quality=quality))

    for b64 in b64_list:
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    with _llm_semaphore:
        resp = llm.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT_BATCH},
                {"role": "user", "content": content_parts},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=_abort_timeout() or _get_timeout(),
        )
        raw = _strip_markdown_json(resp.choices[0].message.content or "")
    try:
        data = json.loads(raw)
        pages = data.get("pages", [])
        results = []
        for page_data in pages:
            results.append(ExtractedPage(**page_data))
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"LLM returned invalid JSON for batch page extraction: {e}\nResponse snippet: {raw[:200]}")
    except Exception as e:
        raise LLMResponseError(f"LLM response parsing failed for batch page extraction: {e}\nResponse snippet: {raw[:200]}")

    # R3-12: verify every requested page came back EXACTLY once. The prompt
    # instructs the LLM to include "page_index" (0-based) per page; a short,
    # duplicated or mismatched response must surface as an error instead of
    # silently dropping/misaligning pages. A set-based check would let
    # duplicates through (e.g. [0,1,1] for 3 pages → one page's data lost), so
    # we require the returned indices to be exactly 0..N-1 with no repeats.
    expected = list(range(len(page_images)))
    indices = [p.page_index for p in results]
    if all(idx is None for idx in indices):
        # No page_index at all: only accept when the count matches exactly,
        # trusting positional order. Anything else is unverifiable → error.
        if len(results) != len(page_images):
            raise LLMResponseError(
                f"Batch page extraction incomplete: requested {len(page_images)} pages, "
                f"got {len(results)} (no page_index provided to verify). "
                f"Refusing to continue with partial data — retry or use batch size 1."
            )
        return results
    if any(idx is None for idx in indices):
        # Mixed list: some entries carry page_index, some don't — the response
        # cannot be verified positionally, and sorted() would raise TypeError
        # comparing None with int.
        raise LLMResponseError(
            f"Batch page extraction returned page_index for only some pages "
            f"({indices!r}). Refusing to continue with partial data — retry or "
            f"use batch size 1."
        )
    if len(results) != len(page_images) or sorted(indices) != expected:
        returned = sorted(i for i in indices if i is not None)
        missing = sorted(set(expected) - set(returned))
        extra = sorted(set(returned) - set(expected))
        dupes = sorted({i for i in returned if returned.count(i) > 1})
        raise LLMResponseError(
            f"Batch page extraction incomplete: requested {len(page_images)} pages, "
            f"got indices {returned} (missing: {missing}, unexpected: {extra}, "
            f"duplicated: {dupes}). "
            f"Refusing to continue with partial data — retry or use batch size 1."
        )
    # Correct order for downstream positional consumers (extractor maps results
    # back to pages by index).
    results.sort(key=lambda p: p.page_index)
    return results


def extract_page(llm: OpenAI, page_image: Image.Image | str, quality: int = 75) -> ExtractedPage:
    """Extract data from a single page.

    Accepts either a PIL Image or a pre-encoded base64 string.
    """
    if isinstance(page_image, str):
        b64 = page_image
    else:
        b64 = image_to_base64_jpeg(page_image, quality=quality)

    with _llm_semaphore:
        resp = llm.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all lab data from this page."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=_abort_timeout() or _get_timeout(),
        )
        raw = _strip_markdown_json(resp.choices[0].message.content or "")
    try:
        return ExtractedPage(**json.loads(raw))
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"LLM returned invalid JSON for page extraction: {e}\nResponse snippet: {raw[:200]}")
    except Exception as e:
        raise LLMResponseError(f"LLM response parsing failed for page extraction: {e}\nResponse snippet: {raw[:200]}")


def normalize_names(llm: OpenAI, extracted_names: list[dict] | list[str], canonical_list: list[str] | None = None, registry: dict | None = None) -> NormalizationMapping:
    """Ask the LLM to map extracted biomarker entries to canonical names.

    ``extracted_names`` is a deduplicated list of entries, each either a plain
    name string (unit unknown → treated as None) or a dict
    ``{"name": str, "unit": str | None}``. The LLM sees the units so it can
    disambiguate unit-qualified canonical variants (e.g. "HbA1c (%)" vs
    "HbA1c (mmol/mol)"). Each returned mapping echoes the unit of the entry it
    maps, so callers can match mappings back to entries per unit.
    """
    # Normalize plain strings to dicts (backward compatibility)
    entries = [
        {"name": e, "unit": None} if isinstance(e, str) else e
        for e in extracted_names
    ]
    # Build a compact registry string for the LLM with aliases + units
    registry_lines = []
    # R3-09: fall back to the keys of _CANONICAL_BIOMARKERS — the bare
    # CANONICAL_BIOMARKERS list is not imported here (would be a NameError).
    # ``registry`` (active DB registry) takes precedence over the static
    # config baseline so accepted proposals are honored.
    active_registry = registry if registry is not None else _CANONICAL_BIOMARKERS
    for name in canonical_list or list(active_registry.keys()):
        info = active_registry.get(name, {})
        aliases = info.get("aliases", [])
        units = info.get("units", [])
        parts = [f"{name}"]
        if aliases:
            parts.append(f"aliases: {', '.join(aliases[:10])}")  # cap at 10 to avoid bloating prompt
        if units:
            parts.append(f"units: {', '.join(units)}")
        registry_lines.append(" | ".join(parts))
    registry_str = "\n".join(registry_lines)

    prompt = (
        f"Extracted names (with unit where known): {json.dumps(entries)}\n\n"
        f"Canonical registry (name | aliases | units):\n{registry_str}\n\n"
        "Map each extracted entry to the closest canonical name. "
        "Echo the unit of each entry in its mapping."
    )
    with _llm_semaphore:
        resp = llm.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": NORMALIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=_abort_timeout() or _get_timeout(),
        )
        raw = _strip_markdown_json(resp.choices[0].message.content or "")
    try:
        return NormalizationMapping(**json.loads(raw))
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"LLM returned invalid JSON for name normalization: {e}\nResponse snippet: {raw[:200]}")
    except Exception as e:
        raise LLMResponseError(f"LLM response parsing failed for name normalization: {e}\nResponse snippet: {raw[:200]}")


# ── Data quality check (split into per-biomarker + duplicate phases) ─────────
#
# The former single call over the whole registry + every measurement was split
# into two phases so each LLM request stays small and focused:
#   Phase 1 — run_biomarker_quality_check(): ONE call per distinct biomarker,
#     carrying that biomarker's measurements (registry as context). It can only
#     return data_point findings (ref_range / unit / value / missing).
#   Phase 2 — run_duplicate_check(): ONE call over a compact list of all the
#     patient's biomarkers + units. It can only return registry findings about
#     duplicates (merge / name).
# Every finding carries a machine-applicable fix (action + params) that the user
# accepts per-finding in the UI; apply_data_quality_finding() in database.py
# executes it. The job (llm_jobs._job_data_quality_check) runs Phase 1
# sequentially per biomarker, then Phase 2, and maps the LLM's 1-based
# measurement numbers to real DataPoint ids before persisting.

BIOMARKER_QUALITY_SYSTEM_PROMPT_DE = """Du bist ein medizinischer Daten-Qualitätsexperte. Du bekommst (1) den Registry-Eintrag für EINEN Biomarker (übliche Einheiten + Standard-Referenzbereich) und (2) ALLE Messwerte dieses Biomarkers für einen Patienten. Prüfe die DATENQUALITÄT dieses Biomarkers — also ob die gespeicherten Werte, Einheiten und Referenzbereiche plausibel und konsistent sind.

Prüfe insbesondere:
- Referenzbereiche pro Messwert: Ist der gespeicherte Bereich (ref_low–ref_high) für den Wert und die Einheit plausibel? Fehlt er komplett? Stimmt die Einheit des Bereichs mit der Einheit des Werts überein?
- Einheiten: Passt die Einheit eines Werts zu dem, was für diesen Biomarker üblich ist (siehe Registry-Eintrag)? Gibt es widersprüchliche Einheiten zwischen den Messwerten desselben Biomarkers?
- Werte: Ist ein Messwert physiologisch unmöglich oder offensichtlich fehlerhaft (z.B. Faktor 10/100 falsch, Negativ bei nicht-negativem Marker)?

Gib eine Liste von Befunden ("findings") zurück — EINER pro konkretes Problem. Jedes Befund hat die Form:
{"category": "ref_range"|"unit"|"value"|"missing", "target_type": "data_point", "biomarker": "<Name>", "description": "...", "action": "...", "params": {...}, "data_points": [1,2]}
- "category": Art des Problems (ref_range=Referenzbereich, unit=Einheit, value=unplausibler Messwert, missing=fehlende Daten)
- "target_type": immer "data_point" — die Befunde betreffen konkrete Messwerte dieses Biomarkers
- "biomarker": der Name des geprüften Biomarkers
- "description": kurze Erklärung auf Deutsch (max 2 Sätze), WAS das Problem ist und WARUM
- "action" + "params": der automatisch anwendbare Fix:
  - "set_ref_range": params {"low": Zahl|null, "high": Zahl|null, "unit": "..."} — korrigiert die gespeicherten Referenzgrenzen der genannten Messwerte (Wert bleibt unverändert)
  - "set_unit": params {"unit": "..."} — korrigiert die Einheit der genannten Messwerte; gespeicherter Wert und ggf. Referenzbereich werden umgerechnet. Vorschlage dies NUR, wenn der UMGERECHNETE Wert für diesen Biomarker physiologisch plausibel ist (vergleiche mit dem Standard-Referenzbereich aus dem Registry-Eintrag) — ergibt die Umrechnung keinen plausiblen Wert, nutze stattdessen "disable_value". Schreibe-Varianten derselben Einheit (z.B. "/nl" vs. "10^3/µl", "Tsd/µl" vs. "10³/µl", "mU/l" vs. "mIU/l") sind immer sicher zu normalisieren — der Zahlenwert bleibt dabei unverändert
  - "disable_value": params {} — markiert die genannten Messwerte als ungültig/ausgeblendet
- "data_points": die Nummern (#1, #2, ...) der betroffenen Messwerte aus der Eingabeliste

Gib NUR Befunde für Probleme, die du mit hoher Sicherheit benennen und beheben kannst. Wenn alles in Ordnung ist, gib {"findings": []} zurück.
Gib NUR ein gültiges JSON zurück, keine Erklärungen:
{"findings": []}"""

BIOMARKER_QUALITY_SYSTEM_PROMPT_EN = """You are a medical data quality expert. You are given (1) the registry entry for ONE biomarker (typical units + default reference range) and (2) ALL measurements of that biomarker for one patient. Check the DATA QUALITY of this biomarker — i.e. whether the stored values, units and reference ranges are plausible and consistent.

Check in particular:
- Per-measurement reference ranges: is the stored range (ref_low–ref_high) plausible for the value and its unit? Is it missing entirely? Does the range's unit match the value's unit?
- Units: does a value's unit match what is typical for that biomarker (see registry entry)? Are there conflicting units among this biomarker's measurements?
- Values: is a measurement physiologically impossible or obviously wrong (e.g. off by a factor of 10/100, negative for a non-negative marker)?

Return a list of findings ("findings") — ONE per concrete problem. Each finding has the form:
{"category": "ref_range"|"unit"|"value"|"missing", "target_type": "data_point", "biomarker": "<name>", "description": "...", "action": "...", "params": {...}, "data_points": [1,2]}
- "category": kind of problem (ref_range=reference range, unit=unit, value=implausible measurement, missing=missing data)
- "target_type": always "data_point" — the findings concern specific measurements of this biomarker
- "biomarker": the name of the biomarker being checked
- "description": short explanation in English (max 2 sentences) of WHAT the problem is and WHY
- "action" + "params": the automatically applicable fix:
  - "set_ref_range": params {"low": number|null, "high": number|null, "unit": "..."} — corrects the stored reference bounds of the named measurements (the value itself is unchanged)
  - "set_unit": params {"unit": "..."} — corrects the unit of the named measurements; the stored value and reference bounds (if any) are converted. Only propose this if the CONVERTED value is physiologically plausible for this biomarker (compare against the default reference range from the registry entry) — if the conversion does not yield a plausible value, use "disable_value" instead. Notation variants of the same unit (e.g. "/nl" vs. "10^3/µl", "Tsd/µl" vs. "10³/µl", "mU/l" vs. "mIU/l") are always safe to normalize — the numeric value stays unchanged
  - "disable_value": params {} — marks the named measurements as invalid/hidden
- "data_points": the numbers (#1, #2, ...) of the affected measurements from the input list

Only report problems you can name and fix with high confidence. If everything is fine, return {"findings": []}.
Return ONLY valid JSON, no explanations:
{"findings": []}"""

DUPLICATE_CHECK_SYSTEM_PROMPT_DE = """Du bist ein medizinischer Terminologie-Assistent. Dir wird die Liste der Biomarker gegeben, für die ein Patient tatsächlich Messwerte hat — jeder Eintrag mit Name und den gefundenen Einheiten. Prüfe NUR auf Duplikate: Tippfehler-Varianten oder alternative Schreibweisen desselben Biomarkers (z.B. "CRP" vs "C-reaktives Protein", "HbA1c" vs "HBA1c").

Gib eine Liste von Befunden ("findings") zurück — EINER pro redundanter Eintrag. Jedes Befund hat die Form:
{"category": "merge"|"name", "target_type": "registry", "biomarker": "<redundanter Name>", "description": "...", "action": "...", "params": {...}}
- "category": "merge" wenn der Eintrag in einen bestehenden Zielnamen zusammengeführt werden soll, "name" wenn nur der Name korrigiert werden soll
- "target_type": immer "registry" — die Befunde betreffen die Biomarker-Namen/Registry
- "biomarker": der redundante bzw. falsch benannte Eintrag (exakt wie in der Liste)
- "description": kurze Erklärung auf Deutsch (max 2 Sätze), WARUM es ein Duplikat ist
- "action" + "params":
  - "merge": params {"target": "Zielname"} — redundanten Eintrag in Ziel zusammenführen (Ziel muss existieren)
  - "registry_name": params {"proposed": "korrigierter Name"}

Gib NUR Befunde für Duplikate, die du mit hoher Sicherheit benennen kannst. Wenn es keine Duplikate gibt, gib {"findings": []} zurück.
Gib NUR ein gültiges JSON zurück, keine Erklärungen:
{"findings": []}"""

DUPLICATE_CHECK_SYSTEM_PROMPT_EN = """You are a medical terminology assistant. You are given the list of biomarkers for which a patient actually has measurements — each entry with its name and the units found. Check ONLY for duplicates: typo variants or alternative spellings of the same biomarker (e.g. "CRP" vs "C-reactive protein", "HbA1c" vs "HBA1c").

Return a list of findings ("findings") — ONE per redundant entry. Each finding has the form:
{"category": "merge"|"name", "target_type": "registry", "biomarker": "<redundant name>", "description": "...", "action": "...", "params": {...}}
- "category": "merge" if the entry should be folded into an existing target name, "name" if only the name needs correcting
- "target_type": always "registry" — the findings concern biomarker names/the registry
- "biomarker": the redundant or misspelled entry (exactly as in the list)
- "description": short explanation in English (max 2 sentences) of WHY it is a duplicate
- "action" + "params":
  - "merge": params {"target": "target name"} — fold the redundant entry into the target (target must exist)
  - "registry_name": params {"proposed": "corrected name"}

Only report duplicates you can name with high confidence. If there are no duplicates, return {"findings": []}.
Return ONLY valid JSON, no explanations:
{"findings": []}"""


# Categories a finding may carry (kept in sync with the UI grouping).
DATA_QUALITY_CATEGORIES = ("ref_range", "unit", "name", "merge", "value", "missing")
_DATA_POINT_ACTIONS = ("set_ref_range", "set_unit", "disable_value")
_REGISTRY_ACTIONS = ("registry_ref_range", "registry_unit", "registry_name", "merge")


def _parse_findings(raw, allowed_categories=None, allowed_target_types=None) -> list[dict]:
    """Normalize the 'findings' field of a data-quality response.

    Each finding is {'category', 'target_type', 'biomarker', 'description',
    'action', 'params', 'data_points'}. Malformed entries are dropped silently —
    findings are advisory and must never break the response contract.

    ``allowed_categories`` / ``allowed_target_types`` optionally restrict which
    categories/target types a phase may return (the per-biomarker call only
    yields data_point findings, the duplicate check only registry findings).
    """
    if not isinstance(raw, list):
        return []
    out = []
    for f in raw:
        if not isinstance(f, dict):
            continue
        category = str(f.get("category", "")).strip()
        target_type = str(f.get("target_type", "")).strip()
        biomarker = str(f.get("biomarker", "")).strip()
        description = str(f.get("description", "")).strip()
        action = str(f.get("action", "")).strip()
        if category not in DATA_QUALITY_CATEGORIES or target_type not in ("registry", "data_point"):
            continue
        if allowed_categories is not None and category not in allowed_categories:
            continue
        if allowed_target_types is not None and target_type not in allowed_target_types:
            continue
        if not biomarker or not description:
            continue

        params_raw = f.get("params")
        params = params_raw if isinstance(params_raw, dict) else {}

        data_points = None
        if target_type == "data_point":
            if action not in _DATA_POINT_ACTIONS:
                continue
            raw_ids = f.get("data_points")
            ids = []
            if isinstance(raw_ids, list):
                for v in raw_ids:
                    try:
                        ids.append(int(v))
                    except (TypeError, ValueError):
                        continue
            if not ids:
                continue
            if action == "set_ref_range":
                # Coerce the bounds to float NOW: database._apply_data_point_fix
                # does float(params["low"]) when the user accepts, so a
                # non-numeric value ("n/a", "", "low") from the LLM would crash
                # the Accept button. Drop the finding if a bound is not numeric.
                coerced = {}
                bad = False
                for key in ("low", "high"):
                    val = params.get(key)
                    if val is None:
                        coerced[key] = None
                        continue
                    try:
                        coerced[key] = float(val)
                    except (TypeError, ValueError):
                        bad = True
                        break
                if bad or (coerced["low"] is None and coerced["high"] is None):
                    continue
                params = {**params, "low": coerced["low"], "high": coerced["high"]}
            if action == "set_unit" and not str(params.get("unit", "")).strip():
                continue
            data_points = ids
        else:  # registry
            if action not in _REGISTRY_ACTIONS:
                continue
            if action == "merge":
                if not str(params.get("target", "")).strip():
                    continue
            elif not str(params.get("proposed", "")).strip():
                continue

        out.append({
            "category": category,
            "target_type": target_type,
            "biomarker": biomarker,
            "description": description,
            "action": action,
            "params": params,
            "data_points": data_points,
        })
    return out


def _dq_call(llm: OpenAI, system_prompt: str, prompt: str, what: str,
             allowed_categories=None, allowed_target_types=None) -> dict:
    """Shared LLM call + JSON contract for the data-quality phases.

    Returns {'findings': [...]} (parsed via _parse_findings, optionally
    restricted to the phase's categories/target types). Raises
    LLMResponseError on empty/invalid responses — callers must surface that as
    an error, NOT as "no issues found".
    """
    raw = ""
    try:
        with _llm_semaphore:
            resp = llm.chat.completions.create(
                model=_get_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=_abort_timeout() or _get_timeout(),
            )
        raw = resp.choices[0].message.content or ""
        raw = _strip_markdown_json(raw).strip()

        if not raw:
            raise LLMResponseError(f"LLM returned an empty response for {what}")

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise LLMResponseError(
                f"LLM returned invalid JSON for {what}: expected object, got {type(data).__name__}\n"
                f"Response snippet: {raw[:200]}"
            )

        return {"findings": _parse_findings(
            data.get("findings"),
            allowed_categories=allowed_categories,
            allowed_target_types=allowed_target_types,
        )}
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise LLMResponseError(
            f"LLM returned invalid JSON for {what}: {e}\n"
            f"Response snippet: {raw[:200]}"
        ) from e
    except LLMResponseError:
        raise
    except Exception as e:
        raise LLMResponseError(f"{what} failed: {e}") from e


def run_biomarker_quality_check(llm: OpenAI, registry: dict, biomarker_name: str,
                                data_points: list[dict], lang: str | None = None) -> dict:
    """Phase 1: check ONE biomarker's measurements against the registry.

    The LLM sees ONLY this biomarker's registry entry (units + default ref
    range) plus its data points (numbered #1..#n in the order given). Sending
    just the single relevant entry — not the whole ~90-entry registry — keeps
    each call small enough for small models to finish well within the timeout.

    Args:
        llm: OpenAI client instance
        registry: active registry dict (get_registry shape); only the entry for
            ``biomarker_name`` is used as context.
        biomarker_name: canonical name of the biomarker being checked.
        data_points: ordered list of dicts with keys id, name, value, unit,
            ref_low, ref_high, report_date. The 1-based position in this list is
            the "#n" token the LLM references in findings' 'data_points'.
        lang: 'de' or 'en'; defaults to the app language via _get_lang().

    Returns:
        {'findings': [...]}. Only data_point findings (categories ref_range /
        unit / value / missing); 'data_points' holds 1-based indices into the
        input list.

    Raises:
        LLMResponseError: if the LLM call fails or returns unparseable content.
            Callers must surface this as an error, NOT as "no issues found".
    """
    # Only the single registry entry for THIS biomarker is relevant context —
    # sending the whole registry in every per-biomarker call was what made each
    # request too large/slow for small models.
    info = registry.get(biomarker_name) or {}
    units = ", ".join(info.get("units", [])) or "(none)"
    ref = info.get("default_ref") or {}
    if ref and (ref.get("low") is not None or ref.get("high") is not None):
        low, high, r_unit = ref.get("low"), ref.get("high"), ref.get("unit", "")
        reg_line = f"- {biomarker_name} | units: {units} | ref range: {low}–{high} {r_unit}".strip()
    else:
        reg_line = f"- {biomarker_name} | units: {units}"

    dp_lines = []
    for i, dp in enumerate(data_points, start=1):
        value = dp.get("value")
        unit = (dp.get("unit") or "").strip()
        ref_low, ref_high = dp.get("ref_low"), dp.get("ref_high")
        ref_txt = "(none)"
        if ref_low is not None or ref_high is not None:
            ref_txt = f"{ref_low}–{ref_high} {unit}".strip()
        date = dp.get("report_date") or ""
        dp_lines.append(
            f"#{i}  {dp.get('name')} | value {value} {unit} | ref {ref_txt} | report {date}".strip()
        )

    if lang is None:
        lang = _get_lang()
    if lang == "de":
        system_prompt = BIOMARKER_QUALITY_SYSTEM_PROMPT_DE
        reg_header = "REGISTRY-EINTRAG FÜR DIESEN BIOMARKER:\n"
        dp_header = f"MESSWERTE FÜR {biomarker_name}:\n"
    else:
        system_prompt = BIOMARKER_QUALITY_SYSTEM_PROMPT_EN
        reg_header = "REGISTRY ENTRY FOR THIS BIOMARKER:\n"
        dp_header = f"MEASUREMENTS FOR {biomarker_name}:\n"

    prompt = (
        f"{reg_header}\n{reg_line}\n"
        + f"\n\n{dp_header}\n({len(data_points)} measurements)\n"
        + ("\n".join(dp_lines) if dp_lines else "(none)")
        + '\n\nReturn ONLY valid JSON:\n'
          '{"findings": []}'
    )

    return _dq_call(
        llm, system_prompt, prompt, "biomarker quality check",
        allowed_categories=("ref_range", "unit", "value", "missing"),
        allowed_target_types=("data_point",),
    )


def run_duplicate_check(llm: OpenAI, biomarker_entries: list[tuple[str, str]],
                        registry: dict | None = None, lang: str | None = None) -> dict:
    """Phase 2: check the patient's biomarker names for duplicates.

    The LLM sees a compact list of every distinct biomarker the patient has
    measurements for (name + units found), plus the canonical registry names so
    merge targets can be chosen from real entries. It may only return registry
    findings about duplicates (merge / name).

    Args:
        llm: OpenAI client instance
        biomarker_entries: ordered list of (canonical_name, unit) pairs — one
            per distinct measurement of the patient.
        registry: active registry dict (get_registry shape); its names are
            listed so merge targets must be real entries. Optional.
        lang: 'de' or 'en'; defaults to the app language via _get_lang().

    Returns:
        {'findings': [...]}. Only registry findings (categories merge / name).

    Raises:
        LLMResponseError: if the LLM call fails or returns unparseable content.
            Callers must surface this as an error, NOT as "no issues found".
    """
    units_by_name: dict[str, list[str]] = {}
    for name, unit in biomarker_entries:
        u = (unit or "").strip()
        if not name:
            continue
        lst = units_by_name.setdefault(name, [])
        if u and u not in lst:
            lst.append(u)

    lines = [
        f"- {name} | units: {', '.join(units_by_name[name]) or '(none)'}"
        for name in sorted(units_by_name)
    ]

    if lang is None:
        lang = _get_lang()
    if lang == "de":
        system_prompt = DUPLICATE_CHECK_SYSTEM_PROMPT_DE
        header = "BIOMARKER DES PATIENTEN:\n"
        reg_header = "KANONISCHE BIOMARKER-NAMEN (Registry):\n"
    else:
        system_prompt = DUPLICATE_CHECK_SYSTEM_PROMPT_EN
        header = "PATIENT'S BIOMARKERS:\n"
        reg_header = "CANONICAL BIOMARKER NAMES (registry):\n"

    prompt = (
        f"{header}\n({len(lines)} entries)\n"
        + ("\n".join(lines) if lines else "(none)")
    )
    if registry:
        prompt += (
            f"\n\n{reg_header}\n({len(registry)} entries)\n"
            + "\n".join(f"- {name}" for name in sorted(registry))
        )
    prompt += '\n\nReturn ONLY valid JSON:\n' '{"findings": []}'

    return _dq_call(
        llm, system_prompt, prompt, "duplicate check",
        allowed_categories=("merge", "name"),
        allowed_target_types=("registry",),
    )


# ── Merge-target suggestion (single LLM call over all variant groups) ────────
#
# Replaces the former per-group LLM calls in database.auto_merge_biomarkers():
# one request carries ALL of a patient's biomarker variant groups plus the
# canonical list, and the LLM answers with one plan entry per group.

MERGE_TARGETS_SYSTEM_PROMPT_DE = """Du bist ein medizinischer Terminologie-Assistent. Dir werden Varianten-Gruppen von Biomarkern gegeben, die aus Laborberichten extrahiert wurden — jede Gruppe enthält Namensvarianten desselben Biomarkers (mit den gefundenen Einheiten) sowie eine Liste kanonischer Biomarkernamen.

Für JEDE Gruppe wähle:
- "target": den besten kanonischen Namen aus der kanonischen Liste, der auf ALLE Varianten der Gruppe zutrifft (exakt so geschrieben wie in der Liste).
- "unit": die beste Einheit aus den für diese Gruppe gefundenen Einheiten.

Gib NUR ein gültiges JSON zurück, ohne Markdown oder Erklärungen:
{"plans": [{"group_index": 0, "target": "...", "unit": "..."}]}

"group_index" ist die 0-basierte Position der Gruppe in der Eingabeliste. Gib genau einen Eintrag pro Gruppe zurück."""

MERGE_TARGETS_SYSTEM_PROMPT_EN = """You are a medical terminology assistant. You will be given variant groups of biomarkers extracted from lab reports — each group contains name variants of the same biomarker (with the units found) plus a list of canonical biomarker names.

For EACH group pick:
- "target": the best canonical name from the canonical list that fits ALL variants of that group (exactly as written in the list).
- "unit": the best unit among the units found for that group.

Return ONLY valid JSON, no markdown or explanations:
{"plans": [{"group_index": 0, "target": "...", "unit": "..."}]}

"group_index" is the 0-based position of the group in the input list. Return exactly one entry per group."""


def _build_merge_targets_prompt(groups: list[dict], canonical_names: list[str], lang: str) -> str:
    """Build the user prompt for suggest_merge_targets()."""
    if lang == "de":
        header = "Die folgenden Varianten-Gruppen wurden für denselben Patienten extrahiert:\n\n"
        label_variants = "Varianten"
        label_units = "Einheiten"
        canonical_header = "Kanonicaliste:"
        instruction = ("Wähle für jede Gruppe den besten kanonischen Namen und die beste Einheit. "
                       "Gib NUR das JSON zurück.")
    else:
        header = "The following variant groups were extracted for the same patient:\n\n"
        label_variants = "Variants"
        label_units = "Units"
        canonical_header = "Canonical list:"
        instruction = ("For each group pick the best canonical target name and unit. "
                       "Return ONLY the JSON.")

    parts = [header]
    for i, group in enumerate(groups):
        variant_descs = []
        for v in group["variants"]:
            if v.get("unit"):
                variant_descs.append(f"{v['name']} ({v['unit']})")
            else:
                variant_descs.append(v["name"])
        units = list({v["unit"] for v in group["variants"] if v.get("unit")})
        parts.append(
            f"Group {i}:\n"
            f"- {label_variants}: {json.dumps(variant_descs, ensure_ascii=False)}\n"
            f"- {label_units}: {json.dumps(units, ensure_ascii=False)}"
        )
    parts.append(f"\n{canonical_header} {json.dumps(canonical_names, ensure_ascii=False)}")
    parts.append(f"\n{instruction}")
    return "\n\n".join(parts)


def suggest_merge_targets(llm: OpenAI, groups: list[dict], canonical_names: list[str], lang: str | None = None) -> list:
    """Suggest a canonical merge target (name + unit) for ALL variant groups in one LLM call.

    Args:
        llm: OpenAI-compatible client.
        groups: List of group dicts as returned by
            database.get_unique_biomarker_groups() — each has "variants":
            [{"name": str, "unit": str|None, "count": int}, ...].
        canonical_names: Canonical biomarker names the LLM may pick from.
        lang: 'de' or 'en'; defaults to the app language via _get_lang().

    Returns a list aligned to ``groups``: for each group either
    {"target": str, "unit": str} or None when the LLM omitted that group or
    returned an unusable entry (e.g. target not in canonical_names). Per-group
    gaps never raise — callers fall back to their own heuristics.

    Raises:
        LLMResponseError: if the whole LLM call fails or the response is
            unusable (unparseable JSON / wrong shape).
    """
    if lang is None:
        lang = _get_lang()
    system_prompt = MERGE_TARGETS_SYSTEM_PROMPT_DE if lang == "de" else MERGE_TARGETS_SYSTEM_PROMPT_EN
    prompt = _build_merge_targets_prompt(groups, canonical_names, lang)

    raw = ""
    try:
        with _llm_semaphore:
            resp = llm.chat.completions.create(
                model=_get_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=_abort_timeout() or _get_timeout(),
            )
        raw = resp.choices[0].message.content or ""
        raw = _strip_markdown_json(raw).strip()

        if not raw:
            raise LLMResponseError("LLM returned an empty response for merge target suggestion")

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise LLMResponseError(
                f"LLM returned invalid JSON for merge target suggestion: expected object, got {type(data).__name__}\n"
                f"Response snippet: {raw[:200]}"
            )

        plans_raw = data.get("plans")
        if not isinstance(plans_raw, list):
            raise LLMResponseError(
                f"LLM returned invalid JSON for merge target suggestion: 'plans' must be a list\n"
                f"Response snippet: {raw[:200]}"
            )

        # Align plans to the input groups; per-group gaps become None.
        results: list = [None] * len(groups)
        for entry in plans_raw:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("group_index")
            if isinstance(idx, bool) or not isinstance(idx, int):
                try:
                    idx = int(str(idx).strip())
                except (ValueError, TypeError):
                    continue
            if idx < 0 or idx >= len(groups) or results[idx] is not None:
                continue
            target = str(entry.get("target", "")).strip()
            unit = str(entry.get("unit", "")).strip()
            # Unknown targets are unusable — treat the group as omitted.
            if not target or target not in canonical_names:
                continue
            results[idx] = {"target": target, "unit": unit}
        return results
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise LLMResponseError(
            f"LLM returned invalid JSON for merge target suggestion: {e}\n"
            f"Response snippet: {raw[:200]}"
        ) from e
    except LLMResponseError:
        raise
    except Exception as e:
        raise LLMResponseError(f"Merge target suggestion failed: {e}") from e


PATIENT_SUMMARY_SYSTEM_PROMPT_DE = """Du bist ein erfahrener Arzt, der die Gesundheitszusammenfassung eines Patienten erstellst.

Deine Aufgabe: Erstelle eine strukturierte, zusammenhängende Beurteilung des GESAMTEN Gesundheitszustands auf Basis der Laborbefunde. Kreuzprüfe die Marker miteinander: Ein einzelner Wert ist selten aussagekräftig — die Kombination mehrerer Marker entscheidet.

EINGABEDATEN:
- Patientenprofil (Alter, Geschlecht, Vorerkrankungen/Medikamente) — soweit vorhanden. Referenzbereiche sind alters- und geschlechtsabhängig; ordne Werte entsprechend ein (z.B. Hb, Kreatinin, eGFR, Ferritin). Ein bereitgestellter klinischer Kontext (z.B. "L-Thyroxin") verändert die Interpretation deutlich — beziehe ihn ein.
- Pro Biomarker ein vorberechneter Verlauf: Anzahl der Messungen, letzter Wert mit Datum, Abweichung vom Referenzbereich, Trend (erste→letzte Messung), Anzahl der Messungen außerhalb des Referenzbereichs. Nutze diese Signale, statt selbst zu schätzen.
- "Vom System erkannte Muster": Hinweise eines Regelwerks, KEINE unfehlbare Diagnose. Prüfe jedes Muster an den Rohdaten: Stützen die Werte das Muster (gleicher Bericht, plausible Werte)? Wenn nicht, erwähne es kritisch statt es blind zu übernehmen. Identifiziere AKTIV weitere Korrelationen, die das Regelwerk nicht nennt (z.B. MCV+Hb+Ferritin für Anämie-Ursache, Neutrophile/Lymphozyten, Albumin+Kreatinin, Eisen+Ferritin+CRP, GGT+ALP+Bilirubin).
- Risikomarkierungen mit Dringlichkeit (critical = aktueller Wert akut problematisch, long_term = Beobachtung).

WICHTIGE REGELN:
- Stütze jede Aussage strikt auf die vorliegenden Messwerte: Erfinde oder unterstelle keine Werte, Trends, Diagnosen oder Zustände, die nicht durch die gegebenen Daten belegt sind. Formuliere Verdachtsdiagnosen als Möglichkeit ("passt zu", "vereinbar mit"), nie als Gewissheit. Liegen zu wenige Messungen für eine belastbare Trendaussage vor, sage dies ausdrücklich statt eines Trends zu inferieren.
- Sicherheitsregel: Wenn der NEUESTE Wert eines Markers akut kritisch ist (weit außerhalb des Referenzbereichs oder als critical markiert), stelle dies an den Anfang und empfehle eine zeitnahe ärztliche Abklärung. Beurteile Dringlichkeit nach dem neuesten Wert, nicht nach historischen Extremen.
- Betone besorgniserregende Tendenzen, aber nenne ebenso Beruhigendes: welche Systeme sind unauffällig oder gebessert? Eine Gesamtbilanz ohne die gesunden Befunde ist irreführend.
- Keine Panikmache, aber klar über Risiken aufklären.
- Schreibe auf Deutsch in professioneller, aber für Laien verständlicher Sprache.
- Empfehlungen sind patientenorientiert ("Bitte lassen Sie ... bei der nächsten Arztvisite prüfen", "Besprechen Sie mit Ihrem Arzt ..."), niemals Dosierungs- oder Verschreibungsempfehlungen.

ANTWORTFORMAT (Markdown, gesamt 150-300 Wörter):
## Gesamtbild
1-2 Sätze: der wichtigste Gesamteindruck des Gesundheitszustands.
## Systembefunde
Ein Absatz pro Organsystem/Markergruppe mit Daten (z.B. Blutbild, Schilddrüse, Stoffwechsel, Lipide, Niere, Leber, Elektrolyte, Entzündung) — nur Systeme mit Daten. Je: Status (unauffällig / auffällig / gebessert / verschlechtert) + belegende Werte + ggf. Kreuzkorrelation.
## Trends & Verlauf
Nur belegbare Trends; bei dünner Datenlage explizit sagen.
## Empfehlungen
1-3 konkrete nächste Schritte (arztorientiert)."""

PATIENT_SUMMARY_SYSTEM_PROMPT_EN = """You are an experienced physician creating a health summary for a patient.

Your task: Create a structured, coherent assessment of the OVERALL health status based on the lab results. Cross-examine the markers against each other: a single value is rarely meaningful — the combination of markers decides.

INPUT DATA:
- Patient profile (age, sex, conditions/medication) where available. Reference ranges depend on age and sex; interpret values accordingly (e.g. Hb, creatinine, eGFR, ferritin). Provided clinical context (e.g. "levothyroxine") changes interpretation substantially — use it.
- Per biomarker a precomputed course: measurement count, latest value with date, deviation from the reference range, trend (first→last measurement), count of measurements outside the reference range. Use these signals instead of estimating yourself.
- "System-detected patterns": hints from a rule engine, NOT infallible diagnoses. Check each pattern against the raw data: do the values support it (same report, plausible values)? If not, mention it critically instead of adopting it blindly. ACTIVELY identify further correlations the rule set does not name (e.g. MCV+Hb+ferritin for anemia cause, neutrophils/lymphocytes, albumin+creatinine, iron+ferritin+CRP, GGT+ALP+bilirubin).
- Risk flags with urgency (critical = current value acutely problematic, long_term = observation).

IMPORTANT RULES:
- Base every statement strictly on the provided measurements: do not invent or assume values, trends, diagnoses, or conditions that are not supported by the given data. Phrase suspected diagnoses as possibilities ("consistent with", "compatible with"), never as certainty. If there are too few measurements to establish a trend, say so explicitly instead of inferring one.
- Safety rule: if the LATEST value of a marker is acutely critical (far outside the reference range or flagged critical), lead with it and recommend prompt medical evaluation. Judge urgency by the latest value, not by historical extremes.
- Highlight concerning trends, but also state what is reassuring: which systems are unremarkable or improved? An overall picture without the healthy findings is misleading.
- Avoid panic but clearly communicate risks.
- Write in English, professional yet accessible to laypeople.
- Recommendations should be patient-oriented ("Please ask your doctor to check... at your next visit", "Discuss with your physician..."), never dosage or prescribing advice.

ANSWER FORMAT (Markdown, 150-300 words total):
## Overall picture
1-2 sentences: the most important overall impression of the health status.
## Findings by system
One paragraph per organ system / marker group with data (e.g. blood count, thyroid, metabolism, lipids, kidney, liver, electrolytes, inflammation) — only systems with data. Each: status (unremarkable / abnormal / improved / worsening) + supporting values + cross-correlations if any.
## Trends & course
Only supportable trends; explicitly note thin data.
## Recommendations
1-3 concrete next steps (doctor-oriented)."""


RISK_ASSESSMENT_SYSTEM_PROMPT_DE = """Du bist ein erfahrener Arzt, der Laborbefunde eines Patienten bewertet.

Du erhältst eine nummerierte Liste von Risikofindern (Kategorie + Beschreibung, oft mit Wert-Kontext: aktueller Wert, Referenzbereich, Abweichung, Messanzahl und Trend). Bewerte die Dringlichkeit anhand dieser Zahlen, nicht nur des Beschreibungstextes. Deine Aufgabe:
1. Wähle die klinisch relevantesten Befunde aus — maximal 10 Risiken insgesamt.
2. Ordne sie nach Dringlichkeit: Rang 1 = dringendstes Risiko, dann absteigend.
3. Klassifiziere jedes Risiko als "critical" oder "long_term":
   - "critical" = Sofortmaßnahme empfohlen: Das Risiko sollte innerhalb der nächsten Tage bis Wochen adressiert werden (z.B. akut kritische Werte, akute Organschädigung, sofort behandlungsbedürftige Muster).
   - "long_term" = Langzeitbeobachtung empfohlen: Eher ein langfristiges Problem, das aktuell noch nicht gefährlich ist (z.B. chronisch leicht erhöhte Werte, Trends, überfällige Kontrollen).
   - WICHTIG: Beurteile die Dringlichkeit nach dem AKTUELLEN Wert (Stand des letzten Berichts), nicht nach historischen Extremwerten. Ein Marker, der früher kritisch war, aber im neuesten Bericht normal oder deutlich verbessert ist, ist NICHT mehr "critical" — klassifiziere ihn als "long_term" und erwähne die Besserung in der Erklärung.
4. Erstelle für jedes Risiko eine kurze Erklärung (2-3 Sätze): Was bedeutet der Befund klinisch, welche Konsequenzen können daraus resultieren und was sollte konkret getan werden?

Gib NUR gültiges JSON in genau diesem Format zurück, ohne Markdown oder zusätzliche Kommentare:
{"risks": [{"index": 1, "urgency": "critical", "explanation": "..."}, {"index": 3, "urgency": "long_term", "explanation": "..."}]}

- "index" ist die Nummer des Eingabebefunds (1-basiert).
- Die Liste ist in Rangreihenfolge sortiert (dringendstes Risiko zuerst).
- Schreibe die Erklärungen auf Deutsch, sachlich und präzise. Keine Panikmache, aber klar über Risiken aufklären."""

RISK_ASSESSMENT_SYSTEM_PROMPT_EN = """You are an experienced physician evaluating a patient's lab results.

You will receive a numbered list of risk findings (category + description, often with a value context: current value, reference range, deviation, measurement count and trend). Judge urgency from these numbers, not from the description text alone. Your task:
1. Select the most clinically relevant findings — at most 10 risks in total.
2. Rank them by urgency: rank 1 = most urgent risk, then descending.
3. Classify each risk as "critical" or "long_term":
   - "critical" = Immediate Action Recommended: the risk should be addressed within the next days to weeks (e.g. acutely critical values, acute organ damage, patterns requiring immediate treatment).
   - "long_term" = Long-Term Observation Recommended: rather a long-term problem that is not yet dangerous (e.g. chronically mildly elevated values, trends, overdue check-ups).
   - IMPORTANT: Judge urgency by the CURRENT value (latest report), not by historical extremes. A marker that was critical before but is normal or clearly improved in the most recent report is NO LONGER "critical" — classify it as "long_term" and mention the improvement in the explanation.
4. Write a brief explanation (2-3 sentences) for each risk: what the finding means clinically, what consequences may result, and what should concretely be done.

Return ONLY valid JSON in exactly this format, no markdown or additional comments:
{"risks": [{"index": 1, "urgency": "critical", "explanation": "..."}, {"index": 3, "urgency": "long_term", "explanation": "..."}]}

- "index" is the number of the input finding (1-based).
- The list is sorted by rank order (most urgent risk first).
- Write the explanations in English, objectively and precisely. Avoid panic but clearly communicate risks."""


def _get_risk_assessment_prompt(lang: str | None = None) -> str:
    """Return the risk assessment system prompt in the requested language."""
    if lang is None:
        lang = _get_lang()
    return RISK_ASSESSMENT_SYSTEM_PROMPT_DE if lang == "de" else RISK_ASSESSMENT_SYSTEM_PROMPT_EN


def _normalize_urgency(value) -> str | None:
    """Map a raw LLM urgency value to 'critical' or 'long_term' (None if unusable)."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in ("critical", "immediate", "immediate_action", "immediate action recommended"):
        return "critical"
    if v in ("long_term", "long-term", "long term", "observation", "long_term_observation",
             "long-term observation recommended"):
        return "long_term"
    return None


def _format_finding_context(finding: dict, lang: str) -> str:
    """Render optional numeric context of a candidate finding for the ranking prompt.

    Returns '' when the finding carries no context keys (older callers).
    """
    bits = []
    value = finding.get("value")
    if value is not None:
        unit = finding.get("unit") or ""
        ref_low = finding.get("ref_low")
        ref_high = finding.get("ref_high")
        item = f"{value} {unit}".strip()
        if ref_low is not None and ref_high is not None:
            item += f" (Ref {ref_low}–{ref_high})"
        dev = _fmt_deviation(value, ref_low, ref_high, lang)
        if dev:
            item += dev
        bits.append(item)
    if finding.get("n_measurements"):
        n = finding["n_measurements"]
        first, last = finding.get("trend_first"), finding.get("trend_last")
        if n >= 2 and first is not None and last is not None:
            trend_word = "Trend"
            pct = f" ({(last - first) / abs(first) * 100:+.0f}%)" if first else ""
            bits.append(f"{n} Messungen | {trend_word} {first}→{last}{pct}"
                        if lang == "de"
                        else f"{n} measurements | {trend_word} {first}→{last}{pct}")
        else:
            bits.append(f"{n} Messungen" if lang == "de" else f"{n} measurements")
    if finding.get("latest_date"):
        bits.append(str(finding["latest_date"]))
    return "; ".join(bits)


def assess_risks(llm: OpenAI, findings_data: list[dict], lang: str | None = None) -> list[dict]:
    """Ask the LLM to find, rank and explain up to 10 risks from candidate findings.

    Args:
        llm: OpenAI-compatible client.
        findings_data: List of dicts in input order (1-based index = position).
            Required keys: "category", "description". Optional context keys
            ("value", "unit", "ref_low", "ref_high", "latest_date",
            "n_measurements", "trend_first", "trend_last") are rendered into
            the prompt: judging urgency from the description text alone is
            guesswork — the LLM needs the actual numbers, bounds and trend to
            decide "critical" vs "long_term" reliably.
        lang: 'de' or 'en'.

    Returns a list of {"index": int, "urgency": "critical"|"long_term",
    "explanation": str} sorted by rank order (most urgent first). Returns [] when
    there are no findings or the LLM response is unusable — callers must treat an
    empty result as "no LLM assessment available".
    """
    if not findings_data:
        return []

    if lang is None:
        lang = _get_lang()

    label_cat = "Kategorie" if lang == "de" else "Category"
    label_desc = "Beschreibung" if lang == "de" else "Description"
    label_ctx = "Wert-Kontext" if lang == "de" else "Value context"
    prompt_parts = []
    for i, f in enumerate(findings_data, 1):
        part = f"{i}. {label_cat}: {f['category']}\n   {label_desc}: {f['description']}"
        ctx = _format_finding_context(f, lang)
        if ctx:
            part += f"\n   {label_ctx}: {ctx}"
        prompt_parts.append(part)
    prompt = "\n\n".join(prompt_parts)

    with _llm_semaphore:
        resp = llm.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": _get_risk_assessment_prompt(lang)},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=_abort_timeout() or _get_timeout(),
        )
        raw = resp.choices[0].message.content or ""

    try:
        data = json.loads(_strip_markdown_json(raw))
    except (json.JSONDecodeError, ValueError):
        return []

    items: list | None = None
    if isinstance(data, dict):
        for key in ("risks", "results", "data", "items"):
            val = data.get(key)
            if isinstance(val, list):
                items = val
                break
        if items is None:
            for val in data.values():
                if isinstance(val, list):
                    items = val
                    break
    elif isinstance(data, list):
        items = data

    if not items:
        return []

    valid_indices = set(range(1, len(findings_data) + 1))
    seen: set[int] = set()
    risks: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            try:
                idx = int(str(idx).strip())
            except (ValueError, TypeError):
                continue
        if idx not in valid_indices or idx in seen:
            continue
        urgency = _normalize_urgency(item.get("urgency"))
        explanation = item.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            continue
        seen.add(idx)
        risks.append({"index": idx, "urgency": urgency, "explanation": explanation.strip()})
    return risks


def _fmt_deviation(value, ref_low, ref_high, lang):
    """Short human-readable deviation from the reference range, or '' if in range."""
    if value is None:
        return ""
    if ref_high is not None and value > ref_high:
        if ref_high > 0:
            dev = (value - ref_high) / ref_high * 100
            return f" +{dev:.0f}%" + (" über Obergrenze" if lang == "de" else " above upper limit")
        return " über Obergrenze" if lang == "de" else " above upper limit"
    if ref_low is not None and value < ref_low:
        if ref_low > 0:
            dev = (ref_low - value) / ref_low * 100
            return f" -{dev:.0f}%" + (" unter Untergrenze" if lang == "de" else " below lower limit")
        return " unter Untergrenze" if lang == "de" else " below lower limit"
    return ""


def _summarize_biomarker_series(series, lang):
    """Precompute per-biomarker trend signals from a chronological series.

    ``series`` is a list of (report_date, value, unit, ref_low, ref_high,
    is_abnormal) tuples for ONE biomarker, oldest first. Returns a compact
    line the LLM can use directly — small models are unreliable at
    reconstructing trajectories and doing percentage arithmetic from a flat
    dump of hundreds of values, so the arithmetic is done here instead.

    Example (de):
      Kreatinin (mg/dl): 5 Messungen 03.01.2023→12.06.2025 | letzter 1.35
      (12.06.2025, +15% über Obergrenze, Ref 0.7–1.3) | Trend 0.9→1.35
      (+50%) | 2/5 Messungen außerhalb Referenz
    """
    values = [v for (_d, v, *_r) in series if v is not None]
    if not values:
        return ""
    unit = next((u for (_d, _v, u, *_r) in series if u), "")
    first_date, last_date = series[0][0], series[-1][0]
    last_value = values[-1]
    # Reference bounds of the LATEST measurement — labs reissue ranges and the
    # current one is what the latest value must be judged against.
    _ld, _lv, _lu, ref_low, ref_high, _ab = series[-1]
    n_abn = sum(1 for row in series if row[5])

    d_first = first_date.strftime("%d.%m.%Y") if first_date else "?"
    d_last = last_date.strftime("%d.%m.%Y") if last_date else "?"
    ref_str = "—"
    if ref_low is not None and ref_high is not None:
        ref_str = f"{ref_low}–{ref_high}"
    elif ref_high is not None:
        ref_str = f"< {ref_high}"
    elif ref_low is not None:
        ref_str = f"> {ref_low}"

    dev = _fmt_deviation(last_value, ref_low, ref_high, lang)
    count_word = ("Messung" if len(values) == 1 else "Messungen") if lang == "de" \
        else ("measurement" if len(values) == 1 else "measurements")
    parts = [
        f"{len(values)} {count_word}"
        + f" {d_first}→{d_last}",
        ("letzter" if lang == "de" else "latest")
        + (f" {last_value} ({d_last},{dev}, Ref {ref_str})" if dev
           else f" {last_value} ({d_last}, Ref {ref_str})"),
    ]
    if len(values) >= 2:
        first_val, last_val = values[0], values[-1]
        if first_val:
            pct = (last_val - first_val) / abs(first_val) * 100
            parts.append(
                ("Trend" if lang == "de" else "Trend")
                + f" {first_val}→{last_val} ({pct:+.0f}%)"
            )
        else:
            parts.append(
                ("Trend" if lang == "de" else "Trend")
                + f" {first_val}→{last_val}"
            )
    parts.append(
        f"{n_abn}/{len(values)} "
        + ("außerhalb Referenz" if lang == "de" else "outside reference range")
    )
    return " | ".join(parts)


def build_patient_summary_prompt(patient_name, recent_dps, historical_dps, risk_flags, lang, profile=None):
    """Build the user prompt for the patient health summary LLM call.

    Args:
        patient_name: Patient name
        recent_dps: List of (report_date, canonical_name, value, unit, ref_low, ref_high, is_abnormal) for last 2 years
        historical_dps: List of (report_date, canonical_name, value, unit, ref_low, ref_high, is_abnormal) for chronically abnormal markers
        risk_flags: List of (category, description) tuples, or 4-tuples
            (category, description, urgency, explanation) as produced by
            analyzer.refresh_risk_flags() — urgency/explanation let the summary
            agree with the Risks tab instead of contradicting it.
        lang: 'de' or 'en'
        profile: optional dict {"age": int|None, "sex": str|None,
            "clinical_context": str|None} — demographics the LLM needs to
            interpret age/sex-dependent reference ranges.
    """
    if lang == "de":
        header = f"Gesundheitszusammenfassung für Patient: {patient_name}\n\n"
        profile_section = "=== Patientenprofil ===\n"
        recent_section = ("=== Biomarker-Verläufe (letzte 2 Jahre, pro Marker "
                          "zusammengefasst: Anzahl, letzter Wert, Abweichung, Trend) ===\n")
        detail_section = "\n=== Letzte Messwerte pro Marker (Details) ===\n"
        historical_section = "\n=== Langzeit-Verlauf chronisch abnormaler Biomarker (vor 2 Jahren) ===\n"
        cross_section = "\n=== Vom System erkannte Muster (Regelwerk-Hinweise — an den Rohdaten prüfen, nicht blind übernehmen) ===\n"
        risk_section = "\n=== Risikomarkierungen ===\n"
        instruction = ("Erstelle auf Basis dieser Daten die strukturierte "
                       "Gesundheitszusammenfassung im vorgegebenen Format.")
        sex_labels = {"f": "weiblich", "m": "männlich", "d": "divers"}
    else:
        header = f"Health summary for patient: {patient_name}\n\n"
        profile_section = "=== Patient profile ===\n"
        recent_section = ("=== Biomarker courses (last 2 years, summarised per marker: "
                          "count, latest value, deviation, trend) ===\n")
        detail_section = "\n=== Latest measurement per marker (details) ===\n"
        historical_section = "\n=== Long-term trend of chronically abnormal biomarkers (older than 2 years) ===\n"
        cross_section = "\n=== System-detected patterns (rule-engine hints — verify against raw data, do not adopt blindly) ===\n"
        risk_section = "\n=== Risk flags ===\n"
        instruction = "Based on this data, create the structured health summary in the required format."
        sex_labels = {"f": "female", "m": "male", "d": "diverse"}

    # ── Patient profile section (age / sex / clinical context) ──────────────
    profile_text = ""
    if profile:
        prof_lines = []
        if profile.get("age") is not None:
            prof_lines.append(
                ("Alter: " if lang == "de" else "Age: ") + f"{profile['age']}"
            )
        if profile.get("sex") in sex_labels:
            prof_lines.append(
                ("Geschlecht: " if lang == "de" else "Sex: ") + sex_labels[profile["sex"]]
            )
        if profile.get("clinical_context"):
            prof_lines.append(
                ("Klinischer Kontext (Vorerkrankungen/Medikamente): " if lang == "de"
                 else "Clinical context (conditions/medication): ")
                + str(profile["clinical_context"]).strip()
            )
        if prof_lines:
            profile_text = profile_section + "\n".join(prof_lines) + "\n"

    # ── Group recent data points per biomarker (chronological) ──────────────
    # The old prompt dumped every value grouped by report date: for a
    # multi-year patient that is 1,000+ flat lines the model must reassemble
    # into trajectories itself. Grouping per biomarker + precomputed signals
    # serves the cross-examination goal directly and bounds the token count.
    series_by_marker: dict[tuple[str, str], list] = {}
    latest_by_marker: dict[tuple[str, str], tuple] = {}
    for rdate, cname, value, unit, ref_low, ref_high, is_abn in recent_dps:
        key = (cname, unit or "")
        series_by_marker.setdefault(key, []).append(
            (rdate, value, unit, ref_low, ref_high, is_abn)
        )
        prev = latest_by_marker.get(key)
        if prev is None or (rdate and (prev[0] is None or rdate >= prev[0])):
            latest_by_marker[key] = (rdate, cname, value, unit, ref_low, ref_high, is_abn)

    summary_lines = []
    for key in sorted(series_by_marker):
        series = sorted(series_by_marker[key], key=lambda r: r[0] or datetime.min)
        cname, unit = key
        line = _summarize_biomarker_series(series, lang)
        if line:
            summary_lines.append(f"- {cname}{' (' + unit + ')' if unit else ''}: {line}")
    recent_text = "\n".join(summary_lines) + "\n" if summary_lines else "(none)\n"

    # ── Compact detail: the latest measurement of every marker ──────────────
    detail_lines = []
    for key in sorted(latest_by_marker):
        rdate, cname, value, unit, ref_low, ref_high, is_abn = latest_by_marker[key]
        date_str = rdate.strftime("%d.%m.%Y") if rdate else "?"
        ref_str = ""
        if ref_low is not None and ref_high is not None:
            ref_str = f" (Ref: {ref_low}-{ref_high} {unit or ''})".rstrip()
        elif ref_high is not None:
            ref_str = f" (Ref: < {ref_high} {unit or ''})".rstrip()
        elif ref_low is not None:
            ref_str = f" (Ref: > {ref_low} {unit or ''})".rstrip()
        marker = " [ABNORMAL]" if is_abn else ""
        detail_lines.append(f"  {cname} = {value} {unit or ''}{ref_str}{marker} ({date_str})")
    detail_text = "\n".join(detail_lines) + "\n" if detail_lines else "(none)\n"

    # Format historical (chronically abnormal) data
    historical_groups = {}
    for rdate, cname, value, unit, ref_low, ref_high, is_abn in historical_dps:
        date_key = rdate.strftime("%d.%m.%Y") if rdate else "?"
        if date_key not in historical_groups:
            historical_groups[date_key] = []
        historical_groups[date_key].append(f"  {cname} = {value} {unit}")

    historical_text = ""
    for cname in sorted(historical_groups.keys()):
        entries = historical_groups[cname]
        historical_text += f"{cname}:\n" + "\n".join(entries) + "\n"

    # Format risk flags — separate cross-biomarker patterns from individual
    # findings. Flags may carry the LLM's urgency + clinical explanation from
    # refresh_risk_flags(); including them keeps the summary consistent with
    # the Risks tab (same findings, same urgency, no contradiction).
    cross_biomarker_flags = []
    individual_flags = []
    for flag in risk_flags:
        category, description = flag[0], flag[1]
        urgency = flag[2] if len(flag) > 2 else None
        explanation = flag[3] if len(flag) > 3 else None
        entry = (category, description, urgency, explanation)
        if category == "cross_biomarker_correlation":
            cross_biomarker_flags.append(entry)
        else:
            individual_flags.append(entry)

    # Build cross-biomarker section
    cross_text = ""
    for _cat, desc, _urg, _expl in cross_biomarker_flags:
        cross_text += f"  ⚠️ {desc}\n"
    if not cross_text:
        cross_text = "(none)\n"

    # Build individual flags section
    risk_text = ""
    for category, description, urgency, explanation in individual_flags:
        line = f"  [{category}]"
        if urgency:
            line += f" [{urgency}]"
        line += f" {description}"
        risk_text += line + "\n"
        if explanation:
            label = "Einordnung" if lang == "de" else "Assessment"
            risk_text += f"      {label}: {explanation}\n"
    if not risk_text:
        risk_text = "(none)\n"

    return (
        f"{header}"
        f"{profile_text}"
        f"{recent_section}{recent_text}"
        f"{detail_section}{detail_text}"
        f"{historical_section}{historical_text}"
        f"{cross_section}{cross_text}"
        f"{risk_section}{risk_text}"
        f"\n{instruction}"
    )


def generate_patient_summary(llm, patient_name, recent_dps, historical_dps, risk_flags, lang=None, profile=None):
    """Generate a patient health summary using the LLM.

    Args:
        llm: OpenAI client instance
        patient_name: Patient name
        recent_dps: List of (report_date, canonical_name, value, unit, ref_low, ref_high, is_abnormal)
        historical_dps: List of (report_date, canonical_name, value, unit, ref_low, ref_high, is_abnormal)
        risk_flags: List of (category, description) — optionally extended with
            (urgency, explanation) from the LLM risk assessment
        lang: 'de' or 'en' (defaults to current app language)
        profile: optional dict {"age", "sex", "clinical_context"} — patient
            demographics; reference ranges are age/sex dependent, so the LLM
            needs these to judge values correctly.

    Returns:
        The generated summary text, or None on failure.
    """
    if lang is None:
        lang = _get_lang()

    if lang == "de":
        system_prompt = PATIENT_SUMMARY_SYSTEM_PROMPT_DE
    else:
        system_prompt = PATIENT_SUMMARY_SYSTEM_PROMPT_EN

    user_prompt = build_patient_summary_prompt(
        patient_name, recent_dps, historical_dps, risk_flags, lang, profile=profile
    )

    with _llm_semaphore:
        resp = llm.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            timeout=_abort_timeout() or _get_timeout(),
        )
    content = resp.choices[0].message.content
    return content.strip() if content else None


# ── Biomarker Description & Interpretation ────────────────────────────────────

BIOMARKER_DESCRIPTION_SYSTEM_PROMPT_DE = """Du bist ein medizinischer Experte. Deine Aufgabe ist es, eine kurze, verständliche Erklärung zu einem spezifischen Biomarker zu erstellen.

Schreibe maximal 5 Sätze auf Deutsch. Beantworte:
1. Was misst dieser Biomarker?
2. Was können zu hohe Werte bedeuten?
3. Was können zu niedrige Werte bedeuten?

Schreibe in einer professionellen, aber für Laien verständlichen Sprache. Keine Panikmache."""

BIOMARKER_DESCRIPTION_SYSTEM_PROMPT_EN = """You are a medical expert. Your task is to create a short, understandable explanation for a specific biomarker.

Write a maximum of 5 sentences in English. Answer:
1. What does this biomarker measure?
2. What might too high values mean?
3. What might too low values mean?

Write in professional but layman-accessible language. Avoid panic-inducing statements."""

BIOMARKER_INTERPRETATION_SYSTEM_PROMPT_DE = """Du bist ein erfahrener Arzt, der Laborbefunde eines Patienten interpretiert.

Deine Aufgabe: Erstelle eine patientenspezifische medizinische Interpretation der gemessenen Werte für den genannten Biomarker im Zeitverlauf.

Berücksichtige:
- Den gesamten Verlauf aller Messwerte (Datum, Wert, Referenzbereich)
- Tendenzen (steigend, fallend, stabil)
- Wie viele Messungen es gibt und wie weit sie zurückreichen
- Ob Werte außerhalb des Referenzbereichs liegen und wie stark abweichend
- Klinische Relevanz der gefundenen Muster
- Falls klinisch verwandte Marker mit angegeben sind: nutze sie zur Einordnung (z.B. MCV + Ferritin bei Anämie, ALT + GGT + Bilirubin bei Leberwerten), aber interpretiere weiterhin nur den genannten Biomarker — die verwandten Werte sind Kontext, nicht Gegenstand der Interpretation
- Stütze jede Aussage strikt auf die vorliegenden Messwerte: Erfinde oder unterstelle keine Werte, Trends, Diagnosen oder Zustände, die nicht durch die gegebenen Daten belegt sind. Liegen zu wenige Messungen für eine belastbare Trendaussage vor (oder wurde kein Langzeitverlauf bereitgestellt), sage dies ausdrücklich statt eines Trends zu inferieren.

Zeilen mit [ABNORMAL] liegen außerhalb des Referenzbereichs.

Schreibe 3-5 Sätze auf Deutsch in professioneller, aber verständlicher Sprache. Keine Panikmache, aber klar über Risiken aufklären."""

BIOMARKER_INTERPRETATION_SYSTEM_PROMPT_EN = """You are an experienced physician interpreting a patient's lab results.

Your task: Create a patient-specific medical interpretation of the measured values for the named biomarker over time.

Consider:
- The entire course of all measurements (date, value, reference range)
- Trends (rising, falling, stable)
- How many measurements exist and how far back they go
- Whether values are outside the reference range and by how much
- Clinical relevance of observed patterns
- If clinically related markers are provided, use them to contextualise (e.g. MCV + ferritin in anaemia, ALT + GGT + bilirubin in liver values) but still interpret only the named biomarker — the related values are context, not the subject of the interpretation
- Base every statement strictly on the provided measurements: do not invent or assume values, trends, diagnoses, or conditions that are not supported by the given data. If there are too few measurements to establish a trend (or no long-term history is provided), say so explicitly instead of inferring one.

Lines marked [ABNORMAL] are outside the reference range.

Write 3-5 sentences in English in professional but accessible language. Avoid panic-inducing statements, but clearly communicate risks."""


def _get_biomarker_description_prompt(lang: str | None = None) -> str:
    """Return the biomarker description system prompt in the requested language."""
    if lang is None:
        lang = _get_lang()
    return BIOMARKER_DESCRIPTION_SYSTEM_PROMPT_DE if lang == "de" else BIOMARKER_DESCRIPTION_SYSTEM_PROMPT_EN


def _get_biomarker_interpretation_prompt(lang: str | None = None) -> str:
    """Return the biomarker interpretation system prompt in the requested language."""
    if lang is None:
        lang = _get_lang()
    return BIOMARKER_INTERPRETATION_SYSTEM_PROMPT_DE if lang == "de" else BIOMARKER_INTERPRETATION_SYSTEM_PROMPT_EN


def generate_biomarker_description(llm, biomarker_name: str, unit: str | None, canonical_list: list[str], lang: str | None = None) -> str | None:
    """Generate a general explanation for a biomarker (what it measures, implications of abnormal values).

    Args:
        llm: OpenAI client instance
        biomarker_name: The canonical name of the biomarker
        unit: Unit of measurement (e.g. 'g/dl')
        canonical_list: Full list of canonical biomarkers for context
        lang: 'de' or 'en'

    Returns:
        Generated description text, or None on failure.
    """
    if lang is None:
        lang = _get_lang()

    unit_str = f" ({unit})" if unit else ""
    display_name = f"{biomarker_name}{unit_str}"

    prompt = (
        f"Biomarker: {display_name}\n\n"
        f"Erstelle eine allgemeine Erklärung dieses Biomarkers. "
        f"Canonical-Liste zur Einordnung: {json.dumps(canonical_list)}"
    ) if lang == "de" else (
        f"Biomarker: {display_name}\n\n"
        f"Create a general explanation of this biomarker. "
        f"Canonical list for context: {json.dumps(canonical_list)}"
    )

    try:
        with _llm_semaphore:
            resp = llm.chat.completions.create(
                model=_get_model(),
                messages=[
                    {"role": "system", "content": _get_biomarker_description_prompt(lang)},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                timeout=_abort_timeout() or _get_timeout(),
            )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def generate_biomarker_interpretation(
    llm,
    biomarker_name: str,
    unit: str | None,
    data_points: list[tuple],  # [(report_date, value, ref_low, ref_high), ...]
    lang: str | None = None,
    context_lines: list[str] | None = None,
) -> str | None:
    """Generate a patient-specific medical interpretation of biomarker measurements over time.

    Args:
        llm: OpenAI client instance
        biomarker_name: The canonical name of the biomarker
        unit: Unit of measurement (e.g. 'g/dl')
        data_points: List of (report_date, value, ref_low, ref_high) tuples
        lang: 'de' or 'en'
        context_lines: optional latest values of clinically related markers
            (config.RELATED_BIOMARKERS), e.g.
            ["Ferritin = 8 ng/ml (Ref 30-400) [ABNORMAL] (12.06.2025)"].
            A single marker in isolation cannot be interpreted correctly
            (low MCV + low ferritin vs. normal ferritin are different
            diagnoses), so these anchor the interpretation in the panel.

    Returns:
        Generated interpretation text, or None on failure.
    """
    if lang is None:
        lang = _get_lang()

    unit_str = f" ({unit})" if unit else ""
    display_name = f"{biomarker_name}{unit_str}"

    # Format data points for the prompt
    if lang == "de":
        header = f"Biomarker: {display_name}\n\nPatientenspezifische Interpretation des Verlaufs:\n\n"
        label_date = "Datum"
        label_value = "Wert"
        label_ref = "Referenzbereich"
    else:
        header = f"Biomarker: {display_name}\n\nPatient-specific interpretation of the trend:\n\n"
        label_date = "Date"
        label_value = "Value"
        label_ref = "Reference Range"

    measurements = []
    for report_date, value, ref_low, ref_high in data_points:
        date_str = report_date.strftime("%d.%m.%Y") if hasattr(report_date, 'strftime') else str(report_date)
        ref_str = f"{ref_low}–{ref_high}" if (ref_low is not None and ref_high is not None) else "—"
        # Same [ABNORMAL] convention as build_patient_summary_prompt: one-sided
        # bounds count; a missing value can never be flagged.
        abnormal = (
            value is not None
            and ((ref_low is not None and value < ref_low)
                 or (ref_high is not None and value > ref_high))
        )
        marker = " [ABNORMAL]" if abnormal else ""
        measurements.append(f"  - {label_date}: {date_str}, {label_value}: {value} {unit or ''}, {label_ref}: {ref_str}{marker}")

    prompt = header + "\n".join(measurements)
    if context_lines:
        ctx_header = (
            "\n\nAktuelle Werte klinisch verwandter Marker (zur Einordnung — "
            "nicht selbst Gegenstand der Interpretation):"
            if lang == "de"
            else "\n\nLatest values of clinically related markers (context only — "
            "not the subject of this interpretation):"
        )
        prompt += ctx_header + "\n" + "\n".join(f"  - {line}" for line in context_lines)

    try:
        with _llm_semaphore:
            resp = llm.chat.completions.create(
                model=_get_model(),
                messages=[
                    {"role": "system", "content": _get_biomarker_interpretation_prompt(lang)},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                timeout=_abort_timeout() or _get_timeout(),
            )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


# ── "Chat with your data" ─────────────────────────────────────────────────────
# Free-form Q&A over the patient's measurements. Unlike the summary/
# interpretation helpers, errors are NOT swallowed here: the chat tab renders
# them inline so the user can retry the question.

CHAT_SYSTEM_PROMPT_DE = """Du bist ein medizinischer Assistent, der einem Patienten Fragen zu dessen eigenen Laborbefunden beantwortet.

Du erhältst als Kontext:
- Patientenprofil (Alter, Geschlecht, Vorerkrankungen/Medikamente) — soweit vorhanden. Referenzbereiche sind alters- und geschlechtsabhängig; ordne Werte entsprechend ein.
- Alle nicht deaktivierten Messwerte des Patienten der letzten 5 Jahre, gruppiert nach Biomarker: Messdatum, Wert, Einheit und Referenzbereich. Mit [ABNORMAL] markierte Werte liegen außerhalb des Referenzbereichs.
- Die bisherige Unterhaltung und die aktuelle Frage.

REGELN:
- Beantworte Fragen ausschließlich auf Basis der bereitgestellten Messwerte und des Chatverlaufs. Erfinde oder unterstelle keine Werte, Trends, Diagnosen oder Zustände, die nicht durch die Daten belegt sind.
- Fehlende Daten explizit benennen: Wurde ein Marker nie gemessen oder liegt die Messung außerhalb des 5-Jahres-Fensters, sage das ausdrücklich statt zu spekulieren.
- Belege Aussagen mit konkreten Werten und Messdaten (z.B. "LDL stieg von 130 mg/dl am 03.04.2023 auf 190 mg/dl am 12.06.2025").
- Beurteile den aktuellen Stand nach dem NEUESTEN Wert pro Marker; historische Extreme sind nicht der aktuelle Stand.
- Kreuzkorrelationen mehrerer Marker sind ausdrücklich erwünscht (z.B. MCV + Ferritin bei Anämie, ALT + GGT + Bilirubin bei Leberwerten), aber als Möglichkeit formulieren ("passt zu", "vereinbar mit"), nie als sichere Diagnose.
- Keine Dosierungs-, Medikamenten- oder Therapieempfehlungen. Bei auffälligen oder kritischen Befunden ärztliche Abklärung empfehlen.
- Liegen zu wenige Messungen für eine belastbare Trendaussage vor, sage das ausdrücklich statt einen Trend zu inferieren.
- Antworte auf Deutsch in professioneller, aber für Laien verständlicher Sprache. Halte die Antwort fokussiert (max. ~250 Wörter), außer die Frage verlangt ausdrücklich einen ausführlichen Vergleich."""

CHAT_SYSTEM_PROMPT_EN = """You are a medical assistant answering a patient's questions about their own lab results.

You are given as context:
- Patient profile (age, sex, conditions/medication) where available. Reference ranges depend on age and sex; interpret values accordingly.
- All non-disabled measurements of the patient from the last 5 years, grouped per biomarker: measurement date, value, unit, and reference range. Lines marked [ABNORMAL] are outside the reference range.
- The conversation so far and the current question.

RULES:
- Answer strictly from the provided measurements and the chat history. Do not invent or assume values, trends, diagnoses, or conditions that are not supported by the data.
- Explicitly name missing data: if a marker was never measured or a measurement falls outside the 5-year window, say so instead of speculating.
- Support statements with concrete values and dates (e.g. "LDL rose from 130 mg/dl on 04/03/2023 to 190 mg/dl on 06/12/2025").
- Judge the current status by the LATEST value per marker; historical extremes are not the current state.
- Cross-correlating markers is explicitly encouraged (e.g. MCV + ferritin in anaemia, ALT + GGT + bilirubin in liver values), but phrase findings as possibilities ("consistent with", "compatible with"), never as certain diagnoses.
- No dosage, medication, or treatment recommendations. For abnormal or critical findings, recommend medical evaluation.
- If there are too few measurements to establish a trend, say so explicitly instead of inferring one.
- Answer in English, professional yet accessible to laypeople. Keep the answer focused (max ~250 words) unless the question explicitly asks for an in-depth comparison."""


def _get_chat_system_prompt(lang: str | None = None) -> str:
    """Return the chat system prompt in the requested language."""
    if lang is None:
        lang = _get_lang()
    return CHAT_SYSTEM_PROMPT_DE if lang == "de" else CHAT_SYSTEM_PROMPT_EN


def chat_with_data(llm, data_context: str, messages: list[dict], lang: str | None = None) -> str:
    """Answer free-form questions about a patient's lab data ("chat with your data").

    Args:
        llm: OpenAI client instance
        data_context: preformatted patient dataset injected into the system
            prompt (see ui.tabs.chat.format_chat_data_context)
        messages: chat history [{"role": "user"|"assistant", "content": str}, ...],
            ending with the new user question
        lang: 'de' or 'en' (defaults to current app language)

    Returns:
        The assistant's answer text ("" when the model returned nothing).

    Raises:
        Exception: LLM/network errors propagate — the chat tab shows them
        inline (unlike the background-job helpers, which record failures
        into job state).
    """
    if lang is None:
        lang = _get_lang()
    system_prompt = _get_chat_system_prompt(lang) + "\n\n" + data_context
    with _llm_semaphore:
        resp = llm.chat.completions.create(
            model=_get_model(),
            messages=[{"role": "system", "content": system_prompt}, *messages],
            temperature=0.3,
            timeout=_abort_timeout() or _get_timeout(),
        )
    content = resp.choices[0].message.content
    return content.strip() if content else ""
