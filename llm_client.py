import json
import threading
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
# (set by llm_jobs._run_job) because LLM call sites don't take a job id. Job
# ids are unique, so finished entries never match a new job — no clearing.
_aborted_job_ids: set[int] = set()
_abort_lock = threading.Lock()
_current_job_id_local = threading.local()


def request_job_abort(job_id: int):
    """Mark one job as aborted; its worker stops after the current step."""
    with _abort_lock:
        _aborted_job_ids.add(job_id)


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
    if cfg:
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
        raw = _strip_markdown_json(resp.choices[0].message.content)
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

    # R3-12: verify every requested page came back. The prompt instructs the
    # LLM to include "page_index" (0-based) per page; a short or mismatched
    # response must surface as an error instead of silently dropping pages.
    expected = set(range(len(page_images)))
    returned = {p.page_index for p in results if p.page_index is not None}
    if all(p.page_index is None for p in results):
        # No page_index at all: only accept when the count matches exactly,
        # trusting positional order. Anything else is unverifiable → error.
        if len(results) != len(page_images):
            raise LLMResponseError(
                f"Batch page extraction incomplete: requested {len(page_images)} pages, "
                f"got {len(results)} (no page_index provided to verify). "
                f"Refusing to continue with partial data — retry or use batch size 1."
            )
        return results
    missing = sorted(expected - returned)
    extra = sorted(returned - expected)
    if missing or extra:
        raise LLMResponseError(
            f"Batch page extraction incomplete: requested {len(page_images)} pages, "
            f"got indices {sorted(returned)} (missing: {missing}, unexpected: {extra}). "
            f"Refusing to continue with partial data — retry or use batch size 1."
        )
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
        raw = _strip_markdown_json(resp.choices[0].message.content)
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
        raw = _strip_markdown_json(resp.choices[0].message.content)
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
                has_bound = params.get("low") is not None or params.get("high") is not None
                if not has_bound:
                    continue
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

Deine Aufgabe: Erstelle eine prägnante, zusammenhängende Zusammenfassung des Gesundheitszustands des Patienten basierend auf dessen Laborbefunden.

WICHTIGE REGELN:
- Die Zusammenfassung soll 100-200 Wörter umfassen, maximal 250 Wörter.
- Fokussiere dich auf die letzten 2 Jahre der Befunde, betrachte aber auch den Langzeitverlauf auffälliger Werte.
- PRIORISIERE die Sektion "Kreuz-Biomarker-Muster": Diese wurden vom System als klinisch relevante Korrelationen identifiziert (z.B. CK + Troponin für Herzmuskelverletzung, TSH + fT3/fT4 für Schilddrüsenfunktionsstörung, LDL + HDL für kardiovaskuläres Risiko). Vertraue diesen vorab erkannten Mustern und beziehe sie zentral in die Zusammenfassung ein.
- Verbinde zusätzlich eigenständig die Punkte zwischen verschiedenen Biomarkern: Identifiziere gemeinsame Ursachen oder zugrunde liegende Probleme (z.B. niedriges Eisen + hohes Ferritin = mögliche Eisenmangelanämie mit Entzündung).
- Hebe besorgniserregende Tendenzen hervor.
- Stütze jede Aussage strikt auf die vorliegenden Messwerte: Erfinde oder unterstelle keine Werte, Trends, Diagnosen oder Zustände, die nicht durch die gegebenen Daten belegt sind. Liegen zu wenige Messungen für eine belastbare Trendaussage vor (oder wurde kein Langzeitverlauf bereitgestellt), sage dies ausdrücklich statt eines Trends zu inferieren.
- Schließe mit 1-3 konkreten, praxisnahen Empfehlungen für den Patienten als nächstes zu tun ab.
- Schreibe auf Deutsch.
- Verwende eine professionelle, aber verständliche Sprache. Keine Panikmache, aber klar über Risiken aufklären.
- Die Empfehlungen sollen patientenorientiert sein (z.B. "Bitte lassen Sie... in Ihrer nächsten Arztvisite prüfen lassen", "Besprechen Sie mit Ihrem Arzt...", nicht "verschreiben Sie...")."""

PATIENT_SUMMARY_SYSTEM_PROMPT_EN = """You are an experienced physician creating a health summary for a patient.

Your task: Create a concise, coherent summary of the patient's health status based on their lab results.

IMPORTANT RULES:
- The summary should be 100-200 words, maximum 250 words.
- Focus on the last 2 years of results, but also consider the long-term trend of abnormal values.
- PRIORITIZE the "Cross-Biomarker Patterns" section: These have been identified by the system as clinically relevant correlations (e.g., CK + Troponin for myocardial injury, TSH + fT3/fT4 for thyroid dysfunction, LDL + HDL for cardiovascular risk). Trust these pre-detected patterns and feature them prominently in the summary.
- Additionally connect the dots between individual biomarkers: identify common causes or underlying issues (e.g., low iron + high ferritin = possible iron deficiency anemia with inflammation).
- Highlight concerning trends.
- Base every statement strictly on the provided measurements: do not invent or assume values, trends, diagnoses, or conditions that are not supported by the given data. If there are too few measurements to establish a trend (or no long-term history is provided), say so explicitly instead of inferring one.
- Conclude with 1-3 concrete, practical recommendations for the patient on what to do next.
- Write in English.
- Use professional yet accessible language. Avoid panic but clearly communicate risks.
- Recommendations should be patient-oriented (e.g., "Please ask your doctor to check... at your next visit", "Discuss with your physician...", not "prescribe...")."""




RISK_ASSESSMENT_SYSTEM_PROMPT_DE = """Du bist ein erfahrener Arzt, der Laborbefunde eines Patienten bewertet.

Du erhältst eine nummerierte Liste von Risikofindern (Kategorie + Beschreibung). Deine Aufgabe:
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

You will receive a numbered list of risk findings (category + description). Your task:
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


def assess_risks(llm: OpenAI, findings_data: list[dict], lang: str | None = None) -> list[dict]:
    """Ask the LLM to find, rank and explain up to 10 risks from candidate findings.

    Args:
        llm: OpenAI-compatible client.
        findings_data: List of {"category": str, "description": str} in input order
            (1-based index = position in this list).
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
    prompt_parts = [
        f"{i}. {label_cat}: {f['category']}\n   {label_desc}: {f['description']}"
        for i, f in enumerate(findings_data, 1)
    ]
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
        raw = resp.choices[0].message.content

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


def build_patient_summary_prompt(patient_name, recent_dps, historical_dps, risk_flags, lang):
    """Build the user prompt for the patient health summary LLM call.

    Args:
        patient_name: Patient name
        recent_dps: List of (report_date, canonical_name, value, unit, ref_low, ref_high, is_abnormal) for last 2 years
        historical_dps: List of (report_date, canonical_name, value, unit, ref_low, ref_high, is_abnormal) for chronically abnormal markers
        risk_flags: List of (category, description) risk flags
        lang: 'de' or 'en'
    """
    if lang == "de":
        header = f"Gesundheitszusammenfassung für Patient: {patient_name}\n\n"
        recent_section = "=== Aktuelle Berichte (letzte 2 Jahre) ===\n"
        historical_section = "\n=== Langzeit-Verlauf chronisch abnormaler Biomarker ===\n"
        risk_section = "\n=== Risikomarkierungen ===\n"
        instruction = "Erstelle auf Basis dieser Daten eine zusammenhängende Gesundheitszusammenfassung."
    else:
        header = f"Health summary for patient: {patient_name}\n\n"
        recent_section = "=== Recent Reports (last 2 years) ===\n"
        historical_section = "\n=== Long-term trend of chronically abnormal biomarkers ===\n"
        risk_section = "\n=== Risk flags ===\n"
        instruction = "Based on this data, create a coherent health summary."

    # Format recent data points grouped by report date
    recent_groups = {}
    for rdate, cname, value, unit, ref_low, ref_high, is_abn in recent_dps:
        date_key = rdate.strftime("%d.%m.%Y") if rdate else "?"
        if date_key not in recent_groups:
            recent_groups[date_key] = []
        abnormal_marker = " [ABNORMAL]" if is_abn else ""
        ref_str = f" (Ref: {ref_low}-{ref_high} {unit})" if ref_low is not None and ref_high is not None else ""
        recent_groups[date_key].append(f"  {cname} = {value} {unit}{ref_str}{abnormal_marker}")

    recent_text = ""
    for date_key, entries in sorted(recent_groups.items()):
        recent_text += f"{date_key}:\n" + "\n".join(entries) + "\n"

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

    # Format risk flags — separate cross-biomarker patterns from individual findings
    cross_biomarker_flags = []
    individual_flags = []
    for category, description in risk_flags:
        if category == "cross_biomarker_correlation":
            cross_biomarker_flags.append(description)
        else:
            individual_flags.append((category, description))

    if lang == "de":
        cross_section = "\n=== Kreuz-Biomarker-Muster (klinische Korrelationen) ===\n"
        risk_section = "\n=== Einzelne Risikomarkierungen ===\n"
    else:
        cross_section = "\n=== Cross-Biomarker Patterns (Clinical Correlations) ===\n"
        risk_section = "\n=== Individual Risk Flags ===\n"

    # Build cross-biomarker section
    cross_text = ""
    for desc in cross_biomarker_flags:
        cross_text += f"  ⚠️ {desc}\n"

    # Build individual flags section
    risk_text = ""
    for category, description in individual_flags:
        risk_text += f"  [{category}] {description}\n"

    return (
        f"{header}"
        f"{recent_section}{recent_text}"
        f"{historical_section}{historical_text}"
        f"{cross_section}{cross_text}"
        f"{risk_section}{risk_text}"
        f"\n{instruction}"
    )


def generate_patient_summary(llm, patient_name, recent_dps, historical_dps, risk_flags, lang=None):
    """Generate a patient health summary using the LLM.

    Args:
        llm: OpenAI client instance
        patient_name: Patient name
        recent_dps: List of (report_date, canonical_name, value, unit, ref_low, ref_high, is_abnormal)
        historical_dps: List of (report_date, canonical_name, value, unit, ref_low, ref_high, is_abnormal)
        risk_flags: List of (category, description)
        lang: 'de' or 'en' (defaults to current app language)

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
        patient_name, recent_dps, historical_dps, risk_flags, lang
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
    return resp.choices[0].message.content.strip()


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
) -> str | None:
    """Generate a patient-specific medical interpretation of biomarker measurements over time.

    Args:
        llm: OpenAI client instance
        biomarker_name: The canonical name of the biomarker
        unit: Unit of measurement (e.g. 'g/dl')
        data_points: List of (report_date, value, ref_low, ref_high) tuples
        lang: 'de' or 'en'

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
