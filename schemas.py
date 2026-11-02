from pydantic import BaseModel, Field, model_validator


# ── Table layout (column-confusion guard) ────────────────────────────────────
# The extractor's characteristic failure: a lab report table has several numeric
# columns (often one per draw date) plus ONE reference column, and a vision
# model silently reads the reference cell as the measurement. The fix is
# structural — the model must first declare which columns exist and what each
# one means, and every entry must name the result column it came from. Roles
# are normalised because models word them inconsistently (German/English,
# "Ref.", "Normbereich", …).
_COLUMN_ROLE_ALIASES = {
    "name": "name", "bezeichnung": "name", "parameter": "name", "test": "name",
    "analyse": "name", "biomarker": "name", "marker": "name", "label": "name",
    "result": "result", "value": "result", "messwert": "result", "wert": "result",
    "messergebnis": "result", "ergebnis": "result", "measured": "result",
    "measurement": "result",
    "reference": "reference", "ref": "reference", "referenz": "reference",
    "referenzbereich": "reference", "references": "reference",
    "reference range": "reference", "sollbereich": "reference",
    "normbereich": "reference", "norm": "reference", "target": "reference",
    "unit": "unit", "einheit": "unit", "uom": "unit",
    "flag": "flag", "status": "flag", "hinweis": "flag", "note": "flag",
    "abnormal": "flag", "abnormality": "flag",
}


def _normalise_column_role(raw: object) -> str:
    """Map a free-form role word to one of name|result|reference|unit|flag|other."""
    key = str(raw or "").strip().lower()
    if key in _COLUMN_ROLE_ALIASES:
        return _COLUMN_ROLE_ALIASES[key]
    compact = key.replace("_", "").replace("-", "").replace(" ", "")
    if compact in _COLUMN_ROLE_ALIASES:
        return _COLUMN_ROLE_ALIASES[compact]
    # Prefix match either way: "reference value" → reference, "result column" → result.
    for alias, role in _COLUMN_ROLE_ALIASES.items():
        if compact and (compact.startswith(alias) or alias.startswith(compact)):
            return role
    return "other"


def is_number(s: str) -> bool:
    s = s.strip().lstrip("<>=").strip().replace(",", ".")
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def looks_like_range(s: str) -> bool:
    """True for a reference-range shaped string, e.g. '4.2-6.1' or '120 - 200'."""
    s = s.strip()
    for sep in ("–", "—", "−", "-", "~"):
        head, _, tail = s.partition(sep)
        if _ and is_number(head) and is_number(tail):
            return True
    return False


def bare_number(s: str | None) -> float | None:
    """A single number with no range separator and no comparison sign.

    '5.2' → 5.2. '4.2-6.1', '< 5.2', 'negativ' → None. A bare number sitting in
    the reference field is the signature of a column mix-up: reference cells are
    ranges or qualitative words, result cells are single numbers.
    """
    if not s:
        return None
    s = s.strip()
    if not s or s.startswith(("<", ">", "=")):
        return None
    for sep in ("–", "—", "−", "~"):
        if sep in s:
            return None
    # A '-' only starts a range when a number follows it ('-5' is negative,
    # '4-6' is a range).
    if "-" in s[1:] or s.count("-") > 1:
        return None
    return float(s) if is_number(s) else None


def _parse_value(v: str | float | None) -> float | None:
    """Parse a biomarker value string, handling LLM quirks like '<5,0' or '1,2'.

    The LLM sometimes returns values with German comma decimals or with
    comparison prefixes (<, >, <=, >=) that should be stripped.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v if isinstance(v, float) else float(v)

    s = str(v).strip()
    if not s:
        return None

    # Strip comparison prefixes the LLM may attach: <, >, <=, >=
    s = s.lstrip("<>=").strip()

    # Convert German comma to dot
    s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return None


def _normalise_confidence(raw: object) -> float | None:
    """Coerce a model-reported confidence into a float in [0, 1], or None.

    Models word/scale this inconsistently ('high', '0.9', 90, 'unsicher'). A
    number >1 is read as a percentage; a word maps to a fixed point; anything
    unrecognisable becomes None so the extractor never flags an entry the model
    did not actually rate.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
    else:
        text = str(raw).strip().lower()
        if not text:
            return None
        words = {
            "high": 0.95, "sehr hoch": 0.95, "sicher": 0.95, "certain": 0.95,
            "medium": 0.6, "mittel": 0.6, "medium-high": 0.75, "unsicher": 0.4,
            "low": 0.3, "niedrig": 0.3, "gering": 0.3, "uncertain": 0.4,
        }
        if text in words:
            return words[text]
        try:
            val = float(text.replace(",", "."))
        except ValueError:
            return None
    if val > 1.0:  # reported as a percentage (e.g. 90)
        val = val / 100.0
    return max(0.0, min(1.0, val))


class TableColumn(BaseModel):
    """One column of a report's results table, as declared by the extractor.

    The model lists these left-to-right BEFORE reading any row, which is what
    makes a reference/measurement mix-up detectable: an entry whose
    ``column_index`` is not a declared ``result`` column, or a page whose entry
    count does not match its declared result-column count, is a layout error
    rather than a plausible-looking number.
    """
    index: int | None = Field(default=None, description="0-based column position, left to right")
    header: str | None = Field(
        default=None,
        description="Column header exactly as printed, e.g. '02.05.2024' or 'Referenzbereich'")
    role: str = Field(default="other", description="name | result | reference | unit | flag | other")
    result_date: str | None = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) this result column was measured on, when its "
                    "header is a date (reports with several draw dates). Null otherwise.")

    @model_validator(mode="before")
    @classmethod
    def coerce_role(cls, data):
        if isinstance(data, dict):
            data["role"] = _normalise_column_role(data.get("role"))
        return data


class BiomarkerEntry(BaseModel):
    name: str = Field(description="Name of the biomarker/test")
    value: float | None = Field(default=None, description="Numeric value of the result")
    unit: str | None = Field(default=None, description="Unit of measurement (e.g. g/dl, U/l)")
    reference_range: str | None = Field(default=None, description="Reference range as string, e.g. '4.2-6.1' or '< 5.2'")
    # Which declared result column this entry was read from. None when the page
    # declared no layout at all (single-column tables, older models) — the
    # extractor only validates it when a layout IS declared.
    column_index: int | None = Field(
        default=None,
        description="Index of the result column this value was read from; must match a "
                    "declared column whose role is 'result'")
    # The un-coerced value string, kept whenever the model returned a string so a
    # value/reference swap can be undone AND the review UI can show exactly what
    # was printed. A range-shaped string in "value" becomes None via _parse_value,
    # and without the original text the range would be unrecoverable; for an
    # ordinary number it lets the Mapping tab show the printed form ('13,18')
    # beside the parsed value (13.18) so the user can eyeball the read.
    value_raw: str | None = Field(default=None, description="Raw value string as returned, before numeric coercion")
    # The model's own confidence (0..1) that THIS value came from the right cell.
    # Opt-in: only present when the extraction prompt asks for it. The extractor
    # turns a low value into a review finding; None (not reported) is never flagged.
    confidence: float | None = Field(
        default=None,
        description="Model self-reported confidence 0..1 that this value was read "
                    "correctly; null when the model did not report one")

    @model_validator(mode="before")
    @classmethod
    def coerce_value(cls, data):
        if isinstance(data, dict):
            raw = data.get("value")
            if raw is not None and not isinstance(raw, (int, float)):
                text = str(raw).strip()
                # Keep the printed text whenever the model returned a string:
                # always when coercion loses content (a range/qualitative string),
                # and otherwise too so the review UI can show the exact reading.
                if text:
                    data["value_raw"] = text
                data["value"] = _parse_value(raw)
            if "confidence" in data:
                data["confidence"] = _normalise_confidence(data.get("confidence"))
        return data


class ArtifactEntry(BaseModel):
    """A non-numeric finding from a report's written text (not a lab value).

    Qualitative results, clinical observations, conditions or general health
    notes. Stored as a ReportArtifact tied to the report; the user can disable
    any single one to exclude it from LLM analyses.
    """
    category: str = Field(
        description="Kind of artifact: 'qualitative_result' (a non-numeric test "
                    "result, e.g. 'Urinkultur: negativ'), 'clinical_observation' "
                    "(a finding/observation stated in the text), 'condition' (a "
                    "diagnosed/pre-existing condition or medication mentioned), "
                    "or 'general_insight' (an overall remark about the patient's health)."
    )
    text: str = Field(description="Concise, faithful statement of the finding (one bullet).")
    source_excerpt: str | None = Field(
        default=None,
        description="Short verbatim snippet from the report the finding was derived from (null if none)."
    )


class ExtractedPage(BaseModel):
    patient_name: str | None = Field(default=None, description="Full name of the patient as printed on the report")
    report_date: str | None = Field(default=None, description="Date of the lab report (as printed)")
    panel_type: str | None = Field(default=None, description="Type of panel, e.g. 'Blutbild', 'Leberwerte', 'Lipidstatus'")
    entries: list[BiomarkerEntry] = Field(description="All extracted biomarker entries from this page")
    # The declared table layout (STEP 1 of the prompt). Empty when the model
    # returned no layout — the extractor then skips layout validation instead of
    # failing, so a weaker model degrades to the old flat behaviour.
    columns: list[TableColumn] = Field(
        default_factory=list,
        description="The results table's columns, declared left-to-right before any row "
                    "is read, each with its role (name/result/reference/unit/flag).")
    # Non-numeric findings from the report's written text. Default empty so
    # pages without artifacts (and older callers/mocks) need no change.
    artifacts: list[ArtifactEntry] = Field(
        default_factory=list,
        description="Qualitative results / clinical observations / conditions / "
                    "general health insights extracted from this page's written text.",
    )
    # Batch mode only (R3-12): 0-based position of this page within the
    # requested image batch. Used to verify the LLM returned every page.
    page_index: int | None = Field(default=None, description="Zero-based index of this page within the batch (batch mode only)")


# ── Second-pass column audit (see llm_client.verify_page_extraction) ─────────

class VerifyProblem(BaseModel):
    """One disagreement the audit pass found on an already-extracted page.

    Never applied automatically — each problem becomes a DataQualityFinding the
    user accepts or dismisses in the Mapping tab.
    """
    entry_index: int | None = Field(default=None, description="0-based position in the page's entries list")
    problem: str = Field(default="", description="value_from_reference_column | reference_holds_result | wrong_result_column | missing_result_column | layout_mismatch")
    value: str | None = Field(default=None, description="Correct value exactly as printed, or null")
    reference_range: str | None = Field(default=None, description="Correct reference range exactly as printed, or null")
    column_index: int | None = Field(default=None, description="Result column the value actually belongs to")
    evidence: str | None = Field(default=None, description="Short verbatim quote of the printed row")

    @model_validator(mode="before")
    @classmethod
    def coerce_str_entry(cls, data):
        # Some models return a bare string per problem instead of an object.
        if isinstance(data, str):
            return {"problem": data, "evidence": data}
        return data


class VerifyPageResult(BaseModel):
    columns_ok: bool | None = Field(default=None, description="False when the declared layout does not match the printed table")
    columns: list[TableColumn] = Field(default_factory=list, description="Corrected layout, when the audit disagrees")
    problems: list[VerifyProblem] = Field(default_factory=list, description="Per-entry column problems; empty when the page is correct")


class ExtractedReport(BaseModel):
    pages: list[ExtractedPage]


class NormalizationMapping(BaseModel):
    # Values are str | None: the "unit" key echoes the unit of the extracted
    # entry a mapping was made for and is null when that entry had no unit.
    mappings: list[dict] = Field(
        description="List of {'original': '...', 'unit': '... or null', 'canonical': '...'} mappings"
    )
