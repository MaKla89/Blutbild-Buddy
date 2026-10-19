from pydantic import BaseModel, Field, model_validator


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


class BiomarkerEntry(BaseModel):
    name: str = Field(description="Name of the biomarker/test")
    value: float | None = Field(default=None, description="Numeric value of the result")
    unit: str | None = Field(default=None, description="Unit of measurement (e.g. g/dl, U/l)")
    reference_range: str | None = Field(default=None, description="Reference range as string, e.g. '4.2-6.1' or '< 5.2'")

    @model_validator(mode="before")
    @classmethod
    def coerce_value(cls, data):
        if isinstance(data, dict):
            raw = data.get("value")
            if raw is not None and not isinstance(raw, (int, float)):
                data["value"] = _parse_value(raw)
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


class ExtractedReport(BaseModel):
    pages: list[ExtractedPage]


class NormalizationMapping(BaseModel):
    # Values are str | None: the "unit" key echoes the unit of the extracted
    # entry a mapping was made for and is null when that entry had no unit.
    mappings: list[dict] = Field(
        description="List of {'original': '...', 'unit': '... or null', 'canonical': '...'} mappings"
    )
