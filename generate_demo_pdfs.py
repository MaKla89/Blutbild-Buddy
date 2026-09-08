#!/usr/bin/env python3
"""Generate 4 demo lab-report PDFs for a single fictional patient.

Patient: Anna Beispiel (clearly fictional)
Span:   June 2025 → July 2026 (~13 months, 4 reports)

Narrative arc:
  R1 (2025-06): Baseline — mostly normal, LDL borderline high
  R2 (2025-09): Early changes — lipids worsening, glucose creeping up, Vit D low
  R3 (2026-01): Worsening — liver enzymes elevated, CRP/BSG up (inflammation),
                cardio cross-biomarker pattern active
  R4 (2026-07): Mixed recovery — liver/inflammation improved, lipids still high

Run:  .venv/bin/python generate_demo_pdfs.py
Output: reports_input/demo_anna_beispiel_YYYY-MM-DD.pdf
"""

from pathlib import Path
from fpdf import FPDF  # LGPL-3.0 PDF writer (replaces AGPL PyMuPDF; used unmodified as a separate library)

OUT_DIR = Path(__file__).parent / "reports_input"
OUT_DIR.mkdir(exist_ok=True)

LAB_NAME = "Medizinisches Labor Dr. Fiktiv"
LAB_ADDR = "Musterstraße 42, 12345 Beispielstadt"
LAB_PHONE = "Tel: 01234-567890 | Fax: 01234-567891"
PATIENT_NAME = "Beispiel, Anna"
PATIENT_DOB = "15.03.1985"
PATIENT_ID = "MB-2025-0042"


def fmt_ref(low, high, unit):
    """Format a reference range string like German labs do."""
    if low is None and high is not None:
        return f"< {high} {unit}"
    if low is not None and high is None:
        return f"> {low} {unit}"
    if low is not None and high is not None:
        return f"{low} - {high} {unit}"
    return ""


def _rgb(color):
    """Convert a 0-1 float RGB tuple to fpdf2's 0-255 ints."""
    return tuple(int(round(v * 255)) for v in color)


def _text(pdf, x, y_baseline, s, size, bold=False, color=(0, 0, 0)):
    """Place text so its baseline sits at ~y_baseline (fitz-style coordinates)."""
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.set_text_color(*_rgb(color))
    pdf.set_xy(x, y_baseline - size * 0.8)
    pdf.cell(w=0, h=size, text=s)


def _line(pdf, x1, y1, x2, y2, color=(0, 0, 0), width=0.5):
    pdf.set_draw_color(*_rgb(color))
    pdf.set_line_width(width)
    pdf.line(x1, y1, x2, y2)


def build_report(date_str: str, panel_title: str, entries: list[dict], notes: str = "") -> FPDF:
    """Build a single-page PDF that looks like a German lab report.

    Each entry: {"name": str, "value": float|str, "unit": str, "ref_low": float|None, "ref_high": float|None}
    """
    pdf = FPDF(unit="pt", format="A4")  # same coordinate system as the old fitz version
    pdf.add_page()

    # ── Header ──────────────────────────────────────────────────────────────
    y = 40
    _text(pdf, 40, y, LAB_NAME, size=16, bold=True, color=(0.1, 0.2, 0.5))
    y += 18
    _text(pdf, 40, y, LAB_ADDR, size=9, color=(0.3, 0.3, 0.3))
    y += 14
    _text(pdf, 40, y, LAB_PHONE, size=9, color=(0.3, 0.3, 0.3))

    # Horizontal rule
    y += 8
    _line(pdf, 40, y, 555, y, color=(0.1, 0.2, 0.5), width=1.5)

    # ── Patient / meta block ────────────────────────────────────────────────
    y += 22
    meta_lines = [
        f"Patient: {PATIENT_NAME}",
        f"Geb.: {PATIENT_DOB}    |    Patient-ID: {PATIENT_ID}",
        f"Untersuchungsdatum: {date_str}",
        f"Panel: {panel_title}",
    ]
    for line in meta_lines:
        _text(pdf, 40, y, line, size=10)
        y += 15

    # ── Table header ────────────────────────────────────────────────────────
    y += 8
    col_name = 40
    col_value = 320
    col_unit = 395
    col_ref = 470

    _line(pdf, 40, y, 555, y, color=(0.1, 0.2, 0.5), width=0.8)
    y += 14
    _text(pdf, col_name, y, "Parameter", size=9, bold=True, color=(0.1, 0.2, 0.5))
    _text(pdf, col_value, y, "Ergebnis", size=9, bold=True, color=(0.1, 0.2, 0.5))
    _text(pdf, col_unit, y, "Einheit", size=9, bold=True, color=(0.1, 0.2, 0.5))
    _text(pdf, col_ref, y, "Referenzbereich", size=9, bold=True, color=(0.1, 0.2, 0.5))
    y += 4
    _line(pdf, 40, y, 555, y, color=(0.1, 0.2, 0.5), width=0.8)

    # ── Data rows ───────────────────────────────────────────────────────────
    y += 16
    for e in entries:
        name = e["name"]
        val_str = str(e["value"]) if not isinstance(e["value"], float) else (f"{e['value']:g}")
        unit = e.get("unit", "")
        ref = fmt_ref(e.get("ref_low"), e.get("ref_high"), unit)

        # Flag abnormal values in red (int or float)
        color = (0, 0, 0)
        v = e["value"]
        if isinstance(v, (int, float)):
            lo, hi = e.get("ref_low"), e.get("ref_high")
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                color = (0.85, 0.1, 0.1)

        _text(pdf, col_name, y, name, size=9, color=color)
        _text(pdf, col_value, y, val_str, size=9, bold=True, color=color)
        _text(pdf, col_unit, y, unit, size=9, color=(0.3, 0.3, 0.3))
        _text(pdf, col_ref, y, ref, size=8, color=(0.4, 0.4, 0.4))
        y += 15

    # ── Footer / notes ──────────────────────────────────────────────────────
    if notes:
        y += 10
        _text(pdf, 40, y, "Bemerkung:", size=9, bold=True)
        y += 13
        # Word-wrap the note
        words = notes.split()
        line = ""
        for w in words:
            if len(line) + len(w) > 75:
                _text(pdf, 40, y, line, size=8, color=(0.3, 0.3, 0.3))
                y += 12
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            _text(pdf, 40, y, line, size=8, color=(0.3, 0.3, 0.3))

    # Bottom rule + disclaimer
    y += 25
    _line(pdf, 40, y, 555, y, color=(0.6, 0.6, 0.6), width=0.5)
    y += 14
    _text(pdf, 40, y, "Dies ist ein fiktives Dokument zur Demonstration. Alle Daten sind erfunden.",
          size=7, color=(0.5, 0.5, 0.5))

    pdf.set_title(f"Lab Report {date_str} - {PATIENT_NAME}")
    pdf.set_author(LAB_NAME)
    return pdf


# ═══════════════════════════════════════════════════════════════════════════
# REPORT DATA
# ═══════════════════════════════════════════════════════════════════════════

R1_DATE = "15.06.2025"
R1_ENTRIES = [
    # CBC
    {"name": "Leukozyten (WBC)", "value": 7.2, "unit": "10³/µl", "ref_low": 4.0, "ref_high": 11.0},
    {"name": "Erythrozyten (RBC)", "value": 4.8, "unit": "Mio./µl", "ref_low": 4.5, "ref_high": 5.5},
    {"name": "Hämoglobin (Hb)", "value": 13.8, "unit": "g/dl", "ref_low": 12.0, "ref_high": 17.5},
    {"name": "Hämatokrit (Hkt)", "value": 41, "unit": "%", "ref_low": 36.0, "ref_high": 54.0},
    {"name": "MCV", "value": 92, "unit": "fl", "ref_low": 80.0, "ref_high": 100.0},
    {"name": "MCH", "value": 28.7, "unit": "pg", "ref_low": 27.0, "ref_high": 33.0},
    {"name": "Thrombozyten (PLT)", "value": 245, "unit": "10³/µl", "ref_low": 150.0, "ref_high": 400.0},
    # Lipids
    {"name": "Cholesterin gesamt", "value": 195, "unit": "mg/dl", "ref_low": None, "ref_high": 200.0},
    {"name": "HDL-Cholesterin", "value": 52, "unit": "mg/dl", "ref_low": 40.0, "ref_high": 90.0},
    {"name": "LDL-Cholesterin", "value": 118, "unit": "mg/dl", "ref_low": None, "ref_high": 116.0},
    {"name": "Triglyceride", "value": 130, "unit": "mg/dl", "ref_low": None, "ref_high": 150.0},
    # Glucose / Diabetes
    {"name": "Glukose (nüchtern)", "value": 92, "unit": "mg/dl", "ref_low": 70.0, "ref_high": 100.0},
    {"name": "HbA1c (%)", "value": 5.4, "unit": "%", "ref_low": 4.0, "ref_high": 5.7},
    # Kidney
    {"name": "Kreatinin", "value": 0.8, "unit": "mg/dl", "ref_low": 0.6, "ref_high": 1.2},
    {"name": "eGFR (CKD-EPI)", "value": 95, "unit": "ml/min/1.73m²", "ref_low": 90.0, "ref_high": None},
    # Liver
    {"name": "ALT (GPT)", "value": 28, "unit": "U/l", "ref_low": 7.0, "ref_high": 46.0},
    {"name": "AST (GOT)", "value": 24, "unit": "U/l", "ref_low": 10.0, "ref_high": 40.0},
    {"name": "Gamma-GT (GGT)", "value": 35, "unit": "U/l", "ref_low": 8.0, "ref_high": 61.0},
    # Thyroid
    {"name": "TSH basal", "value": 2.1, "unit": "µIU/ml", "ref_low": 0.27, "ref_high": 4.2},
    # Iron / Vitamins
    {"name": "Ferritin", "value": 85, "unit": "ng/ml", "ref_low": 15.0, "ref_high": 200.0},
    {"name": "Vitamin D (25-OH)", "value": 32, "unit": "ng/ml", "ref_low": 30.0, "ref_high": 70.0},
    # Inflammation
    {"name": "CRP", "value": 0.3, "unit": "mg/dl", "ref_low": None, "ref_high": 0.5},
    # Electrolytes
    {"name": "Natrium", "value": 141, "unit": "mmol/l", "ref_low": 136.0, "ref_high": 145.0},
    {"name": "Kalium", "value": 4.1, "unit": "mmol/l", "ref_low": 3.5, "ref_high": 5.0},
]

R2_DATE = "10.09.2025"
R2_ENTRIES = [
    # CBC
    {"name": "Leukozyten (WBC)", "value": 8.1, "unit": "10³/µl", "ref_low": 4.0, "ref_high": 11.0},
    {"name": "Erythrozyten (RBC)", "value": 4.7, "unit": "Mio./µl", "ref_low": 4.5, "ref_high": 5.5},
    {"name": "Hämoglobin (Hb)", "value": 13.2, "unit": "g/dl", "ref_low": 12.0, "ref_high": 17.5},
    {"name": "Hämatokrit (Hkt)", "value": 40, "unit": "%", "ref_low": 36.0, "ref_high": 54.0},
    {"name": "MCV", "value": 94, "unit": "fl", "ref_low": 80.0, "ref_high": 100.0},
    {"name": "Thrombozyten (PLT)", "value": 260, "unit": "10³/µl", "ref_low": 150.0, "ref_high": 400.0},
    # Lipids — worsening
    {"name": "Cholesterin gesamt", "value": 218, "unit": "mg/dl", "ref_low": None, "ref_high": 200.0},
    {"name": "HDL-Cholesterin", "value": 44, "unit": "mg/dl", "ref_low": 40.0, "ref_high": 90.0},
    {"name": "LDL-Cholesterin", "value": 142, "unit": "mg/dl", "ref_low": None, "ref_high": 116.0},
    {"name": "Triglyceride", "value": 175, "unit": "mg/dl", "ref_low": None, "ref_high": 150.0},
    # Glucose — creeping up
    {"name": "Glukose (nüchtern)", "value": 104, "unit": "mg/dl", "ref_low": 70.0, "ref_high": 100.0},
    {"name": "HbA1c (%)", "value": 5.8, "unit": "%", "ref_low": 4.0, "ref_high": 5.7},
    # Kidney
    {"name": "Kreatinin", "value": 0.9, "unit": "mg/dl", "ref_low": 0.6, "ref_high": 1.2},
    {"name": "eGFR (CKD-EPI)", "value": 88, "unit": "ml/min/1.73m²", "ref_low": 90.0, "ref_high": None},
    # Liver — upper range
    {"name": "ALT (GPT)", "value": 42, "unit": "U/l", "ref_low": 7.0, "ref_high": 46.0},
    {"name": "AST (GOT)", "value": 35, "unit": "U/l", "ref_low": 10.0, "ref_high": 40.0},
    {"name": "Gamma-GT (GGT)", "value": 55, "unit": "U/l", "ref_low": 8.0, "ref_high": 61.0},
    # Thyroid
    {"name": "TSH basal", "value": 1.8, "unit": "µIU/ml", "ref_low": 0.27, "ref_high": 4.2},
    # Iron / Vitamins
    {"name": "Ferritin", "value": 72, "unit": "ng/ml", "ref_low": 15.0, "ref_high": 200.0},
    {"name": "Vitamin D (25-OH)", "value": 24, "unit": "ng/ml", "ref_low": 30.0, "ref_high": 70.0},
    # Inflammation — mild elevation
    {"name": "CRP", "value": 0.6, "unit": "mg/dl", "ref_low": None, "ref_high": 0.5},
    # Electrolytes
    {"name": "Natrium", "value": 140, "unit": "mmol/l", "ref_low": 136.0, "ref_high": 145.0},
    {"name": "Kalium", "value": 4.0, "unit": "mmol/l", "ref_low": 3.5, "ref_high": 5.0},
]

R3_DATE = "20.01.2026"
R3_ENTRIES = [
    # CBC
    {"name": "Leukozyten (WBC)", "value": 9.5, "unit": "10³/µl", "ref_low": 4.0, "ref_high": 11.0},
    {"name": "Erythrozyten (RBC)", "value": 4.6, "unit": "Mio./µl", "ref_low": 4.5, "ref_high": 5.5},
    {"name": "Hämoglobin (Hb)", "value": 12.8, "unit": "g/dl", "ref_low": 12.0, "ref_high": 17.5},
    {"name": "Hämatokrit (Hkt)", "value": 39, "unit": "%", "ref_low": 36.0, "ref_high": 54.0},
    {"name": "MCV", "value": 96, "unit": "fl", "ref_low": 80.0, "ref_high": 100.0},
    {"name": "Thrombozyten (PLT)", "value": 280, "unit": "10³/µl", "ref_low": 150.0, "ref_high": 400.0},
    # Lipids — clearly abnormal
    {"name": "Cholesterin gesamt", "value": 245, "unit": "mg/dl", "ref_low": None, "ref_high": 200.0},
    {"name": "HDL-Cholesterin", "value": 38, "unit": "mg/dl", "ref_low": 40.0, "ref_high": 90.0},
    {"name": "LDL-Cholesterin", "value": 168, "unit": "mg/dl", "ref_low": None, "ref_high": 116.0},
    {"name": "Triglyceride", "value": 210, "unit": "mg/dl", "ref_low": None, "ref_high": 150.0},
    # Glucose — prediabetic range
    {"name": "Glukose (nüchtern)", "value": 118, "unit": "mg/dl", "ref_low": 70.0, "ref_high": 100.0},
    {"name": "HbA1c (%)", "value": 6.4, "unit": "%", "ref_low": 4.0, "ref_high": 5.7},
    # Kidney
    {"name": "Kreatinin", "value": 1.0, "unit": "mg/dl", "ref_low": 0.6, "ref_high": 1.2},
    {"name": "eGFR (CKD-EPI)", "value": 82, "unit": "ml/min/1.73m²", "ref_low": 90.0, "ref_high": None},
    # Liver — elevated
    {"name": "ALT (GPT)", "value": 58, "unit": "U/l", "ref_low": 7.0, "ref_high": 46.0},
    {"name": "AST (GOT)", "value": 44, "unit": "U/l", "ref_low": 10.0, "ref_high": 40.0},
    {"name": "Gamma-GT (GGT)", "value": 72, "unit": "U/l", "ref_low": 8.0, "ref_high": 61.0},
    # Thyroid
    {"name": "TSH basal", "value": 1.5, "unit": "µIU/ml", "ref_low": 0.27, "ref_high": 4.2},
    # Iron / Vitamins
    {"name": "Ferritin", "value": 65, "unit": "ng/ml", "ref_low": 15.0, "ref_high": 200.0},
    {"name": "Vitamin D (25-OH)", "value": 19, "unit": "ng/ml", "ref_low": 30.0, "ref_high": 70.0},
    # Inflammation — elevated
    {"name": "CRP", "value": 1.8, "unit": "mg/dl", "ref_low": None, "ref_high": 0.5},
    {"name": "BSG", "value": 28, "unit": "mm/h", "ref_low": None, "ref_high": 20.0},
    # Electrolytes
    {"name": "Natrium", "value": 139, "unit": "mmol/l", "ref_low": 136.0, "ref_high": 145.0},
    {"name": "Kalium", "value": 3.9, "unit": "mmol/l", "ref_low": 3.5, "ref_high": 5.0},
]

R4_DATE = "15.07.2026"
R4_ENTRIES = [
    # CBC
    {"name": "Leukozyten (WBC)", "value": 8.8, "unit": "10³/µl", "ref_low": 4.0, "ref_high": 11.0},
    {"name": "Erythrozyten (RBC)", "value": 4.7, "unit": "Mio./µl", "ref_low": 4.5, "ref_high": 5.5},
    {"name": "Hämoglobin (Hb)", "value": 13.0, "unit": "g/dl", "ref_low": 12.0, "ref_high": 17.5},
    {"name": "Hämatokrit (Hkt)", "value": 40, "unit": "%", "ref_low": 36.0, "ref_high": 54.0},
    {"name": "MCV", "value": 95, "unit": "fl", "ref_low": 80.0, "ref_high": 100.0},
    {"name": "Thrombozyten (PLT)", "value": 255, "unit": "10³/µl", "ref_low": 150.0, "ref_high": 400.0},
    # Lipids — improving but still high
    {"name": "Cholesterin gesamt", "value": 228, "unit": "mg/dl", "ref_low": None, "ref_high": 200.0},
    {"name": "HDL-Cholesterin", "value": 46, "unit": "mg/dl", "ref_low": 40.0, "ref_high": 90.0},
    {"name": "LDL-Cholesterin", "value": 138, "unit": "mg/dl", "ref_low": None, "ref_high": 116.0},
    {"name": "Triglyceride", "value": 165, "unit": "mg/dl", "ref_low": None, "ref_high": 150.0},
    # Glucose — improving but still elevated
    {"name": "Glukose (nüchtern)", "value": 108, "unit": "mg/dl", "ref_low": 70.0, "ref_high": 100.0},
    {"name": "HbA1c (%)", "value": 6.0, "unit": "%", "ref_low": 4.0, "ref_high": 5.7},
    # Kidney
    {"name": "Kreatinin", "value": 0.95, "unit": "mg/dl", "ref_low": 0.6, "ref_high": 1.2},
    {"name": "eGFR (CKD-EPI)", "value": 85, "unit": "ml/min/1.73m²", "ref_low": 90.0, "ref_high": None},
    # Liver — back to normal
    {"name": "ALT (GPT)", "value": 38, "unit": "U/l", "ref_low": 7.0, "ref_high": 46.0},
    {"name": "AST (GOT)", "value": 32, "unit": "U/l", "ref_low": 10.0, "ref_high": 40.0},
    {"name": "Gamma-GT (GGT)", "value": 48, "unit": "U/l", "ref_low": 8.0, "ref_high": 61.0},
    # Thyroid
    {"name": "TSH basal", "value": 1.9, "unit": "µIU/ml", "ref_low": 0.27, "ref_high": 4.2},
    # Iron / Vitamins — Vit D recovered
    {"name": "Ferritin", "value": 78, "unit": "ng/ml", "ref_low": 15.0, "ref_high": 200.0},
    {"name": "Vitamin D (25-OH)", "value": 34, "unit": "ng/ml", "ref_low": 30.0, "ref_high": 70.0},
    # Inflammation — resolved
    {"name": "CRP", "value": 0.4, "unit": "mg/dl", "ref_low": None, "ref_high": 0.5},
    {"name": "BSG", "value": 15, "unit": "mm/h", "ref_low": None, "ref_high": 20.0},
    # Electrolytes
    {"name": "Natrium", "value": 142, "unit": "mmol/l", "ref_low": 136.0, "ref_high": 145.0},
    {"name": "Kalium", "value": 4.2, "unit": "mmol/l", "ref_low": 3.5, "ref_high": 5.0},
]


def main():
    reports = [
        (R1_DATE, "Blutbild + Lipidstatus + Nierenwerte + Leberwerte + Schilddrüse + Vitamine",
         R1_ENTRIES,
         "Erstuntersuchung. LDL-Cholesterin grenzwertig erhöht."),
        (R2_DATE, "Blutbild + Lipidstatus + Nierenwerte + Leberwerte + Schilddrüse + Vitamine",
         R2_ENTRIES,
         "Lipidprofil verschlechtert sich. Glukose leicht erhöht. Vitamin D insuffizient."),
        (R3_DATE, "Blutbild + Lipidstatus + Nierenwerte + Leberwerte + Schilddrüse + Vitamine + Entzündung",
         R3_ENTRIES,
         "Leberenzyme und Entzündungswerte deutlich erhöht. Kardiovaskuläres Risiko durch LDL/HDL-Diskrepanz."),
        (R4_DATE, "Blutbild + Lipidstatus + Nierenwerte + Leberwerte + Schilddrüse + Vitamine + Entzündung",
         R4_ENTRIES,
         "Leberwerte und Entzündungsmarker normalisiert. Lipidprofil und Glukose weiterhin erhöht - Ernährungsberatung empfohlen."),
    ]

    for date_str, panel, entries, notes in reports:
        doc = build_report(date_str, panel, entries, notes)
        # Filename: demo_anna_beispiel_YYYY-MM-DD.pdf (ISO date for sorting)
        iso_date = date_str.replace(".", "-")  # "15.06.2025" → "15-06-2025"
        # Reorder to YYYY-MM-DD
        parts = date_str.split(".")
        iso = f"{parts[2]}-{parts[1]}-{parts[0]}"
        fname = f"demo_anna_beispiel_{iso}.pdf"
        out_path = OUT_DIR / fname
        doc.output(str(out_path))
        print(f"  ✓ {out_path.name}  ({len(entries)} parameters)")

    print(f"\nDone. {len(reports)} PDFs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
