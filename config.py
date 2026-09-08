import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports_input"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "blutbild_buddy.db"

# Default values (used when env var is not set)
_DEFAULT_UNSLOTH_BASE_URL = "http://localhost:8888/v1"
_DEFAULT_UNSLOTH_MODEL = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
_DEFAULT_UNSLOTH_API_KEY = ""
_DEFAULT_LLM_TIMEOUT = "300"


def get_unsloth_base_url() -> str:
    """Lazily read LLM_BASE_URL so env changes after import are picked up."""
    return os.environ.get("LLM_BASE_URL", _DEFAULT_UNSLOTH_BASE_URL)


def get_unsloth_model() -> str:
    """Lazily read LLM_MODEL."""
    return os.environ.get("LLM_MODEL", _DEFAULT_UNSLOTH_MODEL)


def get_unsloth_api_key() -> str:
    """Lazily read LLM_API_KEY."""
    return os.environ.get("LLM_API_KEY", _DEFAULT_UNSLOTH_API_KEY)


def get_llm_timeout() -> int:
    """Lazily read LLM_TIMEOUT."""
    return int(os.environ.get("LLM_TIMEOUT", _DEFAULT_LLM_TIMEOUT))

# ── 3-Tier Biomarker Registry ────────────────────────────────────────────────
# Each entry: { canonical_name: { units: [...], aliases: [...], default_ref: {...} } }
# - canonical_name: single source of truth for display/storage
# - units: expected measurement units (may differ across labs)
# - aliases: lab-report names that map to this canonical name
# - default_ref: {low, high, unit} adult reference range (fallback when PDF omits it)
#   Units use the most common German lab convention; conversion handled by get_default_ref_range().
# ──────────────────────────────────────────────────────────────────────────────
_CANONICAL_BIOMARKERS: dict[str, dict] = {
    # Blutbild (CBC)
    "Leukozyten (WBC)": {"units": ["10³/µl", "/µl", "G/l"], "aliases": ["WBC", "Leukozyten", "weiße Blutkörperchen", "Leuko"],
        "default_ref": {"low": 4.0, "high": 11.0, "unit": "10³/µl"}},
    "Erythrozyten (RBC)": {"units": ["Mio./µl", "10⁶/µl", "T/l"], "aliases": ["RBC", "Erythrozyten", "rote Blutkörperchen", "Erythro"],
        "default_ref": {"low": 4.5, "high": 5.5, "unit": "Mio./µl"}},
    "Hämoglobin (Hb)": {"units": ["g/dl", "mmol/l"], "aliases": ["Hb", "Haemoglobin", "hemo globin", "Hgb", "Hämoglobin"],
        "default_ref": {"low": 12.0, "high": 17.5, "unit": "g/dl"}},
    "Hämatokrit (Hkt)": {"units": ["%", "l/l"], "aliases": ["Hkt", "Hct", "Hämo", "Hämatokrit", "Packed Cell Volume", "PCV"],
        "default_ref": {"low": 36.0, "high": 54.0, "unit": "%"}},
    "MCH": {"units": ["pg"], "aliases": ["Mean Corpuscular Hemoglobin", "MCH", "corpusculärer Hämoglobingehalt"],
        "default_ref": {"low": 27.0, "high": 33.0, "unit": "pg"}},
    "MCHC": {"units": ["g/dl", "g/l"], "aliases": ["Mean Corpuscular Hb Conc.", "MCHC", "corpusculäre Hämoglobinkonzentration"],
        "default_ref": {"low": 30.0, "high": 36.0, "unit": "g/dl"}},
    "MCV": {"units": ["fl"], "aliases": ["Mean Corpuscular Volume", "MCV", "corpusculäres Volumen"],
        "default_ref": {"low": 80.0, "high": 100.0, "unit": "fl"}},
    "Thrombozyten (PLT)": {"units": ["10³/µl", "G/l"], "aliases": ["PLT", "Thrombozyten", "Blutplättchen", "Platelets"],
        "default_ref": {"low": 150.0, "high": 400.0, "unit": "10³/µl"}},
    "Neutrophile Granulozyten": {"units": ["10³/µl", "/µl", "%"], "aliases": ["Neutrophile", "NEUT", "Seg", "segmentierte", "Stabsegende"],
        "default_ref": {"low": 40.0, "high": 75.0, "unit": "%"}},
    "Lymphozyten": {"units": ["10³/µl", "/µl", "%"], "aliases": ["LYM", "Lympho"],
        "default_ref": {"low": 20.0, "high": 45.0, "unit": "%"}},
    "Monozyten": {"units": ["10³/µl", "/µl", "%"], "aliases": ["MONO", "Mono"],
        "default_ref": {"low": 2.0, "high": 10.0, "unit": "%"}},
    "Eosinophile Granulozyten": {"units": ["10³/µl", "/µl", "%"], "aliases": ["EOS", "Eosinophile", "Eosino"],
        "default_ref": {"low": 0.0, "high": 4.0, "unit": "%"}},
    "Basophile Granulozyten": {"units": ["10³/µl", "/µl", "%"], "aliases": ["BASO", "Basophile", "Baso"],
        "default_ref": {"low": 0.0, "high": 1.0, "unit": "%"}},
    # Eisen & Vitamine
    "Ferritin": {"units": ["ng/ml", "µg/l", "ng/mL", "µg/L"], "aliases": ["Fer", "Ferritin", "Ferr"],
        "default_ref": {"low": 15.0, "high": 200.0, "unit": "ng/ml"}},
    "Eisen": {"units": ["µg/dl", "nmol/l"], "aliases": ["Serum-Eisen", "Fe", "serum iron", "Iron", "Eisenspiegel", "serum Fe", "BFE"],
        "default_ref": {"low": 60.0, "high": 170.0, "unit": "µg/dl"}},
    "Transferrin": {"units": ["mg/dl", "g/l"], "aliases": ["TFR", "Transferrin", "Tf"],
        "default_ref": {"low": 200.0, "high": 360.0, "unit": "mg/dl"}},
    "Transferrinsättigung": {"units": ["%"], "aliases": ["TSAT", "Transferrin-Sättigung", "Transferrin saturation", "TS", "Eisensättigung"],
        "default_ref": {"low": 20.0, "high": 55.0, "unit": "%"}},
    "Vitamin B12": {"units": ["pg/ml", "pmol/l"], "aliases": ["B12", "Cobalamin", "Vitamin B 12", "Cyanocobalamin"],
        "default_ref": {"low": 200.0, "high": 900.0, "unit": "pg/ml"}},
    "Vitamin D (25-OH)": {"units": ["ng/ml", "nmol/l", "µg/l"], "aliases": ["Vit D", "Vitamin D", "25-OH-Vitamin D", "Calcidiol", "VD", "Vitamin D3"],
        "default_ref": {"low": 30.0, "high": 70.0, "unit": "ng/ml"}},
    "Folsäure": {"units": ["ng/ml", "nmol/l"], "aliases": ["Folat", "Folate", "Folic Acid", "Vitamin B9", "Folat", "Folsäure"],
        "default_ref": {"low": 4.0, "high": 20.0, "unit": "ng/ml"}},
    # Lipidstatus
    "Cholesterin gesamt": {"units": ["mg/dl", "mmol/l"], "aliases": ["Gesamtcholesterin", "Total Cholesterol", "Cholesterin", "TC", "Chol"],
        "default_ref": {"low": 0.0, "high": 200.0, "unit": "mg/dl"}},
    "HDL-Cholesterin": {"units": ["mg/dl", "mmol/l"], "aliases": ["HDL", "HDL-Chol", "high-density lipoprotein"],
        "default_ref": {"low": 40.0, "high": 90.0, "unit": "mg/dl"}},
    "LDL-Cholesterin": {"units": ["mg/dl", "mmol/l"], "aliases": ["LDL", "LDL-Chol", "low-density lipoprotein"],
        "default_ref": {"low": 0.0, "high": 116.0, "unit": "mg/dl"}},
    "Triglyceride": {"units": ["mg/dl", "mmol/l"], "aliases": ["TG", "Triglyzeride", "Triacylglycerole", "TAG"],
        "default_ref": {"low": 0.0, "high": 150.0, "unit": "mg/dl"}},
    "VLDL-Cholesterin": {"units": ["mg/dl", "mmol/l"], "aliases": ["VLDL", "very-low-density lipoprotein"],
        "default_ref": {"low": 2.0, "high": 10.0, "unit": "mg/dl"}},
    "Lipoprotein (a)": {"units": ["mg/dl", "nmol/l", "mg/L"], "aliases": ["Lp(a)", "Lipoprotein a", "LPa"],
        "default_ref": {"low": 0.0, "high": 30.0, "unit": "mg/dl"}},
    # Leberwerte
    "ALT (GPT)": {"units": ["U/l", "IU/l", "U/L"], "aliases": ["ALT", "GPT", "Alanin-Aminotransferase", "SGPT"],
        "default_ref": {"low": 7.0, "high": 46.0, "unit": "U/l"}},
    "AST (GOT)": {"units": ["U/l", "IU/l", "U/L"], "aliases": ["AST", "GOT", "Aspartat-Aminotransferase", "SGOT"],
        "default_ref": {"low": 10.0, "high": 40.0, "unit": "U/l"}},
    "Gamma-GT (GGT)": {"units": ["U/l", "IU/l", "U/L"], "aliases": ["GGT", "Gamma-GT", "γ-GT", "Gamma-glutamyltransferase"],
        "default_ref": {"low": 8.0, "high": 61.0, "unit": "U/l"}},
    "Alkalische Phosphatase (AP)": {"units": ["U/l", "IU/l", "U/L"], "aliases": ["AP", "ALP", "Alkaline Phosphatase", "alk. Phosphatase"],
        "default_ref": {"low": 40.0, "high": 130.0, "unit": "U/l"}},
    "Bilirubin gesamt": {"units": ["mg/dl", "µmol/l", "mg/dL", "µmol/L"], "aliases": ["Bili ges.", "Gesamtbilirubin", "Total Bilirubin", "TBil"],
        "default_ref": {"low": 0.2, "high": 1.2, "unit": "mg/dl"}},
    "Bilirubin direkt": {"units": ["mg/dl", "µmol/l", "mg/dL", "µmol/L"], "aliases": ["Bili dir.", "Direktes Bilirubin", "Conjugated Bilirubin", "DBil"],
        "default_ref": {"low": 0.0, "high": 0.3, "unit": "mg/dl"}},
    "Total Protein": {"units": ["g/dl", "g/l"], "aliases": ["Gesamtprotein", "TP", "Protein gesamt", "Protein"],
        "default_ref": {"low": 6.0, "high": 8.3, "unit": "g/dl"}},
    "Albumin": {"units": ["g/dl", "g/l"], "aliases": ["Alb", "Albumin"],
        "default_ref": {"low": 3.5, "high": 5.5, "unit": "g/dl"}},
    # Nierenwerte
    "Kreatinin": {"units": ["mg/dl", "mg/dL", "µmol/l", "µmol/L"], "aliases": ["Crea", "Creatinine", "Kreat"],
        "default_ref": {"low": 0.6, "high": 1.2, "unit": "mg/dl"}},
    "Harnstoff": {"units": ["mg/dl", "mmol/l", "mg/dL", "mmol/L"], "aliases": ["BUN", "Urea", "Harnstoff", "Blutharnstoff"],
        "default_ref": {"low": 17.0, "high": 43.0, "unit": "mg/dl"}},
    "Harnsäure": {"units": ["mg/dl", "µmol/l", "mg/dL", "µmol/L"], "aliases": ["Harns.", "UA", "Uric Acid", "Harnsaure", "SA"],
        "default_ref": {"low": 2.0, "high": 7.5, "unit": "mg/dl"}},
    "eGFR (CKD-EPI)": {"units": ["ml/min/1.73m²", "ml/min/1.73m2", "ml/min"], "aliases": ["eGFR", "CKD-EPI", "EGFR", "estimated GFR", "Estimated Glomerular Filtration Rate", "glomeruläre Filtrationsrate"],
        "default_ref": {"low": 90.0, "high": None, "unit": "ml/min/1.73m²"}},
    "eGFR (MDRD)": {"units": ["ml/min/1.73m²", "ml/min/1.73m2"], "aliases": ["eGFR MDRD", "MDRD", "estimated GFR MDRD"],
        "default_ref": {"low": 90.0, "high": None, "unit": "ml/min/1.73m²"}},
    "Kreatinin-Clearance": {"units": ["ml/min"], "aliases": ["CrCl", "Krea-Clearance", "Creatinine Clearance"],
        "default_ref": {"low": 90.0, "high": None, "unit": "ml/min"}},
    "Cystatin C": {"units": ["mg/l", "mg/L", "mg/dl", "mg/dL", "µg/l", "µg/L"], "aliases": ["Cystatin", "Cyst C", "Zystatin"],
        "default_ref": {"low": 0.6, "high": 1.2, "unit": "mg/l"}},
    # Elektrolyte
    "Natrium": {"units": ["mmol/l", "mEq/l", "mmol/L", "mEq/L"], "aliases": ["Na", "Na+", "Natrium"],
        "default_ref": {"low": 136.0, "high": 145.0, "unit": "mmol/l"}},
    "Kalium": {"units": ["mmol/l", "mEq/l", "mmol/L", "mEq/L"], "aliases": ["K", "K+", "Kalium"],
        "default_ref": {"low": 3.5, "high": 5.0, "unit": "mmol/l"}},
    "Calcium": {"units": ["mg/dl", "mmol/l", "mg/dL", "mmol/L"], "aliases": ["Ca", "Ca2+", "Calcium", "Gesamtkalzium"],
        "default_ref": {"low": 8.5, "high": 10.5, "unit": "mg/dl"}},
    "Magnesium": {"units": ["mg/dl", "mmol/l", "mg/dL", "mmol/L"], "aliases": ["Mg", "Mg2+", "Magnesium"],
        "default_ref": {"low": 1.7, "high": 2.2, "unit": "mg/dl"}},
    "Chlorid": {"units": ["mmol/l", "mEq/l", "mmol/L", "mEq/L"], "aliases": ["Cl", "Cl-", "Chlorid"],
        "default_ref": {"low": 98.0, "high": 107.0, "unit": "mmol/l"}},
    "Phosphat": {"units": ["mg/dl", "mmol/l", "mg/dL", "mmol/L"], "aliases": ["P", "PO4", "Phosphor", "Inorganisches Phosphat"],
        "default_ref": {"low": 2.5, "high": 4.5, "unit": "mg/dl"}},
    # Schilddrüse
    "TSH basal": {"units": ["µIU/ml", "mIU/l", "µIU/mL", "mIU/L", "uIU/ml"], "aliases": ["TSH", "Thyreoidea-stimulierendes Hormon", "Thyrotropin"],
        "default_ref": {"low": 0.27, "high": 4.2, "unit": "µIU/ml"}},
    "fT3": {"units": ["pg/ml", "pmol/l", "pg/mL", "pmol/L"], "aliases": ["ft3", "Freies T3", "Free T3", "Triiodthyronin frei"],
        "default_ref": {"low": 2.0, "high": 4.4, "unit": "pg/ml"}},
    "fT4": {"units": ["ng/dl", "pmol/l", "ng/dL", "pmol/L"], "aliases": ["ft4", "Freies T4", "Free T4", "Thyroxin frei"],
        "default_ref": {"low": 0.93, "high": 1.7, "unit": "ng/dl"}},
    "TPO-Antikörper": {"units": ["IU/ml", "IU/L", "IE/ml", "IE/L", "U/ml", "U/L"], "aliases": ["TPO-Ab", "Anti-TPO", "TPO-AK", "Thyroperoxidase-Antikörper"],
        "default_ref": {"low": 0.0, "high": 34.0, "unit": "IU/ml"}},
    "TRAK": {"units": ["IU/l", "IU/L", "IE/l", "IE/L"], "aliases": ["TRAK", "TSI", "TSH-Rezeptor-Antikörper", "TSHR-Ab"],
        "default_ref": {"low": 0.0, "high": 1.5, "unit": "IU/l"}},
    # Entzündung & Immun
    "CRP": {"units": ["mg/dl", "mg/l", "mg/L", "mg/dL"], "aliases": ["C-reaktives Protein", "CRP", "hs-CRP", "high-sensitivity CRP"],
        "default_ref": {"low": 0.0, "high": 0.5, "unit": "mg/dl"}},
    "BSG": {"units": ["mm/r", "mm/h"], "aliases": ["Blutsenkungsgeschwindigkeit", "BSG", "ESR", "Sedimentation rate"],
        "default_ref": {"low": 0.0, "high": 20.0, "unit": "mm/h"}},
    "IL-6": {"units": ["pg/ml", "pg/mL"], "aliases": ["Interleukin-6", "IL6"],
        "default_ref": {"low": 0.0, "high": 7.0, "unit": "pg/ml"}},
    "TNF-alpha": {"units": ["pg/ml", "pg/mL"], "aliases": ["TNF-a", "TNF-alpha", "Tumor Necrosis Factor alpha", "TNF-α"],
        "default_ref": {"low": 0.0, "high": 8.0, "unit": "pg/ml"}},
    # Antikörper & Immunologie
    "Lactatdehydrogenase (LDH)": {"units": ["U/l", "IU/l", "U/L", "IU/L"], "aliases": ["LDH", "Lactatdehydrogenase", "Lactat dehydrogenase", "Lactic Dehydrogenase"],
        "default_ref": {"low": 100.0, "high": 250.0, "unit": "U/l"}},
    "IgE gesamt": {"units": ["IU/ml", "kU/l", "IU/ml", "kU/L"], "aliases": ["IgE", "Gesamt IgE", "Total IgE", "Immunoglobulin E"],
        "default_ref": {"low": 0.0, "high": 100.0, "unit": "IU/ml"}},
    "Rheumafaktor": {"units": ["IU/ml", "IU/L", "IE/ml", "IE/L", "U/ml", "U/L"], "aliases": ["RF", "Rheumatoid Factor", "Rheuma Faktor", "RF-IgM"],
        "default_ref": {"low": 0.0, "high": 15.0, "unit": "IU/ml"}},
    "Anti-Streptolysin O (ASO)": {"units": ["IU/ml", "IE/ml", "IU/L", "IE/L", "U/ml", "U/L"], "aliases": ["ASO", "ASL", "Antistreptolysin O", "Antistreptolysin", "Streptozym"],
        "default_ref": {"low": 0.0, "high": 200.0, "unit": "IU/ml"}},
    "ANA (Antikörper)": {"units": ["Ratio", "titer", "Einh./ml", "IE/ml"], "aliases": ["ANA", "Antinukleäre Antikörper", "Antinuclear Antibodies", "AK", "Autoantikörper"],
        "default_ref": {"low": 0.0, "high": 0.4, "unit": "Ratio"}},
    "ENA Pool": {"units": ["IU/ml", "IE/ml", "Einh./ml", "U/ml"], "aliases": ["ENA Pool Plus", "ENA-Pool", "Extractable Nuclear Antigens", "ENA Pool"],
        "default_ref": {"low": 0.0, "high": 7.0, "unit": "IU/ml"}},
    "Anti-Thyroglobulin (TgAb)": {"units": ["IU/ml", "IE/ml", "Einh./ml", "U/ml"], "aliases": ["Anti-Tg", "Anti TG", "Anti-Thyroglobulin Antibody", "TgAb", "Thyroglobulin-Antikörper"],
        "default_ref": {"low": 0.0, "high": 40.0, "unit": "IU/ml"}},
    # Nierenfunktion & Säure-Basen
    "Base Excess (BE)": {"units": ["mmol/l", "mEq/l", "mmol/L", "mEq/L"], "aliases": ["BE", "Base Excess", "Basenüberschuss", "Base Deficit", "BD"],
        "default_ref": {"low": -3.0, "high": 3.0, "unit": "mmol/l"}},
    # Blutzucker / Diabetes
    "Glukose (nüchtern)": {"units": ["mg/dl", "mmol/l", "mg/dL", "mmol/L"], "aliases": ["Glukose", "Blutzucker", "BZ", "Glucose", "fasting glucose", "Nüchternglukose"],
        "default_ref": {"low": 70.0, "high": 100.0, "unit": "mg/dl"}},
    "HbA1c (%)": {"units": ["%"], "aliases": ["HbA1c", "Glykogenisiertes Hämoglobin", "A1c", "Hemoglobin A1c", "HbA1c (TDH)"],
        "default_ref": {"low": 4.0, "high": 5.7, "unit": "%"}},
    "HbA1c (mmol/mol)": {"units": ["mmol/mol"], "aliases": ["HbA1c IFCC", "HbA1c (IFCC)", "HbA1c (SI)", "Hemoglobin A1c IFCC"],
        "default_ref": {"low": 20.0, "high": 39.0, "unit": "mmol/mol"}},
    "Insulin (nüchtern)": {"units": ["µIU/ml", "mIE/l", "µIU/mL", "mIE/L", "pmol/l", "pmol/L"], "aliases": ["Insulin", "fasting insulin", "Nüchterinsulin"],
        "default_ref": {"low": 2.6, "high": 24.9, "unit": "µIU/ml"}},
    "HOMA-Index": {"units": [], "aliases": ["HOMA", "HOMA-IR", "Homeostasis Model Assessment"],
        "default_ref": {"low": 0.0, "high": 2.5, "unit": ""}},
    # Herz & Gefäße
    "Homocystein": {"units": ["µmol/l", "µmol/L", "umol/l", "umol/L"], "aliases": ["Hcy", "Homocyst(e)in"],
        "default_ref": {"low": 5.0, "high": 15.0, "unit": "µmol/l"}},
    "CK gesamt": {"units": ["U/l", "IU/l", "U/L"], "aliases": ["CK", "CPK", "Creatin-Kinase", "Creatine Kinase"],
        "default_ref": {"low": 30.0, "high": 200.0, "unit": "U/l"}},
    "CK-MB": {"units": ["U/l", "IU/l", "U/L", "ng/ml", "ng/mL"], "aliases": ["CK-MB", "CK-Mass", "CK-MB Mass"],
        "default_ref": {"low": 0.0, "high": 5.0, "unit": "U/l"}},
    "Troponin T": {"units": ["ng/ml", "ng/mL", "µg/l", "µg/L"], "aliases": ["TnT", "cTnT", "cardiac Troponin T", "Troponin"],
        "default_ref": {"low": 0.0, "high": 0.01, "unit": "ng/ml"}},
    "NT-proBNP": {"units": ["pg/ml", "pg/mL", "ng/l", "ng/L"], "aliases": ["NT-pro-BNP", "N-terminal pro B-type Natriuretic Peptide"],
        "default_ref": {"low": 0.0, "high": 125.0, "unit": "pg/ml"}},
    "Fibrinogen": {"units": ["mg/dl", "g/l", "mg/dL", "g/L"], "aliases": ["Fibrinogen", "Factor I", "FG"],
        "default_ref": {"low": 200.0, "high": 400.0, "unit": "mg/dl"}},
    "D-Dimere": {"units": ["µg/ml", "mg/l", "µg/mL", "mg/L", "FEU", "DDU"], "aliases": ["DDimer", "D-Dimer", "D-Dimers", "D-Dim"],
        "default_ref": {"low": 0.0, "high": 500.0, "unit": "µg/ml"}},
    # Hormone
    "Cortisol": {"units": ["µg/dl", "nmol/l", "µg/dL", "nmol/L"], "aliases": ["Cortisol", "Hydrocortison", "Cort"],
        "default_ref": {"low": 5.0, "high": 23.0, "unit": "µg/dl"}},
    "Testosteron gesamt": {"units": ["ng/dl", "nmol/l", "ng/dL", "nmol/L"], "aliases": ["Testosteron", "Total Testosterone", "T"],
        "default_ref": {"low": 280.0, "high": 800.0, "unit": "ng/dl"}},
    "Testosteron frei": {"units": ["pg/ml", "pmol/l", "pg/mL", "pmol/L"], "aliases": ["Freies Testosteron", "Free Testosterone", "fT"],
        "default_ref": {"low": 5.0, "high": 21.0, "unit": "pg/ml"}},
    "Östradiol (E2)": {"units": ["pg/ml", "pmol/l", "pg/mL", "pmol/L"], "aliases": ["E2", "Östradiol", "Estradiol", "Estradiol E2"],
        "default_ref": {"low": 10.0, "high": 400.0, "unit": "pg/ml"}},
    "Progesteron": {"units": ["ng/dl", "nmol/l", "ng/dL", "nmol/L"], "aliases": ["Progesteron", "P"],
        "default_ref": {"low": 0.1, "high": 25.0, "unit": "ng/dl"}},
    "SHBG": {"units": ["nmol/l", "nmol/L", "IU/l", "IU/L"], "aliases": ["SHBG", "Sexual Hormone Binding Globulin", "HBG", "Sexualhormon-bindendes Globulin"],
        "default_ref": {"low": 15.0, "high": 115.0, "unit": "nmol/l"}},
    "LH": {"units": ["IU/l", "IU/L", "mIU/ml", "mIU/ml"], "aliases": ["LH", "Luteinisierendes Hormon", "Lutropin"],
        "default_ref": {"low": 1.5, "high": 9.3, "unit": "IU/l"}},
    "FSH": {"units": ["IU/l", "IU/L", "mIU/ml", "mIU/ml"], "aliases": ["FSH", "Follikel-stimulierendes Hormon", "Follicle Stimulating Hormone"],
        "default_ref": {"low": 1.5, "high": 12.4, "unit": "IU/l"}},
    "Prolaktin": {"units": ["ng/ml", "µg/l", "pg/ml", "ng/mL", "µg/L", "pg/mL"], "aliases": ["PRL", "Prolactin", "Prolaktin"],
        "default_ref": {"low": 4.0, "high": 23.0, "unit": "ng/ml"}},
    # Urin
    "Urin-pH": {"units": [], "aliases": ["pH Urin", "Urin pH", "pH im Urin"],
        "default_ref": {"low": 5.0, "high": 7.0, "unit": ""}},
    "Urin-spezifisches Gewicht": {"units": [], "aliases": ["spez. Gewicht Urin", "Dichte Urin", "SG Urin", "Specific Gravity"],
        "default_ref": {"low": 1.005, "high": 1.030, "unit": ""}},
    "Urin-Protein": {"units": ["mg/dl", "g/l", "mg/24h", "mg/24h", "mg/dL", "g/L"], "aliases": ["Protein Urin", "Urinprotein", "Albumin Urin", "UP"],
        "default_ref": {"low": 0.0, "high": 30.0, "unit": "mg/24h"}},
    "Urin-Glukose": {"units": ["mg/dl", "mmol/l", "mg/dL", "mmol/L"], "aliases": ["Glukose Urin", "Zucker Urin", "Glykosurie"],
        "default_ref": {"low": 0.0, "high": 0.0, "unit": "mg/dl"}},
    "Urin-Keton": {"units": [], "aliases": ["Keton Urin", "Ketone Urine", "Ketonurie"],
        "default_ref": {"low": 0.0, "high": 0.0, "unit": ""}},
    "Urin-Bilirubin": {"units": [], "aliases": ["Bilirubin Urin", "Urobilinogen Urin"],
        "default_ref": {"low": 0.0, "high": 0.0, "unit": ""}},
    "Urin-Urobilinogen": {"units": ["mg/dl", "mg/l", "mg/dL", "mg/L"], "aliases": ["Urobilinogen", "Uro"],
        "default_ref": {"low": 0.1, "high": 1.0, "unit": "mg/dl"}},
    "Urin-Nitrit": {"units": [], "aliases": ["Nitrit Urin", "Nitrite Urine"],
        "default_ref": {"low": 0.0, "high": 0.0, "unit": ""}},
    "Urin-Leukozyten": {"units": [], "aliases": ["Leukozyten Urin", "Pyurie", "WBC Urine"],
        "default_ref": {"low": 0.0, "high": 0.0, "unit": ""}},
}


def get_canonical_names() -> list[str]:
    """Return the list of canonical biomarker names (backward-compatible)."""
    return list(_CANONICAL_BIOMARKERS.keys())


def get_biomarker_info(canonical_name: str, registry: dict | None = None) -> dict | None:
    """Return {units, aliases} for a canonical name, or None.

    ``registry`` defaults to the static config baseline; pass the active
    DB registry (database.get_registry) to honor accepted proposals.
    """
    return (registry if registry is not None else _CANONICAL_BIOMARKERS).get(canonical_name)


def resolve_canonical(name: str, unit: str | None = None, registry: dict | None = None) -> str | None:
    """Deterministically resolve an extracted biomarker name to its canonical name.

    Returns the canonical name if found via alias match, or None.
    If multiple aliases match, returns the first one found.

    ``registry`` defaults to the static config baseline; pass the active
    DB registry (database.get_registry) to honor accepted proposals.
    """
    name_lower = name.strip().lower()
    for canonical, info in (registry if registry is not None else _CANONICAL_BIOMARKERS).items():
        # Check exact canonical name match (case-insensitive)
        if canonical.lower() == name_lower:
            return canonical
        # Check aliases
        for alias in info.get("aliases", []):
            if alias.lower() == name_lower:
                return canonical
    return None


def get_default_ref_range(canonical_name: str, unit: str | None = None, registry: dict | None = None) -> tuple[float | None, float | None, str | None]:
    """Return (ref_low, ref_high, unit) from the default reference range for a biomarker.

    If the biomarker has no default_ref or the provided unit doesn't match the
    stored unit, returns (None, None, None).

    ``registry`` defaults to the static config baseline; pass the active
    DB registry (database.get_registry) to honor accepted proposals.

    Unit conversion is supported for common lab conversions:
      - Glucose: mg/dl → mmol/l  (÷ 18)
      - Cholesterol: mg/dl → mmol/l  (÷ 38.67)
      - Triglycerides: mg/dl → mmol/l  (÷ 88.57)
      - Creatinine: mg/dl → µmol/l  (× 88.4)
      - Bilirubin: mg/dl → µmol/l  (× 17.1)
      - Urea/BUN: mg/dl → mmol/l  (÷ 6.006)
      - Iron: µg/dl → nmol/l  (× 179.067)
      - Vitamin B12: pg/ml → pmol/l  (× 0.738)
      - Vitamin D: ng/ml → nmol/l  (× 2.496)
      - Folate: ng/ml → nmol/l  (× 2.266)
      - Testosterone: ng/dl → nmol/l  (× 0.0347)
      - Estradiol: pg/ml → pmol/l  (× 3.671)
      - Progesterone: ng/dl → nmol/l  (× 0.0318)
      - Cortisol: µg/dl → nmol/l  (× 27.59)
      - HbA1c % → mmol/mol  ((% - 2.15) × 10.929)
    """
    active_registry = registry if registry is not None else _CANONICAL_BIOMARKERS
    info = active_registry.get(canonical_name)
    if not info or "default_ref" not in info:
        return None, None, None

    default_ref = info["default_ref"]
    ref_unit = default_ref.get("unit", "")
    ref_low = default_ref.get("low")
    ref_high = default_ref.get("high")

    if unit is None or unit.strip().lower() == ref_unit.strip().lower():
        # Unit matches (or no unit specified) — return as-is
        return ref_low, ref_high, ref_unit

    # Attempt unit conversion (name-aware). If unsupported, return no range
    # rather than a value in the wrong unit (which would cause false flags).
    converted_low, converted_high = _convert_ref_range(canonical_name, ref_low, ref_high, ref_unit, unit)
    if converted_low is None and converted_high is None:
        return None, None, None
    return converted_low, converted_high, unit.strip().lower()


# Per-biomarker linear conversion factors. Each entry is
# (canonical_name_lower, unit_a, unit_b, k) meaning ``value_in_unit_b = value_in_unit_a * k``.
# Both directions are indexed (the reverse uses 1/k). This makes the converter
# name-aware so biomarkers sharing a unit pair (e.g. glucose vs. cholesterol,
# both mg/dl ↔ mmol/l) use their own factor instead of the first branch winning.
_LINEAR_CONVERSIONS = [
    ("glukose (nüchtern)", "mg/dl", "mmol/l", 1.0 / 18.0),
    ("cholesterin gesamt", "mg/dl", "mmol/l", 1.0 / 38.67),
    ("hdl-cholesterin", "mg/dl", "mmol/l", 1.0 / 38.67),
    ("ldl-cholesterin", "mg/dl", "mmol/l", 1.0 / 38.67),
    ("vldl-cholesterin", "mg/dl", "mmol/l", 1.0 / 38.67),
    ("triglyceride", "mg/dl", "mmol/l", 1.0 / 88.57),
    ("harnstoff", "mg/dl", "mmol/l", 1.0 / 6.006),
    ("kreatinin", "mg/dl", "umol/l", 88.4),
    ("bilirubin gesamt", "mg/dl", "umol/l", 17.1),
    ("bilirubin direkt", "mg/dl", "umol/l", 17.1),
    ("harnsäure", "mg/dl", "umol/l", 59.48),
    ("eisen", "ug/dl", "nmol/l", 179.067),
    ("cortisol", "ug/dl", "nmol/l", 27.59),
    ("vitamin b12", "pg/ml", "pmol/l", 0.738),
    ("östradiol (e2)", "pg/ml", "pmol/l", 3.671),
    ("testosteron frei", "pg/ml", "pmol/l", 3.467),
    ("vitamin d (25-oh)", "ng/ml", "nmol/l", 2.496),
    ("folsäure", "ng/ml", "nmol/l", 2.266),
    ("testosteron gesamt", "ng/dl", "nmol/l", 0.0347),
    ("progesteron", "ng/dl", "nmol/l", 0.0318),
    # ng/ml is a common lab-report variant of ng/dl (1 dl = 100 ml → ×100).
    ("testosteron gesamt", "ng/ml", "ng/dl", 100.0),
    ("progesteron", "ng/ml", "ng/dl", 100.0),
]


def _norm_unit(unit: str) -> str:
    """Normalize a unit string for conversion-table lookup (lower-case, µ→u)."""
    return unit.strip().lower().replace("µ", "u")


# Notation variants of the SAME physical unit. Keys are _norm_unit() forms;
# values are group ids. Two units in the same group need NO numeric conversion
# — only a label change (e.g. "/nl" → "10^3/µl", "tsd/µl" → "10³/µl").
_UNIT_EQUIV: dict[str, str] = {
    # Leukozyten / Thrombozyten: 10³ per µl (per nl is the same magnitude:
    # 1 µl = 1000 nl → X/nl == X·10³/µl)
    "/nl": "1e3_per_ul",
    "tsd/ul": "1e3_per_ul",
    "10^3/ul": "1e3_per_ul",
    "10³/ul": "1e3_per_ul",
    # Erythrozyten: 10⁶ per µl (1 µl = 10⁶ pl → X/pl == X·Mio./µl)
    "/pl": "1e6_per_ul",
    "mio./ul": "1e6_per_ul",
    "10^6/ul": "1e6_per_ul",
    "10⁶/ul": "1e6_per_ul",
    # TSH: mU/l == mIU/l (U/IU used interchangeably) and µIU/ml == mIU/l
    # (1 µIU/ml = 1000 µIU/l = 1 mIU/l). "ulu/ml" is a known OCR/typo of µIU/ml.
    "mu/l": "miu_per_l",
    "miu/l": "miu_per_l",
    "uiu/ml": "miu_per_l",
    "ulu/ml": "miu_per_l",
    # Flow: ml per minute (spacing variants)
    "ml/min": "ml_per_min",
    "ml/ min": "ml_per_min",
}


def _units_equivalent(from_unit: str, to_unit: str) -> bool:
    """True when both units are notation variants of the same physical unit."""
    a = _norm_unit(from_unit)
    b = _norm_unit(to_unit)
    if not a or not b:
        return False
    ga = _UNIT_EQUIV.get(a)
    gb = _UNIT_EQUIV.get(b)
    return ga is not None and ga == gb


# Index of (name, from_unit, to_unit) → multiplier. Built once at import time.
_CONVERSION_INDEX = {}
for _name, _ua, _ub, _k in _LINEAR_CONVERSIONS:
    _CONVERSION_INDEX[(_name, _ua, _ub)] = _k
    _CONVERSION_INDEX[(_name, _ub, _ua)] = 1.0 / _k


def _convert_ref_range(canonical_name: str | None, low: float | None, high: float | None, from_unit: str, to_unit: str) -> tuple[float | None, float | None]:
    """Convert reference range values between compatible units (name-aware).

    ``canonical_name`` disambiguates biomarkers that share a unit pair (e.g.
    glucose vs. cholesterol, both mg/dl ↔ mmol/l). Returns
    (converted_low, converted_high), or (None, None) when the conversion is not
    supported — callers must treat that as "no range", never as an unchanged value.
    """
    if low is None and high is None:
        return None, None

    from_u = _norm_unit(from_unit)
    to_u = _norm_unit(to_unit)
    if not from_u or not to_u:
        return None, None
    if from_u == to_u:
        # Identical unit — nothing to convert; values pass through unchanged.
        return low, high

    # Dimensionless target ("none", "—", "-"): the value is a pure ratio and
    # needs no numeric conversion — only the label changes.
    if to_u in ("none", "-", "—", ""):
        return low, high

    # Notation variants of the same physical unit (e.g. "/nl" → "10^3/µl"):
    # numerically identical, so values pass through unchanged.
    if _units_equivalent(from_u, to_u):
        return low, high

    # HbA1c: % ↔ mmol/mol — affine (offset + scale), handled before the linear table.
    if {from_u, to_u} == {"%", "mmol/mol"}:
        if to_u == "mmol/mol":
            return ((low - 2.15) * 10.929 if low is not None else None,
                    (high - 2.15) * 10.929 if high is not None else None)
        return ((low / 10.929) + 2.15 if low is not None else None,
                (high / 10.929) + 2.15 if high is not None else None)

    # Linear conversions, keyed by (canonical name, from_unit, to_unit).
    key = (canonical_name.strip().lower() if canonical_name else "", from_u, to_u)
    factor = _CONVERSION_INDEX.get(key)
    if factor is None:
        return None, None
    return _apply(low, high, factor, False)


def _apply(low: float | None, high: float | None, factor: float, divide: bool) -> tuple[float | None, float | None]:
    """Apply a conversion factor to low/high values.

    If divide=True, divides by factor; otherwise multiplies.
    """
    if divide:
        return (low / factor if low is not None else None), (high / factor if high is not None else None)
    else:
        return (low * factor if low is not None else None), (high * factor if high is not None else None)


CANONICAL_BIOMARKERS = get_canonical_names()  # backward-compatible list

# Per-biomarker critical-value direction. For most markers a value far beyond
# EITHER bound is dangerous ("both"). Some markers are only clinically
# meaningful in one direction, so a violation on the other side must NOT be
# flagged as a critical value:
#   "high" -> only values ABOVE the upper limit matter (e.g. COHb: elevated
#             carboxyhemoglobin = CO poisoning; a low value is not dangerous)
#   "low"  -> only values BELOW the lower limit matter (e.g. eGFR: reduced
#             kidney function is bad, an elevated value is not)
# Keyed by base name (case-insensitive, unit suffix in parens stripped).
# Any marker not listed here defaults to "both".
CRITICAL_DIRECTIONS = {
    "cohb": "high",
}


def get_critical_direction(canonical_name: str | None) -> str:
    """Return the clinically-meaningful direction for critical-value detection.

    Returns 'high' (only above-range is dangerous), 'low' (only below-range),
    or 'both' (default). Keyed by base name, case-insensitive, with any unit
    suffix in parens stripped so "COHb" and "COHb (%)" both resolve.
    """
    if not canonical_name:
        return "both"
    base = canonical_name.strip().lower()
    if "(" in base and base.endswith(")"):
        base = base[:base.rfind("(")].strip()
    return CRITICAL_DIRECTIONS.get(base, "both")


RISK_CONFIG = {
    "critical_margin_pct": 20,
    "concerning_trend_pct": 15,
    "concerning_trend_min_points": 2,
    "max_flags": 10,
    "improvement_lookback_years": 2,
    # Missing-marker detection
    "missing_marker_lookback_days": 180,
    "missing_marker_required_markers": [
        "Glukose (nüchtern)", "HbA1c (%)",
        "Cholesterin gesamt", "HDL-Cholesterin", "LDL-Cholesterin", "Triglyceride",
        "Kreatinin", "eGFR (CKD-EPI)",
        "TSH basal",
        "Ferritin", "Vitamin D (25-OH)", "Vitamin B12",
        "ALT (GPT)", "AST (GOT)", "Gamma-GT (GGT)",
        "Natrium", "Kalium",
    ],
    # Cross-biomarker correlation rules
    "cross_biomarker_rules": [
        {
            "id": "ck_troponin_cardiac",
            "name": "Herzinfarkt-Muster (CK + Troponin)",
            "description_template": "risk_ck_troponin",
            "pattern": "any_high",
            "markers": [
                {"canonical": "CK gesamt", "condition": "abnormal"},
                {"canonical": "Troponin T", "condition": "abnormal"},
            ],
            "severity_modifier": 1,  # upgrade severity
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "ck_troponin_isolated_ck",
            "name": "Isolierte CK-Erhöhung (nicht-kardial)",
            "description_template": "risk_ck_isolated",
            "pattern": "any_high",
            "markers": [
                {"canonical": "CK gesamt", "condition": "abnormal"},
                {"canonical": "Troponin T", "condition": "normal_or_missing"},
            ],
            "severity_modifier": -1,  # downgrade severity
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "ck_mb_troponin_cardiac",
            "name": "Herzmuskelverletzung (CK-MB + Troponin)",
            "description_template": "risk_ckmb_troponin",
            "pattern": "any_high",
            "markers": [
                {"canonical": "CK-MB", "condition": "abnormal"},
                {"canonical": "Troponin T", "condition": "abnormal"},
            ],
            "severity_modifier": 1,
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "ldl_hdl_cardio",
            "name": "Kardiovaskuläres Risiko (LDL hoch + HDL niedrig)",
            "description_template": "risk_ldl_hdl",
            "pattern": "any_high",
            "markers": [
                {"canonical": "LDL-Cholesterin", "condition": "high"},
                {"canonical": "HDL-Cholesterin", "condition": "low"},
            ],
            "severity_modifier": 1,
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "tsh_ft3_ft4_hyper",
            "name": "Schilddrüsenüberfunktion (TSH niedrig + fT3/fT4 hoch)",
            "description_template": "risk_tsh_ft",
            "pattern": "any_high",
            "markers": [
                {"canonical": "TSH basal", "condition": "low"},
                {"canonical": "fT3", "condition": "high"},
            ],
            "severity_modifier": 1,
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "tsh_ft4_hyper",
            "name": "Schilddrüsenüberfunktion (TSH niedrig + fT4 hoch)",
            "description_template": "risk_tsh_ft",
            "pattern": "any_high",
            "markers": [
                {"canonical": "TSH basal", "condition": "low"},
                {"canonical": "fT4", "condition": "high"},
            ],
            "severity_modifier": 1,
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "tsh_ft3_ft4_hypo",
            "name": "Schilddrüsenunterfunktion (TSH hoch + fT3/fT4 niedrig)",
            "description_template": "risk_tsh_ft",
            "pattern": "any_high",
            "markers": [
                {"canonical": "TSH basal", "condition": "high"},
                {"canonical": "fT3", "condition": "low"},
            ],
            "severity_modifier": 1,
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "tsh_ft4_hypo",
            "name": "Schilddrüsenunterfunktion (TSH hoch + fT4 niedrig)",
            "description_template": "risk_tsh_ft",
            "pattern": "any_high",
            "markers": [
                {"canonical": "TSH basal", "condition": "high"},
                {"canonical": "fT4", "condition": "low"},
            ],
            "severity_modifier": 1,
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "crp_bsg_inflammation",
            "name": "Entzündungsmarker (CRP + BSG erhöht)",
            "description_template": "risk_crp_bsg",
            "pattern": "any_high",
            "markers": [
                {"canonical": "CRP", "condition": "abnormal"},
                {"canonical": "BSG", "condition": "abnormal"},
            ],
            "severity_modifier": 1,
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "creatinine_egfr_renal",
            "name": "Nierenfunktion (Kreatinin hoch + eGFR niedrig)",
            "description_template": "risk_crea_egfr",
            "pattern": "any_high",
            "markers": [
                {"canonical": "Kreatinin", "condition": "high"},
                {"canonical": "eGFR (CKD-EPI)", "condition": "low"},
            ],
            "severity_modifier": 1,
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "glucose_hba1c_diabetes",
            "name": "Diabetes-Muster (Glukose + HbA1c erhöht)",
            "description_template": "risk_glucose_hba1c",
            "pattern": "any_high",
            "markers": [
                {"canonical": "Glukose (nüchtern)", "condition": "high"},
                {"canonical": "HbA1c (%)", "condition": "high"},
            ],
            "severity_modifier": 1,
            "category": "cross_biomarker_correlation",
        },
        {
            "id": "iron_transferrin",
            "name": "Eisenstoffwechsel (Eisen + Transferrinsättigung abnormal)",
            "description_template": "risk_iron_tsat",
            "pattern": "any_high",
            "markers": [
                {"canonical": "Eisen", "condition": "abnormal"},
                {"canonical": "Transferrinsättigung", "condition": "abnormal"},
            ],
            "severity_modifier": 0,
            "category": "cross_biomarker_correlation",
        },
    ],
}
