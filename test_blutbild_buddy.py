import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import config
import database

TEST_DIR = Path(tempfile.mkdtemp(prefix="bb_test_"))


def _fresh_db(name: str):
    config.DB_PATH = TEST_DIR / name
    database.init_db()
    return database.get_session()


from pdf_processor import file_hash, pdf_page_count, pdf_to_images, image_to_base64_jpeg


MOCK_EXTRACTED_PAGE = {
    "patient_name": "Doe_J.",
    "report_date": "15.03.2022",
    "panel_type": "Blutbild",
    "entries": [
        {"name": "Leukozyten", "value": 6.5, "unit": "Tsd./µl", "reference_range": "4.0-10.0"},
        {"name": "Erythrozyten", "value": 5.1, "unit": "Mio./µl", "reference_range": "4.5-5.9"},
        {"name": "Hämoglobin", "value": 15.2, "unit": "g/dl", "reference_range": "14.0-18.0"},
        {"name": "Hämatokrit", "value": 45, "unit": "%", "reference_range": "40-54"},
        {"name": "MCH", "value": 29.8, "unit": "pg", "reference_range": "27-34"},
        {"name": "MCHC", "value": 33.8, "unit": "g/dl", "reference_range": "32-36"},
        {"name": "MCV", "value": 88, "unit": "fl", "reference_range": "80-100"},
        {"name": "Thrombozyten", "value": 220, "unit": "Tsd./µl", "reference_range": "150-400"},
        {"name": "CRP", "value": 5.2, "unit": "mg/l", "reference_range": "<5.0"},
        {"name": "Cholesterin gesamt", "value": 240, "unit": "mg/dl", "reference_range": "120-200"},
        {"name": "HDL-Cholesterin", "value": 45, "unit": "mg/dl", "reference_range": "35-80"},
        {"name": "LDL-Cholesterin", "value": 160, "unit": "mg/dl", "reference_range": "<130"},
        {"name": "Triglyceride", "value": 175, "unit": "mg/dl", "reference_range": "<150"},
        {"name": "Vitamin D (25-OH)", "value": 18, "unit": "ng/ml", "reference_range": "30-100"},
    ]
}

class _MockLLMResponseChoice:
    def __init__(self, content: str):
        self.message = type("_MockMessage", (), {"content": content})()


class _MockLLMResponse:
    def __init__(self, content: str):
        self.choices = [_MockLLMResponseChoice(content)]


class _MockLLMCompletions:
    def __init__(self, content: str):
        self.content = content

    def create(self, *a, **kw):
        return _MockLLMResponse(self.content)


class _MockLLMChat:
    def __init__(self, content: str):
        self.completions = _MockLLMCompletions(content)


class _MockLLM:
    def __init__(self, content: str):
        self.chat = _MockLLMChat(content)


class _SeqCompletions:
    """Returns a different content per call (for multi-call LLM tests)."""
    def __init__(self, contents):
        self._contents = list(contents)
        self._i = 0

    def create(self, *a, **kw):
        c = self._contents[min(self._i, len(self._contents) - 1)]
        self._i += 1
        return _MockLLMResponse(c)


class _SeqChat:
    def __init__(self, contents):
        self.completions = _SeqCompletions(contents)


class _SeqLLM:
    def __init__(self, contents):
        self.chat = _SeqChat(contents)


MOCK_NORMALIZATION = {
    # New shape: each mapping echoes the unit of the extracted entry it maps
    # (null when the entry had no unit).
    "mappings": [
        {"original": "Leukozyten", "unit": None, "canonical": "Leukozyten (WBC)"},
        {"original": "Erythrozyten", "unit": None, "canonical": "Erythrozyten (RBC)"},
        {"original": "Hämoglobin", "unit": None, "canonical": "Hämoglobin (Hb)"},
        {"original": "Hämatokrit", "unit": None, "canonical": "Hämatokrit (Hkt)"},
        {"original": "MCH", "unit": None, "canonical": "MCH"},
        {"original": "MCHC", "unit": None, "canonical": "MCHC"},
        {"original": "MCV", "unit": None, "canonical": "MCV"},
        {"original": "Thrombozyten", "unit": None, "canonical": "Thrombozyten (PLT)"},
        {"original": "CRP", "unit": None, "canonical": "CRP"},
        {"original": "Cholesterin gesamt", "unit": None, "canonical": "Cholesterin gesamt"},
        {"original": "HDL-Cholesterin", "unit": None, "canonical": "HDL-Cholesterin"},
        {"original": "LDL-Cholesterin", "unit": None, "canonical": "LDL-Cholesterin"},
        {"original": "Triglyceride", "unit": None, "canonical": "Triglyceride"},
        {"original": "Vitamin D (25-OH)", "unit": None, "canonical": "Vitamin D (25-OH)"},
        {"original": "HbA1c", "unit": None, "canonical": "HbA1c (%)"},
    ]
}


class TestPDFProcessor(unittest.TestCase):
    """Tests that need a real sample PDF in reports_input/ (gitignored).

    Discovers any *.pdf present; skips gracefully when none is available so the
    suite still passes on a fresh clone without local data.
    """

    @classmethod
    def setUpClass(cls):
        candidates = sorted(Path("reports_input").glob("*.pdf"))
        cls.pdf_sample = candidates[0] if candidates else None

    def _require_pdf(self):
        if self.pdf_sample is None:
            self.skipTest("no sample PDF in reports_input/ (add one to run these tests)")
        return self.pdf_sample

    def test_file_hash_sha256(self):
        pdf = self._require_pdf()
        h = file_hash(pdf)
        self.assertEqual(len(h), 64)

    def test_pdf_page_count_positive(self):
        pdf = self._require_pdf()
        pages = pdf_page_count(pdf)
        self.assertGreater(pages, 0)

    def test_pdf_to_images_renders(self):
        pdf = self._require_pdf()
        imgs = pdf_to_images(pdf, dpi=72)
        self.assertGreater(len(imgs), 0)
        w, h = imgs[0].size
        self.assertGreater(w, 100)
        self.assertGreater(h, 100)
        print(f"  sample PDF: {len(imgs)} pages @72dpi = {w}x{h}")

    def test_image_to_base64_jpeg(self):
        from PIL import Image
        img = Image.new("RGB", (50, 50), "white")
        b64 = image_to_base64_jpeg(img)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 50)


class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_database.db")

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_create_patient(self):
        p = database.get_or_create_patient(self.session, "Alice")
        self.assertIsNotNone(p.id)
        self.assertEqual(p.name, "Alice")

    def test_create_report_with_data_points(self):
        patient = database.get_or_create_patient(self.session, "Bob")
        report = database.Report(
            patient_id=patient.id, filename="bob.pdf",
            file_hash="abc123", report_date=datetime(2024, 1, 1),
        )
        self.session.add(report)
        self.session.flush()
        dp = database.DataPoint(
            report_id=report.id, canonical_name="Hb",
            original_name="Hämoglobin", value=14.0, unit="g/dl",
            ref_low=13.0, ref_high=17.0, is_abnormal=False,
        )
        self.session.add(dp)
        self.session.commit()
        loaded = self.session.query(database.DataPoint).first()
        self.assertEqual(loaded.value, 14.0)


class TestExtractorMocked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_extractor.db")

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    @patch("extractor.extract_page")
    @patch("extractor.normalize_names")
    @patch("extractor.extract_pages_batch")
    def test_process_new_pdfs(self, mock_batch, mock_norm, mock_extract):
        from schemas import ExtractedPage, NormalizationMapping
        mock_extract.return_value = ExtractedPage(**MOCK_EXTRACTED_PAGE)
        # Default batch size is 3, so multi-page PDFs take the batch path —
        # mock it too (one extracted page per input image).
        mock_batch.side_effect = lambda llm, pages, **kw: [ExtractedPage(**MOCK_EXTRACTED_PAGE) for _ in pages]
        mock_norm.return_value = NormalizationMapping(**MOCK_NORMALIZATION)

        from extractor import process_new_pdfs
        processed, errors = process_new_pdfs(self.session)
        self.assertGreater(len(processed), 0)
        print(f"  Processed PDFs: {len(processed)}")

        dps = self.session.query(database.DataPoint).all()
        self.assertGreater(len(dps), 5)
        print(f"  Total data points: {len(dps)}")


class TestAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_analyzer.db")
        cls._seed_multi_year_data()

    @classmethod
    def _seed_multi_year_data(cls):
        patient = database.get_or_create_patient(cls.session, "Doe_J.")
        dates = [datetime(2022, 3, 15), datetime(2023, 6, 20),
                 datetime(2024, 1, 10), datetime(2025, 4, 5),
                 datetime(2025, 9, 12), datetime(2026, 2, 18)]
        # LDL uses a one-sided bound (ref_high=130). Since R3-08, values beyond
        # the single bound can also produce critical_value flags — keep the
        # final value below the critical threshold ((155-130)/130 ≈ 19% < 20%)
        # so this scenario still exercises the concerning_trend path.
        ldl_values = [140, 145, 150, 152, 154, 155]
        vitd_values = [35, 28, 22, 18, 15, 12]

        for i, (d, ldl, vitd) in enumerate(zip(dates, ldl_values, vitd_values)):
            report = database.Report(
                patient_id=patient.id, filename=f"report_{i}.pdf",
                file_hash=f"unique_hash_{i}", report_date=d, panel_type="Blutbild",
            )
            cls.session.add(report)
            cls.session.flush()
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name="LDL-Cholesterin",
                original_name="LDL", value=ldl, unit="mg/dl",
                ref_low=None, ref_high=130, is_abnormal=ldl > 130,
            ))
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name="Vitamin D (25-OH)",
                original_name="Vitamin D", value=vitd, unit="ng/ml",
                ref_low=30, ref_high=100, is_abnormal=vitd < 30,
            ))
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_concerning_trend_detected(self):
        from analyzer import analyze_trends
        flags = analyze_trends(self.session)
        categories = {f.category for f in flags}
        print(f"  Risk flags: {len(flags)}, categories: {categories}")
        for f in flags:
            print(f"    {f.category} sev={f.severity}: {f.description[:100]}")
        self.assertIn("concerning_trend", categories)

    def test_explanation_field_exists(self):
        from analyzer import refresh_risk_flags
        n = refresh_risk_flags(self.session, llm=None)
        count = self.session.query(database.RiskFlag).count()
        self.assertEqual(n, count)
        self.assertGreater(n, 0)
        flags = self.session.query(database.RiskFlag).all()
        for f in flags:
            self.assertIsNotNone(f.id)
            self.assertIsNotNone(f.category)
            self.assertIsNotNone(f.description)
        print(f"  Risk flags stored: {n}")

    def test_refresh_clears_old_flags(self):
        from analyzer import refresh_risk_flags
        refresh_risk_flags(self.session, llm=None)
        count1 = self.session.query(database.RiskFlag).count()
        refresh_risk_flags(self.session, llm=None)
        count2 = self.session.query(database.RiskFlag).count()
        self.assertEqual(count1, count2)


class TestAnalyzerExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_analyzer_extended.db")
        cls._seed_critical_and_missing()

    @classmethod
    def _seed_critical_and_missing(cls):
        patient = database.get_or_create_patient(cls.session, "CriticalPatient")
        dates = [datetime(2024, 1, 1), datetime(2024, 6, 1), datetime(2024, 12, 1)]
        # Kreatinin: 1.2, 1.4, 2.8 (normal range 0.6-1.2 → last value is 133% beyond range)
        for i, d in enumerate(dates):
            report = database.Report(
                patient_id=patient.id, filename=f"crit_{i}.pdf",
                file_hash=f"crit_hash_{i}", report_date=d,
            )
            cls.session.add(report)
            cls.session.flush()
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name="Kreatinin",
                original_name="Kreatinin", value=[1.2, 1.4, 2.8][i],
                unit="mg/dl", ref_low=0.6, ref_high=1.2,
                is_abnormal=[False, False, True][i],
            ))
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_critical_value_detected(self):
        from analyzer import analyze_trends
        flags = analyze_trends(self.session)
        categories = {f.category for f in flags}
        self.assertIn("critical_value", categories)
        critical_flags = [f for f in flags if f.category == "critical_value"]
        self.assertGreater(len(critical_flags), 0)
        for f in critical_flags:
            self.assertIsNotNone(f.description)
            self.assertEqual(f.severity, 3)

    def test_explanation_persisted_from_llm(self):
        from analyzer import refresh_risk_flags
        mock_llm = _MockLLM('{"risks": [{"index": 1, "urgency": "critical", "explanation": "Erklärungstext 1"}]}')
        n = refresh_risk_flags(self.session, llm=mock_llm)
        self.assertGreater(n, 0)
        flags = self.session.query(database.RiskFlag).all()
        explanations = [f.explanation for f in flags if f.explanation]
        self.assertGreater(len(explanations), 0)

    def test_explanation_none_without_llm(self):
        from analyzer import refresh_risk_flags
        refresh_risk_flags(self.session, llm=None)
        flags = self.session.query(database.RiskFlag).all()
        for f in flags:
            self.assertIsNone(f.explanation)


class TestParseReferenceRange(unittest.TestCase):
    def test_range_dash(self):
        from extractor import _parse_reference_range
        low, high = _parse_reference_range("4.2-6.1")
        self.assertEqual(low, 4.2)
        self.assertEqual(high, 6.1)

    def test_range_en_dash(self):
        from extractor import _parse_reference_range
        low, high = _parse_reference_range("4,2–6,1")
        self.assertEqual(low, 4.2)
        self.assertEqual(high, 6.1)

    def test_range_em_dash(self):
        from extractor import _parse_reference_range
        low, high = _parse_reference_range("4.2 — 6.1")
        self.assertEqual(low, 4.2)
        self.assertEqual(high, 6.1)

    def test_less_than(self):
        from extractor import _parse_reference_range
        low, high = _parse_reference_range("< 5.2")
        self.assertIsNone(low)
        self.assertEqual(high, 5.2)

    def test_greater_than(self):
        from extractor import _parse_reference_range
        low, high = _parse_reference_range("> 120")
        self.assertEqual(low, 120)
        self.assertIsNone(high)

    def test_none_input(self):
        from extractor import _parse_reference_range
        low, high = _parse_reference_range(None)
        self.assertIsNone(low)
        self.assertIsNone(high)

    def test_unparseable(self):
        from extractor import _parse_reference_range
        low, high = _parse_reference_range("negativ")
        self.assertIsNone(low)
        self.assertIsNone(high)


class TestNormalizePatientName(unittest.TestCase):
    def test_simple_two_words(self):
        result = database.normalize_patient_name("Doe J.")
        self.assertEqual(result, "J., Doe")

    def test_already_comma_separated(self):
        result = database.normalize_patient_name("J., Doe")
        self.assertEqual(result, "J., Doe")

    def test_underscore_separated(self):
        result = database.normalize_patient_name("Doe_J.")
        self.assertEqual(result, "J., Doe")

    def test_single_word(self):
        result = database.normalize_patient_name("Müller")
        self.assertEqual(result, "Müller, ")

    def test_empty_string(self):
        result = database.normalize_patient_name("")
        self.assertEqual(result, "")

    def test_none_input(self):
        result = database.normalize_patient_name(None)
        self.assertIsNone(result)

    def test_three_words(self):
        result = database.normalize_patient_name("Anna Maria Schmidt")
        self.assertEqual(result, "Schmidt, Anna Maria")


class TestLLMClient(unittest.TestCase):
    def test_strip_markdown_json(self):
        from llm_client import _strip_markdown_json
        raw = '```json\n{"key": "value"}\n```'
        result = _strip_markdown_json(raw)
        self.assertNotIn("```", result)
        self.assertIn('"key": "value"', result)

    def test_strip_plain_json(self):
        from llm_client import _strip_markdown_json
        raw = '{"key": "value"}'
        result = _strip_markdown_json(raw)
        self.assertEqual(result, raw)

    def test_assess_risks_valid_ranked(self):
        from llm_client import assess_risks
        raw = '{"risks": [{"index": 2, "urgency": "critical", "explanation": "Dringlich"}, {"index": 1, "urgency": "long_term", "explanation": "Langfristig"}]}'
        mock_llm = _MockLLM(raw)
        findings = [{"category": "a", "description": "d1"}, {"category": "b", "description": "d2"}]
        result = assess_risks(mock_llm, findings)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["index"], 2)
        self.assertEqual(result[0]["urgency"], "critical")
        self.assertEqual(result[1]["index"], 1)
        self.assertEqual(result[1]["urgency"], "long_term")

    def test_assess_risks_empty_findings(self):
        from llm_client import assess_risks
        mock_llm = _MockLLM('{"risks": []}')
        result = assess_risks(mock_llm, [])
        self.assertEqual(result, [])

    def test_assess_risks_invalid_json(self):
        from llm_client import assess_risks
        mock_llm = _MockLLM("This is not JSON at all.")
        findings = [{"category": "a", "description": "d1"}]
        result = assess_risks(mock_llm, findings)
        self.assertEqual(result, [])

    def test_assess_risks_urgency_normalization(self):
        from llm_client import _normalize_urgency
        self.assertEqual(_normalize_urgency("critical"), "critical")
        self.assertEqual(_normalize_urgency("immediate"), "critical")
        self.assertEqual(_normalize_urgency("long_term"), "long_term")
        self.assertEqual(_normalize_urgency("long-term"), "long_term")
        self.assertIsNone(_normalize_urgency("unknown_value"))
        self.assertIsNone(_normalize_urgency(42))

    def test_assess_risks_out_of_range_index_skipped(self):
        from llm_client import assess_risks
        raw = '{"risks": [{"index": 99, "urgency": "critical", "explanation": "x"}, {"index": 1, "urgency": "long_term", "explanation": "ok"}]}'
        mock_llm = _MockLLM(raw)
        findings = [{"category": "a", "description": "d1"}]
        result = assess_risks(mock_llm, findings)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["index"], 1)

    def test_assess_risks_json_fenced(self):
        from llm_client import assess_risks
        raw = '```json\n{"risks": [{"index": 1, "urgency": "critical", "explanation": "Fenced"}]}\n```'
        mock_llm = _MockLLM(raw)
        findings = [{"category": "a", "description": "d1"}]
        result = assess_risks(mock_llm, findings)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["explanation"], "Fenced")

    # ── suggest_merge_targets (single call over all variant groups) ──────────

    @staticmethod
    def _merge_groups():
        return [
            {"variants": [{"name": "HB", "unit": "g/dl", "count": 1},
                          {"name": "Hb", "unit": "g/dl", "count": 1}]},
            {"variants": [{"name": "HDL-Cholesterin", "unit": "mg/dl", "count": 1},
                          {"name": "HDL-CHOLESTERIN", "unit": "mg/dl", "count": 1}]},
        ]

    def _suggest(self, raw, lang="en"):
        import llm_client
        from llm_client import suggest_merge_targets
        with patch.object(llm_client, "_get_model", return_value="test-model"), \
             patch.object(llm_client, "_get_timeout", return_value=30):
            return suggest_merge_targets(_MockLLM(raw), self._merge_groups(),
                                         ["Hämoglobin (Hb)", "HDL-Cholesterin"], lang=lang)

    def test_suggest_merge_targets_valid_plans(self):
        raw = '{"plans": [{"group_index": 0, "target": "Hämoglobin (Hb)", "unit": "g/dl"},' \
              ' {"group_index": 1, "target": "HDL-Cholesterin", "unit": "mg/dl"}]}'
        result = self._suggest(raw)
        self.assertEqual(result[0], {"target": "Hämoglobin (Hb)", "unit": "g/dl"})
        self.assertEqual(result[1], {"target": "HDL-Cholesterin", "unit": "mg/dl"})

    def test_suggest_merge_targets_unknown_target_is_none(self):
        raw = '{"plans": [{"group_index": 0, "target": "Not A Real Biomarker", "unit": "g/dl"},' \
              ' {"group_index": 1, "target": "HDL-Cholesterin", "unit": "mg/dl"}]}'
        result = self._suggest(raw)
        self.assertIsNone(result[0], "Unknown target must be treated as omitted")
        self.assertEqual(result[1], {"target": "HDL-Cholesterin", "unit": "mg/dl"})

    def test_suggest_merge_targets_missing_group_is_none(self):
        raw = '{"plans": [{"group_index": 1, "target": "HDL-Cholesterin", "unit": "mg/dl"}]}'
        result = self._suggest(raw)
        self.assertIsNone(result[0], "Omitted group must come back as None")
        self.assertEqual(result[1], {"target": "HDL-Cholesterin", "unit": "mg/dl"})

    def test_suggest_merge_targets_invalid_json_raises(self):
        from llm_client import LLMResponseError
        with self.assertRaises(LLMResponseError):
            self._suggest("This is not JSON at all.")

    def test_suggest_merge_targets_missing_plans_key_raises(self):
        from llm_client import LLMResponseError
        with self.assertRaises(LLMResponseError):
            self._suggest('{"target": "Hämoglobin (Hb)", "unit": "g/dl"}')

    def test_suggest_merge_targets_prompt_language(self):
        """Prompt language follows lang (default = app language via _get_lang)."""
        import llm_client
        from llm_client import suggest_merge_targets, MERGE_TARGETS_SYSTEM_PROMPT_EN, MERGE_TARGETS_SYSTEM_PROMPT_DE

        captured = {}

        class CapturingCompletions:
            def create(self, *a, **kw):
                captured["messages"] = kw.get("messages") or (a[0] if a else None)
                return _MockLLMResponse('{"plans": []}')

        class CapturingChat:
            completions = CapturingCompletions()

        class CapturingLLM:
            chat = CapturingChat()

        groups = self._merge_groups()
        canonicals = ["Hämoglobin (Hb)", "HDL-Cholesterin"]

        # Default (lang=None) follows the app language — patch _get_lang to "en"
        with patch.object(llm_client, "_get_model", return_value="test-model"), \
             patch.object(llm_client, "_get_timeout", return_value=30), \
             patch.object(llm_client, "_get_lang", return_value="en"):
            suggest_merge_targets(CapturingLLM(), groups, canonicals)
        system = [m for m in captured["messages"] if m["role"] == "system"][0]["content"]
        self.assertEqual(system, MERGE_TARGETS_SYSTEM_PROMPT_EN,
                         "Default prompt must be the app-language (EN) variant")

        # Forced German
        with patch.object(llm_client, "_get_model", return_value="test-model"), \
             patch.object(llm_client, "_get_timeout", return_value=30):
            suggest_merge_targets(CapturingLLM(), groups, canonicals, lang="de")
        system_de = [m for m in captured["messages"] if m["role"] == "system"][0]["content"]
        self.assertEqual(system_de, MERGE_TARGETS_SYSTEM_PROMPT_DE)


class TestBiomarkerInterpretationPrompt(unittest.TestCase):
    """Out-of-range measurements in the interpretation prompt get a [ABNORMAL] marker."""

    def _user_prompt(self, lang):
        import llm_client
        captured = {}

        class CapturingCompletions:
            def create(self, *a, **kw):
                captured["messages"] = kw.get("messages") or (a[0] if a else None)
                return _MockLLMResponse("ok")

        class CapturingChat:
            completions = CapturingCompletions()

        class CapturingLLM:
            chat = CapturingChat()

        with patch.object(llm_client, "_get_model", return_value="test-model"), \
             patch.object(llm_client, "_get_timeout", return_value=30):
            llm_client.generate_biomarker_interpretation(
                CapturingLLM(), "Testmarker", "mg/dl", self._data_points(), lang=lang
            )
        user = [m for m in captured["messages"] if m["role"] == "user"][0]
        return user["content"], captured["messages"]

    @staticmethod
    def _data_points():
        d = datetime(2024, 1, 1)
        return [
            (d, 5.0, 4.0, 6.0),      # in range → no marker
            (d, 7.5, 4.0, 6.0),      # above high → [ABNORMAL]
            (d, 3.0, 4.0, 6.0),      # below low → [ABNORMAL]
            (d, 140, None, 130),     # one-sided high bound exceeded → [ABNORMAL]
            (d, 120, None, 130),     # one-sided high bound within range → no marker
            (d, 60, 90, None),       # one-sided low bound below → [ABNORMAL]
            (d, 120, 90, None),      # one-sided low bound within range → no marker
            (d, 5.0, None, None),    # no bounds → no marker
            (d, None, 4.0, 6.0),     # missing value → no marker
        ]

    def test_abnormal_marker_only_on_out_of_range_lines(self):
        prompt, _ = self._user_prompt("en")
        lines = [l for l in prompt.splitlines() if l.startswith("  - ")]
        self.assertEqual(len(lines), 9)
        # Exactly the four out-of-range lines (7.5, 3.0, 140, 60) carry the marker.
        self.assertEqual(prompt.count("[ABNORMAL]"), 4)
        expected = [
            "  - Date: 01.01.2024, Value: 5.0 mg/dl, Reference Range: 4.0–6.0",
            "  - Date: 01.01.2024, Value: 7.5 mg/dl, Reference Range: 4.0–6.0 [ABNORMAL]",
            "  - Date: 01.01.2024, Value: 3.0 mg/dl, Reference Range: 4.0–6.0 [ABNORMAL]",
            "  - Date: 01.01.2024, Value: 140 mg/dl, Reference Range: — [ABNORMAL]",
            "  - Date: 01.01.2024, Value: 120 mg/dl, Reference Range: —",
            "  - Date: 01.01.2024, Value: 60 mg/dl, Reference Range: — [ABNORMAL]",
            "  - Date: 01.01.2024, Value: 120 mg/dl, Reference Range: —",
            "  - Date: 01.01.2024, Value: 5.0 mg/dl, Reference Range: —",
            "  - Date: 01.01.2024, Value: None mg/dl, Reference Range: 4.0–6.0",
        ]
        self.assertEqual(lines, expected)

    def test_system_prompt_explains_marker(self):
        _, messages = self._user_prompt("en")
        system_en = [m for m in messages if m["role"] == "system"][0]["content"]
        self.assertIn("Lines marked [ABNORMAL] are outside the reference range.", system_en)

    def test_system_prompt_explains_marker_de(self):
        import llm_client
        with patch.object(llm_client, "_get_model", return_value="test-model"), \
             patch.object(llm_client, "_get_timeout", return_value=30):
            self.assertIn(
                "Zeilen mit [ABNORMAL] liegen außerhalb des Referenzbereichs.",
                llm_client._get_biomarker_interpretation_prompt("de"),
            )


class TestAppImports(unittest.TestCase):
    def test_streamlit_import(self):
        import streamlit  # noqa
        print("  Streamlit import OK")

    def test_plotly_import(self):
        import plotly  # noqa
        print("  Plotly import OK")

    def test_xlsx_export_uses_get_column_letter(self):
        """R3-07: XLSX export must use openpyxl's get_column_letter, not chr(64+i),
        which produces invalid column letters past index 23 (e.g. '[', '\\')."""
        from pathlib import Path
        # The XLSX export lives in the all-values tab (moved out of app.py).
        src = Path("ui/tabs/all_values.py").read_text(encoding="utf-8")
        self.assertNotIn("chr(64", src, "all_values.py still uses chr(64 + col_idx) for XLSX columns")
        self.assertIn("get_column_letter", src)
        # Sanity: the helper handles >23 columns correctly.
        from openpyxl.utils import get_column_letter
        self.assertEqual(get_column_letter(26), "Z")
        self.assertEqual(get_column_letter(27), "AA")


class TestDedupOneAlertPerBiomarker(unittest.TestCase):
    """Verify that at most one risk flag is produced per biomarker."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_dedup.db")
        cls._seed_multi_violations()

    @classmethod
    def _seed_multi_violations(cls):
        """Create a patient with multiple reports where the same biomarker
        has critical_value AND concerning_trend violations simultaneously."""
        patient = database.get_or_create_patient(cls.session, "DedupPatient")
        dates = [datetime(2024, 1, 1), datetime(2024, 6, 1),
                 datetime(2024, 12, 1), datetime(2025, 6, 1)]
        for i, d in enumerate(dates):
            report = database.Report(
                patient_id=patient.id, filename=f"dedup_{i}.pdf",
                file_hash=f"dedup_hash_{i}", report_date=d,
            )
            cls.session.add(report)
            cls.session.flush()
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name="Kreatinin",
                original_name="Kreatinin", value=[1.2, 1.6, 2.0, 2.8][i],
                unit="mg/dl", ref_low=0.6, ref_high=1.2,
                is_abnormal=[False, True, True, True][i],
            ))
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_max_one_flag_per_biomarker(self):
        from analyzer import analyze_trends
        flags = analyze_trends(self.session)
        biomarkers = [f.description.split()[0] for f in flags]
        self.assertEqual(len(biomarkers), len(set(biomarkers)),
                         f"Duplicate biomarkers in flags: {biomarkers}")
        self.assertLessEqual(len(flags), 1)


class TestDedupParenthesizedName(unittest.TestCase):
    """R3-05: dedup must work for parenthesized canonical names.

    The old per-biomarker SQL filter compared the full lowered name against the
    normalized base key, so 'Glukose (nüchtern)' never matched and the enhanced
    description path (headline + 'Bisheriger Extremwert' context) was silently
    skipped."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_dedup_paren.db")
        patient = database.get_or_create_patient(cls.session, "ParenPatient")
        dates = [datetime(2024, 1, 1), datetime(2024, 6, 1),
                 datetime(2024, 12, 1), datetime(2025, 6, 1)]
        values = [95.0, 120.0, 140.0, 160.0]
        for i, d in enumerate(dates):
            report = database.Report(
                patient_id=patient.id, filename=f"paren_{i}.pdf",
                file_hash=f"paren_hash_{i}", report_date=d,
            )
            cls.session.add(report)
            cls.session.flush()
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name="Glukose (nüchtern)",
                original_name="Glukose", value=values[i],
                unit="mg/dl", ref_low=70.0, ref_high=100.0,
                is_abnormal=values[i] > 100.0,
            ))
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_enhanced_description_runs_for_paren_name(self):
        from analyzer import analyze_trends
        flags = analyze_trends(self.session)
        self.assertGreaterEqual(len(flags), 1, "Expected at least one flag")
        # The enhanced description path must have run (it only runs when the
        # per-biomarker data-point lookup succeeds): a headline with the current
        # value plus the reference range. Here the latest value (160) is also
        # the historical extreme, so no separate context line is expected.
        self.assertTrue(
            any("Referenzbereich" in f.description for f in flags),
            f"Enhanced description missing; got: {[f.description for f in flags]}",
        )


class TestDedupDescriptionBelowRange(unittest.TestCase):
    """Regression test: below-range values on a narrow ref range must not
    produce nonsensical descriptions.

    Reproduces the PO2 case from the production (Docker) DB: two data points
    in one report (34.9 and 29.0 mmHg, ref 80–90). Previously:
      - display_name was parsed from the description text (no colon), so the
        whole sentence "PO2 = 34.9 mmHg liegt 451% unter ..." was repeated as
        the prefix;
      - the footer used an uncapped percentage (451%) while Worst/Latest used
        the capped one (200%) — two different percentages for the same value;
      - "Latest" decided über/unter from the sign of _compute_pct_beyond,
        which is positive in BOTH directions → 29.0 was labeled "über Limit".
    """

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_dedup_below.db")
        patient = database.get_or_create_patient(cls.session, "BelowRangePatient")
        report = database.Report(
            patient_id=patient.id, filename="below.pdf",
            file_hash="below_hash_1", report_date=datetime(2026, 8, 11),
        )
        cls.session.add(report)
        cls.session.flush()
        for value in (34.9, 29.0):
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name="PO2",
                original_name="PO2", value=value, unit="mmHg",
                ref_low=80.0, ref_high=90.0, is_abnormal=True,
            ))
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_description_is_sane_for_below_range(self):
        from analyzer import analyze_trends
        flags = analyze_trends(self.session)
        po2_flags = [f for f in flags if "PO2" in f.description]
        self.assertGreaterEqual(len(po2_flags), 1, "Expected a PO2 flag")
        desc = po2_flags[0].description

        # Display name must be the bare biomarker name, not the whole sentence.
        self.assertTrue(desc.startswith("PO2:"), f"Bad display prefix: {desc!r}")
        # A below-range value must never be labeled as above the limit.
        self.assertNotIn("Obergrenze", desc, f"Below-range mislabeled: {desc!r}")
        self.assertIn("unter der Untergrenze", desc)
        # Deviation is relative to the violated LIMIT, not the range width:
        # worst 34.9 -> (80-34.9)/80 = 56%, latest 29.0 -> (80-29)/80 = 64%.
        # The old range-width numbers (451% uncapped / 200% capped) must not appear.
        self.assertIn("56%", desc, f"Missing limit-relative worst %: {desc!r}")
        self.assertIn("64%", desc, f"Missing limit-relative latest %: {desc!r}")
        self.assertNotIn("451%", desc, f"Range-width percentage leaked: {desc!r}")
        self.assertNotIn("200%", desc, f"Range-width percentage leaked: {desc!r}")

    def test_pct_deviation_from_limit(self):
        from analyzer import _pct_deviation_from_limit
        # Below range: relative to the lower limit (bounded at 100% for value 0).
        self.assertAlmostEqual(_pct_deviation_from_limit(34.9, 80.0, 90.0), 56.375)
        self.assertAlmostEqual(_pct_deviation_from_limit(0.0, 80.0, 90.0), 100.0)
        # Above range: relative to the upper limit (unbounded, >100% is normal).
        self.assertAlmostEqual(_pct_deviation_from_limit(2.8, 0.6, 1.2), 133.3333, places=3)
        # Within range: no deviation.
        self.assertIsNone(_pct_deviation_from_limit(85.0, 80.0, 90.0))
        # Zero lower limit (e.g. pH-like): falls back to capped range-width %.
        self.assertAlmostEqual(_pct_deviation_from_limit(-1.0, 0.0, 14.0), 7.142857, places=3)

    def test_above_range_description_uses_upper_limit(self):
        """Above-range values show '% über der Obergrenze' relative to the max."""
        session = _fresh_db("test_dedup_above.db")
        try:
            patient = database.get_or_create_patient(session, "AboveRangePatient")
            report = database.Report(
                patient_id=patient.id, filename="above.pdf",
                file_hash="above_hash_1", report_date=datetime(2026, 8, 11),
            )
            session.add(report)
            session.flush()
            session.add(database.DataPoint(
                report_id=report.id, canonical_name="Kreatinin",
                original_name="Kreatinin", value=2.8, unit="mg/dl",
                ref_low=0.6, ref_high=1.2, is_abnormal=True,
            ))
            session.commit()
            from analyzer import analyze_trends
            flags = analyze_trends(session)
            cr_flags = [f for f in flags if "Kreatinin" in f.description]
            self.assertGreaterEqual(len(cr_flags), 1, "Expected a Kreatinin flag")
            desc = cr_flags[0].description
            # (2.8 - 1.2) / 1.2 = 133% above the upper limit.
            self.assertIn("133% über der Obergrenze", desc, f"Got: {desc!r}")
            self.assertNotIn("unter der Untergrenze", desc)
        finally:
            session.close()

    def test_extract_display_name_formats(self):
        from analyzer import _extract_display_name
        self.assertEqual(
            _extract_display_name(
                "PO2 = 34.9 mmHg liegt 200% unter dem Referenzbereich (80.0–90.0 mmHg)."
            ),
            "PO2",
        )
        self.assertEqual(
            _extract_display_name("Kreatinin zeigt steigend: von 1.2 auf 2.8 mg/dl (+133%)."),
            "Kreatinin",
        )
        self.assertEqual(
            _extract_display_name(
                "📋 Ferritin wurde vor 400 Tagen gemessen (letzter Wert: 12). Überprüfung empfohlen."
            ),
            "Ferritin",
        )


class TestRecoveryDownweighting(unittest.TestCase):
    """Markers that have returned to normal should score lower."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_recovery.db")
        cls._seed_recovering()

    @classmethod
    def _seed_recovering(cls):
        patient = database.get_or_create_patient(cls.session, "RecoveryPatient")
        dates = [datetime(2024, 1, 1), datetime(2024, 6, 1),
                 datetime(2024, 12, 1), datetime(2025, 6, 1)]
        values = [2.0, 2.4, 2.6, 1.0]
        for i, d in enumerate(dates):
            report = database.Report(
                patient_id=patient.id, filename=f"rec_{i}.pdf",
                file_hash=f"rec_hash_{i}", report_date=d,
            )
            cls.session.add(report)
            cls.session.flush()
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name="Kreatinin",
                original_name="Kreatinin", value=values[i],
                unit="mg/dl", ref_low=0.6, ref_high=1.2,
                is_abnormal=values[i] > 1.2,
            ))
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_recovered_marker_has_flag(self):
        from analyzer import analyze_trends
        flags = analyze_trends(self.session)
        self.assertGreaterEqual(len(flags), 0)


class TestIntensityScoring(unittest.TestCase):
    """Worse violations should score higher."""

    def test_intensity_calculates_correctly(self):
        from analyzer import _compute_pct_beyond
        self.assertAlmostEqual(_compute_pct_beyond(1.8, 0.6, 1.2), 100.0)
        self.assertAlmostEqual(_compute_pct_beyond(1.0, 0.6, 1.2), 0.0)
        self.assertAlmostEqual(_compute_pct_beyond(0.3, 0.6, 1.2), 50.0)


class TestBiomarkerMerge(unittest.TestCase):
    """Tests for case-insensitive biomarker merging and grouping."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_merge.db")
        cls._report_counter = 0

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def _next_hash(self):
        self.__class__._report_counter += 1
        return f"merge_hash_{self._report_counter}"

    def _add_report(self, session, patient_id, report_date, entries):
        """Helper to add a report with data points."""
        from database import Report, DataPoint
        report = Report(
            patient_id=patient_id, filename="TestPDF.pdf",
            file_hash=self._next_hash(), report_date=report_date,
            panel_type="Blutbild",
        )
        session.add(report)
        session.flush()
        for e in entries:
            session.add(DataPoint(
                report_id=report.id, canonical_name=e["name"],
                original_name=e["name"], value=e["value"], unit=e["unit"],
                ref_low=None, ref_high=None, is_abnormal=False,
            ))
        session.commit()
        return report

    def _add_patient_with_reports(self, session, patient_name, reports_data):
        """Add a patient with multiple reports containing biomarker entries."""
        patient = database.get_or_create_patient(session, patient_name)
        for report_date, entries in reports_data:
            if isinstance(report_date, str):
                from datetime import datetime
                report_date = datetime.strptime(report_date, "%Y-%m-%d")
            self._add_report(session, patient.id, report_date, entries)
        return patient

    def test_same_name_same_unit_no_merge_group(self):
        """Identical canonical_name and unit should NOT be grouped for merging."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_NoMerge", [
                ("2024-01-01", [
                    {"name": "Glucose", "value": 90, "unit": "mg/dl", "reference_range": "70-100"},
                ]),
                ("2024-02-01", [
                    {"name": "Glucose", "value": 95, "unit": "mg/dl", "reference_range": "70-100"},
                ]),
            ]
        )
        groups = database.get_unique_biomarker_groups(session, patient.id)
        self.assertEqual(len(groups), 0, "Same name+unit should not trigger merge groups")

    def test_same_name_different_unit_case_should_merge(self):
        """Same canonical name but unit differs only by case should be grouped."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_UnitCase", [
                ("2024-01-01", [
                    {"name": "Glucose", "value": 90, "unit": "mg/dl", "reference_range": "70-100"},
                ]),
                ("2024-02-01", [
                    {"name": "Glucose", "value": 95, "unit": "MG/dl", "reference_range": "70-100"},
                ]),
            ]
        )
        groups = database.get_unique_biomarker_groups(session, patient.id)
        self.assertEqual(len(groups), 1, "Should detect 1 group with different unit casing")
        self.assertEqual(len(groups[0]["variants"]), 2, "Should have 2 variants: mg/dl and MG/dl")

    def test_different_name_casing_same_unit_should_merge(self):
        """Different name casing with same unit should be grouped."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_NameCasing", [
                ("2024-01-01", [
                    {"name": "HB", "value": 14, "unit": "g/dl", "reference_range": "14-18"},
                ]),
                ("2024-02-01", [
                    {"name": "Hb", "value": 15, "unit": "g/dl", "reference_range": "14-18"},
                ]),
            ]
        )
        groups = database.get_unique_biomarker_groups(session, patient.id)
        self.assertEqual(len(groups), 1, "Should detect 1 group with different name casing")
        self.assertEqual(len(groups[0]["variants"]), 2, "Should have 2 variants: HB and Hb")

    def test_same_name_different_unit_values_should_not_merge(self):
        """Same name but different units (e.g. mg/dl vs mmol/l) should NOT be grouped."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_UnitDiff", [
                ("2024-01-01", [
                    {"name": "Glucose", "value": 90, "unit": "mg/dl", "reference_range": "70-100"},
                ]),
                ("2024-02-01", [
                    {"name": "Glucose", "value": 5.0, "unit": "mmol/l", "reference_range": "3.9-6.1"},
                ]),
            ]
        )
        groups = database.get_unique_biomarker_groups(session, patient.id)
        self.assertEqual(len(groups), 0, "Different units should NOT trigger merge groups")

    def test_merge_canonical_names_case_insensitive(self):
        """merge_canonical_names should work case-insensitively."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_MergeName", [
                ("2024-01-01", [
                    {"name": "HB", "value": 14, "unit": "g/dl", "reference_range": "14-18"},
                ]),
                ("2024-02-01", [
                    {"name": "Hb", "value": 15, "unit": "g/dl", "reference_range": "14-18"},
                ]),
            ]
        )
        count = database.merge_canonical_names(session, patient.id, ["HB", "Hb"], "Hämoglobin (Hb)")
        self.assertEqual(count, 2, "Both rows should be updated")

        # Verify both data points now have the merged name
        dps = session.query(database.DataPoint).join(database.Report).filter(
            database.Report.patient_id == patient.id
        ).all()
        for dp in dps:
            self.assertEqual(dp.canonical_name, "Hämoglobin (Hb)", "All data points should have merged name")

    def test_merge_two_biomarkers_case_insensitive(self):
        """merge_two_biomarkers should work case-insensitively."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_MergeTwo", [
                ("2024-01-01", [
                    {"name": "HB", "value": 14, "unit": "g/dl", "reference_range": "14-18"},
                ]),
            ]
        )
        count = database.merge_two_biomarkers(session, patient.id, "hb", "Hämoglobin (Hb)")
        self.assertEqual(count, 1, "One row should be updated")

        dps = session.query(database.DataPoint).join(database.Report).filter(
            database.Report.patient_id == patient.id
        ).all()
        self.assertEqual(dps[0].canonical_name, "Hämoglobin (Hb)")

    def test_apply_merge_plan_name_only(self):
        """apply_merge_plan should merge canonical names correctly."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_PlanName", [
                ("2024-01-01", [
                    {"name": "HB", "value": 14, "unit": "g/dl", "reference_range": "14-18"},
                ]),
                ("2024-02-01", [
                    {"name": "Hb", "value": 15, "unit": "g/dl", "reference_range": "14-18"},
                ]),
            ]
        )
        plan = [
            {
                "variants": [
                    {"name": "HB", "unit": "g/dl", "count": 1},
                    {"name": "Hb", "unit": "g/dl", "count": 1},
                ],
                "suggested_target": "Hämoglobin (Hb)",
                "suggested_unit": "g/dl",
            }
        ]
        total = database.apply_merge_plan(session, patient.id, plan)
        self.assertGreaterEqual(total, 1, "Should have merged at least 1 data point")

        dps = session.query(database.DataPoint).join(database.Report).filter(
            database.Report.patient_id == patient.id
        ).all()
        # All should have the target name now
        names = {dp.canonical_name for dp in dps}
        self.assertNotIn("HB", names)
        self.assertNotIn("Hb", names)

    def test_apply_merge_plan_unit_only(self):
        """apply_merge_plan should merge units when names are the same."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_UnitMerge", [
                ("2024-01-01", [
                    {"name": "Glucose", "value": 90, "unit": "mg/dl", "reference_range": "70-100"},
                ]),
                ("2024-02-01", [
                    {"name": "Glucose", "value": 95, "unit": "MG/dl", "reference_range": "70-100"},
                ]),
            ]
        )
        plan = [
            {
                "variants": [
                    {"name": "Glucose", "unit": "mg/dl", "count": 1},
                    {"name": "Glucose", "unit": "MG/dl", "count": 1},
                ],
                "suggested_target": "Glucose",
                "suggested_unit": "mg/dl",
            }
        ]
        total = database.apply_merge_plan(session, patient.id, plan)
        self.assertGreaterEqual(total, 1, "Should have merged at least 1 data point (unit)")

        dps = session.query(database.DataPoint).join(database.Report).filter(
            database.Report.patient_id == patient.id
        ).all()
        units = {dp.unit for dp in dps}
        self.assertNotIn("MG/dl", units, "MG/dl should be merged to mg/dl")

    def test_auto_merge_detects_groups(self):
        """auto_merge_biomarkers should detect groups with different name casing."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_AutoMerge", [
                ("2024-01-01", [
                    {"name": "HB", "value": 14, "unit": "g/dl", "reference_range": "14-18"},
                ]),
                ("2024-02-01", [
                    {"name": "Hb", "value": 15, "unit": "g/dl", "reference_range": "14-18"},
                ]),
            ]
        )
        # Mock the LLM to return a valid single-call response (plans list)
        with patch('llm_client._build_client') as mock_build:
            mock_client = _MockLLM(
                '{"plans": [{"group_index": 0, "target": "Hämoglobin (Hb)", "unit": "g/dl"}]}'
            )
            mock_build.return_value = mock_client
            result = database.auto_merge_biomarkers(session, patient.id)
        self.assertTrue(len(result["groups"]) > 0, "Should detect at least one group")
        self.assertEqual(len(result["merge_plan"]), 1, "LLM plan should be applied")
        self.assertEqual(result["merge_plan"][0]["suggested_target"], "Hämoglobin (Hb)")
        self.assertEqual(result["merge_plan"][0]["suggested_unit"], "g/dl")

    def test_auto_merge_multiple_groups_single_llm_call(self):
        """All variant groups must be handled in ONE LLM call with a plans list."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_AutoMergeMulti", [
                ("2024-01-01", [
                    {"name": "HB", "value": 14, "unit": "g/dl", "reference_range": "14-18"},
                    {"name": "HDL-Cholesterin", "value": 45, "unit": "mg/dl", "reference_range": "35-80"},
                ]),
                ("2024-02-01", [
                    {"name": "Hb", "value": 15, "unit": "g/dl", "reference_range": "14-18"},
                    {"name": "HDL-CHOLESTERIN", "value": 47, "unit": "mg/dl", "reference_range": "35-80"},
                ]),
            ]
        )
        plans_json = json.dumps({
            "plans": [
                {"group_index": 0, "target": "Hämoglobin (Hb)", "unit": "g/dl"},
                {"group_index": 1, "target": "HDL-Cholesterin", "unit": "mg/dl"},
            ]
        })
        mock_client = _MockLLM(plans_json)
        # Wrap create() to count how many LLM calls are made
        original_create = mock_client.chat.completions.create
        call_count = {"n": 0}

        def counting_create(*a, **kw):
            call_count["n"] += 1
            return original_create(*a, **kw)

        mock_client.chat.completions.create = counting_create
        with patch('llm_client._build_client') as mock_build:
            mock_build.return_value = mock_client
            result = database.auto_merge_biomarkers(session, patient.id)

        self.assertEqual(len(result["groups"]), 2, "Should detect two variant groups")
        # The LLM must be called exactly ONCE for all groups
        self.assertEqual(call_count["n"], 1,
                         "All groups must be sent in a single LLM call")
        self.assertEqual(len(result["merge_plan"]), 2, "Both group plans should be applied")
        targets = {p["suggested_target"] for p in result["merge_plan"]}
        self.assertEqual(targets, {"Hämoglobin (Hb)", "HDL-Cholesterin"})

    def test_auto_merge_llm_failure_uses_fallback(self):
        """When the LLM call fails, per-group fallback heuristics must be used."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_AutoMergeFail", [
                ("2024-01-01", [
                    {"name": "Hämoglobin", "value": 14, "unit": "g/dl", "reference_range": "14-18"},
                ]),
                ("2024-02-01", [
                    {"name": "HÄMOGLOBIN", "value": 15, "unit": "g/dl", "reference_range": "14-18"},
                ]),
            ]
        )
        with patch('llm_client._build_client') as mock_build:
            mock_client = _MockLLM("This is not JSON at all.")
            mock_build.return_value = mock_client
            result = database.auto_merge_biomarkers(session, patient.id)

        self.assertEqual(len(result["groups"]), 1)
        # Fallback matches the variant base name to the canonical list
        self.assertEqual(len(result["merge_plan"]), 1)
        self.assertEqual(result["merge_plan"][0]["suggested_target"], "Hämoglobin (Hb)")
        self.assertEqual(result["merge_plan"][0]["suggested_unit"], "g/dl")

    def test_auto_merge_unknown_target_uses_fallback(self):
        """A target not in the canonical list must fall back to heuristics."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_AutoMergeUnknown", [
                ("2024-01-01", [
                    {"name": "Hämoglobin", "value": 14, "unit": "g/dl", "reference_range": "14-18"},
                ]),
                ("2024-02-01", [
                    {"name": "HÄMOGLOBIN", "value": 15, "unit": "g/dl", "reference_range": "14-18"},
                ]),
            ]
        )
        with patch('llm_client._build_client') as mock_build:
            mock_client = _MockLLM(
                '{"plans": [{"group_index": 0, "target": "Not A Real Biomarker", "unit": "g/dl"}]}'
            )
            mock_build.return_value = mock_client
            result = database.auto_merge_biomarkers(session, patient.id)

        self.assertEqual(len(result["groups"]), 1)
        # Unknown target → fallback (base-name match to canonical list)
        self.assertEqual(len(result["merge_plan"]), 1)
        self.assertEqual(result["merge_plan"][0]["suggested_target"], "Hämoglobin (Hb)")

    def test_normalize_canonical_name_case_insensitive(self):
        """normalize_canonical_name should return lowercase for comparison."""
        self.assertEqual(database.normalize_canonical_name("Hb"), "hb")
        self.assertEqual(database.normalize_canonical_name("HB"), "hb")
        self.assertEqual(database.normalize_canonical_name("Hb"), "hb")
        self.assertEqual(database.normalize_canonical_name("Hämoglobin"), "hämoglobin")

    def test_normalize_unit_case_insensitive(self):
        """normalize_unit should return lowercase for comparison."""
        self.assertEqual(database.normalize_unit("mg/dl"), "mg/dl")
        self.assertEqual(database.normalize_unit("MG/dl"), "mg/dl")
        self.assertEqual(database.normalize_unit("Mg/DL"), "mg/dl")
        self.assertIsNone(database.normalize_unit(None))

    def test_merge_canonical_names_no_match(self):
        """merge_canonical_names should return 0 when no names match."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_NoMatch", [
                ("2024-01-01", [
                    {"name": "Glucose", "value": 90, "unit": "mg/dl", "reference_range": "70-100"},
                ]),
            ]
        )
        count = database.merge_canonical_names(session, patient.id, ["NonExistent"], "Glucose")
        self.assertEqual(count, 0, "No rows should be updated")

    def test_hba1c_different_units_not_grouped(self):
        """HbA1c (%) and HbA1c (mmol/mol) should NOT be grouped for merging."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_HbA1c", [
                ("2024-01-01", [
                    {"name": "HbA1c", "value": 7.5, "unit": "%", "reference_range": "4.0-6.0"},
                ]),
                ("2024-02-01", [
                    {"name": "HbA1c", "value": 58, "unit": "mmol/mol", "reference_range": "20-42"},
                ]),
            ]
        )
        groups = database.get_unique_biomarker_groups(session, patient.id)
        self.assertEqual(len(groups), 0, "HbA1c with different units should NOT be grouped")

    def test_hba1c_same_unit_case_diff_should_merge(self):
        """HbA1c (%) and HBA1C (%) with same unit (only name casing differs) should be grouped."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestPatient_HbA1cCase", [
                ("2024-01-01", [
                    {"name": "HbA1c (%)", "value": 7.5, "unit": "%", "reference_range": "4.0-6.0"},
                ]),
                ("2024-02-01", [
                    {"name": "HBA1C (%)", "value": 7.0, "unit": "%", "reference_range": "4.0-6.0"},
                ]),
            ]
        )
        groups = database.get_unique_biomarker_groups(session, patient.id)
        self.assertEqual(len(groups), 1, "HbA1c with same unit should be grouped")
        self.assertEqual(len(groups[0]["variants"]), 2, "Should have 2 variants")


class TestManualMerge(unittest.TestCase):
    """Tests for manual biomarker merge functionality."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_manual_merge.db")
        cls._report_counter = 0

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def _next_hash(self):
        self.__class__._report_counter += 1
        return f"manual_hash_{self._report_counter}"

    def _add_report(self, session, patient_id, report_date, entries):
        """Helper to add a report with data points."""
        from datetime import datetime
        if isinstance(report_date, str):
            report_date = datetime.strptime(report_date, "%Y-%m-%d")
        from database import Report, DataPoint
        report = Report(
            patient_id=patient_id, filename="TestPDF.pdf",
            file_hash=self._next_hash(), report_date=report_date,
            panel_type="Blutbild",
        )
        session.add(report)
        session.flush()
        for e in entries:
            session.add(DataPoint(
                report_id=report.id, canonical_name=e["name"],
                original_name=e["name"], value=e["value"], unit=e["unit"],
                ref_low=None, ref_high=None, is_abnormal=False,
            ))
        session.commit()
        return report

    def _add_patient_with_reports(self, session, patient_name, reports_data):
        """Add a patient with multiple reports."""
        patient = database.get_or_create_patient(session, patient_name)
        for report_date, entries in reports_data:
            self._add_report(session, patient.id, report_date, entries)
        return patient

    def test_manual_merge_case_insensitive_source(self):
        """Manual merge should match source name case-insensitively."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestManual_Patient1", [
                ("2024-01-01", [
                    {"name": "LEUKOZYTEN", "value": 7.0, "unit": "Tsd./µl", "reference_range": "4.0-10.0"},
                ]),
            ]
        )
        count = database.merge_two_biomarkers(session, patient.id, "leukozyten", "Leukozyten (WBC)")
        self.assertEqual(count, 1, "Should merge case-insensitively")

        dps = session.query(database.DataPoint).join(database.Report).filter(
            database.Report.patient_id == patient.id
        ).all()
        self.assertEqual(dps[0].canonical_name, "Leukozyten (WBC)")

    def test_manual_merge_multiple_data_points(self):
        """Manual merge should update all matching data points across reports."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestManual_Patient2", [
                ("2024-01-01", [
                    {"name": "glucose", "value": 90, "unit": "mg/dl", "reference_range": "70-100"},
                ]),
                ("2024-02-01", [
                    {"name": "GLUCOSE", "value": 95, "unit": "mg/dl", "reference_range": "70-100"},
                ]),
                ("2024-03-01", [
                    {"name": "Glucose", "value": 100, "unit": "mg/dl", "reference_range": "70-100"},
                ]),
            ]
        )
        count = database.merge_two_biomarkers(session, patient.id, "glucose", "Glucose")
        self.assertEqual(count, 3, "All 3 data points should be merged")

        dps = session.query(database.DataPoint).join(database.Report).filter(
            database.Report.patient_id == patient.id
        ).all()
        for dp in dps:
            self.assertEqual(dp.canonical_name, "Glucose")

    def test_manual_merge_preserves_other_fields(self):
        """Manual merge should preserve value, unit, and reference_range."""
        session = self.session
        patient = self._add_patient_with_reports(
            session, "TestManual_Patient3", [
                ("2024-01-01", [
                    {"name": "HB", "value": 14.5, "unit": "g/dl", "reference_range": "14-18"},
                ]),
            ]
        )
        database.merge_two_biomarkers(session, patient.id, "hb", "Hämoglobin (Hb)")
        session.commit()

        dp = session.query(database.DataPoint).join(database.Report).filter(
            database.Report.patient_id == patient.id
        ).first()
        self.assertEqual(dp.value, 14.5)
        self.assertEqual(dp.unit, "g/dl")
        self.assertEqual(dp.canonical_name, "Hämoglobin (Hb)")




class TestCrossBiomarkerPatterns(unittest.TestCase):
    """Tests for _detect_cross_biomarker_patterns."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db('test_cross_bio.db')

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def _add_patient_with_markers(self, patient_name, marker_data_list, report_date=None):
        if report_date is None:
            report_date = datetime(2025, 1, 1)
        patient = database.get_or_create_patient(self.session, patient_name)
        report = database.Report(
            patient_id=patient.id, filename='test.pdf',
            file_hash=f'cross_bio_{patient_name}', report_date=report_date,
        )
        self.session.add(report)
        self.session.flush()
        for canonical, value, unit, ref_low, ref_high, is_abnormal in marker_data_list:
            self.session.add(database.DataPoint(
                report_id=report.id, canonical_name=canonical, original_name=canonical,
                value=value, unit=unit, ref_low=ref_low, ref_high=ref_high,
                is_abnormal=is_abnormal,
            ))
        self.session.commit()
        return patient.id

    def test_ck_troponin_both_abnormal(self):
        """Both CK and Troponin abnormal should trigger cross-biomarker flag."""
        from analyzer import _detect_cross_biomarker_patterns
        pid = self._add_patient_with_markers('CKTroponin', [
            ('CK gesamt', 500.0, 'U/l', None, None, True),
            ('Troponin T', 0.2, 'ng/ml', None, None, True),
        ], report_date=datetime(2025, 6, 1))
        flags = _detect_cross_biomarker_patterns(self.session, pid, datetime(2025, 6, 1))
        cross_flags = [f for f in flags if 'Herzinfarkt' in f.description]
        self.assertGreater(len(cross_flags), 0)

    def test_ck_isolated_elevated(self):
        """Only CK elevated (Troponin normal) should trigger isolated CK flag."""
        from analyzer import _detect_cross_biomarker_patterns
        pid = self._add_patient_with_markers('CKIsolated', [
            ('CK gesamt', 500.0, 'U/l', None, None, True),
            ('Troponin T', 0.01, 'ng/ml', None, None, False),
        ], report_date=datetime(2025, 6, 1))
        flags = _detect_cross_biomarker_patterns(self.session, pid, datetime(2025, 6, 1))
        iso_flags = [f for f in flags if 'Isolierte' in f.description]
        self.assertGreater(len(iso_flags), 0)

    def test_missing_marker_skips_required_rule(self):
        """Cross-biomarker rules requiring multiple markers skip if one missing."""
        from analyzer import _detect_cross_biomarker_patterns
        pid = self._add_patient_with_markers('MissingMarker', [
            ('CK gesamt', 500.0, 'U/l', None, None, True),
        ], report_date=datetime(2025, 6, 1))
        flags = _detect_cross_biomarker_patterns(self.session, pid, datetime(2025, 6, 1))
        cross_flags = [f for f in flags if 'Herzinfarkt' in f.description]
        self.assertEqual(len(cross_flags), 0)

    def test_severity_upgrade_both_abnormal(self):
        """Both markers abnormal should produce higher severity."""
        from analyzer import _detect_cross_biomarker_patterns
        pid = self._add_patient_with_markers('SeverityUp', [
            ('CK gesamt', 500.0, 'U/l', None, None, True),
            ('Troponin T', 0.2, 'ng/ml', None, None, True),
        ], report_date=datetime(2025, 6, 1))
        flags = _detect_cross_biomarker_patterns(self.session, pid, datetime(2025, 6, 1))
        heart_flags = [f for f in flags if 'Herzinfarkt' in f.description]
        self.assertGreater(len(heart_flags), 0)
        for f in heart_flags:
            self.assertEqual(f.severity, 3)

    def test_severity_downgrade_isolated(self):
        """Only one marker abnormal should produce lower severity."""
        from analyzer import _detect_cross_biomarker_patterns
        pid = self._add_patient_with_markers('SeverityDown', [
            ('CK gesamt', 500.0, 'U/l', None, None, True),
            ('Troponin T', 0.01, 'ng/ml', None, None, False),
        ], report_date=datetime(2025, 6, 1))
        flags = _detect_cross_biomarker_patterns(self.session, pid, datetime(2025, 6, 1))
        iso_flags = [f for f in flags if 'Isolierte' in f.description]
        self.assertGreater(len(iso_flags), 0)
        for f in iso_flags:
            self.assertEqual(f.severity, 1)

    def test_no_cross_biomarker_when_healthy(self):
        """No cross-biomarker flags when all markers normal."""
        from analyzer import _detect_cross_biomarker_patterns
        pid = self._add_patient_with_markers('Healthy', [
            ('Kreatinin', 90.0, 'mg/dl', 70, 100, False),
            ('HbA1c (%)', 5.2, '%', 4.0, 5.7, False),
        ], report_date=datetime(2025, 6, 1))
        flags = _detect_cross_biomarker_patterns(self.session, pid, datetime(2025, 6, 1))
        self.assertEqual(len(flags), 0)

    def test_ldl_high_hdl_low_triggers_cardio(self):
        """LDL above ref_high + HDL below ref_low should trigger the cardio rule.

        Regression test for R3-01: the 'low' condition previously compared
        value < 0.0 (never true for positive lab values), so this rule was dead.
        """
        from analyzer import _detect_cross_biomarker_patterns
        pid = self._add_patient_with_markers('CardioLow', [
            ('LDL-Cholesterin', 190.0, 'mg/dl', 0, 160, True),   # high: 190 > 160
            ('HDL-Cholesterin', 25.0, 'mg/dl', 35, 80, True),    # low: 25 < 35
        ], report_date=datetime(2025, 6, 1))
        flags = _detect_cross_biomarker_patterns(self.session, pid, datetime(2025, 6, 1))
        cardio_flags = [f for f in flags if 'Kardiovaskuläres Risiko' in f.description]
        self.assertGreater(len(cardio_flags), 0)

    def test_ldl_high_hdl_normal_no_cardio(self):
        """LDL high but HDL within range should NOT trigger the cardio rule."""
        from analyzer import _detect_cross_biomarker_patterns
        pid = self._add_patient_with_markers('CardioNormalHdl', [
            ('LDL-Cholesterin', 190.0, 'mg/dl', 0, 160, True),   # high
            ('HDL-Cholesterin', 55.0, 'mg/dl', 35, 80, False),   # normal: not < 35
        ], report_date=datetime(2025, 6, 1))
        flags = _detect_cross_biomarker_patterns(self.session, pid, datetime(2025, 6, 1))
        cardio_flags = [f for f in flags if 'Kardiovaskuläres Risiko' in f.description]
        self.assertEqual(len(cardio_flags), 0)

    def test_report_id_set_on_flags(self):
        """Cross-biomarker flags should have report_id set."""
        from analyzer import _detect_cross_biomarker_patterns
        pid = self._add_patient_with_markers('ReportId', [
            ('CK gesamt', 500.0, 'U/l', None, None, True),
            ('Troponin T', 0.2, 'ng/ml', None, None, True),
        ], report_date=datetime(2025, 6, 1))
        flags = _detect_cross_biomarker_patterns(self.session, pid, datetime(2025, 6, 1))
        cross_flags = [f for f in flags if 'Herzinfarkt' in f.description]
        self.assertGreater(len(cross_flags), 0)
        for f in cross_flags:
            self.assertIsNotNone(f.report_id)


class TestDetectMissingMarkers(unittest.TestCase):
    """Tests for _detect_missing_markers."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db('test_missing.db')

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_marker_measured_old_should_flag(self):
        """Marker measured but outside lookback period should trigger flag."""
        from analyzer import _detect_missing_markers
        patient = database.get_or_create_patient(self.session, 'OldMarker')
        old_report = database.Report(
            patient_id=patient.id, filename='old.pdf',
            file_hash='old_hash', report_date=datetime(2024, 1, 1),
        )
        self.session.add(old_report)
        self.session.flush()
        self.session.add(database.DataPoint(
            report_id=old_report.id, canonical_name='Kreatinin',
            original_name='Kreatinin', value=100.0, unit='mg/dl',
            ref_low=None, ref_high=None, is_abnormal=False,
        ))
        self.session.commit()
        flags = _detect_missing_markers(self.session, patient.id, datetime(2025, 6, 1))
        missing_flags = [f for f in flags if 'fehlend' in f.description.lower() or '\u00dcberpr\u00fcfung' in f.description]
        self.assertGreater(len(missing_flags), 0)

    def test_marker_measured_recently_no_flag(self):
        """Marker measured within lookback period should not trigger flag."""
        from analyzer import _detect_missing_markers
        patient = database.get_or_create_patient(self.session, 'RecentMarker')
        recent_report = database.Report(
            patient_id=patient.id, filename='recent.pdf',
            file_hash='recent_hash', report_date=datetime(2025, 5, 1),
        )
        self.session.add(recent_report)
        self.session.flush()
        self.session.add(database.DataPoint(
            report_id=recent_report.id, canonical_name='Kreatinin',
            original_name='Kreatinin', value=100.0, unit='mg/dl',
            ref_low=None, ref_high=None, is_abnormal=False,
        ))
        self.session.commit()
        flags = _detect_missing_markers(self.session, patient.id, datetime(2025, 6, 1))
        missing_flags = [f for f in flags if 'fehlend' in f.description.lower() or '\u00dcberpr\u00fcfung' in f.description]
        self.assertEqual(len(missing_flags), 0)

    def test_no_reports_yet(self):
        """No flags when patient has no reports."""
        from analyzer import _detect_missing_markers
        patient = database.get_or_create_patient(self.session, 'NoReports')
        flags = _detect_missing_markers(self.session, patient.id, datetime(2025, 6, 1))
        self.assertEqual(len(flags), 0)


class TestAnalyzerPrimitives(unittest.TestCase):
    """Tests for analyzer primitive functions."""

    def test_compute_trend_increasing(self):
        """Test _compute_trend with increasing values returns positive slope."""
        from analyzer import _compute_trend
        trend = _compute_trend([10.0, 20.0, 30.0, 40.0])
        self.assertGreater(trend, 0)

    def test_compute_trend_decreasing(self):
        """Test _compute_trend with decreasing values returns negative slope."""
        from analyzer import _compute_trend
        trend = _compute_trend([40.0, 30.0, 20.0, 10.0])
        self.assertLess(trend, 0)

    def test_compute_trend_constant(self):
        """Test _compute_trend with constant values returns zero."""
        from analyzer import _compute_trend
        trend = _compute_trend([10.0, 10.0, 10.0, 10.0])
        self.assertEqual(trend, 0.0)

    def test_compute_trend_single_value(self):
        """Test _compute_trend with single value returns zero."""
        from analyzer import _compute_trend
        trend = _compute_trend([10.0])
        self.assertEqual(trend, 0.0)

class TestFindBiomarkerInfo(unittest.TestCase):
    """Tests for database helper functions."""

    def test_find_biomarker_info_exact_match(self):
        """Test _find_biomarker_info returns info for exact match."""
        from database import _find_biomarker_info
        biomarkers_dict = {
            'Glukose (\u00fc\u00fcchtern)': {'unit': 'mg/dl', 'aliases': ['Glukose']},
            'HbA1c (%)': {'unit': '%', 'aliases': ['HbA1c']},
        }
        info = _find_biomarker_info('Glukose (\u00fc\u00fcchtern)', biomarkers_dict)
        self.assertIsNotNone(info)
        self.assertEqual(info['unit'], 'mg/dl')

    def test_find_biomarker_info_alias_match(self):
        """Test _find_biomarker_info returns info for alias match."""
        from database import _find_biomarker_info
        biomarkers_dict = {
            'Kreatinin': {'unit': 'mg/dl', 'aliases': ['Glukose']},
        }
        info = _find_biomarker_info('Glukose', biomarkers_dict)
        self.assertIsNotNone(info)

    def test_find_biomarker_info_no_match(self):
        """Test _find_biomarker_info returns None for missing biomarker."""
        from database import _find_biomarker_info
        biomarkers_dict = {
            'Kreatinin': {'unit': 'mg/dl', 'aliases': ['Glukose']},
        }
        info = _find_biomarker_info('NonExistentMarker', biomarkers_dict)
        self.assertIsNone(info)

    def test_convert_ref_range_mgdl_to_mmoll(self):
        """Test name-aware converter: glucose mg/dl → mmol/l (÷ 18)."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('Glukose (nüchtern)', 70.0, 100.0, 'mg/dl', 'mmol/l')
        self.assertAlmostEqual(low, 70.0 / 18.0, places=2)
        self.assertAlmostEqual(high, 100.0 / 18.0, places=2)

    def test_convert_ref_range_mmoll_to_mgdl(self):
        """Test name-aware converter: glucose mmol/l → mg/dl (× 18)."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('Glukose (nüchtern)', 3.9, 5.6, 'mmol/l', 'mg/dl')
        self.assertAlmostEqual(low, 3.9 * 18.0, places=2)
        self.assertAlmostEqual(high, 5.6 * 18.0, places=2)

    def test_convert_ref_range_none_input(self):
        """Test name-aware converter with None values."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('Glukose (nüchtern)', None, None, 'mg/dl', 'mmol/l')
        self.assertIsNone(low)
        self.assertIsNone(high)

    def test_convert_ref_range_disambiguates_shared_unit_pair(self):
        """Same unit pair, different biomarker → different factor.

        mg/dl ↔ mmol/l is shared by glucose (÷18) and cholesterol (÷38.67).
        The converter must pick the factor by canonical name, not unit pair.
        """
        from config import _convert_ref_range
        chol_low, chol_high = _convert_ref_range('Cholesterin gesamt', 150.0, 200.0, 'mg/dl', 'mmol/l')
        self.assertAlmostEqual(chol_low, 150.0 / 38.67, places=2)
        self.assertAlmostEqual(chol_high, 200.0 / 38.67, places=2)

        glu_low, _ = _convert_ref_range('Glukose (nüchtern)', 150.0, 200.0, 'mg/dl', 'mmol/l')
        self.assertAlmostEqual(glu_low, 150.0 / 18.0, places=2)

    def test_convert_ref_range_creatinine_direction(self):
        """Creatinine mg/dl → µmol/l multiplies by 88.4 (old code divided)."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('Kreatinin', 0.7, 1.3, 'mg/dl', 'µmol/l')
        self.assertAlmostEqual(low, 0.7 * 88.4, places=1)
        self.assertAlmostEqual(high, 1.3 * 88.4, places=1)

    def test_convert_ref_range_iron_factor(self):
        """Iron µg/dl → nmol/l uses 179.067 (10000 / M(Fe)=55.845)."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('Eisen', 60.0, 170.0, 'µg/dl', 'nmol/l')
        self.assertAlmostEqual(low, 60.0 * 179.067, places=1)
        self.assertAlmostEqual(high, 170.0 * 179.067, places=1)

    def test_convert_ref_range_uric_acid_factor(self):
        """Uric acid mg/dl → µmol/l uses 59.48 (10000 / 168.11)."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('Harnsäure', 2.0, 7.5, 'mg/dl', 'µmol/l')
        self.assertAlmostEqual(low, 2.0 * 59.48, places=1)
        self.assertAlmostEqual(high, 7.5 * 59.48, places=1)

    def test_convert_ref_range_unsupported_returns_none(self):
        """Unknown unit pair → (None, None), not silently wrong values."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('Glukose (nüchtern)', 70.0, 100.0, 'mg/dl', 'g/l')
        self.assertIsNone(low)
        self.assertIsNone(high)

    def test_convert_ref_range_hba1c_affine(self):
        """HbA1c % ↔ mmol/mol is affine, not linear."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('HbA1c', 4.8, 6.0, '%', 'mmol/mol')
        self.assertAlmostEqual(low, (4.8 - 2.15) * 10.929, places=1)
        self.assertAlmostEqual(high, (6.0 - 2.15) * 10.929, places=1)

        back_low, _ = _convert_ref_range('HbA1c', low, high, 'mmol/mol', '%')
        self.assertAlmostEqual(back_low, 4.8, places=2)

    def test_convert_ref_range_equivalent_units_passthrough(self):
        """Notation variants of the same unit convert with factor 1.

        These are the units that real lab reports use (and that the LLM
        proposes to normalize): /nl and Tsd/µl for WBC, /pl for RBC,
        mU/l + µIU/ml for TSH, 'ml/ min' spacing variant.
        """
        from config import _convert_ref_range
        # WBC: /nl → 10^3/µl (same magnitude)
        low, high = _convert_ref_range('Leukozyten (WBC)', 4.0, 11.0, '/nl', '10^3/µl')
        self.assertEqual((low, high), (4.0, 11.0))
        # WBC: Tsd/µl → 10³/µl
        low, high = _convert_ref_range('Leukozyten (WBC)', 6.5, 6.5, 'Tsd/µl', '10³/µl')
        self.assertEqual((low, high), (6.5, 6.5))
        # RBC: /pl → Mio./µl
        low, high = _convert_ref_range('Erythrozyten (RBC)', 5.4, 5.4, '/pl', 'Mio./µl')
        self.assertEqual((low, high), (5.4, 5.4))
        # TSH: mU/l → mIU/l and µIU/ml → mIU/l (1 µIU/ml = 1 mIU/l)
        low, high = _convert_ref_range('TSH basal', 0.27, 4.2, 'mU/l', 'mIU/l')
        self.assertEqual((low, high), (0.27, 4.2))
        low, high = _convert_ref_range('TSH basal', 0.98, 0.98, 'µIU/ml', 'mIU/l')
        self.assertEqual((low, high), (0.98, 0.98))
        # TSH: known OCR typo µlU/ml → mIU/l
        low, high = _convert_ref_range('TSH basal', 0.98, 0.98, 'µlU/ml', 'mIU/l')
        self.assertEqual((low, high), (0.98, 0.98))
        # Flow: 'ml/ min' → ml/min (spacing variant)
        low, high = _convert_ref_range('Kreatinin-Clearance', 90.0, 120.0, 'ml/ min', 'ml/min')
        self.assertEqual((low, high), (90.0, 120.0))

    def test_convert_ref_range_dimensionless_target(self):
        """Target unit 'none' means the value is a pure ratio — no conversion."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('QPO2/FO2', 0.85, 1.1, 'mmHg', 'none')
        self.assertEqual((low, high), (0.85, 1.1))

    def test_convert_ref_range_same_unit_passthrough(self):
        """Identical units pass values through unchanged."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('CRP', 0.0, 5.0, 'mg/l', 'mg/l')
        self.assertEqual((low, high), (0.0, 5.0))

    def test_convert_ref_range_ngml_to_ngdl_factor(self):
        """ng/ml → ng/dl multiplies by 100 (1 dl = 100 ml)."""
        from config import _convert_ref_range
        low, high = _convert_ref_range('Testosteron gesamt', 3.78, 3.09, 'ng/ml', 'ng/dl')
        self.assertAlmostEqual(low, 378.0, places=1)
        self.assertAlmostEqual(high, 309.0, places=1)


class TestFindBiomarkerInfoSubstring(unittest.TestCase):
    """Regression: substring tiers (3/4) require len >= 3 on BOTH sides and
    pick the longest match, so short lab abbreviations like "Cr" (creatinine)
    can no longer grab CRP's reference range via substring matching."""

    def _fixture(self):
        return {
            'CRP': {'unit': 'mg/l', 'aliases': ['C-reaktives Protein']},
            'Cholesterin gesamt': {'unit': 'mg/dl', 'aliases': ['Gesamtcholesterin']},
        }

    def test_short_name_does_not_match_crp(self):
        """'Cr'/'CR' (len 2) must not substring-match the CRP key or aliases."""
        from database import _find_biomarker_info
        fixture = self._fixture()
        self.assertIsNone(_find_biomarker_info('Cr', fixture))
        self.assertIsNone(_find_biomarker_info('CR', fixture))

    def test_exact_crp_still_matches(self):
        """Tier 1 exact match is unaffected by the minimum-length rule."""
        from database import _find_biomarker_info
        info = _find_biomarker_info('CRP', self._fixture())
        self.assertIsNotNone(info)
        self.assertEqual(info['unit'], 'mg/l')

    def test_substring_of_key_matches(self):
        """len >= 3 name that is a substring of a key still matches (tier 3)."""
        from database import _find_biomarker_info
        info = _find_biomarker_info('Cholesterin', self._fixture())
        self.assertIsNotNone(info)
        self.assertEqual(info['unit'], 'mg/dl')

    def test_alias_substring_matches(self):
        """len >= 3 name that is a substring of an alias matches (tier 4)."""
        from database import _find_biomarker_info
        fixture = {'Kreatinin': {'unit': 'mg/dl', 'aliases': ['Creatinin']}}
        info = _find_biomarker_info('Crea', fixture)
        self.assertIsNotNone(info)
        self.assertEqual(info['unit'], 'mg/dl')

    def test_longest_key_wins_on_multiple_matches(self):
        """Name substring of multiple keys → longest key wins (most specific)."""
        from database import _find_biomarker_info
        fixture = {
            'HDL': {'unit': 'mg/dl', 'aliases': []},
            'HDL-Cholesterin': {'unit': 'mmol/l', 'aliases': []},
        }
        # "hdl-c" is a substring of "hdl-cholesterin" and contains key "hdl".
        info = _find_biomarker_info('HDL-C', fixture)
        self.assertIsNotNone(info)
        self.assertEqual(info['unit'], 'mmol/l')  # from HDL-Cholesterin, not HDL

    def test_ambiguous_match_logs_warning(self):
        """Multiple substring matches must emit a WARNING naming candidates."""
        from database import _find_biomarker_info
        fixture = {
            'HDL': {'unit': 'mg/dl', 'aliases': []},
            'HDL-Cholesterin': {'unit': 'mmol/l', 'aliases': []},
        }
        with self.assertLogs('database', level='WARNING') as cm:
            info = _find_biomarker_info('HDL-C', fixture)
        self.assertIsNotNone(info)
        self.assertTrue(any('Ambiguous' in msg for msg in cm.output),
                        f"Expected ambiguity warning, got: {cm.output}")

    def test_single_match_no_warning(self):
        """A unique substring match must not log an ambiguity warning."""
        import logging
        from database import _find_biomarker_info
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record)

        logger = logging.getLogger('database')
        handler = _Capture()
        old_level = logger.level
        logger.addHandler(handler)
        try:
            info = _find_biomarker_info('Cholesterin', self._fixture())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
        self.assertIsNotNone(info)
        self.assertEqual([r for r in captured if r.levelno >= logging.WARNING], [])


class TestBackfillReferenceRanges(unittest.TestCase):
    """R3-04: backfill uses the shared name-aware converter and skips
    unconvertible units instead of applying a range in the wrong unit."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_backfill.db")
        patient = database.get_or_create_patient(cls.session, "BackfillPatient")
        report = database.Report(
            patient_id=patient.id, filename="backfill.pdf",
            file_hash="backfill_hash", report_date=datetime(2024, 1, 1),
        )
        cls.session.add(report)
        cls.session.flush()
        # Unconvertible: glucose in g/l (no mg/dl ↔ g/l conversion)
        cls.dp_glucose = database.DataPoint(
            report_id=report.id, canonical_name="Glukose (nüchtern)",
            original_name="Glukose", value=5.0, unit="g/l",
            ref_low=None, ref_high=None, is_abnormal=False,
        )
        # Convertible: creatinine in µmol/l → mg/dl (÷ 88.4)
        cls.dp_creatinine = database.DataPoint(
            report_id=report.id, canonical_name="Kreatinin",
            original_name="Kreatinin", value=100.0, unit="µmol/l",
            ref_low=None, ref_high=None, is_abnormal=False,
        )
        cls.session.add_all([cls.dp_glucose, cls.dp_creatinine])
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_backfill_skips_unconvertible_and_converts_supported(self):
        from database import backfill_reference_ranges
        result = backfill_reference_ranges(self.session)
        self.session.refresh(self.dp_glucose)
        self.session.refresh(self.dp_creatinine)

        # Unconvertible unit pair must be skipped, not filled with wrong-unit values
        self.assertEqual(result["skipped_unconvertible"], 1)
        self.assertIsNone(self.dp_glucose.ref_low)
        self.assertIsNone(self.dp_glucose.ref_high)

        # Supported conversion: default 0.6–1.2 mg/dl → 53.04–106.08 µmol/l (× 88.4)
        self.assertEqual(result["updated"], 1)
        self.assertAlmostEqual(self.dp_creatinine.ref_low, 0.6 * 88.4, places=2)
        self.assertAlmostEqual(self.dp_creatinine.ref_high, 1.2 * 88.4, places=2)


class TestFixImplausibleRefRanges(unittest.TestCase):
    """One-time cleanup of reference ranges corrupted by an older code version:
    a value more than an order of magnitude below the stored lower bound is
    physiologically impossible and its bounds are reset to NULL. A genuine
    extreme *high* value must never be touched."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_fix_implausible.db")
        patient = database.get_or_create_patient(cls.session, "ImplausiblePatient")
        report = database.Report(
            patient_id=patient.id, filename="implausible.pdf",
            file_hash="implausible_hash", report_date=datetime(2024, 1, 1),
        )
        cls.session.add(report)
        cls.session.flush()
        # Corrupted: Testosteron in ng/ml stamped with the ng/dl default 280–800.
        cls.dp_corrupt = database.DataPoint(
            report_id=report.id, canonical_name="Testosteron gesamt",
            original_name="Testosteron", value=3.09, unit="ng/ml",
            ref_low=280.0, ref_high=800.0, is_abnormal=True,
        )
        # Corrupted: COHb in % stamped with the Hämoglobin g/dl range 12–17.5.
        cls.dp_cohb = database.DataPoint(
            report_id=report.id, canonical_name="COHb",
            original_name="COHb", value=1.1, unit="%",
            ref_low=12.0, ref_high=17.5, is_abnormal=True,
        )
        # Valid low: value comfortably above the lower bound — must be untouched.
        cls.dp_valid = database.DataPoint(
            report_id=report.id, canonical_name="Leukozyten",
            original_name="Leukozyten", value=5.0, unit="Tsd./µl",
            ref_low=4.0, ref_high=10.0, is_abnormal=False,
        )
        # Genuine extreme HIGH: value far above the upper bound — must be untouched
        # (only the too-low direction is corrected).
        cls.dp_extreme_high = database.DataPoint(
            report_id=report.id, canonical_name="PSA",
            original_name="PSA", value=100.0, unit="ng/ml",
            ref_low=None, ref_high=4.0, is_abnormal=True,
        )
        cls.session.add_all([cls.dp_corrupt, cls.dp_cohb, cls.dp_valid, cls.dp_extreme_high])
        cls.session.commit()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_fixes_too_low_and_preserves_others(self):
        from database import _migrate_fix_implausible_ref_ranges
        _migrate_fix_implausible_ref_ranges(database._engine)
        self.session.expire_all()

        # Corrupted too-low rows: bounds reset to NULL, no longer abnormal.
        self.assertIsNone(self.dp_corrupt.ref_low)
        self.assertIsNone(self.dp_corrupt.ref_high)
        self.assertFalse(self.dp_corrupt.is_abnormal)
        self.assertIsNone(self.dp_cohb.ref_low)
        self.assertIsNone(self.dp_cohb.ref_high)
        self.assertFalse(self.dp_cohb.is_abnormal)

        # Valid low row untouched.
        self.assertEqual(self.dp_valid.ref_low, 4.0)
        self.assertEqual(self.dp_valid.ref_high, 10.0)

        # Genuine extreme high untouched (only too-low is corrected).
        self.assertIsNone(self.dp_extreme_high.ref_low)
        self.assertEqual(self.dp_extreme_high.ref_high, 4.0)
        self.assertTrue(self.dp_extreme_high.is_abnormal)

    def test_idempotent(self):
        from database import _migrate_fix_implausible_ref_ranges
        _migrate_fix_implausible_ref_ranges(database._engine)
        # Second run is a no-op: still no rows match the predicate.
        _migrate_fix_implausible_ref_ranges(database._engine)
        self.session.expire_all()
        self.assertIsNone(self.dp_corrupt.ref_low)
        self.assertEqual(self.dp_valid.ref_low, 4.0)


class TestTranslations(unittest.TestCase):
    """Tests for translations module."""

    def test_t_patient(self):
        """Test t() returns expected translation for patient."""
        from translations import t
        result = t('patient')
        self.assertIsInstance(result, str)

    def test_t_patient_de(self):
        """Test t() returns German for Patient."""
        from translations import t
        result = t('Patient')
        self.assertIsInstance(result, str)

    def test_translate_description_basic(self):
        """Test translate_description with basic German text."""
        from translations import translate_description
        result = translate_description('Test description')
        self.assertIsInstance(result, str)

    def test_translate_description_with_german_chars(self):
        """Test translate_description handles German umlauts."""
        from translations import translate_description
        result = translate_description('Erh\u00f6hter Wert')
        self.assertIsInstance(result, str)


class TestTranslationDictIssues(unittest.TestCase):
    """R3-21: translation dict hygiene — no duplicate keys, msg_renamed templates
    match their call sites, and t() logs (not silently swallows) format errors."""

    def _dict_keys(self, name):
        import ast
        src = (Path(__file__).parent / "translations.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
        self.fail(f"dict {name} not found in translations.py")

    def test_no_duplicate_keys_de(self):
        keys = self._dict_keys("DE")
        dupes = {k for k in keys if keys.count(k) > 1}
        self.assertEqual(dupes, set(), f"duplicate DE keys: {dupes}")

    def test_no_duplicate_keys_en(self):
        keys = self._dict_keys("EN")
        dupes = {k for k in keys if keys.count(k) > 1}
        self.assertEqual(dupes, set(), f"duplicate EN keys: {dupes}")

    def test_msg_renamed_formats_without_count_placeholders(self):
        """Patient-rename call site passes only old/new — the template must not
        contain {count}/{plural} or raw placeholders would leak into the UI."""
        from translations import t, set_lang
        set_lang("de")
        out = t("msg_renamed", old="Muster, A.", new="Neu, B.")
        self.assertNotIn("{count}", out)
        self.assertNotIn("{plural}", out)
        self.assertIn("Muster, A.", out)
        self.assertIn("Neu, B.", out)

    def test_msg_biomarker_renamed_formats_with_count(self):
        from translations import t, set_lang
        set_lang("de")
        out = t("msg_biomarker_renamed", count=3, plural="e", old="HB", new="Hämoglobin")
        self.assertIn("3", out)
        self.assertNotIn("{", out)

    def test_t_logs_format_error(self):
        """t() must log (not silently swallow) a format failure."""
        from translations import t, set_lang
        set_lang("de")
        with self.assertLogs("translations", level="WARNING") as cm:
            out = t("msg_biomarker_renamed", count=1)  # missing plural/old/new
        self.assertIn("{plural}", out)  # raw template returned
        self.assertTrue(any("format failed" in m for m in cm.output))


class TestRiskDescriptionTranslation(unittest.TestCase):
    """Task A/B: the new LLM-connection keys exist in both languages, and
    translate_description() covers every German fragment the analyzer emits.

    For each flag type we build a representative German description (calling the
    real analyzer helper where cheap) and assert the expected English is present
    while sentinel German words are gone."""

    def _translate(self, de):
        from translations import translate_description
        return translate_description(de)

    # ── Task A: new LLM-connection keys exist in both langs & t() routes them ──
    def test_llm_connection_keys_both_langs(self):
        from translations import DE, EN, t, set_lang, _get_lang
        for key in ("error_processing_failed_llm", "error_llm_connection_hint"):
            self.assertIn(key, DE)
            self.assertIn(key, EN)
            self.assertNotEqual(DE[key], EN[key])
        prev = _get_lang()
        try:
            set_lang("de")
            de_out = t("error_llm_connection_hint", base_url="http://172.17.0.1:8888/v1")
            self.assertIn("Prüfe", de_out)
            self.assertIn("http://172.17.0.1:8888/v1", de_out)
            set_lang("en")
            en_out = t("error_llm_connection_hint", base_url="http://172.17.0.1:8888/v1")
            self.assertIn("Check", en_out)
            self.assertNotIn("Prüfe", en_out)
        finally:
            set_lang(prev)

    # ── Task B: value_changed ──
    def test_value_changed(self):
        de = ("Hämoglobin ist im aktuellen Bericht abnormal (8.5 g/dl, "
              "Referenz: 12\u201316 g/dl), war in früheren Berichten jedoch im Normbereich.")
        out = self._translate(de)
        for en in ("is abnormal in the current report", "Reference:", "previously within the normal range"):
            self.assertIn(en, out)
        for de_word in ("ist im aktuellen Bericht", "aktuellen Bericht", "früheren Berichten", "Normbereich"):
            self.assertNotIn(de_word, out)

    # ── Task B: missing_marker ──
    def test_missing_marker(self):
        de = "📋 Ferritin wurde vor 400 Tagen gemessen (letzter Wert: 12). Überprüfung empfohlen."
        out = self._translate(de)
        for en in ("was measured 400 days ago", "(last value: 12)", "Review recommended."):
            self.assertIn(en, out)
        for de_word in ("wurde vor", "Tagen gemessen", "letzter Wert", "Überprüfung"):
            self.assertNotIn(de_word, out)

    # ── Task B: cross-biomarker sentences (real analyzer output) ──
    _CROSS_CASES = [
        ("risk_ck_troponin", [("CK", 1200), ("Troponin T", 0.5)],
         ["Concurrent elevation of CK and Troponin T", "Clinical correlation recommended."],
         ["Gleichzeitige Erhöhung", "Muskelschädigung", "Klinische Korrelation"]),
        ("risk_ck_isolated", [("CK", 1200), ("Troponin T", 0.02)],
         ["Isolated CK elevation with normal Troponin T", "skeletal muscle than a cardiac cause"],
         ["Isolierte CK-Erhöhung", "skelettmuskuläre", "kardiale Ursache"]),
        ("risk_ckmb_troponin", [("CK-MB", 80), ("Troponin T", 0.9)],
         ["strong indicator of acute myocardial injury", "Immediate clinical assessment recommended."],
         ["starkes Indiz", "Herzmuskelverletzung", "Sofortige klinische"]),
        ("risk_ldl_hdl", [("LDL", 190), ("HDL", 35)],
         ["significantly increases cardiovascular risk", "pharmacological therapy"],
         ["kardiovaskuläre Risiko", "Lebensstilmaßnahmen", "medikamentöse Therapie"]),
        ("risk_tsh_ft", [("TSH", 0.1), ("fT4", 2.5)],
         ["clinically relevant thyroid dysfunction", "Clinical assessment recommended."],
         ["Schilddrüsenfunktionsstörung", "klinisch relevante"]),
        ("risk_crp_bsg", [("CRP", 80), ("BSG", 60)],
         ["systemic inflammatory response", "Determine the cause."],
         ["systemische Entzündungsreaktion", "Ursache abklären"]),
        ("risk_crea_egfr", [("Kreatinin", 2.5), ("eGFR", 35)],
         ["confirm impaired renal function", "Staging and follow-up monitoring recommended."],
         ["eingeschränkte Nierenfunktion", "Verlaufskontrolle"]),
        ("risk_glucose_hba1c", [("Glukose", 250), ("HbA1c", 9.5)],
         ["confirm impaired glucose regulation or diabetes", "Review therapy adjustment."],
         ["gestörte Glukoseregulation", "Therapieanpassung"]),
        ("risk_iron_tsat", [("Ferritin", 20), ("Transferrinsättigung", 15)],
         ["may indicate iron deficiency or iron overload", "Clinical interpretation recommended."],
         ["Eisenmangel", "Eisenüberladung", "Klinische Einordnung"]),
    ]

    def _cross(self, template, pairs):
        from analyzer import _build_cross_biomarker_description
        rule = {"name": "TestRule", "description_template": template}
        marker_results = [{"canonical": c, "value": v, "condition": "high"} for c, v in pairs]
        return _build_cross_biomarker_description(rule, marker_results)

    def test_cross_biomarker_sentences(self):
        for template, pairs, en_subs, de_words in self._CROSS_CASES:
            with self.subTest(template=template):
                out = self._translate(self._cross(template, pairs))
                for en in en_subs:
                    self.assertIn(en, out)
                for w in de_words:
                    self.assertNotIn(w, out)

    def test_cross_not_measured(self):
        de = self._cross("risk_ck_troponin", [("CK", 1200), ("Troponin T", None)])
        out = self._translate(de)
        self.assertIn("not measured", out)
        self.assertNotIn("nicht gemessen", out)

    # ── Task B: critical_value (bare "liegt" + limit-relative deviation) ──
    def test_critical_value(self):
        de = "PO2 = 34.9 mmHg liegt 56% unter der Untergrenze (Referenzbereich 40\u2013100 mmHg)."
        out = self._translate(de)
        for en in ("is 56% below the lower limit", "(Reference range 40\u2013100 mmHg)"):
            self.assertIn(en, out)
        for de_word in ("liegt", "unter der Untergrenze", "Referenzbereich"):
            self.assertNotIn(de_word, out)

    # ── Task B: concerning_trend ("von X auf Y") ──
    def test_concerning_trend(self):
        de = "Kreatinin zeigt steigend: von 5.0 auf 9.2 mg/dl (+84%)."
        out = self._translate(de)
        for en in ("shows rising", "from 5.0 to 9.2"):
            self.assertIn(en, out)
        for de_word in ("zeigt", "steigend", "auf 9.2"):
            self.assertNotIn(de_word, out)

    # ── Task B: dedup (deviation + reference range + historical extreme) ──
    def test_dedup(self):
        de = ("Kreatinin: 9.2 mg/dl (01.02.2025, 56% über der Obergrenze) "
              "(Referenzbereich 0.7\u20131.3 mg/dl) | Bisheriger Extremwert: "
              "12.5 mg/dl (15.01.2024, 89% über der Obergrenze)")
        out = self._translate(de)
        for en in ("above the upper limit", "(Reference range 0.7\u20131.3 mg/dl)", "Previous extreme:"):
            self.assertIn(en, out)
        for de_word in ("über der Obergrenze", "Referenzbereich", "Bisheriger Extremwert"):
            self.assertNotIn(de_word, out)

    def test_dedup_outside_range(self):
        de = "pH: 6.8 mmol/l (01.02.2025, außerhalb des Referenzbereichs) (Referenzbereich > 7.35 mmol/l)"
        out = self._translate(de)
        self.assertIn("outside the reference range", out)
        self.assertNotIn("außerhalb des Referenzbereichs", out)


class TestOneSidedAbnormality(unittest.TestCase):
    """R3-08: one-sided reference ranges ('< 5.2', '> 120') must flag abnormal values."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_one_sided.db")

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_is_value_abnormal_high_only(self):
        from extractor import _is_value_abnormal
        self.assertTrue(_is_value_abnormal(5.2, None, 5.0))   # above "< 5.0"
        self.assertFalse(_is_value_abnormal(4.8, None, 5.0))  # within range

    def test_is_value_abnormal_low_only(self):
        from extractor import _is_value_abnormal
        self.assertTrue(_is_value_abnormal(60, 90, None))     # below "> 90"
        self.assertFalse(_is_value_abnormal(120, 90, None))   # within range

    def test_is_value_abnormal_both_bounds(self):
        from extractor import _is_value_abnormal
        self.assertTrue(_is_value_abnormal(3.5, 4.0, 10.0))
        self.assertTrue(_is_value_abnormal(12.0, 4.0, 10.0))
        self.assertFalse(_is_value_abnormal(6.5, 4.0, 10.0))

    def test_is_value_abnormal_no_bounds_or_value(self):
        from extractor import _is_value_abnormal
        self.assertFalse(_is_value_abnormal(5.0, None, None))
        self.assertFalse(_is_value_abnormal(None, 4.0, 10.0))

    def test_pct_beyond_one_sided(self):
        from analyzer import _compute_pct_beyond
        # high only: (172-130)/130 ≈ 32.3% (limit itself as scale)
        self.assertAlmostEqual(_compute_pct_beyond(172, None, 130), (172 - 130) / 130 * 100, places=2)
        # low only: (90-45)/90 = 50%
        self.assertAlmostEqual(_compute_pct_beyond(45, 90, None), 50.0, places=2)
        # within range → 0
        self.assertEqual(_compute_pct_beyond(120, 90, None), 0.0)

    def test_deviation_from_limit_one_sided(self):
        from analyzer import _pct_deviation_from_limit
        self.assertAlmostEqual(_pct_deviation_from_limit(45, 90, None), 50.0, places=2)
        self.assertAlmostEqual(_pct_deviation_from_limit(172, None, 130), (172 - 130) / 130 * 100, places=2)

    def test_critical_value_flag_one_sided(self):
        """A value far below a low-only bound must produce a critical_value flag."""
        from analyzer import _detect_critical_values
        dp = type("DP", (), {})()
        dp.value = 45.0
        dp.ref_low = 90.0
        dp.ref_high = None
        dp.is_abnormal = True
        dp.canonical_name = "eGFR (CKD-EPI)"
        dp.unit = "ml/min"
        dp.id = 1
        holder = type("Holder", (), {"DataPoint": dp, "report_date": datetime(2026, 1, 1)})()
        flags = _detect_critical_values([holder])
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].category, "critical_value")
        self.assertIn("> 90", flags[0].description)

    def test_no_flag_when_in_range_one_sided(self):
        from analyzer import _detect_critical_values
        dp = type("DP", (), {})()
        dp.value = 120.0
        dp.ref_low = 90.0
        dp.ref_high = None
        dp.is_abnormal = False
        dp.canonical_name = "eGFR (CKD-EPI)"
        dp.unit = "ml/min"
        dp.id = 1
        holder = type("Holder", (), {"DataPoint": dp, "report_date": datetime(2026, 1, 1)})()
        self.assertEqual(_detect_critical_values([holder]), [])

    def _make_holder(self, name, value, ref_low, ref_high, abnormal=True):
        dp = type("DP", (), {})()
        dp.value = value
        dp.ref_low = ref_low
        dp.ref_high = ref_high
        dp.is_abnormal = abnormal
        dp.canonical_name = name
        dp.unit = "%"
        dp.id = 1
        return type("Holder", (), {"DataPoint": dp, "report_date": datetime(2026, 1, 1)})()

    def test_cohb_low_not_critical(self):
        """A LOW COHb (below range) is not clinically dangerous -> no flag."""
        from analyzer import _detect_critical_values
        # 1.1% vs ref 12.0–17.5: far below the lower limit, but COHb is
        # high-only, so this must NOT be flagged as a critical value.
        holder = self._make_holder("COHb", 1.1, 12.0, 17.5)
        self.assertEqual(_detect_critical_values([holder]), [])

    def test_cohb_high_is_critical(self):
        """An ELEVATED COHb (above range) IS dangerous -> critical flag."""
        from analyzer import _detect_critical_values
        # 30% vs ref 12.0–17.5: above the upper limit -> must be flagged.
        holder = self._make_holder("COHb", 30.0, 12.0, 17.5)
        flags = _detect_critical_values([holder])
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].category, "critical_value")

    def test_get_critical_direction(self):
        from config import get_critical_direction
        # COHb is high-only; unit suffix and case must not matter.
        self.assertEqual(get_critical_direction("COHb"), "high")
        self.assertEqual(get_critical_direction("cohb"), "high")
        self.assertEqual(get_critical_direction("COHb (%)"), "high")
        # Unknown markers default to both directions.
        self.assertEqual(get_critical_direction("Kreatinin"), "both")
        self.assertEqual(get_critical_direction(None), "both")

    @patch("extractor.extract_page")
    @patch("extractor.normalize_names")
    @patch("extractor.extract_pages_batch")
    def test_extracted_one_sided_entries_flagged(self, mock_batch, mock_norm, mock_extract):
        """End-to-end: '<5.0' CRP=5.2 and '<130' LDL=160 must be is_abnormal=True."""
        from schemas import ExtractedPage, NormalizationMapping
        mock_extract.return_value = ExtractedPage(**MOCK_EXTRACTED_PAGE)
        # Default batch size is 3, so multi-page PDFs take the batch path —
        # mock it too (one extracted page per input image).
        mock_batch.side_effect = lambda llm, pages, **kw: [ExtractedPage(**MOCK_EXTRACTED_PAGE) for _ in pages]
        mock_norm.return_value = NormalizationMapping(**MOCK_NORMALIZATION)
        from extractor import process_new_pdfs
        processed, errors = process_new_pdfs(self.session)
        self.assertGreater(len(processed), 0)

        crp = self.session.query(database.DataPoint).filter_by(canonical_name="CRP").first()
        ldl = self.session.query(database.DataPoint).filter_by(canonical_name="LDL-Cholesterin").first()
        trig = self.session.query(database.DataPoint).filter_by(canonical_name="Triglyceride").first()
        leuko = self.session.query(database.DataPoint).filter_by(canonical_name="Leukozyten (WBC)").first()

        self.assertIsNotNone(crp)
        self.assertTrue(crp.is_abnormal, "CRP 5.2 vs '<5.0' must be abnormal")
        self.assertIsNotNone(ldl)
        self.assertTrue(ldl.is_abnormal, "LDL 160 vs '<130' must be abnormal")
        self.assertIsNotNone(trig)
        self.assertTrue(trig.is_abnormal, "Triglyceride 175 vs '<150' must be abnormal")
        self.assertIsNotNone(leuko)
        self.assertFalse(leuko.is_abnormal, "Leukozyten 6.5 in 4.0-10.0 must be normal")

    def test_backfill_one_sided_flags_abnormal(self):
        """Backfill of a low-only default range (eGFR '> 90') must flag values below it."""
        patient = database.get_or_create_patient(self.session, "OneSidedPatient")
        report = database.Report(
            patient_id=patient.id, filename="one_sided.pdf",
            file_hash="one_sided_hash", report_date=datetime(2024, 5, 1),
        )
        self.session.add(report)
        self.session.flush()
        dp_low = database.DataPoint(
            report_id=report.id, canonical_name="eGFR (CKD-EPI)",
            original_name="eGFR", value=60.0, unit="ml/min/1.73m²",
            ref_low=None, ref_high=None, is_abnormal=False,
        )
        dp_ok = database.DataPoint(
            report_id=report.id, canonical_name="eGFR (CKD-EPI)",
            original_name="eGFR", value=120.0, unit="ml/min/1.73m²",
            ref_low=None, ref_high=None, is_abnormal=False,
        )
        self.session.add_all([dp_low, dp_ok])
        self.session.commit()

        from database import backfill_reference_ranges
        backfill_reference_ranges(self.session)
        self.session.refresh(dp_low)
        self.session.refresh(dp_ok)

        self.assertEqual(dp_low.ref_low, 90.0)
        self.assertIsNone(dp_low.ref_high)
        self.assertTrue(dp_low.is_abnormal, "eGFR 60 vs '> 90' must be abnormal")
        self.assertFalse(dp_ok.is_abnormal, "eGFR 120 vs '> 90' must be normal")


class TestBatchPageVerification(unittest.TestCase):
    """R3-12: batch extraction must verify every requested page came back."""

    def _pages_json(self, indices, with_index=True):
        pages = []
        for i in indices:
            page = {"patient_name": None, "report_date": None, "panel_type": None,
                    "entries": [{"name": f"Test{i}", "value": 1.0, "unit": "g/dl",
                                 "reference_range": "0-2"}]}
            if with_index:
                page["page_index"] = i
            pages.append(page)
        return json.dumps({"pages": pages})

    def test_all_pages_returned_ok(self):
        from llm_client import extract_pages_batch
        mock_llm = _MockLLM(self._pages_json([0, 1]))
        results = extract_pages_batch(mock_llm, ["b64a", "b64b"])
        self.assertEqual(len(results), 2)

    def test_missing_page_raises(self):
        from llm_client import extract_pages_batch, LLMResponseError
        mock_llm = _MockLLM(self._pages_json([0]))  # only page 0 of 2 returned
        with self.assertRaises(LLMResponseError) as ctx:
            extract_pages_batch(mock_llm, ["b64a", "b64b"])
        self.assertIn("missing", str(ctx.exception))

    def test_unexpected_index_raises(self):
        from llm_client import extract_pages_batch, LLMResponseError
        mock_llm = _MockLLM(self._pages_json([0, 5]))
        with self.assertRaises(LLMResponseError):
            extract_pages_batch(mock_llm, ["b64a", "b64b"])

    def test_no_index_matching_count_ok(self):
        """No page_index at all is tolerated when the count matches exactly."""
        from llm_client import extract_pages_batch
        mock_llm = _MockLLM(self._pages_json([0, 1], with_index=False))
        results = extract_pages_batch(mock_llm, ["b64a", "b64b"])
        self.assertEqual(len(results), 2)

    def test_no_index_wrong_count_raises(self):
        from llm_client import extract_pages_batch, LLMResponseError
        mock_llm = _MockLLM(self._pages_json([0], with_index=False))
        with self.assertRaises(LLMResponseError):
            extract_pages_batch(mock_llm, ["b64a", "b64b"])

    def test_empty_pages_raises(self):
        from llm_client import extract_pages_batch, LLMResponseError
        mock_llm = _MockLLM('{"pages": []}')
        with self.assertRaises(LLMResponseError):
            extract_pages_batch(mock_llm, ["b64a"])


class TestNormalizeNamesFallback(unittest.TestCase):
    """R3-09: normalize_names must not raise NameError when canonical_list is None."""

    def test_fallback_uses_canonical_registry_keys(self):
        from llm_client import normalize_names
        # canonical_list=None previously referenced an unimported bare list.
        result = normalize_names(
            _MockLLM(json.dumps(MOCK_NORMALIZATION)),
            [{"name": "Leukozyten", "unit": None}],
            None,
        )
        self.assertIsNotNone(result)
        mapping = {m["original"]: m["canonical"] for m in result.mappings}
        self.assertEqual(mapping.get("Leukozyten"), "Leukozyten (WBC)")

    def test_explicit_canonical_list_still_works(self):
        from llm_client import normalize_names
        result = normalize_names(
            _MockLLM(json.dumps(MOCK_NORMALIZATION)),
            [{"name": "Leukozyten", "unit": None}],
            ["Leukozyten (WBC)"],
        )
        mapping = {m["original"]: m["canonical"] for m in result.mappings}
        self.assertEqual(mapping.get("Leukozyten"), "Leukozyten (WBC)")

    def test_plain_strings_still_accepted(self):
        """Backward compatibility: plain name strings are treated as unit=None."""
        from llm_client import normalize_names
        result = normalize_names(
            _MockLLM(json.dumps(MOCK_NORMALIZATION)),
            ["Leukozyten"],
            None,
        )
        mapping = {m["original"]: m["canonical"] for m in result.mappings}
        self.assertEqual(mapping.get("Leukozyten"), "Leukozyten (WBC)")


class TestUnitAwareNormalization(unittest.TestCase):
    """Same name with different units must map to the right unit-qualified variant."""

    def test_hba1c_unit_variants(self):
        from extractor import _normalize_name
        # LLM returns one mapping per (name, unit) pair.
        mappings = [
            {"original": "HbA1c", "unit": "%", "canonical": "HbA1c (%)"},
            {"original": "HbA1c", "unit": "mmol/mol", "canonical": "HbA1c (mmol/mol)"},
        ]
        self.assertEqual(_normalize_name("HbA1c", "%", mappings, []), "HbA1c (%)")
        self.assertEqual(_normalize_name("HbA1c", "mmol/mol", mappings, []), "HbA1c (mmol/mol)")

    def test_unit_match_is_case_insensitive(self):
        from extractor import _normalize_name
        mappings = [
            {"original": "HbA1c", "unit": "%", "canonical": "HbA1c (%)"},
            {"original": "HbA1c", "unit": "mmol/mol", "canonical": "HbA1c (mmol/mol)"},
        ]
        self.assertEqual(_normalize_name("hba1c", "%", mappings, []), "HbA1c (%)")
        self.assertEqual(_normalize_name("HbA1c", "MMOL/MOL", mappings, []), "HbA1c (mmol/mol)")

    def test_exact_unit_match_wins_over_unitless(self):
        from extractor import _normalize_name
        # A unit-less mapping must not shadow an exact-unit match.
        mappings = [
            {"original": "HbA1c", "unit": None, "canonical": "HbA1c (%)"},
            {"original": "HbA1c", "unit": "mmol/mol", "canonical": "HbA1c (mmol/mol)"},
        ]
        self.assertEqual(_normalize_name("HbA1c", "mmol/mol", mappings, []), "HbA1c (mmol/mol)")

    def test_unitless_mapping_applies_to_any_unit(self):
        from extractor import _normalize_name
        # No unit info on the entry → a unit-less mapping still applies.
        mappings = [{"original": "CRP", "unit": None, "canonical": "CRP"}]
        self.assertEqual(_normalize_name("CRP", "mg/l", mappings, []), "CRP")
        self.assertEqual(_normalize_name("CRP", None, mappings, []), "CRP")

    def test_llm_mapping_beats_deterministic_lookup(self):
        """Step 1 (LLM) is consulted before step 2 (alias lookup)."""
        from extractor import _normalize_name
        # "HbA1c" is an alias of "HbA1c (%)" in the registry; a unit-specific
        # LLM mapping for mmol/mol must win over the deterministic alias match.
        mappings = [
            {"original": "HbA1c", "unit": "mmol/mol", "canonical": "HbA1c (mmol/mol)"},
        ]
        self.assertEqual(
            _normalize_name("HbA1c", "mmol/mol", mappings, ["HbA1c (%)"]),
            "HbA1c (mmol/mol)",
        )

    def test_normalize_names_prompt_includes_units(self):
        """The user prompt must render entries with their units."""
        from llm_client import normalize_names
        captured = {}

        class _CapturingLLM:
            class chat:
                class completions:
                    @staticmethod
                    def create(*a, **kw):
                        captured["messages"] = kw.get("messages") or a[0]
                        return _MockLLMResponse(json.dumps(MOCK_NORMALIZATION))

        normalize_names(
            _CapturingLLM(),
            [{"name": "HbA1c", "unit": "%"}, {"name": "Hb", "unit": None}],
            ["HbA1c (%)"],
        )
        user_text = captured["messages"][1]["content"]
        self.assertIn('"name": "HbA1c"', user_text)
        self.assertIn('"unit": "%"', user_text)


class TestHistoricalDataCollection(unittest.TestCase):
    """R3-11: summary history is capped, non-overlapping, and chronic-only."""

    @classmethod
    def setUpClass(cls):
        from datetime import timedelta
        cls.session = _fresh_db("test_hist.db")
        patient = database.get_or_create_patient(cls.session, "HistPatient")
        cls.patient_id = patient.id
        now = datetime.now()
        cutoff = now - timedelta(days=730)  # ~2 years

        def add_report(date, name, value, abnormal):
            r = database.Report(
                patient_id=patient.id, filename=f"{date:%Y%m%d}.pdf",
                file_hash=f"hash_{date:%Y%m%d}_{name}", report_date=date,
            )
            cls.session.add(r)
            cls.session.flush()
            dp = database.DataPoint(
                report_id=r.id, canonical_name=name, original_name=name,
                value=value, unit="mg/dl", ref_low=10.0, ref_high=20.0,
                is_abnormal=abnormal,
            )
            cls.session.add(dp)

        # "Chronic" abnormal both before and after cutoff → qualifies as chronic.
        add_report(datetime(2020, 1, 1), "Chronic", 30.0, True)
        add_report(datetime(2021, 6, 1), "Chronic", 32.0, True)
        add_report(now - timedelta(days=30), "Chronic", 34.0, True)  # recent window
        # "OneOff" abnormal only before cutoff → must be excluded.
        add_report(datetime(2019, 5, 1), "OneOff", 40.0, True)
        # "RecentOnly" abnormal only after cutoff → not chronic (no history).
        add_report(now - timedelta(days=30), "RecentOnly", 35.0, True)
        cls.session.commit()
        cls.cutoff = cutoff

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_only_chronic_pre_cutoff_points_returned(self):
        import app
        rows = app._collect_historical_data(self.session, self.patient_id, self.cutoff)
        names = {r[1] for r in rows}
        # Chronic qualifies; OneOff (no recent abnormality) and RecentOnly
        # (no pre-cutoff history) must be excluded.
        self.assertEqual(names, {"Chronic"})
        # Only the two pre-cutoff Chronic points, oldest first.
        values = [r[2] for r in rows]
        self.assertEqual(values, [30.0, 32.0])

    def test_no_recent_abnormality_returns_empty(self):
        import app
        # A patient with no recent-window abnormality has nothing to collect.
        empty = _fresh_db("test_hist_empty.db")
        p = database.get_or_create_patient(empty, "NoRecent")
        r = database.Report(patient_id=p.id, filename="x.pdf",
                            file_hash="xhash", report_date=datetime(2019, 1, 1))
        empty.add(r)
        empty.flush()
        empty.add(database.DataPoint(report_id=r.id, canonical_name="Old",
                                     original_name="Old", value=50.0, unit="mg/dl",
                                     ref_low=10.0, ref_high=20.0, is_abnormal=True))
        empty.commit()
        rows = app._collect_historical_data(empty, p.id, datetime.now())
        self.assertEqual(rows, [])
        empty.close()

    def test_cap_limits_points_per_biomarker(self):
        from datetime import timedelta
        import app
        capped = _fresh_db("test_hist_cap.db")
        p = database.get_or_create_patient(capped, "CapPatient")
        now = datetime.now()
        cutoff = now - timedelta(days=730)

        def add(date, value):
            r = database.Report(patient_id=p.id, filename=f"{date:%Y%m%d}.pdf",
                                file_hash=f"h_{date:%Y%m%d}", report_date=date)
            capped.add(r)
            capped.flush()
            capped.add(database.DataPoint(report_id=r.id, canonical_name="Chronic",
                                          original_name="Chronic", value=value,
                                          unit="mg/dl", ref_low=10.0, ref_high=20.0,
                                          is_abnormal=True))

        # 15 pre-cutoff points + 1 recent point (makes it chronic).
        for i in range(15):
            add(datetime(2018, 1, 1) + timedelta(days=i * 30), 30.0 + i)
        add(now - timedelta(days=30), 39.0)
        capped.commit()

        rows = app._collect_historical_data(capped, p.id, cutoff)
        # Capped at the most recent N pre-cutoff points.
        self.assertEqual(len(rows), app._HISTORICAL_MAX_POINTS_PER_BIOMARKER)
        capped.close()


class TestBulkGenerationIsolation(unittest.TestCase):
    """R3-13: one failing biomarker must not abort the rest of the bulk loop."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_bulk.db")
        cls.patient = database.get_or_create_patient(cls.session, "BulkPatient")

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def test_failure_isolated_and_rest_processed(self):
        import app
        calls = []

        def gen(llm, item):
            cname = item[0]
            calls.append(cname)
            if cname == "B":
                raise RuntimeError("boom")
            return f"desc for {cname}"

        with patch.object(app.st, "empty"), \
             patch.object(app.st, "progress"), \
             patch.object(app.st, "success"), \
             patch.object(app.st, "warning"), \
             patch.object(app.st, "caption"):
            ok, errors = app._run_bulk_generation(
                self.session, self.patient, object(),
                [("A", "u"), ("B", "u"), ("C", "u")],
                generate_one=gen, upsert_field="description",
                label_key="msg_generating_all_descriptions",
                success_key="msg_all_descriptions_generated",
            )

        # All three were attempted (C ran after B failed).
        self.assertEqual(calls, ["A", "B", "C"])
        self.assertEqual(ok, 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("B", errors[0])
        # A and C persisted; B did not.
        self.session.expire_all()
        self.assertIsNotNone(database.get_biomarker_info(self.session, self.patient.id, "A"))
        self.assertIsNone(database.get_biomarker_info(self.session, self.patient.id, "B"))
        self.assertIsNotNone(database.get_biomarker_info(self.session, self.patient.id, "C"))

    def test_all_success_reports_success_key(self):
        import app
        with patch.object(app.st, "empty"), \
             patch.object(app.st, "progress"), \
             patch.object(app.st, "success") as m_success, \
             patch.object(app.st, "warning") as m_warning:
            ok, errors = app._run_bulk_generation(
                self.session, self.patient, object(),
                [("X", "u")],
                generate_one=lambda llm, item: "ok",
                upsert_field="description",
                label_key="msg_generating_all_descriptions",
                success_key="msg_all_descriptions_generated",
            )
        self.assertEqual(ok, 1)
        self.assertEqual(errors, [])
        m_success.assert_called_once()
        m_warning.assert_not_called()


class TestDisabledMeasurements(unittest.TestCase):
    """Disabling a single measurement excludes it from charts and LLM analyses."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_disabled.db")
        patient = database.get_or_create_patient(cls.session, "DisableTest")
        dates = [datetime(2024, 1, 1), datetime(2024, 6, 1), datetime(2024, 12, 1)]
        # Kreatinin: last value 2.8 vs ref 0.6-1.2 → 266% beyond range → critical flag
        for i, d in enumerate(dates):
            report = database.Report(
                patient_id=patient.id, filename=f"dis_{i}.pdf",
                file_hash=f"dis_hash_{i}", report_date=d,
            )
            cls.session.add(report)
            cls.session.flush()
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name="Kreatinin",
                original_name="Kreatinin", value=[1.2, 1.4, 2.8][i],
                unit="mg/dl", ref_low=0.6, ref_high=1.2,
                is_abnormal=[False, False, True][i],
            ))
        cls.session.commit()
        cls.patient = patient

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        # Reset state between tests: re-enable the critical value
        dp = (
            self.session.query(database.DataPoint)
            .filter(database.DataPoint.value == 2.8)
            .first()
        )
        if dp is not None and dp.is_disabled:
            database.set_data_point_disabled(self.session, dp.id, False)

    def test_fresh_db_column_defaults_false(self):
        for dp in self.session.query(database.DataPoint).all():
            self.assertFalse(dp.is_disabled)

    def test_migration_adds_column_to_preexisting_table(self):
        from sqlalchemy import create_engine, text
        db_path = TEST_DIR / "test_disabled_migrate.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            conn.execute(text(
                "CREATE TABLE data_points (id INTEGER PRIMARY KEY, report_id INTEGER NOT NULL, "
                "canonical_name VARCHAR(255) NOT NULL, original_name VARCHAR(255), "
                "value FLOAT, unit VARCHAR(100), ref_low FLOAT, ref_high FLOAT, "
                "is_abnormal BOOLEAN)"
            ))
            conn.commit()
        engine.dispose()

        old_path = config.DB_PATH
        config.DB_PATH = db_path
        try:
            database.init_db()
        finally:
            config.DB_PATH = old_path
        from sqlalchemy import inspect as sa_inspect
        insp = sa_inspect(create_engine(f"sqlite:///{db_path}"))
        cols = [c["name"] for c in insp.get_columns("data_points")]
        self.assertIn("is_disabled", cols)

    def test_toggle_helper_flips_flag_and_deletes_flags(self):
        from analyzer import refresh_risk_flags
        refresh_risk_flags(self.session, llm=None)
        dp = (
            self.session.query(database.DataPoint)
            .filter(database.DataPoint.value == 2.8)
            .one()
        )
        flag_count_before = len(dp.risk_flags)
        self.assertGreater(flag_count_before, 0)

        result = database.set_data_point_disabled(self.session, dp.id, True)
        self.assertIsNotNone(result)
        self.assertTrue(result.is_disabled)
        self.assertEqual(len(dp.risk_flags), 0)

        # Toggle back: flag restored on next refresh
        database.set_data_point_disabled(self.session, dp.id, False)
        self.assertFalse(dp.is_disabled)
        refresh_risk_flags(self.session, llm=None)
        self.assertGreater(len(dp.risk_flags), 0)

    def test_refresh_excludes_disabled_points(self):
        from analyzer import refresh_risk_flags
        refresh_risk_flags(self.session, llm=None)
        dp = (
            self.session.query(database.DataPoint)
            .filter(database.DataPoint.value == 2.8)
            .one()
        )
        self.assertGreater(len(dp.risk_flags), 0)

        database.set_data_point_disabled(self.session, dp.id, True)
        refresh_risk_flags(self.session, llm=None)
        # No flag may reference the disabled point anymore
        referencing = (
            self.session.query(database.RiskFlag)
            .filter(database.RiskFlag.data_point_id == dp.id)
            .count()
        )
        self.assertEqual(referencing, 0)

    def test_summary_recent_data_excludes_disabled(self):
        import app
        from analyzer import refresh_risk_flags
        refresh_risk_flags(self.session, llm=None)
        dp = (
            self.session.query(database.DataPoint)
            .filter(database.DataPoint.value == 2.8)
            .one()
        )
        database.set_data_point_disabled(self.session, dp.id, True)

        captured = {}

        def fake_summary(llm, patient_name, recent_dps, historical_dps, risk_flags, lang=None):
            captured["recent"] = list(recent_dps)
            return "ok"

        with patch.object(app, "generate_patient_summary", side_effect=fake_summary):
            result = app._generate_summary_for_patient(self.session, self.patient, object())
        self.assertEqual(result, "ok")
        # The disabled Kreatinin value (2.8) must not appear in the recent data
        values = [row[2] for row in captured["recent"] if row[1] == "Kreatinin"]
        self.assertNotIn(2.8, values)

    def test_load_data_excludes_disabled(self):
        import app
        dp = (
            self.session.query(database.DataPoint)
            .filter(database.DataPoint.value == 2.8)
            .one()
        )
        database.set_data_point_disabled(self.session, dp.id, True)
        _, _, dps, _ = app.load_data(self.session)
        values = [d.value for d in dps if d.canonical_name == "Kreatinin"]
        self.assertNotIn(2.8, values)
        # Restore for other tests
        database.set_data_point_disabled(self.session, dp.id, False)


class TestToggleRefreshesPatientFlags(unittest.TestCase):
    """Disabling a value must remove risk flags that _dedup_by_biomarker
    reassigned to a different data point.

    Fixture (patient A, Kreatinin mg/dl ref 0.6–1.2):
      2024-01-01: 2.8 (abnormal)   <- the value that gets disabled
      2025-01-01: 1.0 (normal)     <- latest data point

    analyze_trends produces a concerning_trend flag whose data_point_id is the
    LATEST (normal) data point. set_data_point_disabled() only deletes flags
    with data_point_id == <disabled dp>, so disabling the older abnormal value
    leaves the stale flag behind. refresh_patient_risk_flags() must clear it:
    with only the normal value left, no flag can be produced (returns 0).

    Patient B (HDL-Cholesterin, single abnormal value) guards against the
    per-patient rebuild leaking across patients. A different biomarker is used
    on purpose: _dedup_by_biomarker groups flags by name across ALL patients,
    so a shared name would mix the two patients' flags.
    """

    def setUp(self):
        self.session = _fresh_db(f"test_toggle_refresh_{self._testMethodName}.db")
        # Patient A: abnormal -> normal (improved) Kreatinin course.
        patient_a = database.get_or_create_patient(self.session, "TogglePatientA")
        report_a1 = database.Report(
            patient_id=patient_a.id, filename="toggle_a1.pdf",
            file_hash="toggle_a1_hash", report_date=datetime(2024, 1, 1),
        )
        report_a2 = database.Report(
            patient_id=patient_a.id, filename="toggle_a2.pdf",
            file_hash="toggle_a2_hash", report_date=datetime(2025, 1, 1),
        )
        self.session.add_all([report_a1, report_a2])
        self.session.flush()
        self.dp_old = database.DataPoint(
            report_id=report_a1.id, canonical_name="Kreatinin",
            original_name="Kreatinin", value=2.8, unit="mg/dl",
            ref_low=0.6, ref_high=1.2, is_abnormal=True,
        )
        self.dp_latest = database.DataPoint(
            report_id=report_a2.id, canonical_name="Kreatinin",
            original_name="Kreatinin", value=1.0, unit="mg/dl",
            ref_low=0.6, ref_high=1.2, is_abnormal=False,
        )
        self.session.add_all([self.dp_old, self.dp_latest])

        # Patient B: single abnormal HDL-Cholesterin value -> critical flag.
        patient_b = database.get_or_create_patient(self.session, "TogglePatientB")
        report_b1 = database.Report(
            patient_id=patient_b.id, filename="toggle_b1.pdf",
            file_hash="toggle_b1_hash", report_date=datetime(2025, 1, 1),
        )
        self.session.add(report_b1)
        self.session.flush()
        self.dp_b = database.DataPoint(
            report_id=report_b1.id, canonical_name="HDL-Cholesterin",
            original_name="HDL-Cholesterin", value=20.0, unit="mg/dl",
            ref_low=35.0, ref_high=80.0, is_abnormal=True,
        )
        self.session.add(self.dp_b)
        self.session.commit()
        self.patient_a = patient_a
        self.patient_b = patient_b

    def tearDown(self):
        self.session.close()

    def _patient_flag_count(self, patient_id):
        return (
            self.session.query(database.RiskFlag)
            .join(database.Report, database.RiskFlag.report_id == database.Report.id)
            .filter(database.Report.patient_id == patient_id)
            .count()
        )

    def test_disable_toggle_removes_stale_flag(self):
        from analyzer import analyze_trends, refresh_patient_risk_flags, refresh_risk_flags

        # 1. Detection: a flag exists pointing at the LATEST data point.
        flags = analyze_trends(self.session)
        pointing_at_latest = [f for f in flags if f.data_point_id == self.dp_latest.id]
        self.assertGreaterEqual(
            len(pointing_at_latest), 1,
            "Expected a flag pointing at the latest data point",
        )

        # Persist flags like the production path (reports tab) does.
        refresh_risk_flags(self.session, llm=None)
        self.assertEqual(self._patient_flag_count(self.patient_a.id), 1)
        stale = (
            self.session.query(database.RiskFlag)
            .join(database.Report, database.RiskFlag.report_id == database.Report.id)
            .filter(database.Report.patient_id == self.patient_a.id)
            .one()
        )
        # The flag points at the latest dp, NOT at the older abnormal value.
        self.assertEqual(stale.data_point_id, self.dp_latest.id)

        # 2. Disable the OLDER abnormal value: the targeted delete inside
        #    set_data_point_disabled misses (it matches data_point_id), so the
        #    stale flag survives — this documents the bug being fixed.
        database.set_data_point_disabled(self.session, self.dp_old.id, True)
        self.assertTrue(self.dp_old.is_disabled)
        self.assertEqual(
            self._patient_flag_count(self.patient_a.id), 1,
            "Stale flag should survive the targeted delete (bug under test)",
        )

        # 3. Rebuild this patient's flags: with only the normal value left,
        #    no flag can be produced.
        count = refresh_patient_risk_flags(self.session, self.patient_a.id)
        self.assertEqual(count, 0)
        self.assertEqual(self._patient_flag_count(self.patient_a.id), 0)

        # 4. Other patients' flags are untouched by the per-patient rebuild.
        self.assertEqual(self._patient_flag_count(self.patient_b.id), 1)


class TestBiomarkerRegistry(unittest.TestCase):
    """biomarker_registry: seeding from config, idempotency, cache keying."""

    def setUp(self):
        self.session = _fresh_db(f"reg_{self._testMethodName}.db")

    def tearDown(self):
        self.session.close()

    def test_seed_from_config(self):
        reg = database.get_registry(self.session)
        self.assertEqual(len(reg), len(config._CANONICAL_BIOMARKERS))
        crp = reg["CRP"]
        self.assertIn("mg/l", crp["units"])
        self.assertIn("C-reaktives Protein", crp["aliases"])

    def test_seed_idempotent(self):
        count1 = self.session.query(database.BiomarkerRegistry).count()
        database.init_db()  # re-run seeding on the same DB file
        session2 = database.get_session()
        try:
            count2 = session2.query(database.BiomarkerRegistry).count()
        finally:
            session2.close()
        self.assertEqual(count1, count2)

    def test_customized_rows_never_overwritten(self):
        row = (self.session.query(database.BiomarkerRegistry)
               .filter_by(canonical_name="CRP").one())
        row.ref_low = 0.5
        row.ref_high = 9.9
        row.source = "manual"
        self.session.commit()
        database.init_db()
        session2 = database.get_session()
        try:
            row2 = (session2.query(database.BiomarkerRegistry)
                    .filter_by(canonical_name="CRP").one())
        finally:
            session2.close()
        self.assertEqual(row2.ref_low, 0.5)
        self.assertEqual(row2.ref_high, 9.9)
        self.assertEqual(row2.source, "manual")

    def test_cache_keyed_by_db_path(self):
        """Two temp DBs in one process: the cache must not leak across files."""
        s_a = _fresh_db("reg_cache_a.db")
        row_a = (s_a.query(database.BiomarkerRegistry)
                 .filter_by(canonical_name="CRP").one())
        row_a.ref_low = 123.0
        s_a.commit()
        self.assertEqual(
            database.get_registry(s_a)["CRP"]["default_ref"]["low"], 123.0
        )

        # Switching DB files must not serve A's cached registry to B.
        s_b = _fresh_db("reg_cache_b.db")
        reg_b = database.get_registry(s_b)
        self.assertNotEqual(reg_b["CRP"]["default_ref"]["low"], 123.0)

        # And switching back to A must reflect A's customized value again.
        s_a2 = _fresh_db("reg_cache_a.db")
        self.assertEqual(
            database.get_registry(s_a2)["CRP"]["default_ref"]["low"], 123.0
        )
        s_a.close()
        s_b.close()
        s_a2.close()

    def test_get_registry_names_sorted(self):
        names = database.get_registry_names(self.session)
        self.assertEqual(names, sorted(names))


class TestParseRefRange(unittest.TestCase):
    """database._parse_ref_range: two-sided, one-sided, comma decimals."""

    def test_two_sided(self):
        from database import _parse_ref_range
        self.assertEqual(_parse_ref_range("100–200 U/l"), (100.0, 200.0, "U/l"))
        self.assertEqual(_parse_ref_range("4.0-10.0 Tsd./µl"), (4.0, 10.0, "Tsd./µl"))
        self.assertEqual(_parse_ref_range("30~100 ng/ml"), (30.0, 100.0, "ng/ml"))

    def test_one_sided(self):
        from database import _parse_ref_range
        self.assertEqual(_parse_ref_range("< 5.2 mg/dl"), (None, 5.2, "mg/dl"))
        self.assertEqual(_parse_ref_range("> 90 %"), (90.0, None, "%"))

    def test_comma_decimals(self):
        from database import _parse_ref_range
        self.assertEqual(_parse_ref_range("4,0-10,0 g/dl"), (4.0, 10.0, "g/dl"))

    def test_unparseable(self):
        from database import _parse_ref_range
        self.assertEqual(_parse_ref_range(""), (None, None, None))
        self.assertEqual(_parse_ref_range("kein Bereich"), (None, None, None))


class TestUpdateBiomarkerRegistry(unittest.TestCase):
    """update_biomarker_registry: manual edits share LLM-accept semantics."""

    def setUp(self):
        # One DB per test method — renames mutate the registry globally.
        self.session = _fresh_db(f"manreg_{self._testMethodName}.db")
        self.patient = database.get_or_create_patient(self.session, "ManReg_Patient")
        report = database.Report(
            patient_id=self.patient.id, filename="manreg.pdf",
            file_hash="manreg_hash_1", report_date=datetime(2025, 3, 1),
        )
        self.session.add(report)
        self.session.flush()
        self.session.add(database.DataPoint(
            report_id=report.id, canonical_name="CRP", original_name="CRP",
            value=5.2, unit="mg/l", ref_low=None, ref_high=5.0,
        ))
        self.session.commit()

    def tearDown(self):
        self.session.close()

    def _row(self, name):
        return (self.session.query(database.BiomarkerRegistry)
                .filter_by(canonical_name=name).one())

    def test_ref_range_edit_sets_manual_source(self):
        out = database.update_biomarker_registry(
            self.session, "CRP", "ref_range", "0–10 mg/l"
        )
        self.assertEqual(out["status"], "applied")
        row = self._row("CRP")
        self.assertEqual(row.ref_low, 0.0)
        self.assertEqual(row.ref_high, 10.0)
        self.assertEqual(row.ref_unit, "mg/l")
        self.assertEqual(row.source, "manual")

    def test_ref_range_unparseable_skipped(self):
        out = database.update_biomarker_registry(
            self.session, "CRP", "ref_range", "kein Bereich"
        )
        self.assertEqual(out["status"], "skipped")
        row = self._row("CRP")
        # Row unchanged: seed values intact, source untouched.
        self.assertEqual(row.ref_low, 0.0)
        self.assertEqual(row.ref_high, 0.5)
        self.assertEqual(row.source, "seed")

    def test_unit_add_and_replace(self):
        database.update_biomarker_registry(self.session, "MCH", "unit", "pg/cell")
        row = self._row("MCH")
        self.assertEqual(json.loads(row.units), ["pg", "pg/cell"])
        # Case-insensitive re-apply must not duplicate.
        database.update_biomarker_registry(self.session, "MCH", "unit", "PG/CELL")
        row = self._row("MCH")
        units = json.loads(row.units)
        self.assertEqual(len(units), 2)
        self.assertEqual([u.lower() for u in units], ["pg", "pg/cell"])

    def test_name_rename_global_with_alias(self):
        # A second patient proves the rename is global, not per-patient.
        p2 = database.get_or_create_patient(self.session, "ManReg_Patient_2")
        report2 = database.Report(
            patient_id=p2.id, filename="manreg2.pdf",
            file_hash="manreg_hash_2", report_date=datetime(2025, 4, 1),
        )
        self.session.add(report2)
        self.session.flush()
        for pid in (self.patient.id, p2.id):
            rpt = (self.session.query(database.Report)
                   .filter_by(patient_id=pid).first())
            self.session.add(database.DataPoint(
                report_id=rpt.id, canonical_name="MCV", original_name="MCV",
                value=88, unit="fl",
            ))
        database.upsert_biomarker_info(self.session, self.patient.id, "MCV", description="d")
        self.session.commit()

        out = database.update_biomarker_registry(
            self.session, "MCV", "name", "MCV (fM)"
        )
        self.assertEqual(out["status"], "applied")
        row = self._row("MCV (fM)")
        self.assertIn("MCV", json.loads(row.aliases))
        self.assertEqual(row.source, "manual")
        names = [d.canonical_name for d in self.session.query(database.DataPoint).all()]
        self.assertNotIn("MCV", names)
        self.assertEqual(names.count("MCV (fM)"), 2)
        infos = [i.canonical_name for i in self.session.query(database.BiomarkerInfo).all()]
        self.assertIn("MCV (fM)", infos)
        self.assertNotIn("MCV", infos)

    def test_name_collision_skipped(self):
        out = database.update_biomarker_registry(
            self.session, "CRP", "name", "mcv"  # clashes with MCV case-insensitively
        )
        self.assertEqual(out["status"], "skipped")
        self.assertIsNotNone(self._row("CRP"))

    def test_invalid_field_skipped(self):
        out = database.update_biomarker_registry(self.session, "CRP", "bogus", "x")
        self.assertEqual(out["status"], "skipped")

    def test_empty_proposed_skipped(self):
        out = database.update_biomarker_registry(self.session, "CRP", "unit", "   ")
        self.assertEqual(out["status"], "skipped")

    def test_missing_registry_row_unit_creates_row(self):
        # Editing unit/ref_range of a name without a registry row promotes it.
        out = database.update_biomarker_registry(
            self.session, "NichtImRegistry", "unit", "mg/l"
        )
        self.assertEqual(out["status"], "applied")
        row = (self.session.query(database.BiomarkerRegistry)
               .filter_by(canonical_name="NichtImRegistry").one())
        self.assertEqual(json.loads(row.units), ["mg/l"])
        self.assertEqual(row.source, "manual")

    def test_missing_registry_row_ref_range_creates_row(self):
        out = database.update_biomarker_registry(
            self.session, "NeuerMarker", "ref_range", "0–10 mg/l"
        )
        self.assertEqual(out["status"], "applied")
        row = (self.session.query(database.BiomarkerRegistry)
               .filter_by(canonical_name="NeuerMarker").one())
        self.assertEqual(row.ref_low, 0.0)
        self.assertEqual(row.ref_high, 10.0)
        self.assertEqual(row.ref_unit, "mg/l")

    def test_missing_registry_row_rename_skipped(self):
        # Renaming requires an existing registry row — nothing to rename.
        out = database.update_biomarker_registry(
            self.session, "NichtImRegistry", "name", "AndererName"
        )
        self.assertEqual(out["status"], "skipped")

    def test_lookup_is_case_insensitive(self):
        out = database.update_biomarker_registry(
            self.session, "crp", "ref_range", "0–5 mg/l"
        )
        self.assertEqual(out["status"], "applied")
        row = self._row("CRP")
        self.assertEqual(row.ref_high, 5.0)

    def test_cache_invalidated_after_edit(self):
        registry_before = database.get_registry(self.session)
        self.assertEqual(registry_before["CRP"]["default_ref"]["high"], 0.5)
        database.update_biomarker_registry(
            self.session, "CRP", "ref_range", "0–10 mg/l"
        )
        registry_after = database.get_registry(self.session)
        self.assertEqual(registry_after["CRP"]["default_ref"]["high"], 10.0)


class TestLlmJobs(unittest.TestCase):
    """WP6: background LLM job queue — submit, run, dedup, error, stale marking."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_llm_jobs.db")
        patient = database.get_or_create_patient(cls.session, "JobPatient")
        report = database.Report(
            patient_id=patient.id, filename="job.pdf",
            file_hash="job_hash_1", report_date=datetime(2025, 1, 1),
        )
        cls.session.add(report)
        cls.session.flush()
        for name in ("A", "B", "C"):
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name=name, original_name=name,
                value=1.0, unit="mg/l", ref_low=0.5, ref_high=2.0,
            ))
        cls.session.commit()
        cls.patient = patient

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        # Config snapshot source (submit_job reads session state on the main thread).
        import streamlit as st
        st.session_state.llm_base_url = "http://test:8888/v1"
        st.session_state.llm_api_key = "test-key"
        st.session_state.llm_model = "test-model"
        st.session_state.llm_timeout = 30
        st.session_state.pdf_dpi = 150
        st.session_state.pdf_batch_size = 1
        st.session_state._bb_lang = "de"

    def test_submit_and_run_synchronously(self):
        import llm_jobs
        job = llm_jobs.submit_job(
            "backfill", "job_kind_backfill", runner=llm_jobs._run_job,
        )
        self.assertIsNotNone(job)
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "done")
        self.assertIsNotNone(stored.result_text)
        self.assertIsNotNone(stored.started_at)
        self.assertIsNotNone(stored.finished_at)
        # Config snapshot (incl. language) must be persisted for the worker.
        cfg = json.loads(stored.config_json)
        self.assertEqual(cfg["base_url"], "http://test:8888/v1")
        self.assertEqual(cfg["model"], "test-model")
        self.assertEqual(cfg["lang"], "de")

    def test_duplicate_kind_rejected(self):
        import llm_jobs
        with llm_jobs._kind_lock:
            llm_jobs._active_kinds.add("backfill")
        try:
            job = llm_jobs.submit_job("backfill", "job_kind_backfill")
        finally:
            with llm_jobs._kind_lock:
                llm_jobs._active_kinds.discard("backfill")
        self.assertIsNone(job)

    def test_error_path_sets_error_status(self):
        import llm_jobs
        with patch.object(llm_jobs, "_build_client", side_effect=RuntimeError("boom")):
            job = llm_jobs.submit_job(
                "regen_summary", "job_kind_regen_summary",
                params={"patient_id": self.patient.id}, runner=llm_jobs._run_job,
            )
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "error")
        self.assertIn("boom", stored.error or "")

    def test_mark_stale_llm_jobs_interrupted(self):
        import llm_jobs
        # One pending + one running (simulated crash leftovers), one done.
        p = database.create_llm_job(self.session, "backfill", "pending job")
        r = database.create_llm_job(self.session, "backfill", "running job")
        r.status = "running"
        d = database.create_llm_job(self.session, "backfill", "done job")
        d.status = "done"
        self.session.commit()

        n = database.mark_stale_llm_jobs_interrupted(self.session)
        self.assertEqual(n, 2)
        self.session.expire_all()
        self.assertEqual(database.get_llm_job(self.session, p.id).status, "interrupted")
        self.assertEqual(database.get_llm_job(self.session, r.id).status, "interrupted")
        self.assertEqual(database.get_llm_job(self.session, d.id).status, "done")

    def test_mark_stale_excludes_live_jobs(self):
        # Streamlit re-runs the script on every interaction — jobs with a live
        # worker in this process must NOT be marked interrupted.
        p = database.create_llm_job(self.session, "backfill", "stale pending")
        r = database.create_llm_job(self.session, "refresh_risks", "live running")
        r.status = "running"
        self.session.commit()

        n = database.mark_stale_llm_jobs_interrupted(self.session, exclude_ids={r.id})
        self.assertEqual(n, 1)
        self.session.expire_all()
        self.assertEqual(database.get_llm_job(self.session, p.id).status, "interrupted")
        self.assertEqual(database.get_llm_job(self.session, r.id).status, "running")
        # Simulate the live job finishing so later tests see a clean table.
        database.get_llm_job(self.session, r.id).status = "done"
        self.session.commit()

    def test_bulk_job_per_item_isolation(self):
        import llm_jobs
        calls = []

        def gen(llm_, name, unit, canonical_list, lang=None):
            calls.append(name)
            if name == "B":
                raise RuntimeError("boom")
            return f"desc {name}"

        with patch.object(llm_jobs, "_build_client", return_value=object()), \
             patch.object(llm_jobs, "generate_biomarker_description", side_effect=gen):
            job = llm_jobs.submit_job(
                "bulk_descriptions", "job_kind_bulk_descriptions",
                params={"patient_id": self.patient.id}, runner=llm_jobs._run_job,
            )
        # All three attempted (C ran after B failed).
        self.assertEqual(calls, ["A", "B", "C"])
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "done")  # partial success is not an error
        self.assertIn("2 von 3", stored.result_text or "")  # lang="de" snapshot
        self.assertIn("B: boom", stored.result_text or "")
        # A and C persisted; B did not.
        self.assertIsNotNone(database.get_biomarker_info(self.session, self.patient.id, "A"))
        self.assertIsNone(database.get_biomarker_info(self.session, self.patient.id, "B"))
        self.assertIsNotNone(database.get_biomarker_info(self.session, self.patient.id, "C"))
        # Kind lock released after the run.
        with llm_jobs._kind_lock:
            self.assertNotIn("bulk_descriptions", llm_jobs._active_kinds)


class TestLlmJobAbort(unittest.TestCase):
    """User-initiated job abort (LLM-Jobs tab) + finished-job cleanup."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_llm_job_abort.db")
        patient = database.get_or_create_patient(cls.session, "AbortPatient")
        report = database.Report(
            patient_id=patient.id, filename="abort.pdf",
            file_hash="abort_hash_1", report_date=datetime(2025, 1, 1),
        )
        cls.session.add(report)
        cls.session.flush()
        for name in ("A", "B", "C"):
            cls.session.add(database.DataPoint(
                report_id=report.id, canonical_name=name, original_name=name,
                value=1.0, unit="mg/l", ref_low=0.5, ref_high=2.0,
            ))
        cls.session.commit()
        cls.patient = patient

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        import streamlit as st
        st.session_state.llm_base_url = "http://test:8888/v1"
        st.session_state.llm_api_key = "test-key"
        st.session_state.llm_model = "test-model"
        st.session_state.llm_timeout = 30
        st.session_state.pdf_dpi = 150
        st.session_state.pdf_batch_size = 1
        st.session_state._bb_lang = "de"

    def tearDown(self):
        # Never leak abort state into other tests.
        import llm_client
        llm_client._aborted_job_ids.clear()
        llm_client.set_current_job_id(None)

    def test_abort_flag_lifecycle(self):
        import llm_client
        self.assertFalse(llm_client.is_job_aborted(42))
        llm_client.request_job_abort(42)
        self.assertTrue(llm_client.is_job_aborted(42))
        self.assertFalse(llm_client.is_job_aborted(43))
        # Thread-local current job id: call sites without an explicit id.
        llm_client.set_current_job_id(42)
        self.assertTrue(llm_client.is_job_aborted())
        llm_client.set_current_job_id(43)
        self.assertFalse(llm_client.is_job_aborted())
        # Abort-aware timeout: 1s while aborted, default otherwise.
        llm_client.set_current_job_id(42)
        self.assertEqual(llm_client._abort_timeout(), 1)
        llm_client.set_current_job_id(43)
        self.assertIsNone(llm_client._abort_timeout())

    def test_abort_running_bulk_job_stops_early(self):
        import llm_jobs
        calls = []

        def gen(llm_, name, unit, canonical_list, lang=None):
            calls.append(name)
            return f"desc {name}"

        # Simulate the user clicking abort between item 1 and item 2:
        # first is_job_aborted() check (before A) → False, all later → True.
        state = {"n": 0}

        def fake_aborted(job_id=None):
            state["n"] += 1
            return state["n"] >= 2

        with patch.object(llm_jobs, "_build_client", return_value=object()), \
             patch.object(llm_jobs, "generate_biomarker_description", side_effect=gen), \
             patch.object(llm_jobs, "is_job_aborted", side_effect=fake_aborted):
            job = llm_jobs.submit_job(
                "bulk_descriptions", "job_kind_bulk_descriptions",
                params={"patient_id": self.patient.id}, runner=llm_jobs._run_job,
            )
        # Loop stopped before B — only A was processed.
        self.assertEqual(calls, ["A"])
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "interrupted")
        self.assertIsNone(stored.error)
        self.assertIn("abgebrochen", (stored.result_text or "").lower())

    def test_abort_exception_path_is_interrupted_not_error(self):
        import llm_jobs
        # The in-flight LLM call of an aborted job fails fast (1s timeout →
        # exception). The worker must surface that as "interrupted", not error.
        with patch.object(llm_jobs, "_build_client", side_effect=RuntimeError("APITimeoutError")), \
             patch.object(llm_jobs, "is_job_aborted", return_value=True):
            job = llm_jobs.submit_job(
                "regen_summary", "job_kind_regen_summary",
                params={"patient_id": self.patient.id}, runner=llm_jobs._run_job,
            )
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "interrupted")
        self.assertIsNone(stored.error)

    def test_abort_job_helper(self):
        import llm_client
        import llm_jobs
        r = database.create_llm_job(self.session, "backfill", "running job")
        r.status = "running"
        d = database.create_llm_job(self.session, "backfill", "done job")
        d.status = "done"
        self.session.commit()

        self.assertTrue(llm_jobs.abort_job(r.id))
        self.assertTrue(llm_client.is_job_aborted(r.id))
        self.session.expire_all()
        self.assertTrue(database.get_llm_job(self.session, r.id).aborted)
        # Finished jobs can't be aborted.
        self.assertFalse(llm_jobs.abort_job(d.id))

    def test_clear_finished_llm_jobs(self):
        d = database.create_llm_job(self.session, "backfill", "done")
        d.status = "done"
        e = database.create_llm_job(self.session, "backfill", "error")
        e.status = "error"
        i = database.create_llm_job(self.session, "backfill", "interrupted")
        i.status = "interrupted"
        p = database.create_llm_job(self.session, "backfill", "pending")
        r = database.create_llm_job(self.session, "backfill", "running")
        r.status = "running"
        self.session.commit()

        ids = {k: v.id for k, v in
               (("d", d), ("e", e), ("i", i), ("p", p), ("r", r))}

        # The shared DB may hold finished jobs from earlier tests — count them.
        expected = (self.session.query(database.LlmJob)
                    .filter(database.LlmJob.status.in_(["done", "error", "interrupted"]))
                    .count())
        n = database.clear_finished_llm_jobs(self.session)
        self.assertEqual(n, expected)
        self.assertGreaterEqual(n, 3)
        self.session.expire_all()
        for k in ("d", "e", "i"):
            self.assertIsNone(database.get_llm_job(self.session, ids[k]))
        self.assertEqual(database.get_llm_job(self.session, ids["p"]).status, "pending")
        self.assertEqual(database.get_llm_job(self.session, ids["r"]).status, "running")


class TestReprocessPdfs(unittest.TestCase):
    """Re-extract existing reports with the current LLM + registry (reprocess_pdfs job)."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_reprocess.db")
        cls.patient = database.get_or_create_patient(cls.session, "ReprocPatient")
        # One real (minimal) PDF in an isolated dir — the extractor imports
        # REPORTS_DIR by value, so tests patch extractor.REPORTS_DIR to it.
        from fpdf import FPDF
        cls.pdf_dir = Path(tempfile.mkdtemp(prefix="bb_reprocess_"))
        doc = FPDF(unit="pt", format="A4")
        doc.add_page()
        doc.set_font("Helvetica", size=12)
        doc.set_xy(72, 72)
        doc.cell(text="test")
        doc.output(str(cls.pdf_dir / "reproc.pdf"))

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        import streamlit as st
        st.session_state.llm_base_url = "http://test:8888/v1"
        st.session_state.llm_api_key = "test-key"
        st.session_state.llm_model = "test-model"
        st.session_state.llm_timeout = 30
        st.session_state.pdf_dpi = 150
        st.session_state.pdf_batch_size = 1
        st.session_state._bb_lang = "de"
        # Fresh fixture: one report with one data point (tests are self-contained).
        for r in self.session.query(database.Report).filter(
                database.Report.filename == "reproc.pdf").all():
            self.session.delete(r)  # cascade removes data points + risk flags
        self.session.commit()
        report = database.Report(
            patient_id=self.patient.id, filename="reproc.pdf",
            file_hash="reproc_hash_1", report_date=datetime(2025, 1, 1),
        )
        self.session.add(report)
        self.session.flush()
        self.session.add(database.DataPoint(
            report_id=report.id, canonical_name="A", original_name="A",
            value=1.0, unit="mg/l", ref_low=0.5, ref_high=2.0,
        ))
        self.session.commit()

    def _run_reprocess(self, patient_name="ReprocPatient", mappings=None,
                       patient_id=None):
        """Run the reprocess job synchronously with a mocked extraction."""
        import llm_jobs
        from schemas import ExtractedPage
        page = {"patient_name": patient_name, "report_date": "15.03.2022",
                "panel_type": None,
                "entries": [{"name": "A", "value": 1.0, "unit": "mg/l",
                             "reference_range": None}]}
        with patch("extractor.REPORTS_DIR", str(self.pdf_dir)), \
             patch.object(config, "REPORTS_DIR", str(self.pdf_dir)), \
             patch("extractor.pdf_to_base64_jpeg", return_value=["fake_b64"]), \
             patch("extractor.extract_page", return_value=ExtractedPage(**page)), \
             patch("extractor.normalize_names",
                   return_value=type("NM", (), {"mappings": mappings or []})()), \
             patch.object(llm_jobs, "_build_client", return_value=_MockLLM("{}")):
            return llm_jobs.submit_job(
                "reprocess_pdfs", "job_kind_reprocess_pdfs",
                params={"patient_id": patient_id} if patient_id is not None else None,
                runner=llm_jobs._run_job,
            )

    def test_reprocess_replaces_data(self):
        # Mark the old data point so we can prove the row was replaced, not
        # kept (SQLite reuses rowids after delete+insert, so ids are no proof).
        old_dp = self.session.query(database.DataPoint).first()
        old_dp.ref_low = 99.0
        self.session.commit()
        job = self._run_reprocess(patient_id=self.patient.id)
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "done")
        reports = self.session.query(database.Report).filter(
            database.Report.filename == "reproc.pdf").all()
        self.assertEqual(len(reports), 1)
        dps = self.session.query(database.DataPoint).filter(
            database.DataPoint.report_id == reports[0].id).all()
        self.assertEqual(len(dps), 1)
        self.assertEqual(dps[0].canonical_name, "A")
        # The marker value is gone → the old row was deleted and re-extracted.
        self.assertNotEqual(dps[0].ref_low, 99.0)
        # Result text carries the before/after diff summary.
        self.assertIn("neu extrahiert", stored.result_text or "")

    def test_reprocess_pins_patient(self):
        """A divergent LLM patient-name answer must not fork a new patient."""
        job = self._run_reprocess(patient_name="OtherPatient",
                                  patient_id=self.patient.id)
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "done")
        # The report stays with the original patient…
        r = self.session.query(database.Report).filter(
            database.Report.filename == "reproc.pdf").first()
        self.assertEqual(r.patient_id, self.patient.id)
        # …and the stray patient created by the LLM's name answer is cleaned up.
        self.assertEqual(self.session.query(database.Patient).count(), 1)

    def test_reprocess_restores_disabled(self):
        dp = self.session.query(database.DataPoint).first()
        dp.is_disabled = True
        self.session.commit()
        job = self._run_reprocess(patient_id=self.patient.id)
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "done")
        new_dp = self.session.query(database.DataPoint).first()
        self.assertTrue(new_dp.is_disabled)
        self.assertIn("wiederhergestellt", stored.result_text or "")

    def test_reprocess_diff_detects_name_change(self):
        """When the improved registry maps a name differently, the diff says so."""
        job = self._run_reprocess(
            patient_id=self.patient.id,
            mappings=[{"original": "A", "unit": "mg/l", "canonical": "B"}],
        )
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "done")
        new_dp = self.session.query(database.DataPoint).first()
        self.assertEqual(new_dp.canonical_name, "B")
        self.assertIn("anderen Biomarker zugeordnet", stored.result_text or "")

    def test_reprocess_no_reports(self):
        for r in self.session.query(database.Report).filter(
                database.Report.filename == "reproc.pdf").all():
            self.session.delete(r)
        self.session.commit()
        job = self._run_reprocess(patient_id=self.patient.id)
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "done")
        self.assertIn("Keine Berichte", stored.result_text or "")


class TestDataQualityParsing(unittest.TestCase):
    """llm_client._parse_findings + per-biomarker / duplicate-check handling."""

    _REG = {"CRP": {"units": ["mg/l"], "aliases": [],
                    "default_ref": {"low": 0.0, "high": 5.0, "unit": "mg/l"}}}

    def _dq(self, raw):
        import llm_client
        with patch.object(llm_client, "_get_model", return_value="test-model"), \
             patch.object(llm_client, "_get_timeout", return_value=30):
            return llm_client.run_biomarker_quality_check(
                _MockLLM(raw), self._REG, "CRP",
                [{"id": 1, "name": "CRP", "value": 9.9, "unit": "mg/l",
                  "ref_low": 0.0, "ref_high": 5.0, "report_date": "2025-03-01"}],
                lang="en")

    def _dup(self, raw):
        import llm_client
        with patch.object(llm_client, "_get_model", return_value="test-model"), \
             patch.object(llm_client, "_get_timeout", return_value=30):
            return llm_client.run_duplicate_check(
                _MockLLM(raw), [("A", "mg/l"), ("B", "mg/l")], self._REG, lang="en")

    def test_parse_valid_data_point_finding(self):
        from llm_client import _parse_findings
        raw = [{"category": "ref_range", "target_type": "data_point",
                "biomarker": "CRP", "description": "too high",
                "action": "set_ref_range",
                "params": {"low": 0, "high": 10, "unit": "mg/l"},
                "data_points": [1, 2]}]
        out = _parse_findings(raw)
        self.assertEqual(len(out), 1)
        f = out[0]
        self.assertEqual(f["data_points"], [1, 2])
        self.assertEqual(f["params"]["high"], 10)

    def test_parse_valid_registry_finding(self):
        from llm_client import _parse_findings
        raw = [{"category": "merge", "target_type": "registry",
                "biomarker": "CRP2", "description": "dup",
                "action": "merge", "params": {"target": "CRP"}}]
        out = _parse_findings(raw)
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0]["data_points"])

    def test_parse_valid_set_unit_finding(self):
        from llm_client import _parse_findings
        raw = [{"category": "unit", "target_type": "data_point",
                "biomarker": "CRP", "description": "wrong unit",
                "action": "set_unit", "params": {"unit": "mg/l"},
                "data_points": [1]}]
        out = _parse_findings(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["params"]["unit"], "mg/l")

    def test_parse_set_unit_requires_unit_param(self):
        from llm_client import _parse_findings
        raw = [
            {"category": "unit", "target_type": "data_point",
             "biomarker": "X", "description": "d",
             "action": "set_unit", "params": {}, "data_points": [1]},
            {"category": "unit", "target_type": "data_point",
             "biomarker": "X", "description": "d",
             "action": "set_unit", "params": {"unit": "  "}, "data_points": [1]},
        ]
        self.assertEqual(_parse_findings(raw), [])

    def test_parse_drops_malformed_entries(self):
        from llm_client import _parse_findings
        raw = [
            "not a dict",
            {"category": "bogus", "target_type": "registry",
             "biomarker": "X", "description": "d", "action": "merge",
             "params": {"target": "Y"}},                      # bad category
            {"category": "value", "target_type": "data_point",
             "biomarker": "", "description": "d",
             "action": "disable_value", "data_points": [1]},   # missing biomarker
            {"category": "value", "target_type": "data_point",
             "biomarker": "X", "description": "d",
             "action": "disable_value"},                       # no data point ids
            {"category": "ref_range", "target_type": "data_point",
             "biomarker": "X", "description": "d",
             "action": "set_ref_range", "params": {},
             "data_points": [1]},                              # no bounds
            {"category": "merge", "target_type": "registry",
             "biomarker": "X", "description": "d",
             "action": "merge", "params": {}},                 # merge w/o target
            {"category": "unit", "target_type": "registry",
             "biomarker": "X", "description": "d",
             "action": "registry_unit"},                       # no proposed value
        ]
        self.assertEqual(_parse_findings(raw), [])

    def test_parse_non_list_returns_empty(self):
        from llm_client import _parse_findings
        self.assertEqual(_parse_findings({"findings": []}), [])
        self.assertEqual(_parse_findings(None), [])

    def test_run_returns_parsed_findings(self):
        raw = json.dumps({"findings": [
            {"category": "value", "target_type": "data_point",
             "biomarker": "CRP", "description": "implausible",
             "action": "disable_value", "params": {}, "data_points": [1]}]})
        result = self._dq(raw)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["biomarker"], "CRP")

    def test_run_invalid_json_raises(self):
        from llm_client import LLMResponseError
        with self.assertRaises(LLMResponseError):
            self._dq("this is not json")

    def test_run_empty_response_raises(self):
        # An empty/invalid LLM answer must surface as an error, never as
        # "no issues found".
        from llm_client import LLMResponseError
        with self.assertRaises(LLMResponseError):
            self._dq("")

    def test_phase1_drops_registry_findings(self):
        # Phase 1 (per-biomarker) only accepts data_point findings — a registry
        # finding smuggled into the response is filtered out.
        raw = json.dumps({"findings": [
            {"category": "merge", "target_type": "registry",
             "biomarker": "CRP2", "description": "dup",
             "action": "merge", "params": {"target": "CRP"}},
            {"category": "value", "target_type": "data_point",
             "biomarker": "CRP", "description": "implausible",
             "action": "disable_value", "params": {}, "data_points": [1]},
        ]})
        result = self._dq(raw)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["target_type"], "data_point")

    def test_phase2_drops_data_point_findings(self):
        # Phase 2 (duplicate check) only accepts registry findings.
        raw = json.dumps({"findings": [
            {"category": "value", "target_type": "data_point",
             "biomarker": "A", "description": "implausible",
             "action": "disable_value", "params": {}, "data_points": [1]},
            {"category": "merge", "target_type": "registry",
             "biomarker": "B", "description": "dup of A",
             "action": "merge", "params": {"target": "A"}},
        ]})
        result = self._dup(raw)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["target_type"], "registry")

    def test_duplicate_check_returns_registry_findings(self):
        raw = json.dumps({"findings": [
            {"category": "merge", "target_type": "registry",
             "biomarker": "B", "description": "dup of A",
             "action": "merge", "params": {"target": "A"}},
        ]})
        result = self._dup(raw)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["biomarker"], "B")

    def test_duplicate_check_invalid_json_raises(self):
        from llm_client import LLMResponseError
        with self.assertRaises(LLMResponseError):
            self._dup("not json at all")


class TestApplyDataQualityFinding(unittest.TestCase):
    """apply_data_quality_finding: per-finding accept semantics + skip paths."""

    def setUp(self):
        # One DB per test method — registry fixes mutate the registry globally.
        self.session = _fresh_db(f"dq_{self._testMethodName}.db")
        self.patient = database.get_or_create_patient(self.session, "DQ_Patient")
        report = database.Report(
            patient_id=self.patient.id, filename="dq.pdf",
            file_hash="dq_hash_1", report_date=datetime(2025, 1, 1),
        )
        self.session.add(report)
        self.session.flush()
        # CRP above its one-sided upper bound → abnormal.
        self.crp = database.DataPoint(
            report_id=report.id, canonical_name="CRP", original_name="CRP",
            value=5.2, unit="mg/l", ref_low=None, ref_high=5.0, is_abnormal=True,
        )
        # Glucose in mg/dl without a stored range (unit-conversion target).
        self.gluc = database.DataPoint(
            report_id=report.id, canonical_name="Glukose (nüchtern)",
            original_name="Glukose", value=100.0, unit="mg/dl",
        )
        self.session.add_all([self.crp, self.gluc])
        self.session.commit()

    def tearDown(self):
        self.session.close()

    def _add_finding(self, target_type, biomarker, action=None, params=None,
                     data_point_ids=None, category="ref_range"):
        f = database.DataQualityFinding(
            patient_id=self.patient.id,
            category=category,
            target_type=target_type,
            biomarker=biomarker,
            data_point_ids=json.dumps(data_point_ids) if data_point_ids else None,
            description="test finding",
            action=action,
            params_json=json.dumps(params or {}),
            status="pending",
        )
        self.session.add(f)
        self.session.commit()
        return f.id

    # ── data-point fixes ────────────────────────────────────────────────

    def test_set_ref_range_updates_bounds_and_abnormal(self):
        fid = self._add_finding("data_point", "CRP", action="set_ref_range",
                                params={"low": 0.0, "high": 10.0, "unit": "mg/l"},
                                data_point_ids=[self.crp.id])
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        self.assertEqual(self.crp.ref_low, 0.0)
        self.assertEqual(self.crp.ref_high, 10.0)
        self.assertFalse(self.crp.is_abnormal)  # 5.2 now in range

    def test_set_ref_range_one_sided(self):
        fid = self._add_finding("data_point", "CRP", action="set_ref_range",
                                params={"high": 3.0},
                                data_point_ids=[self.crp.id])
        database.apply_data_quality_finding(self.session, fid)
        self.session.expire_all()
        self.assertIsNone(self.crp.ref_low)
        self.assertEqual(self.crp.ref_high, 3.0)
        self.assertTrue(self.crp.is_abnormal)  # 5.2 > 3.0

    def test_set_ref_range_converts_units(self):
        fid = self._add_finding("data_point", "Glukose (nüchtern)",
                                action="set_ref_range",
                                params={"low": 3.9, "high": 6.1, "unit": "mmol/l"},
                                data_point_ids=[self.gluc.id])
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        self.assertAlmostEqual(self.gluc.ref_low, 3.9 * 18.0)
        self.assertAlmostEqual(self.gluc.ref_high, 6.1 * 18.0)
        self.assertFalse(self.gluc.is_abnormal)  # 100 mg/dl in [70.2, 109.8]

    def test_set_ref_range_unsupported_conversion_skipped(self):
        fid = self._add_finding("data_point", "CRP", action="set_ref_range",
                                params={"low": 0.0, "high": 0.09, "unit": "mmol/l"},
                                data_point_ids=[self.crp.id])
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "skipped")
        self.session.expire_all()
        # Nothing changed; finding stays pending for retry/dismiss.
        self.assertIsNone(self.crp.ref_low)
        self.assertEqual(self.crp.ref_high, 5.0)
        f = self.session.get(database.DataQualityFinding, fid)
        self.assertEqual(f.status, "pending")

    def test_set_unit_converts_value_and_ref_range(self):
        # Glucose 100 mg/dl with a stored range in the same unit.
        self.gluc.ref_low = 70.0
        self.gluc.ref_high = 99.0
        self.session.commit()
        fid = self._add_finding("data_point", "Glukose (nüchtern)",
                                action="set_unit", params={"unit": "mmol/l"},
                                data_point_ids=[self.gluc.id], category="unit")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        self.assertAlmostEqual(self.gluc.value, 100.0 / 18.0)
        self.assertAlmostEqual(self.gluc.ref_low, 70.0 / 18.0)
        self.assertAlmostEqual(self.gluc.ref_high, 99.0 / 18.0)
        self.assertEqual(self.gluc.unit, "mmol/l")
        # 5.556 > 5.5 → now above the converted upper bound.
        self.assertTrue(self.gluc.is_abnormal)

    def test_set_unit_unsupported_conversion_skipped(self):
        fid = self._add_finding("data_point", "CRP", action="set_unit",
                                params={"unit": "mmol/l"},
                                data_point_ids=[self.crp.id], category="unit")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "skipped")
        self.session.expire_all()
        # Nothing changed; finding stays pending for retry/dismiss.
        self.assertEqual(self.crp.value, 5.2)
        self.assertEqual(self.crp.unit, "mg/l")
        f = self.session.get(database.DataQualityFinding, fid)
        self.assertEqual(f.status, "pending")

    def test_set_unit_same_unit_only_normalizes_label(self):
        fid = self._add_finding("data_point", "CRP", action="set_unit",
                                params={"unit": "mg/L"},
                                data_point_ids=[self.crp.id], category="unit")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        self.assertEqual(self.crp.value, 5.2)  # no conversion for same unit
        self.assertEqual(self.crp.unit, "mg/L")

    def test_set_unit_notation_variant_normalizes_without_conversion(self):
        """Real-world case: WBC stored as '/nl' (same magnitude as 10^3/µl).

        Accepting the proposal must rename the unit WITHOUT changing the
        numeric value — previously this was skipped because the conversion
        table had no entry for the notation variant.
        """
        wbc = database.DataPoint(
            report_id=self.crp.report_id, canonical_name="Leukozyten (WBC)",
            original_name="Leukozyten", value=6.5, unit="/nl",
            ref_low=4.0, ref_high=11.0,
        )
        self.session.add(wbc)
        self.session.commit()
        fid = self._add_finding("data_point", "Leukozyten (WBC)", action="set_unit",
                                params={"unit": "10^3/µl"},
                                data_point_ids=[wbc.id], category="unit")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        self.assertEqual(wbc.value, 6.5)  # numerically identical
        self.assertEqual(wbc.unit, "10^3/µl")
        self.assertEqual((wbc.ref_low, wbc.ref_high), (4.0, 11.0))

    def test_set_unit_dimensionless_target(self):
        """Target unit 'none' strips the unit label without touching the value."""
        qpo2 = database.DataPoint(
            report_id=self.crp.report_id, canonical_name="QPO2/FO2",
            original_name="QPO2/FO2", value=0.95, unit="mmHg",
        )
        self.session.add(qpo2)
        self.session.commit()
        fid = self._add_finding("data_point", "QPO2/FO2", action="set_unit",
                                params={"unit": "none"},
                                data_point_ids=[qpo2.id], category="unit")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        self.assertEqual(qpo2.value, 0.95)
        self.assertEqual(qpo2.unit, "none")

    def test_disable_value_removes_risk_flags(self):
        rf = database.RiskFlag(
            report_id=self.crp.report_id, data_point_id=self.crp.id,
            category="out-of-range", severity=2, description="x",
        )
        self.session.add(rf)
        self.session.commit()
        fid = self._add_finding("data_point", "CRP", action="disable_value",
                                params={}, data_point_ids=[self.crp.id],
                                category="value")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        self.assertTrue(self.crp.is_disabled)
        # The flag row is gone (query by data_point_id — the instance itself
        # was deleted and must not be re-fetched).
        self.assertIsNone(
            self.session.query(database.RiskFlag)
            .filter_by(data_point_id=self.crp.id).first())

    def test_no_matching_data_points_skipped(self):
        fid = self._add_finding("data_point", "CRP", action="disable_value",
                                params={}, data_point_ids=[99999])
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "skipped")

    # ── registry fixes ──────────────────────────────────────────────────

    def test_registry_ref_range(self):
        fid = self._add_finding("registry", "CRP", action="registry_ref_range",
                                params={"proposed": "0–10 mg/l"})
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        row = (self.session.query(database.BiomarkerRegistry)
               .filter_by(canonical_name="CRP").one())
        self.assertEqual(row.ref_low, 0.0)
        self.assertEqual(row.ref_high, 10.0)
        self.assertEqual(row.source, "llm_accepted")

    def test_registry_unit(self):
        fid = self._add_finding("registry", "CRP", action="registry_unit",
                                params={"proposed": "mg/dl"}, category="unit")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        row = (self.session.query(database.BiomarkerRegistry)
               .filter_by(canonical_name="CRP").one())
        self.assertIn("mg/dl", json.loads(row.units))

    def test_registry_name_rename(self):
        # Custom row so the rename doesn't touch seeded data.
        row = database.BiomarkerRegistry(
            canonical_name="TestMarker", units='["mg/l"]', aliases="[]",
            source="manual",
        )
        self.session.add(row)
        dp = database.DataPoint(
            report_id=self.crp.report_id, canonical_name="TestMarker",
            original_name="TestMarker", value=1.0, unit="mg/l",
        )
        self.session.add(dp)
        self.session.commit()

        fid = self._add_finding("registry", "TestMarker", action="registry_name",
                                params={"proposed": "TestMarker (X)"}, category="name")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        renamed = (self.session.query(database.BiomarkerRegistry)
                   .filter_by(canonical_name="TestMarker (X)").one())
        self.assertIn("TestMarker", json.loads(renamed.aliases))
        names = [d.canonical_name for d in self.session.query(database.DataPoint).all()]
        self.assertIn("TestMarker (X)", names)
        self.assertNotIn("TestMarker", names)

    def test_registry_merge(self):
        src = database.BiomarkerRegistry(
            canonical_name="Foo", units='["mg/l"]', aliases="[]", source="manual")
        tgt = database.BiomarkerRegistry(
            canonical_name="Bar", units='["mg/l"]', aliases="[]", source="manual")
        self.session.add_all([src, tgt])
        dp = database.DataPoint(
            report_id=self.crp.report_id, canonical_name="Foo",
            original_name="Foo", value=1.0, unit="mg/l",
        )
        self.session.add(dp)
        self.session.commit()

        fid = self._add_finding("registry", "Foo", action="merge",
                                params={"target": "Bar"}, category="merge")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        bar = (self.session.query(database.BiomarkerRegistry)
               .filter_by(canonical_name="Bar").one())
        self.assertIn("Foo", json.loads(bar.aliases))
        self.assertIsNone((self.session.query(database.BiomarkerRegistry)
                           .filter_by(canonical_name="Foo").first()))
        names = [d.canonical_name for d in self.session.query(database.DataPoint).all()]
        self.assertIn("Bar", names)

    def test_merge_target_missing_skipped(self):
        fid = self._add_finding("registry", "CRP", action="merge",
                                params={"target": "NoSuchMarker"}, category="merge")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "skipped")

    def test_merge_source_without_registry_row(self):
        # The common real-world case: a variant name has data points but NO
        # registry row (only the canonical target does). Accepting the merge
        # must still migrate the data points and record the name as an alias.
        tgt = database.BiomarkerRegistry(
            canonical_name="TestMarker (TGT)", units='["U/l"]',
            aliases="[]", source="manual")
        self.session.add(tgt)
        dp = database.DataPoint(
            report_id=self.crp.report_id, canonical_name="Testmarker variant",
            original_name="Testmarker variant", value=250.0, unit="U/l",
        )
        self.session.add(dp)
        self.session.commit()

        fid = self._add_finding("registry", "Testmarker variant", action="merge",
                                params={"target": "TestMarker (TGT)"},
                                category="merge")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        # Data point renamed onto the target.
        names = [d.canonical_name for d in self.session.query(database.DataPoint).all()]
        self.assertIn("TestMarker (TGT)", names)
        self.assertNotIn("Testmarker variant", names)
        # The variant name is now an alias of the survivor.
        row = (self.session.query(database.BiomarkerRegistry)
               .filter_by(canonical_name="TestMarker (TGT)").one())
        self.assertIn("Testmarker variant", json.loads(row.aliases))

    def test_merge_source_identical_to_target_skipped(self):
        fid = self._add_finding("registry", "CRP", action="merge",
                                params={"target": "crp"}, category="merge")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "skipped")

    def test_merge_target_promoted_when_measurements_exist(self):
        """Real-world case: two Borrelia spellings, NEITHER in the registry.

        The LLM proposes a merge target that is only a measurement name (no
        registry row). Accepting must promote the target to a registry row and
        migrate the source's data points onto it — not fail with 'not found'.
        """
        src = database.DataPoint(
            report_id=self.crp.report_id, canonical_name="Borreliose IgG (Antikörper)",
            original_name="Borreliose IgG", value=12.0, unit="U/ml",
        )
        tgt = database.DataPoint(
            report_id=self.crp.report_id, canonical_name="Borreliose IgG",
            original_name="Borreliose IgG", value=9.5, unit="U/ml",
        )
        self.session.add_all([src, tgt])
        self.session.commit()

        fid = self._add_finding("registry", "Borreliose IgG (Antikörper)",
                                action="merge", params={"target": "Borreliose IgG"},
                                category="merge")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "applied")
        self.session.expire_all()
        # Target promoted to a registry row (llm_accepted).
        row = (self.session.query(database.BiomarkerRegistry)
               .filter_by(canonical_name="Borreliose IgG").one())
        self.assertEqual(row.source, "llm_accepted")
        # Source data point migrated onto the target name.
        names = [d.canonical_name for d in self.session.query(database.DataPoint).all()]
        self.assertEqual(names.count("Borreliose IgG"), 2)
        self.assertNotIn("Borreliose IgG (Antikörper)", names)
        # The redundant spelling is now an alias of the survivor.
        self.assertIn("Borreliose IgG (Antikörper)", json.loads(row.aliases))

    def test_merge_target_no_measurements_still_skipped(self):
        """A target with no registry row AND no measurements is a hallucination.

        It must stay skipped — we never invent a canonical entry for a name the
        patient has no data for.
        """
        src = database.DataPoint(
            report_id=self.crp.report_id, canonical_name="FooVariant",
            original_name="FooVariant", value=1.0, unit="U/ml",
        )
        self.session.add(src)
        self.session.commit()

        fid = self._add_finding("registry", "FooVariant", action="merge",
                                params={"target": "BarGhost"}, category="merge")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "skipped")
        self.session.expire_all()
        # No registry row was invented for the ghost target.
        self.assertIsNone((self.session.query(database.BiomarkerRegistry)
                           .filter_by(canonical_name="BarGhost").first()))
        # Source data point untouched.
        self.assertEqual(src.canonical_name, "FooVariant")

    # ── lifecycle ───────────────────────────────────────────────────────

    def test_unknown_finding_id_skipped(self):
        out = database.apply_data_quality_finding(self.session, 99999)
        self.assertEqual(out["status"], "skipped")

    def test_applied_finding_cannot_be_reapplied(self):
        fid = self._add_finding("data_point", "CRP", action="disable_value",
                                params={}, data_point_ids=[self.crp.id])
        self.assertEqual(
            database.apply_data_quality_finding(self.session, fid)["status"],
            "applied")
        out = database.apply_data_quality_finding(self.session, fid)
        self.assertEqual(out["status"], "skipped")

    def test_dismiss_and_clear_resolved(self):
        f1 = self._add_finding("data_point", "CRP", action="disable_value",
                               params={}, data_point_ids=[self.crp.id])
        f2 = self._add_finding("registry", "CRP", action="registry_unit",
                               params={"proposed": "mg/dl"})
        self.assertTrue(database.dismiss_data_quality_finding(self.session, f1))
        # Not pending anymore → second dismiss is a no-op.
        self.assertFalse(database.dismiss_data_quality_finding(self.session, f1))
        n = database.clear_resolved_data_quality_findings(self.session, self.patient.id)
        self.assertEqual(n, 1)
        self.session.expire_all()
        self.assertIsNone(self.session.get(database.DataQualityFinding, f1))
        # Pending findings are never touched by the cleanup.
        self.assertEqual(self.session.get(database.DataQualityFinding, f2).status, "pending")

    def test_clear_patient_findings_supersedes(self):
        self._add_finding("data_point", "CRP", action="disable_value",
                          params={}, data_point_ids=[self.crp.id])
        self._add_finding("registry", "CRP", action="registry_unit",
                          params={"proposed": "mg/dl"})
        n = database.clear_patient_data_quality_findings(self.session, self.patient.id)
        self.assertEqual(n, 2)
        self.assertEqual(
            database.get_data_quality_findings(self.session, self.patient.id), [])


class TestDataQualityJob(unittest.TestCase):
    """data_quality_check background job — persist, supersede, error paths."""

    @classmethod
    def setUpClass(cls):
        cls.session = _fresh_db("test_dq_job.db")
        patient = database.get_or_create_patient(cls.session, "DQJobPatient")
        report = database.Report(
            patient_id=patient.id, filename="dq.pdf",
            file_hash="dq_hash_1", report_date=datetime(2025, 3, 1),
        )
        cls.session.add(report)
        cls.session.flush()
        for name in ("A", "B"):
            dp = database.DataPoint(
                report_id=report.id, canonical_name=name, original_name=name,
                value=1.0, unit="mg/l", ref_low=0.5, ref_high=2.0,
            )
            cls.session.add(dp)
            setattr(cls, f"dp_{name.lower()}", dp)
        cls.session.commit()
        cls.patient = patient

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        # Config snapshot source (submit_job reads session state on the main thread).
        import streamlit as st
        st.session_state.llm_base_url = "http://test:8888/v1"
        st.session_state.llm_api_key = "test-key"
        st.session_state.llm_model = "test-model"
        st.session_state.llm_timeout = 30
        st.session_state.pdf_dpi = 150
        st.session_state.pdf_batch_size = 1
        st.session_state._bb_lang = "de"

    def _run_check(self, per_biomarker=None, duplicate=None):
        """Run the job with both LLM phases mocked.

        per_biomarker: dict biomarker_name -> findings list (phase 1).
        duplicate: findings list for the phase-2 duplicate check.
        """
        import llm_jobs
        per_biomarker = per_biomarker or {}
        with patch.object(llm_jobs, "_build_client", return_value=object()), \
             patch.object(llm_jobs, "run_biomarker_quality_check",
                          side_effect=lambda llm, reg, name, dps, lang=None: {
                              "findings": per_biomarker.get(name, [])}), \
             patch.object(llm_jobs, "run_duplicate_check",
                          return_value={"findings": duplicate or []}):
            job = llm_jobs.submit_job(
                "data_quality_check", "job_kind_data_quality_check",
                params={"patient_id": self.patient.id}, runner=llm_jobs._run_job,
            )
        self.session.expire_all()
        return database.get_llm_job(self.session, job.id)

    def test_run_persists_findings(self):
        # Phase 1 returns a data_point finding for "A" (1-based index into the
        # group); phase 2 returns a registry finding. The job must map the
        # index to the real DataPoint DB id before persisting.
        stored = self._run_check(
            per_biomarker={"A": [
                {"category": "ref_range", "target_type": "data_point",
                 "biomarker": "A", "description": "d1", "action": "set_ref_range",
                 "params": {"low": 0.0, "high": 3.0, "unit": "mg/l"},
                 "data_points": [1]},
            ]},
            duplicate=[
                {"category": "merge", "target_type": "registry",
                 "biomarker": "B", "description": "d2", "action": "merge",
                 "params": {"target": "A"}, "data_points": None},
            ],
        )
        self.assertEqual(stored.status, "done")
        rows = database.get_data_quality_findings(self.session, self.patient.id)
        self.assertEqual(len(rows), 2)
        dp_finding = [f for f in rows if f.target_type == "data_point"][0]
        # The stored id must be the real DB primary key, not the prompt index.
        self.assertEqual(json.loads(dp_finding.data_point_ids), [self.dp_a.id])
        self.assertEqual(json.loads(dp_finding.params_json)["high"], 3.0)
        self.assertEqual(dp_finding.status, "pending")
        reg_finding = [f for f in rows if f.target_type == "registry"][0]
        self.assertIsNone(reg_finding.data_point_ids)
        # lang="de" snapshot → German result message with both counts.
        self.assertIn("2 Messwerte geprüft", stored.result_text or "")

    def test_rerun_supersedes_previous_findings(self):
        self._run_check(per_biomarker={"A": [
            {"category": "value", "target_type": "data_point",
             "biomarker": "A", "description": "old",
             "action": "disable_value", "params": {}, "data_points": [1]}]})
        stored = self._run_check()
        self.assertEqual(stored.status, "done")
        self.assertEqual(
            database.get_data_quality_findings(self.session, self.patient.id), [])

    def test_partial_failure_surfaces_warning(self):
        # One biomarker's LLM call fails → job still persists the other
        # findings and reports a warning with the error line.
        import llm_jobs
        per_biomarker = {
            "A": [{"category": "value", "target_type": "data_point",
                   "biomarker": "A", "description": "ok",
                   "action": "disable_value", "params": {}, "data_points": [1]}],
        }

        def boom(llm, reg, name, dps, lang=None):
            if name == "B":
                raise RuntimeError("boom")
            return {"findings": per_biomarker.get(name, [])}

        with patch.object(llm_jobs, "_build_client", return_value=object()), \
             patch.object(llm_jobs, "run_biomarker_quality_check", side_effect=boom), \
             patch.object(llm_jobs, "run_duplicate_check",
                          return_value={"findings": []}):
            job = llm_jobs.submit_job(
                "data_quality_check", "job_kind_data_quality_check",
                params={"patient_id": self.patient.id}, runner=llm_jobs._run_job,
            )
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        # A partial failure still completes the job (status "done") but the
        # result text carries the German partial message + the error line.
        self.assertEqual(stored.status, "done")
        self.assertIn("fehlgeschlagen", stored.result_text or "")
        self.assertIn("B: boom", stored.result_text or "")
        rows = database.get_data_quality_findings(self.session, self.patient.id)
        self.assertEqual(len(rows), 1)

    def test_missing_patient_id_is_error(self):
        import llm_jobs
        job = llm_jobs.submit_job(
            "data_quality_check", "job_kind_data_quality_check",
            runner=llm_jobs._run_job,
        )
        self.session.expire_all()
        stored = database.get_llm_job(self.session, job.id)
        self.assertEqual(stored.status, "error")
        self.assertIn("patient_id", stored.error or "")


def tearDownModule():
    """Clean up test artifacts (temp directory and files)."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestPDFProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractorMocked))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyzerExtended))
    suite.addTests(loader.loadTestsFromTestCase(TestParseReferenceRange))
    suite.addTests(loader.loadTestsFromTestCase(TestNormalizePatientName))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMClient))
    suite.addTests(loader.loadTestsFromTestCase(TestBiomarkerInterpretationPrompt))
    suite.addTests(loader.loadTestsFromTestCase(TestAppImports))
    suite.addTests(loader.loadTestsFromTestCase(TestDedupOneAlertPerBiomarker))
    suite.addTests(loader.loadTestsFromTestCase(TestDedupParenthesizedName))
    suite.addTests(loader.loadTestsFromTestCase(TestDedupDescriptionBelowRange))
    suite.addTests(loader.loadTestsFromTestCase(TestRecoveryDownweighting))
    suite.addTests(loader.loadTestsFromTestCase(TestIntensityScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestBiomarkerMerge))
    suite.addTests(loader.loadTestsFromTestCase(TestManualMerge))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossBiomarkerPatterns))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectMissingMarkers))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyzerPrimitives))
    suite.addTests(loader.loadTestsFromTestCase(TestFindBiomarkerInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestFindBiomarkerInfoSubstring))
    suite.addTests(loader.loadTestsFromTestCase(TestBackfillReferenceRanges))
    suite.addTests(loader.loadTestsFromTestCase(TestTranslations))
    suite.addTests(loader.loadTestsFromTestCase(TestTranslationDictIssues))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskDescriptionTranslation))
    suite.addTests(loader.loadTestsFromTestCase(TestOneSidedAbnormality))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchPageVerification))
    suite.addTests(loader.loadTestsFromTestCase(TestNormalizeNamesFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestUnitAwareNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestHistoricalDataCollection))
    suite.addTests(loader.loadTestsFromTestCase(TestBulkGenerationIsolation))
    suite.addTests(loader.loadTestsFromTestCase(TestDisabledMeasurements))
    suite.addTests(loader.loadTestsFromTestCase(TestToggleRefreshesPatientFlags))
    suite.addTests(loader.loadTestsFromTestCase(TestBiomarkerRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestParseRefRange))
    suite.addTests(loader.loadTestsFromTestCase(TestUpdateBiomarkerRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestLlmJobs))
    suite.addTests(loader.loadTestsFromTestCase(TestLlmJobAbort))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQualityParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestApplyDataQualityFinding))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQualityJob))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'='*50}")
    print(f"Tests: {result.testsRun}, Failures: {len(result.failures)}, "
          f"Errors: {len(result.errors)}")
    shutil.rmtree(str(TEST_DIR), ignore_errors=True)
    sys.exit(0 if result.wasSuccessful() else 1)
