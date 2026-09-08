import logging
import re

import streamlit as st

_logger = logging.getLogger(__name__)

# ── Translation dictionaries ──────────────────────────────────────────────────

DE = {
    # App title
    "app_title": "🩸 Blutbild-Buddy",

    # Sidebar
    "sidebar_header": "Aktionen",
    "sidebar_llm_settings": "LLM-Einstellungen",
    "btn_settings": "⚙️ Einstellungen",
    "btn_process_pdfs": "📄 Neue PDFs verarbeiten",
    "spinner_processing": "Verarbeite PDFs...",
    "msg_processed": "Verarbeitet: {file_list} ({n} Risikomarkierungen)",
    "msg_partial_processing": "{count} PDF(s) erfolgreich, {errors} Fehler aufgetreten.",
    "error_processing_failed": "Verarbeitung fehlgeschlagen.",
    "error_processing_failed_llm": "Verarbeitung fehlgeschlagen (LLM-Verbindung): {error}",
    "error_llm_connection_hint": "Prüfe **LLM_BASE_URL** in den Einstellungen. Aktuell: `{base_url}`. In Docker muss dies die **LAN-IP** deines LLM-Servers sein (nicht localhost).",
    "msg_no_new_pdfs": "Keine neuen PDFs gefunden.",
    "msg_operation_in_progress": "Eine andere Operation läuft bereits. Bitte warten Sie, bis sie abgeschlossen ist.",
    "btn_refresh_risks": "🔄 Risikoanalyse aktualisieren",
    "spinner_refreshing": "Risikoanalyse wird durchgeführt...",
    "msg_risks_refreshed": "{n} Risikomarkierungen aktualisiert.",
    "select_patient": "Patient auswählen",
    "patient_option_with_reports": "{name} — {n} Berichte, letzter {date}",
    "rename_patient": "Patient umbenennen",
    "msg_renamed": "Umbenannt: {old} → {new}",
    "msg_rename_error": "Fehler beim Umbenennen.",
    "caption_stats": "Berichte: {n} | Messwerte: {m}",
    "sidebar_pdf_settings": "PDF-Einstellungen",
    "label_upload_pdf": "PDF-Datei hochladen",
    "label_batch_size": "Seiten pro LLM-Aufruf",
    "btn_merge_patients": "🔗 Patienten zusammenführen",
    "btn_rename_biomarker": "Umbenennen & ggf. zur Masterliste hinzufügen",
    "btn_manage_reports": "📁 Berichte verwalten",
    "col_report_filename": "Dateiname",
    "col_report_date": "Berichtsdatum",
    "col_report_status": "Status",
    "col_biomarker_count": "Biomarker",
    "status_processed": "✅ Verarbeitet",
    "status_not_processed": "⏳ Noch nicht verarbeitet",
    "subheader_merge_patients": "Patienten zusammenführen",
    "caption_merge_patients": "Wähle zwei Patienten, um alle Berichte und Datenpunkte zu einem zusammenzuführen.",
    "select_source_patient": "Quell-Patient (wird gelöscht)",
    "select_target_patient": "Ziel-Patient (behält Daten)",
    "btn_confirm_merge": "✅ Zusammenführen",
    "msg_merge_success": "Erfolgreich: {reports} Bericht(e), {dps} Datenpunkt(e) verschoben. Quell-Patient gelöscht.",
    "error_merge_same_patient": "Quell- und Ziel-Patient dürfen nicht identisch sein.",
    "warning_confirm_merge": "⚠️ Achtung: Dies kann nicht rückgängig gemacht werden!",
    "warning_same_name": "Der neue Name ist identisch mit dem alten.",
    "warning_invalid_file": "Ungültiger Dateityp — nur PDFs erlaubt",
    "warning_file_too_large": "Datei zu groß",

    # Empty state
    "info_empty_state": "👋 Noch keine Daten vorhanden. Lege PDFs in `reports_input/` und klicke auf 'Neue PDFs verarbeiten'.",

    # Tab labels
    "tab_reports": "📁 Berichte",
    "tab_patients": "👥 Patienten",
    "tab_llm_settings": "⚙️ LLM-Einstellungen",
    "tab_overview": "📊 Übersicht",
    "tab_detail": "📋 Bericht-Details",
    "tab_trends": "📈 Biomarker-Details",
    "tab_risks": "⚠️ Risiken",
    "tab_mapping": "🔗 Biomarker-Einstellungen",
    "tab_all_values": "📋 Alle Werte",
    "tab_management": "⚙️ Verwaltung",

    # Reports tab
    "subheader_reports_manage": "Berichte verwalten",
    "caption_reports_manage": "Laden Sie PDFs hoch, prüfen Sie den Verarbeitungsstatus und löschen Sie einzelne Berichte.",
    "btn_upload_and_process": "📤 Hochladen & Verarbeiten",
    "btn_process_all": "📄 Alle PDFs verarbeiten",
    "btn_reprocess_pdfs": "♻️ Vorhandene Berichte neu extrahieren",
    "dialog_reprocess_title": "Berichte neu extrahieren",
    "dialog_reprocess_text": "{n} Bericht(e) werden mit dem aktuellen LLM und Register **neu extrahiert**. Bestehende Messwerte, Referenzbereiche und Risikomarkierungen dieser Berichte werden ersetzt. Der Patient bleibt unverändert.",
    "dialog_reprocess_disabled_note": "⚠️ {n} manuell deaktivierter Wert(e) werden nach der Neuextraktion wiederhergestellt (gemessen an Name + Einheit).",
    "dialog_reprocess_missing_note": "⚠️ Für {n} Bericht(e) wurde die PDF-Datei in `reports_input/` nicht gefunden — diese können nicht neu extrahiert werden.",
    "btn_confirm_reprocess": "♻️ Neu extrahieren",
    "msg_reprocess_nothing": "Keine Berichte zum Neuextrahieren gefunden.",
    "msg_reprocess_done": "{n} Bericht(e) neu extrahiert ({risks} Risikomarkierungen).",
    "reprocess_diff_names": "🔗 {n} Messwert(e) wurden einem anderen Biomarker zugeordnet.",
    "reprocess_diff_refs": "📏 {n} Referenzbereich(e) geändert.",
    "reprocess_diff_values": "⚠️ {n} Messwert(e) unterscheiden sich von der vorherigen Extraktion.",
    "reprocess_diff_none": "Keine Änderungen gegenüber der vorherigen Extraktion.",
    "reprocess_disabled_restored": "♻️ {n} deaktivierter Wert(e) wiederhergestellt.",
    "reprocess_patient_pinned": "📌 {n} Bericht(e) wurden ihrem ursprünglichen Patienten zugeordnet (der LLM hatte einen anderen Namen erkannt).",
    "reprocess_missing_file": "PDF-Datei fehlt: {filename}",
    "btn_delete_report": "🗑️ Löschen",
    "btn_confirm_delete": "✅ Bestätigen",
    "msg_report_deleted": "Bericht '{filename}' gelöscht.",
    "warning_confirm_delete_report": "Bericht wirklich löschen?",
    "msg_all_processed": "Keine neuen PDFs zum Verarbeiten gefunden.",
    "msg_processing_started": "Verarbeitung gestartet…",
    "msg_risks_refreshed_all": "{n} Risikomarkierungen für alle Patienten aktualisiert.",
    "btn_backfill_ref_ranges": "🔧 Fehlbende Referenzbereiche auffüllen",
    "spinner_backfilling": "Referenzbereiche werden aufgefüllt…",
    "msg_backfill_complete": "{n} Datenpunkt(e) mit Referenzbereichen aktualisiert.",
    "msg_no_backfill_needed": "Alle Datenpunkte haben bereits Referenzbereiche.",
    "msg_backfill_skipped": "Keine neuen Referenzbereiche aufgefüllt. {unknown} unbekannte Biomarker, {unconvertible} mit nicht umwandelbarer Einheit, {no_range} ohne Standardbereich im Register.",
    "status_pending": "⏳ Ausstehend",

    # Patients tab
    "subheader_patients_manage": "Patienten verwalten",
    "caption_patients_manage": "Liste aller Patienten, Umbenennen und Zusammenführen.",
    "btn_select_for_merge": "Auswählen",
    "btn_merge_selected": "🔗 Ausgewählte zusammenführen",
    "msg_no_patients": "Noch keine Patienten vorhanden.",
    "info_select_patient_first": "Bitte wählen Sie einen Patienten aus, um Daten anzusehen.",

    # LLM Settings tab
    "subheader_llm_settings": "LLM-Einstellungen",
    "caption_llm_settings": "Konfigurieren Sie die Verbindung zu Ihrem LLM-Server (z.B. Unsloth, Ollama, OpenAI).",
    "subheader_pdf_settings_tab": "PDF-Einstellungen",
    "caption_pdf_settings_tab": "Einstellungen für PDF-Verarbeitung und Bildextraktion.",
    "btn_test_connection": "🔌 Verbindung testen",
    "msg_llm_connected": "✅ LLM-Verbindung erfolgreich!",
    "msg_llm_disconnected": "❌ Keine Verbindung:",
    "msg_llm_error": "❌ Fehler:",
    "tooltip_docker_llm": "In Docker muss Base URL die LAN-IP des LLM-Servers sein (nicht localhost).",

    # All Values tab
    "subheader_all_values": "Alle Biomarker-Werte im Zeitverlauf",
    "caption_all_values": "Matrix: Biomarker (Zeilen) × Berichtstermine (Spalten). Leere Zellen bedeuten, dass der Marker in diesem Bericht nicht gemessen wurde. Gelb hinterlegte Werte sind deaktiviert und fließen nicht in Diagramme oder Analysen ein.",
    "btn_export_csv": "📥 Als XLSX exportieren",
    "msg_csv_exported": "XLSX-Datei heruntergeladen: {filename}",
    "col_reference_min": "Min",
    "col_reference_max": "Max",
    "col_all_values_date": "Datum",

    # Overview tab
    "metric_reports": "Berichte insgesamt",
    "metric_last_report": "Letzter Bericht",
    "metric_abnormal": "Abnormale Werte",
    "metric_risk_flags": "Risikomarkierungen",
    "metric_delta_latest": "{n} im letzten Bericht",
    "metric_delta_critical": "{n} kritisch",
    "metric_delta_none": "✓ keine",
    "subheader_health_summary": "🏥 Gesundheitszusammenfassung",
    "btn_regenerate_summary": "Zusammenfassung neu berechnen",
    "last_updated": "Letzte Aktualisierung",
    "msg_generating_summary": "Erstelle Gesundheitszusammenfassung...",
    "msg_summary_regenerated": "Gesundheitszusammenfassung aktualisiert.",
    "summary_disclaimer": "⚠️ KI-generierte Zusammenfassung ersetzt keine ärztliche Beratung.",
    "subheader_panel_types": "Panel-Typen im Verlauf",
    "panel_unknown": "Unbekannt",
    "subheader_last_abnormal": "Letzter Bericht — Auffällige Werte",
    "msg_no_abnormal": "Keine auffälligen Werte im letzten Bericht.",
    "msg_no_summary": "Klicken Sie auf 'Zusammenfassung neu berechnen', um eine KI-generierte Gesundheitszusammenfassung zu erstellen.",
    "msg_no_summary_auto": "Noch keine Zusammenfassung vorhanden. Sie können eine manuell erstellen oder sie wird automatisch bei der nächsten PDF-Verarbeitung generiert.",
    "error_summary_generation": "Fehler bei der Generierung der Zusammenfassung.",
    "error_risk_flags": "Fehler bei der Risikoanalyse.",
    "warning_summary_failed": "Automatische Zusammenfassung konnte nicht erstellt werden.",
    "error_select_patient": "Bitte einen Patienten auswählen.",
    "multiselect_favorites": "Biomarker als Favorit markieren",

    # Charts
    "chart_trend": "Verlauf",
    "chart_ref_range": "Referenzbereich",
    "chart_ref_boundary": "0% (Referenzgrenze)",
    "chart_relative_ylabel": "Abweichung in % von der Referenzgrenze",
    "chart_above_high": "🔴 Biomarker über dem oberen Referenzlimit (letzte 2 Jahre)",
    "chart_below_low": "🟢 Biomarker unter dem unteren Referenzlimit (letzte 2 Jahre)",
    "chart_log_scale_toggle": "Relative Abweichung mit logarithmischer y-Achse",
    "chart_log_scale_caption": "Zeigt nur Messwerte, die die Referenzgrenze verletzen, als Abweichungsgröße in % auf logarithmischer Skala. Nützlich, wenn Abweichungen um Größenordnungen streuen. Die Referenzgrenze (0 %) liegt am unteren Rand der Achse.",

    # Detail tab
    "select_report": "Bericht wählen",
    "info_no_data_points": "Keine Datenpunkte in diesem Bericht.",
    "info_no_reports": "Keine Berichte für diesen Patienten.",
    "caption_disabled_values": "Deaktivierte Werte werden in allen Diagrammen und KI-Analysen ausgeschlossen (z. B. Messungen nach medizinischen Eingriffen).",
    "btn_disable_value": "🚫 Deaktivieren",
    "btn_enable_value": "✅ Aktivieren",
    "msg_value_disabled": "Wert deaktiviert – wird in Diagrammen und KI-Analysen ausgeschlossen.",
    "msg_value_enabled": "Wert wieder aktiviert.",

    # Table headers
    "col_status": "",
    "col_marker": "Marker",
    "col_action": "Aktion",
    "col_original": "Original",
    "col_active": "Aktiv",
    "col_value": "Wert",
    "col_reference": "Referenz",
    "col_date": "Datum",
    "col_panel": "Panel",
    "col_severity": "Schwere",
    "col_category": "Kategorie",
    "col_rank": "#",
    "col_urgency": "Dringlichkeit",
    "col_description": "Beschreibung",
    "col_report": "Bericht",

    # Risk category labels (card header + table)
    "cat_critical_value": "Kritischer Wert",
    # Plain variant for card headers (no severity wording — the severity is
    # shown by the badge/border, so the category stays purely descriptive).
    "cat_critical_value_plain": "Außerhalb Referenzbereich",
    "cat_concerning_trend": "Verdächtiger Trend",
    "cat_value_changed": "Neu abnormal",
    "cat_missing_marker": "Überfällige Kontrolle",
    "cat_cross_biomarker_correlation": "Muster mehrerer Marker",
    "col_unit": "Einheit",
    "col_canonical": "Canonical",

    # Trends tab
    "select_biomarker": "Biomarker wählen",
    "info_no_biomarker_data": "Keine Daten für diesen Biomarker.",
    "time_range_label": "Zeitraum:",

    # Risks tab
    "severity_critical": "🔴 Kritisch",
    "severity_warning": "🟡 Warnung",
    "severity_info": "🔵 Hinweis",
    "urgency_immediate": "🔴 Sofortmaßnahme empfohlen",
    "urgency_long_term": "🟡 Langzeitbeobachtung empfohlen",
    "urgency_info": "🔵 Langzeitbeobachtung empfohlen",
    "subheader_all_flags": "Alle Risikomarkierungen als Tabelle",
    "explanation_label": "Klinische Einordnung",
    "caption_disclaimer": "⚠️ KI-generierte Einordnungen ersetzen keine ärztliche Beratung.",
    "msg_no_risk_flags": "✅ Keine Risikomarkierungen für diesen Patienten.",

    # Risk flag descriptions (analyzer.py)
    "risk_direction_above": "über",
    "risk_direction_below": "unter",
    "risk_above_upper_limit": "über der Obergrenze",
    "risk_below_lower_limit": "unter der Untergrenze",
    "risk_trend_rising": "steigend",
    "risk_trend_falling": "fallend",

    # Biomarker mapping tab
    "subheader_mapping": "Biomarker-Zuordnung überprüfen & anpassen",
    "caption_mapping": "Zeigt alle extrahierten Biomarker mit ihren Originalnamen, Werten und Referenzbereichen. Canonical-Namen können geändert werden.",
    "select_all_reports": "Alle Berichte",
    "select_filter_report": "Bericht filtern",
    "subheader_rename": "Canonical-Namen ändern",
    "caption_rename": "Wähle einen bestehenden Namen und ersetze ihn durch einen anderen. Alle Datenpunkte des Patienten werden aktualisiert.",
    "select_old_name": "Alter Name",
    "select_new_name": "Neuer Name (aus Masterliste)",
    "input_new_name": "Oder neu eingeben",
    "input_new_name_placeholder": "Freier Name...",
    "error_no_name": "Bitte einen neuen Namen auswählen oder eingeben.",
    "msg_added_to_master": "Neuer Name '{name}' zur Masterliste hinzugefügt.",
    "msg_already_in_master": "Name '{name}' ist bereits in der Masterliste.",
    "msg_biomarker_renamed": "{count} Datenpunkt{plural} von '{old}' auf '{new}' umbenannt.",
    "warning_no_data_points": "Keine Datenpunkte mit dem Namen '{old}' gefunden.",
    "msg_biomarker_renamed_global": "'{old}' wurde global in '{new}' umbenannt — Registry und alle Datenpunkte aller Patienten. Der alte Name bleibt als Alias erhalten.",
    "warning_rename_collision": "'{new}' existiert bereits in der Registry. Es wurde nichts geändert — bitte einen anderen Namen wählen.",

    # Manual registry edit section
    "subheader_registry_edit": "🛠️ Registry manuell bearbeiten",
    "caption_registry_edit": "Direkte Änderungen an der aktiven Biomarker-Registry (Name, Einheit oder Referenzbereich) — mit denselben Prüfungen wie bei akzeptierten LLM-Vorschlägen. Einheiten/Referenzbereiche für Namen ohne Registry-Eintrag erstellen automatisch eine neue Zeile.",
    "select_registry_biomarker": "Biomarker",
    "select_registry_field": "Feld",
    "input_registry_value": "Neuer Wert",
    "placeholder_registry_name": "Neuer Canonical-Name",
    "placeholder_registry_unit": "z.B. mg/l",
    "placeholder_registry_ref_range": "z.B. 0–5 mg/l oder < 5.2 mg/dl",
    "btn_apply_registry_edit": "Übernehmen",
    "error_empty_value": "Bitte einen Wert eingeben.",
    "msg_registry_edit_applied": "Registry aktualisiert ({field}: {proposed}).",
    "info_no_data_for_patient": "Keine Datenpunkte für diesen Patienten.",

    # Auto-merge section
    "subheader_auto_merge": "Automatische Gruppierung",
    "caption_auto_merge": "Erkennt automatisch Biomarker-Varianten (z.B. 'HB', 'Hb', 'hämoglobin') und schlägt Zusammenführungen vor.",
    "btn_detect_groups": "Gruppen erkennen",
    "btn_apply_merges": "Zusammenführungen anwenden",
    "btn_cancel_merge": "Abbrechen",
    "msg_detecting_groups": "Erkenne Biomarker-Gruppen...",
    "msg_calling_llm": "LLM wird konsultiert für Canonical-Namen...",
    "msg_no_groups_found": "Keine Biomarker-Gruppen zum Zusammenführen gefunden.",
    "msg_groups_found": "{n} Biomarker-Gruppe{n_plural} zum Zusammenführen gefunden.",
    "msg_merged_success": "{n} Biomarker-Gruppe{n_plural} zusammengeführt, {m} Datenpunkt{plural} migriert.",
    "label_variants": "Varianten",
    "label_suggested_target": "Vorgeschlagener Zielname",
    "label_data_points": "Datenpunkte",
    "label_auto_merge_plan": "Vorschau der Zusammenführungen",

    # Manual merge section
    "subheader_manual_merge": "Biomarker manuell zusammenführen",
    "caption_manual_merge": "Wähle zwei Biomarker-Namen und führe sie zusammen. Alle Datenpunkte werden auf den Zielnamen migriert.",
    "select_source_name": "Quell-Biomarker (wird zusammengeführt)",
    "select_target_name": "Ziel-Biomarker (bleibt erhalten)",
    "btn_merge": "Zusammenführen",
    "warning_merge_same": "Quelle und Ziel sind identisch.",
    "msg_merged": "{count} Datenpunkt{plural} von '{old}' auf '{new}' migriert.",
    # Manual registry edit (biomarker mapping tab)
    "field_name": "Name",
    "field_unit": "Einheit",
    "field_ref_range": "Referenzbereich",
    "msg_suggestion_skipped": "Vorschlag nicht übernommen: {detail}",
    "caption_backfill_from_registry": "Überträgt die Referenzbereiche aus dem Register auf alle Datenpunkte (aller Patienten), denen noch Werte fehlen — z. B. nach einer Registry-Bearbeitung.",

    # Data quality check (registry + actual patient measurements)
    "subheader_data_quality": "🩺 Datenqualität prüfen",
    "caption_data_quality": "Mehrere LLM-Calls prüfen die Registry UND die tatsächlich extrahierten Messwerte dieses Patienten — ein Call pro Biomarker (Referenzbereiche, Einheiten, unplausible Werte) plus ein Duplikat-Check über alle Biomarker. Jeder Befund kann einzeln akzeptiert (automatisch in der DB korrigiert) oder verworfen werden.",
    "btn_data_quality_check": "🩺 Datenqualität prüfen",
    "spinner_data_quality_check": "LLM prüft Biomarker und Messwerte...",
    "msg_data_quality_checked": "{n} Messwerte geprüft, {s} Befund(e) gespeichert.",
    "msg_data_quality_partial": "{n} Biomarker geprüft, {s} Befund(e) gespeichert, {failed} fehlgeschlagen.",
    "label_duplicate_check": "Duplikat-Check",
    "error_data_quality_failed": "Datenqualitätsprüfung fehlgeschlagen",
    "subheader_findings": "📋 Offene Datenqualitäts-Befunde",
    "caption_no_findings": "Keine offenen Befunde. Nach einer Prüfung erscheinen hier konkrete Probleme — sie werden erst korrigiert, wenn du sie akzeptierst.",
    "btn_accept_finding": "✅ Korrigieren",
    "btn_accept_all_findings": "✅ Alle korrigieren",
    "msg_accept_all_findings": "Alle Befunde verarbeitet: {applied} korrigiert, {skipped} übersprungen",
    "btn_dismiss_finding": "🗑 Verwerfen",
    "col_target": "Ziel",
    "cat_ref_range": "Referenzbereich",
    "cat_unit": "Einheit",
    "cat_name": "Name/Duplikat",
    "cat_merge": "Zusammenführung",
    "cat_value": "Unplausibler Wert",
    "cat_missing": "Fehlende Daten",
    "target_registry": "Registry",
    "target_data_point": "Messwerte",
    "col_biomarker": "Biomarker",
    "col_proposal": "Vorschlag",
    "proposal_set_ref_range": "→ Referenzbereich {range} setzen (Wert bleibt unverändert)",
    "proposal_set_unit": "→ Einheit auf \"{new_unit}\" setzen (Wert wird umgerechnet)",
    "proposal_disable_value": "→ Messwert(e) als ungültig markieren und aus Diagrammen/Analysen ausschließen",
    "proposal_merge": "→ in \"{target}\" zusammenführen (Datenpunkte werden umbenannt, Registry-Eintrag verschmolzen)",
    "proposal_rename": "→ in \"{proposed}\" umbenennen (global, alter Name bleibt als Alias)",
    "proposal_registry_unit": "→ Einheit auf \"{proposed}\" setzen",
    "proposal_registry_ref_range": "→ Standard-Referenzbereich auf {range} setzen",
    "proposal_unknown": "(kein anwendbarer Fix hinterlegt)",
    "msg_finding_applied": "Befund korrigiert: {detail}",
    "msg_finding_skipped": "Befund nicht korrigiert: {detail}",
    "btn_clear_resolved_findings": "🧹 Aufgelöste Befunde löschen",
    "msg_findings_cleared": "{n} aufgelöste(r) Befund(e) entfernt.",

    # Biomarker Info (Trends tab)
    "subheader_biomarker_description": "📖 Biomarker-Erklärung",
    "caption_biomarker_description": "Allgemeine Informationen zu diesem Biomarker, was er misst und was abnormale Werte bedeuten können.",
    "btn_generate_description": "🤖 Biomarker-Erklärung generieren",
    "msg_generating_description": "Erstelle Biomarker-Erklärung...",
    "msg_description_generated": "Biomarker-Erklärung aktualisiert.",
    "msg_no_description": "Klicken Sie auf 'Biomarker-Erklärung generieren', um allgemeine Informationen zu diesem Biomarker zu erstellen.",

    "subheader_biomarker_interpretation": "🔬 Medizinische Interpretation",
    "caption_biomarker_interpretation": "Patientenspezifische Einordnung der gemessenen Werte im Verlauf.",
    "btn_generate_interpretation": "🤖 Medizinische Interpretation generieren",
    "msg_generating_interpretation": "Erstelle medizinische Interpretation...",
    "msg_interpretation_generated": "Medizinische Interpretation aktualisiert.",
    "msg_no_interpretation": "Klicken Sie auf 'Medizinische Interpretation generieren', um eine patientenspezifische Auswertung zu erstellen.",

    # Biomarker Info (bulk actions in Reports tab)
    "btn_generate_all_descriptions": "📖 Alle Biomarker-Erklärungen generieren",
    "btn_generate_all_interpretations": "🔬 Alle Biomarker-Interpretationen generieren",
    "msg_generating_all_descriptions": "Erstelle Biomarker-Erklärungen für alle Marker...",
    "msg_generating_all_interpretations": "Erstelle Biomarker-Interpretationen für alle Marker...",
    "msg_all_descriptions_generated": "Alle Biomarker-Erklärungen aktualisiert.",
    "msg_all_interpretations_generated": "Alle Biomarker-Interpretationen aktualisiert.",
    "msg_bulk_partial": "{ok} von {total} Biomarkern aktualisiert, {failed} fehlgeschlagen.",

    # Confirmation dialogs + misc UX
    "btn_cancel": "Abbrechen",
    "dialog_confirm_delete_report_title": "Bericht löschen?",
    "dialog_confirm_delete_report_text": "Der Bericht '{filename}' wird dauerhaft gelöscht — inklusive aller {n} Messwerte und Risikomarkierungen. Dies kann nicht rückgängig gemacht werden.",
    "dialog_confirm_merge_title": "Patienten zusammenführen?",
    "dialog_confirm_merge_text": "'{source}' wird gelöscht; alle Berichte und Datenpunkte werden zu '{target}' verschoben. Dies kann nicht rückgängig gemacht werden.",
    "caption_settings_live": "Änderungen werden sofort übernommen — ein Speichern ist nicht nötig.",
    "info_files_saved": "📄 PDF(s) gespeichert — klicke auf 'Alle PDFs verarbeiten', um die Werte zu extrahieren.",
    "col_patient": "Patient",
    "sheet_all_values": "Alle Werte",

    # LLM Jobs (background tasks)
    "tab_llm_jobs": "🤖 LLM-Jobs",
    "subheader_llm_jobs": "LLM-Jobs",
    "caption_llm_jobs": "Lange LLM-Operationen laufen im Hintergrund — ein Seiten-Refresh oder Schließen verliert keinen Fortschritt. Status und Ergebnisse erscheinen hier.",
    "msg_no_jobs_yet": "Noch keine Jobs. Starte z.B. 'Alle PDFs verarbeiten' im Tab Berichte.",
    "col_job_label": "Job",
    "col_job_status": "Status",
    "col_job_progress": "Fortschritt",
    "col_job_result": "Ergebnis / Fehler",
    "col_job_created": "Erstellt",
    "job_status_pending": "⏳ Wartend",
    "job_status_running": "🔄 Läuft",
    "job_status_done": "✅ Fertig",
    "job_status_error": "❌ Fehler",
    "job_status_interrupted": "⚠️ Unterbrochen",
    "job_kind_process_pdfs": "PDFs verarbeiten",
    "job_kind_reprocess_pdfs": "Berichte neu extrahieren",
    "job_kind_refresh_risks": "Risikoanalyse aktualisieren (alle)",
    "job_kind_regen_summary": "Gesundheitszusammenfassung neu generieren",
    "job_kind_bulk_descriptions": "Biomarker-Erklärungen generieren",
    "job_kind_bulk_interpretations": "Biomarker-Interpretationen generieren",
    "job_kind_backfill": "Referenzbereiche nachtragen",
    "job_kind_data_quality_check": "Datenqualität prüfen",
    "msg_job_submitted": "Job gestartet — Fortschritt im Tab 'LLM-Jobs'.",
    "msg_job_already_running": "Ein Job dieser Art läuft bereits.",
    "job_interrupted_note": "Unterbrochen (App-Neustart).",
    "btn_abort_job": "⏹ Abbrechen",
    "btn_clear_finished_jobs": "✅ Beendete Jobs aufräumen",
    "msg_jobs_cleared": "{n} beendete(r) Job(s) entfernt.",
    "job_aborted_note": "Job vom Benutzer abgebrochen — bereits verarbeitete Ergebnisse bleiben erhalten.",
}

EN = {
    # App title
    "app_title": "🩸 Blutbild-Buddy",

    # Sidebar
    "sidebar_header": "Actions",
    "sidebar_llm_settings": "LLM Settings",
    "btn_settings": "⚙️ Settings",
    "btn_process_pdfs": "📄 Process New PDFs",
    "spinner_processing": "Processing PDFs...",
    "msg_processed": "Processed: {file_list} ({n} risk flags)",
    "msg_partial_processing": "{count} PDF(s) processed successfully, {errors} error(s) occurred.",
    "error_processing_failed": "Processing failed.",
    "error_processing_failed_llm": "Processing failed (LLM connection): {error}",
    "error_llm_connection_hint": "Check **LLM_BASE_URL** in Settings. Current: `{base_url}`. In Docker this must be the **LAN IP** of your LLM server (not localhost).",
    "msg_no_new_pdfs": "No new PDFs found.",
    "msg_operation_in_progress": "Another operation is already running. Please wait for it to complete.",
    "btn_refresh_risks": "🔄 Refresh Risk Analysis",
    "spinner_refreshing": "Running risk analysis...",
    "msg_risks_refreshed": "{n} risk flags updated.",
    "select_patient": "Select Patient",
    "patient_option_with_reports": "{name} — {n} reports, last {date}",
    "rename_patient": "Rename Patient",
    "msg_renamed": "Renamed: {old} → {new}",
    "msg_rename_error": "Error during rename.",
    "caption_stats": "Reports: {n} | Values: {m}",
    "sidebar_pdf_settings": "PDF Settings",
    "label_upload_pdf": "Upload PDF Files",
    "label_batch_size": "Pages per LLM Call",
    "btn_merge_patients": "🔗 Merge Patients",
    "btn_rename_biomarker": "Rename & Add to Master List if Needed",
    "btn_manage_reports": "📁 Manage Reports",
    "col_report_filename": "Filename",
    "col_report_date": "Report Date",
    "col_report_status": "Status",
    "col_biomarker_count": "Biomarkers",
    "status_processed": "✅ Processed",
    "status_not_processed": "⏳ Not processed yet",
    "subheader_merge_patients": "Merge Patients",
    "caption_merge_patients": "Select two patients to merge all their reports and data points into one.",
    "select_source_patient": "Source Patient (will be deleted)",
    "select_target_patient": "Target Patient (keeps data)",
    "btn_confirm_merge": "✅ Merge",
    "msg_merge_success": "Success: {reports} report(s), {dps} data point(s) moved. Source patient deleted.",
    "error_merge_same_patient": "Source and target patients must be different.",
    "warning_confirm_merge": "⚠️ Warning: This action cannot be undone!",
    "warning_same_name": "The new name is identical to the old one.",
    "warning_invalid_file": "Invalid file type — only PDFs allowed",
    "warning_file_too_large": "File too large",

    # Empty state
    "info_empty_state": "👋 No data available yet. Place PDFs in `reports_input/` and click 'Process New PDFs'.",

    # Tab labels
    "tab_reports": "📁 Reports",
    "tab_patients": "👥 Patients",
    "tab_llm_settings": "⚙️ LLM Settings",
    "tab_overview": "📊 Overview",
    "tab_detail": "📋 Report Details",
    "tab_trends": "📈 Biomarker Details",
    "tab_risks": "⚠️ Risks",
    "tab_mapping": "🔗 Biomarker Settings",
    "tab_all_values": "📋 All Values",
    "tab_management": "⚙️ Management",

    # Reports tab
    "subheader_reports_manage": "Manage Reports",
    "caption_reports_manage": "Upload PDFs, check processing status, and delete individual reports.",
    "btn_upload_and_process": "📤 Upload & Process",
    "btn_process_all": "📄 Process All PDFs",
    "btn_reprocess_pdfs": "♻️ Re-extract Existing Reports",
    "dialog_reprocess_title": "Re-extract reports",
    "dialog_reprocess_text": "{n} report(s) will be **re-extracted** with the current LLM and registry. Existing measurements, reference ranges and risk flags of these reports are replaced. Patients stay unchanged.",
    "dialog_reprocess_disabled_note": "⚠️ {n} manually disabled value(s) will be restored after re-extraction (matched by name + unit).",
    "dialog_reprocess_missing_note": "⚠️ The PDF file for {n} report(s) was not found in `reports_input/` — they cannot be re-extracted.",
    "btn_confirm_reprocess": "♻️ Re-extract",
    "msg_reprocess_nothing": "No reports found to re-extract.",
    "msg_reprocess_done": "{n} report(s) re-extracted ({risks} risk flags).",
    "reprocess_diff_names": "🔗 {n} measurement(s) mapped to a different biomarker.",
    "reprocess_diff_refs": "📏 {n} reference range(s) changed.",
    "reprocess_diff_values": "⚠️ {n} value(s) differ from the previous extraction.",
    "reprocess_diff_none": "No changes compared to the previous extraction.",
    "reprocess_disabled_restored": "♻️ {n} disabled value(s) restored.",
    "reprocess_patient_pinned": "📌 {n} report(s) re-attributed to their original patient (the LLM recognized a different name).",
    "reprocess_missing_file": "PDF file missing: {filename}",
    "btn_delete_report": "🗑️ Delete",
    "btn_confirm_delete": "✅ Confirm",
    "msg_report_deleted": "Report '{filename}' deleted.",
    "warning_confirm_delete_report": "Really delete this report?",
    "msg_all_processed": "No new PDFs found to process.",
    "msg_processing_started": "Processing started…",
    "msg_risks_refreshed_all": "{n} risk flags updated for all patients.",
    "btn_backfill_ref_ranges": "🔧 Backfill Missing Reference Ranges",
    "spinner_backfilling": "Backfilling reference ranges…",
    "msg_backfill_complete": "{n} data point(s) updated with reference ranges.",
    "msg_no_backfill_needed": "All data points already have reference ranges.",
    "msg_backfill_skipped": "No new reference ranges backfilled. {unknown} unknown biomarkers, {unconvertible} with unconvertible units, {no_range} without a default range in the registry.",
    "status_pending": "⏳ Pending",

    # Patients tab
    "subheader_patients_manage": "Manage Patients",
    "caption_patients_manage": "List of all patients, rename and merge operations.",
    "btn_select_for_merge": "Select",
    "btn_merge_selected": "🔗 Merge Selected",
    "msg_no_patients": "No patients available yet.",
    "info_select_patient_first": "Please select a patient to view data.",

    # LLM Settings tab
    "subheader_llm_settings": "LLM Settings",
    "caption_llm_settings": "Configure the connection to your LLM server (e.g., Unsloth, Ollama, OpenAI).",
    "subheader_pdf_settings_tab": "PDF Settings",
    "caption_pdf_settings_tab": "Settings for PDF processing and image extraction.",
    "btn_test_connection": "🔌 Test Connection",
    "msg_llm_connected": "✅ LLM connection successful!",
    "msg_llm_disconnected": "❌ No connection:",
    "msg_llm_error": "❌ Error:",
    "tooltip_docker_llm": "In Docker, Base URL must be the LAN IP of the LLM server (not localhost).",

    # All Values tab
    "subheader_all_values": "All Biomarker Values Over Time",
    "caption_all_values": "Matrix: biomarkers (rows) × report dates (columns). Empty cells mean the marker was not measured in that report. Yellow-highlighted values are disabled and excluded from charts and analyses.",
    "btn_export_csv": "📥 Export as XLSX",
    "msg_csv_exported": "XLSX file downloaded: {filename}",
    "col_reference_min": "Min",
    "col_reference_max": "Max",
    "col_all_values_date": "Date",
    # Overview tab
    "metric_reports": "Total Reports",
    "metric_last_report": "Last Report",
    "metric_abnormal": "Abnormal Values",
    "metric_risk_flags": "Risk Flags",
    "metric_delta_latest": "{n} in latest report",
    "metric_delta_critical": "{n} critical",
    "metric_delta_none": "✓ none",
    "subheader_health_summary": "🏥 Health Summary",
    "btn_regenerate_summary": "Recalculate Summary",
    "last_updated": "Last updated",
    "msg_generating_summary": "Generating health summary...",
    "msg_summary_regenerated": "Health summary updated.",
    "summary_disclaimer": "⚠️ AI-generated summary does not replace medical advice.",
    "subheader_panel_types": "Panel Types Over Time",
    "panel_unknown": "Unknown",
    "subheader_last_abnormal": "Last Report — Abnormal Values",
    "msg_no_abnormal": "No abnormal values in the last report.",
    "msg_no_summary": "Click 'Recalculate Summary' to generate an AI-powered health summary.",
    "msg_no_summary_auto": "No summary yet. You can generate one manually or it will be created automatically with the next PDF processing.",
    "error_summary_generation": "Error generating the summary.",
    "error_risk_flags": "Error during risk analysis.",
    "warning_summary_failed": "Could not generate automatic summary.",
    "error_select_patient": "Please select a patient first.",
    "multiselect_favorites": "Mark Biomarkers as Favorites",

    # Charts
    "chart_trend": "Trend",
    "chart_ref_range": "Reference Range",
    "chart_ref_boundary": "0% (Reference Boundary)",
    "chart_relative_ylabel": "Deviation in % from Reference Boundary",
    "chart_above_high": "🔴 Biomarkers above upper reference limit (last 2 years)",
    "chart_below_low": "🟢 Biomarkers below lower reference limit (last 2 years)",
    "chart_log_scale_toggle": "Relative deviation with logarithmic y-axis",
    "chart_log_scale_caption": "Shows only measurements that cross the reference boundary, as deviation magnitude in % on a log scale. Useful when deviations span orders of magnitude. The reference boundary (0%) sits at the bottom edge of the axis.",

    # Detail tab
    "select_report": "Select Report",
    "info_no_data_points": "No data points in this report.",
    "info_no_reports": "No reports for this patient.",
    "caption_disabled_values": "Disabled values are excluded from all charts and AI analyses (e.g. measurements after medical procedures).",
    "btn_disable_value": "🚫 Disable",
    "btn_enable_value": "✅ Enable",
    "msg_value_disabled": "Value disabled – excluded from charts and AI analyses.",
    "msg_value_enabled": "Value re-enabled.",

    # Table headers
    "col_status": "",
    "col_marker": "Marker",
    "col_action": "Action",
    "col_original": "Original",
    "col_active": "Active",
    "col_value": "Value",
    "col_reference": "Reference",
    "col_date": "Date",
    "col_panel": "Panel",
    "col_severity": "Severity",
    "col_category": "Category",
    "col_rank": "#",
    "col_urgency": "Urgency",
    "col_description": "Description",
    "col_report": "Report",

    # Risk category labels (card header + table)
    "cat_critical_value": "Critical Value",
    # Plain variant for card headers (no severity wording — the severity is
    # shown by the badge/border, so the category stays purely descriptive).
    "cat_critical_value_plain": "Out of Reference Range",
    "cat_concerning_trend": "Concerning Trend",
    "cat_value_changed": "Newly Abnormal",
    "cat_missing_marker": "Overdue Check-up",
    "cat_cross_biomarker_correlation": "Multi-Marker Pattern",
    "col_unit": "Unit",
    "col_canonical": "Canonical",

    # Trends tab
    "select_biomarker": "Select Biomarker",
    "info_no_biomarker_data": "No data for this biomarker.",
    "time_range_label": "Time range:",

    # Risks tab
    "severity_critical": "🔴 Critical",
    "severity_warning": "🟡 Warning",
    "severity_info": "🔵 Info",
    "urgency_immediate": "🔴 Immediate Action Recommended",
    "urgency_long_term": "🟡 Long-Term Observation Recommended",
    "urgency_info": "🔵 Long-Term Observation Recommended",
    "subheader_all_flags": "All Risk Flags as Table",
    "explanation_label": "Clinical Interpretation",
    "caption_disclaimer": "⚠️ AI-generated interpretations do not replace medical advice.",
    "msg_no_risk_flags": "✅ No risk flags for this patient.",

    # Risk flag descriptions (analyzer.py)
    "risk_direction_above": "above",
    "risk_direction_below": "below",
    "risk_above_upper_limit": "above the upper limit",
    "risk_below_lower_limit": "below the lower limit",
    "risk_trend_rising": "rising",
    "risk_trend_falling": "falling",

    # Biomarker mapping tab
    "subheader_mapping": "Review & Adjust Biomarker Mapping",
    "caption_mapping": "Shows all extracted biomarkers with their original names, values, and reference ranges. Canonical names can be changed.",
    "select_all_reports": "All Reports",
    "select_filter_report": "Filter by Report",
    "subheader_rename": "Rename Canonical Names",
    "caption_rename": "Select an existing name and replace it with another. All data points for the patient will be updated.",
    "select_old_name": "Old Name",
    "select_new_name": "New Name (from Master List)",
    "input_new_name": "Or enter new",
    "input_new_name_placeholder": "Free text...",
    "error_no_name": "Please select or enter a new name.",
    "msg_added_to_master": "New name '{name}' added to master list.",
    "msg_already_in_master": "Name '{name}' is already in the master list.",
    "msg_biomarker_renamed": "{count} data point{plural} renamed from '{old}' to '{new}'.",
    "warning_no_data_points": "No data points found with name '{old}'.",
    "msg_biomarker_renamed_global": "'{old}' renamed globally to '{new}' — registry and all data points of all patients. The old name is kept as an alias.",
    "warning_rename_collision": "'{new}' already exists in the registry. Nothing was changed — please pick a different name.",

    # Manual registry edit section
    "subheader_registry_edit": "🛠️ Edit Registry Manually",
    "caption_registry_edit": "Direct changes to the active biomarker registry (name, unit or reference range) — with the same guards as accepted LLM proposals. Setting a unit/reference range for a name without a registry entry creates a new row automatically.",
    "select_registry_biomarker": "Biomarker",
    "select_registry_field": "Field",
    "input_registry_value": "New Value",
    "placeholder_registry_name": "New canonical name",
    "placeholder_registry_unit": "e.g. mg/l",
    "placeholder_registry_ref_range": "e.g. 0–5 mg/l or < 5.2 mg/dl",
    "btn_apply_registry_edit": "Apply",
    "error_empty_value": "Please enter a value.",
    "msg_registry_edit_applied": "Registry updated ({field}: {proposed}).",
    "info_no_data_for_patient": "No data points for this patient.",

    # Auto-merge section
    "subheader_auto_merge": "Auto-Grouping",
    "caption_auto_merge": "Automatically detects biomarker variants (e.g. 'HB', 'Hb', 'hemoglobin') and suggests merges.",
    "btn_detect_groups": "Detect Groups",
    "btn_apply_merges": "Apply Merges",
    "btn_cancel_merge": "Cancel",
    "msg_detecting_groups": "Detecting biomarker groups...",
    "msg_calling_llm": "Consulting LLM for canonical names...",
    "msg_no_groups_found": "No biomarker groups found for merging.",
    "msg_groups_found": "{n} biomarker group{n_plural} found for merging.",
    "msg_merged_success": "{n} biomarker group{n_plural} merged, {m} data point{plural} migrated.",
    "label_variants": "Variants",
    "label_suggested_target": "Suggested Target",
    "label_data_points": "Data Points",
    "label_auto_merge_plan": "Merge Preview",

    # Manual merge section
    "subheader_manual_merge": "Manually Merge Biomarkers",
    "caption_manual_merge": "Select two biomarker names and merge them. All data points will be migrated to the target name.",
    "select_source_name": "Source Biomarker (to merge)",
    "select_target_name": "Target Biomarker (to keep)",
    "btn_merge": "Merge",
    "warning_merge_same": "Source and target are identical.",
    "msg_merged": "{count} data point{plural} migrated from '{old}' to '{new}'.",
    # Manual registry edit (biomarker mapping tab)
    "field_name": "Name",
    "field_unit": "Unit",
    "field_ref_range": "Reference Range",
    "msg_suggestion_skipped": "Proposal not applied: {detail}",
    "caption_backfill_from_registry": "Copies reference ranges from the registry onto all data points (all patients) that are still missing them — e.g. after editing the registry.",

    # Data quality check (registry + actual patient measurements)
    "subheader_data_quality": "🩺 Data Quality Check",
    "caption_data_quality": "Several LLM calls check the registry AND this patient's actually extracted measurements — one call per biomarker (reference ranges, units, implausible values) plus a duplicate check over all biomarkers. Each finding can be accepted (auto-corrected in the DB) or dismissed individually.",
    "btn_data_quality_check": "🩺 Check Data Quality",
    "spinner_data_quality_check": "LLM checking biomarkers and measurements...",
    "msg_data_quality_checked": "{n} measurements checked, {s} finding(s) saved.",
    "msg_data_quality_partial": "{n} biomarkers checked, {s} finding(s) saved, {failed} failed.",
    "label_duplicate_check": "Duplicate check",
    "error_data_quality_failed": "Data quality check failed",
    "subheader_findings": "📋 Open Data Quality Findings",
    "caption_no_findings": "No open findings. After a check, concrete problems appear here — they are only corrected when you accept them.",
    "btn_accept_finding": "✅ Fix",
    "btn_accept_all_findings": "✅ Fix all",
    "msg_accept_all_findings": "All findings processed: {applied} corrected, {skipped} skipped",
    "btn_dismiss_finding": "🗑 Dismiss",
    "col_target": "Target",
    "cat_ref_range": "Reference Range",
    "cat_unit": "Unit",
    "cat_name": "Name/Duplicate",
    "cat_merge": "Merge",
    "cat_value": "Implausible Value",
    "cat_missing": "Missing Data",
    "target_registry": "Registry",
    "target_data_point": "Measurements",
    "col_biomarker": "Biomarker",
    "col_proposal": "Proposed Fix",
    "proposal_set_ref_range": "→ set reference range to {range} (measured value unchanged)",
    "proposal_set_unit": "→ set unit to \"{new_unit}\" (value is converted)",
    "proposal_disable_value": "→ mark measurement(s) as invalid and exclude from charts/analysis",
    "proposal_merge": "→ merge into \"{target}\" (data points renamed, registry entries combined)",
    "proposal_rename": "→ rename to \"{proposed}\" (global, old name kept as alias)",
    "proposal_registry_unit": "→ set unit to \"{proposed}\"",
    "proposal_registry_ref_range": "→ set default reference range to {range}",
    "proposal_unknown": "(no applicable fix stored)",
    "msg_finding_applied": "Finding corrected: {detail}",
    "msg_finding_skipped": "Finding not corrected: {detail}",
    "btn_clear_resolved_findings": "🧹 Clear resolved findings",
    "msg_findings_cleared": "{n} resolved finding(s) removed.",

    # Biomarker Info (Trends tab)
    "subheader_biomarker_description": "📖 Biomarker Explanation",
    "caption_biomarker_description": "General information about this biomarker, what it measures, and what abnormal values may imply.",
    "btn_generate_description": "🤖 Generate Biomarker Explanation",
    "msg_generating_description": "Generating biomarker explanation...",
    "msg_description_generated": "Biomarker explanation updated.",
    "msg_no_description": "Click 'Generate Biomarker Explanation' to create general information about this biomarker.",

    "subheader_biomarker_interpretation": "🔬 Medical Interpretation",
    "caption_biomarker_interpretation": "Patient-specific assessment of measured values over time.",
    "btn_generate_interpretation": "🤖 Generate Medical Interpretation",
    "msg_generating_interpretation": "Generating medical interpretation...",
    "msg_interpretation_generated": "Medical interpretation updated.",
    "msg_no_interpretation": "Click 'Generate Medical Interpretation' to create a patient-specific evaluation.",

    # Biomarker Info (bulk actions in Reports tab)
    "btn_generate_all_descriptions": "📖 Generate All Biomarker Explanations",
    "btn_generate_all_interpretations": "🔬 Generate All Biomarker Interpretations",
    "msg_generating_all_descriptions": "Generating biomarker explanations for all markers...",
    "msg_generating_all_interpretations": "Generating biomarker interpretations for all markers...",
    "msg_all_descriptions_generated": "All biomarker explanations updated.",
    "msg_all_interpretations_generated": "All biomarker interpretations updated.",
    "msg_bulk_partial": "{ok} of {total} biomarkers updated, {failed} failed.",

    # Confirmation dialogs + misc UX
    "btn_cancel": "Cancel",
    "dialog_confirm_delete_report_title": "Delete report?",
    "dialog_confirm_delete_report_text": "The report '{filename}' will be permanently deleted — including all {n} measurements and risk flags. This cannot be undone.",
    "dialog_confirm_merge_title": "Merge patients?",
    "dialog_confirm_merge_text": "'{source}' will be deleted; all reports and data points will be moved to '{target}'. This cannot be undone.",
    "caption_settings_live": "Changes are applied immediately — no saving needed.",
    "info_files_saved": "📄 PDF(s) saved — click 'Process All PDFs' to extract the values.",
    "col_patient": "Patient",
    "sheet_all_values": "All Values",

    # LLM Jobs (background tasks)
    "tab_llm_jobs": "🤖 LLM Jobs",
    "subheader_llm_jobs": "LLM Jobs",
    "caption_llm_jobs": "Long-running LLM operations run in the background — refreshing or closing the page never loses progress. Status and results appear here.",
    "msg_no_jobs_yet": "No jobs yet. Start e.g. 'Process All PDFs' in the Reports tab.",
    "col_job_label": "Job",
    "col_job_status": "Status",
    "col_job_progress": "Progress",
    "col_job_result": "Result / Error",
    "col_job_created": "Created",
    "job_status_pending": "⏳ Pending",
    "job_status_running": "🔄 Running",
    "job_status_done": "✅ Done",
    "job_status_error": "❌ Error",
    "job_status_interrupted": "⚠️ Interrupted",
    "job_kind_process_pdfs": "Process PDFs",
    "job_kind_reprocess_pdfs": "Re-extract reports",
    "job_kind_refresh_risks": "Refresh risk analysis (all)",
    "job_kind_regen_summary": "Regenerate health summary",
    "job_kind_bulk_descriptions": "Generate biomarker explanations",
    "job_kind_bulk_interpretations": "Generate biomarker interpretations",
    "job_kind_backfill": "Backfill reference ranges",
    "job_kind_data_quality_check": "Check data quality",
    "msg_job_submitted": "Job started — progress in the 'LLM Jobs' tab.",
    "msg_job_already_running": "A job of this kind is already running.",
    "job_interrupted_note": "Interrupted (app restart).",
    "btn_abort_job": "⏹ Abort",
    "btn_clear_finished_jobs": "✅ Clear finished jobs",
    "msg_jobs_cleared": "{n} finished job(s) removed.",
    "job_aborted_note": "Job aborted by user — results processed so far are kept.",
}

# ── Language helper ───────────────────────────────────────────────────────────

_LANG_KEY = "_bb_lang"


def _get_lang() -> str:
    """Return the current language code ('de' or 'en')."""
    if _LANG_KEY not in st.session_state:
        st.session_state[_LANG_KEY] = "de"
    return st.session_state[_LANG_KEY]


def set_lang(lang: str):
    """Set the app language. lang must be 'de' or 'en'."""
    if lang in ("de", "en"):
        st.session_state[_LANG_KEY] = lang


def t(key: str, lang: str | None = None, **kwargs) -> str:
    """Look up a translation string and optionally format it with kwargs.

    Falls back to the English string, then to the key itself. A format error
    (e.g. a template placeholder missing from kwargs) is logged and degrades
    to the raw template instead of being silently swallowed.

    ``lang`` overrides the app language — background job workers must pass it
    explicitly because they run outside the Streamlit script context where
    session state is not safe to touch.
    """
    if lang is None:
        lang = _get_lang()
    d = DE if lang == "de" else EN
    template = d.get(key, EN.get(key, key))
    if kwargs:
        try:
            template = template.format(**kwargs)
        except (KeyError, IndexError, ValueError) as e:
            _logger.warning("t(%r): format failed (%s); returning raw template", key, e)
    return template


# ── Risk description translation ──────────────────────────────────────────────
# Risk flag descriptions are stored in German in the DB. When the UI language
# is English we translate them on the fly using regex-based substitution.

_DE_TO_EN_PATTERNS = [
    # Limit-relative deviations (new wording) — must come before the generic
    # "% über dem" / "% unter dem" fallbacks below.
    (r"\b(\d+(?:,\d+)?)%\s+über\s+der\s+Obergrenze\b", r"\1% above the upper limit"),
    (r"\b(\d+(?:,\d+)?)%\s+unter\s+der\s+Untergrenze\b", r"\1% below the lower limit"),
    (r"\b(\d+(?:,\d+)?)%\s+über\s+dem\s+Referenzbereich\b", r"\1% above the reference range"),
    (r"\b(\d+(?:,\d+)?)%\s+unter\s+dem\s+Referenzbereich\b", r"\1% below the reference range"),
    (r"\b(\d+(?:,\d+)?)%\s+über\s+Limit\b", r"\1% above limit"),
    (r"\b(\d+(?:,\d+)?)%\s+unter\s+Limit\b", r"\1% below limit"),
    (r"\b(\d+(?:,\d+)?)%\s+abweichend\b", r"\1% deviating"),
    (r"\b(\d+(?:,\d+)?)%\s+über\s+dem\b", r"\1% above"),
    (r"\b(\d+(?:,\d+)?)%\s+unter\s+dem\b", r"\1% below"),
    (r"liegt\s+(\d+(?:,\d+)?)%\s+über\s+dem\s+Referenzbereich", r"is \1% above the reference range"),
    (r"liegt\s+(\d+(?:,\d+)?)%\s+unter\s+dem\s+Referenzbereich", r"is \1% below the reference range"),
    # ── Cross-biomarker pattern sentences (full-sentence, most specific) ──
    # Whole clinical sentences are translated as a unit so they read naturally
    # instead of word-by-word. These must precede the generic fragments below.
    (re.escape("Gleichzeitige Erhöhung von CK und Troponin T deutet auf eine Muskelschädigung mit möglicher Beteiligung des Herzmuskels hin. Klinische Korrelation empfohlen."), "Concurrent elevation of CK and Troponin T suggests muscle damage with possible involvement of the heart muscle. Clinical correlation recommended."),
    (re.escape("Isolierte CK-Erhöhung bei normalem Troponin T spricht eher für skelettmuskuläre als für kardiale Ursache."), "Isolated CK elevation with normal Troponin T points more to a skeletal muscle than a cardiac cause."),
    (re.escape("Gleichzeitige Erhöhung von CK-MB und Troponin T ist ein starkes Indiz für eine akute Herzmuskelverletzung. Sofortige klinische Bewertung empfohlen."), "Concurrent elevation of CK-MB and Troponin T is a strong indicator of acute myocardial injury. Immediate clinical assessment recommended."),
    (re.escape("Hohes LDL bei niedrigem HDL erhöht das kardiovaskuläre Risiko signifikant. Lebensstilmaßnahmen und ggf. medikamentöse Therapie überprüfen."), "High LDL with low HDL significantly increases cardiovascular risk. Review lifestyle measures and, if necessary, pharmacological therapy."),
    (re.escape("Gleichzeitige Abweichung von TSH und freien Schilddrüsenhormonen spricht für eine klinisch relevante Schilddrüsenfunktionsstörung. Klinische Bewertung empfohlen."), "Concurrent deviation of TSH and free thyroid hormones points to a clinically relevant thyroid dysfunction. Clinical assessment recommended."),
    (re.escape("Gleichzeitig erhöhte Entzündungsmarker (CRP und BSG) deuten auf eine systemische Entzündungsreaktion hin. Ursache abklären."), "Simultaneously elevated inflammatory markers (CRP and ESR) indicate a systemic inflammatory response. Determine the cause."),
    (re.escape("Gleichzeitig erhöhtes Kreatinin und vermindertes eGFR bestätigen eine eingeschränkte Nierenfunktion. Staging und Verlaufskontrolle empfohlen."), "Simultaneously elevated creatinine and reduced eGFR confirm impaired renal function. Staging and follow-up monitoring recommended."),
    (re.escape("Gleichzeitig erhöhte Glukose und HbA1c bestätigen eine gestörte Glukoseregulation bzw. Diabetes. Therapieanpassung überprüfen."), "Simultaneously elevated glucose and HbA1c confirm impaired glucose regulation or diabetes. Review therapy adjustment."),
    (re.escape("Gleichzeitig abnormale Eisenwerte und Transferrinsättigung können auf einen Eisenmangel oder eine Eisenüberladung hinweisen. Klinische Einordnung empfohlen."), "Concurrently abnormal iron values and transferrin saturation may indicate iron deficiency or iron overload. Clinical interpretation recommended."),
    # ── Missing-marker / cross-biomarker fragments ──
    (r"nicht gemessen", r"not measured"),
    (r"wurde vor (\d+) Tagen gemessen", r"was measured \1 days ago"),
    (r"\(letzter Wert: ([^)]+)\)", r"(last value: \1)"),
    (r"Überprüfung empfohlen\.", r"Review recommended."),
    # ── Dedup / critical-value fragments ──
    (r"außerhalb des Referenzbereichs", r"outside the reference range"),
    (r"von\s+([\d.,]+)\s+auf\s+([\d.,]+)", r"from \1 to \2"),
    # Bare "liegt" (critical_value: "... liegt 56% below the lower limit ...").
    # Placed last so the specific "liegt X% ..." patterns above win first.
    (r"\bliegt\b", r"is"),
    (r"zeigt\s+(steigend|fallend)", r"shows \1"),
    (r"steigend", r"rising"),
    (r"fallend", r"falling"),
    # value_changed: "X ist im aktuellen Bericht abnormal ..." — translate the
    # whole verb phrase so no bare "ist" is left behind. Must precede the
    # generic "im aktuellen Bericht abnormal" fallback below.
    (r"ist\s+im\s+aktuellen\s+Bericht\s+abnormal", r"is abnormal in the current report"),
    (r"im\s+aktuellen\s+Bericht\s+abnormal", r"abnormal in the current report"),
    (r"war\s+in\s+früheren\s+Berichten\s+jedoch\s+im\s+Normbereich", r"but was previously within the normal range"),
    (r"Referenz:", r"Reference:"),
    (r"Bisheriger\s+Extremwert:", r"Previous extreme:"),
    (r"\(Referenzbereich\s+([^)]+)\)", r"(Reference range \1)"),
    (r"im\s+Bereich", r"in range"),
    (r"\blag\s+(\d+(?:,\d+)?)%\s+über\s+dem\s+Referenzbereich\s*\(([^)]+)\)\.", r"\1% above the reference range (\2)."),
    (r"\blag\s+(\d+(?:,\d+)?)%\s+unter\s+dem\s+Referenzbereich\s*\(([^)]+)\)\.", r"\1% below the reference range (\2)."),
    (r"\bim\s+Referenzbereich\s*\(([^)]+)\)\.", r"in the reference range (\1)."),
    (r"\blag\s+(\d+(?:,\d+)?)%\s+über\s+dem\b", r"\1% above"),
    (r"\blag\s+(\d+(?:,\d+)?)%\s+unter\s+dem\b", r"\1% below"),
]


def category_label(category: str) -> str:
    """Human-readable label for an internal risk flag category key.

    Maps e.g. "critical_value" -> "Kritischer Wert" / "Critical Value".
    Falls back to the raw category string for unknown keys.
    """
    if not category:
        return ""
    key = f"cat_{category}"
    lang = _get_lang()
    label = DE.get(key) if lang == "de" else EN.get(key, DE.get(key))
    return label or category


def category_label_plain(category: str) -> str:
    """Plain descriptive label for a risk flag category (no severity wording).

    Used in risk card headers, where the severity is shown separately via a
    badge and border color. For categories whose regular label implies a
    severity level (e.g. "critical_value" -> "Kritischer Wert"), a plain
    variant is used so the category stays purely descriptive. Categories
    without a dedicated plain variant fall back to :func:`category_label`.
    """
    if not category:
        return ""
    key = f"cat_{category}_plain"
    lang = _get_lang()
    label = DE.get(key) if lang == "de" else EN.get(key, DE.get(key))
    return label or category_label(category)


def translate_description(de_text: str) -> str:
    """Translate a German risk flag description to English.

    This is a best-effort heuristic translator for risk descriptions.
    It is not perfect but covers the common patterns used by the analyzer.
    """
    result = de_text
    for pattern, replacement in _DE_TO_EN_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result
