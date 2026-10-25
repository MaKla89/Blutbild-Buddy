"""Biomarker Mapping tab (moved verbatim from app.py main(), WP3f).

Renders the Biomarker Mapping tab: mapping table, rename section, manual
registry edit, reference-range backfill, and the per-patient data-quality
check. Since WP6 the long-running operations (backfill, data-quality check)
are submitted as background jobs (llm_jobs.submit_job) — the buttons no
longer block the UI; progress is tracked in the "LLM-Jobs" tab."""

import json

import pandas as pd
import streamlit as st

from database import (
    get_registry_names, get_custom_biomarkers,
    rename_canonical_name, add_custom_biomarker, update_biomarker_registry,
    get_data_quality_findings, apply_data_quality_finding,
    dismiss_data_quality_finding, clear_resolved_data_quality_findings,
)
from translations import t
from ui import queries
from ui.helpers import _flash, _job_button, _show_flash


def _fmt_num(v):
    """Format a numeric bound without a trailing .0 (5.0 -> '5', 4.5 -> '4.5')."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else str(f)


def _finding_proposal(f) -> str:
    """Human-readable summary of the machine-applicable fix a finding carries.

    Lets the user see WHAT 'Korrigieren' will actually change before clicking it."""
    try:
        params = json.loads(f.params_json) if f.params_json else {}
    except (json.JSONDecodeError, TypeError):
        params = {}
    if not isinstance(params, dict):
        params = {}

    action = f.action or ""
    if action == "set_ref_range":
        low, high = params.get("low"), params.get("high")
        unit = str(params.get("unit", "") or "").strip()
        if low is not None and high is not None:
            rng = f"{_fmt_num(low)}–{_fmt_num(high)}"
        elif high is not None:
            rng = f"< {_fmt_num(high)}"
        else:
            rng = f"> {_fmt_num(low)}"
        if unit:
            rng += f" {unit}"
        return t("proposal_set_ref_range", range=rng)
    if action == "set_unit":
        return t("proposal_set_unit", new_unit=str(params.get("unit", "") or "?"))
    if action == "disable_value":
        return t("proposal_disable_value")
    if action == "merge":
        return t("proposal_merge", target=str(params.get("target", "") or "?"))
    if action == "registry_name":
        return t("proposal_rename", proposed=str(params.get("proposed", "") or "?"))
    if action == "registry_unit":
        return t("proposal_registry_unit", proposed=str(params.get("proposed", "") or "?"))
    if action == "registry_ref_range":
        return t("proposal_registry_ref_range", range=str(params.get("proposed", "") or "?"))
    return t("proposal_unknown")


def render_mapping_tab(session, patient, patient_reports):
    st.subheader(t("subheader_mapping"))
    st.caption(t("caption_mapping"))

    report_options = {
        t("select_all_reports"): None,
    }
    for r in patient_reports:
        date_str = r.report_date.strftime("%d.%m.%Y") if r.report_date else "?"
        report_options[date_str + " — " + r.filename] = r.id
    selected_report_label = st.selectbox(t("select_filter_report"), list(report_options.keys()),
                                         key="mapping_report_filter")
    selected_report_id = report_options[selected_report_label]

    dp_results = queries.mapping_rows(patient.id, selected_report_id)

    if dp_results:
        # Build unique canonical names list first (needed for rename dropdown)
        unique_canonicals = sorted({row.canonical_name for row in dp_results})

        # Column labels are resolved once, not per row: a mapping table can run
        # into the thousands of rows and each t() hits session state for the
        # language, which measurably dominates the rerun.
        c_date, c_report = t("col_date"), t("col_report")
        c_original, c_canonical = t("col_original"), t("col_canonical")
        c_value, c_unit, c_reference = t("col_value"), t("col_unit"), t("col_reference")
        mapping_rows = []
        for dp in dp_results:
            ref_str = f"{dp.ref_low}–{dp.ref_high}" if dp.ref_low is not None and dp.ref_high is not None else "—"
            mapping_rows.append({
                c_date: dp.report_date.strftime("%d.%m.%Y") if dp.report_date else "?",
                c_report: dp.filename,
                c_original: dp.original_name or "—",
                c_canonical: dp.canonical_name,
                c_value: f"{dp.value} {dp.unit or ''}" if dp.value is not None else "—",
                c_unit: dp.unit or "—",
                c_reference: ref_str,
            })
        df_mapping = pd.DataFrame(mapping_rows)
        st.dataframe(df_mapping, width='stretch', hide_index=True)

        # --- Rename section ---
        st.divider()
        st.subheader(t("subheader_rename"))
        st.caption(t("caption_rename"))

        # A successful rename sets _rerun_requested, which aborts the run and
        # discards everything rendered in it — so every outcome of the rename is
        # queued via _flash and shown on the following run.
        _show_flash("mapping_rename")

        registry_names = get_registry_names(session)
        custom_names = get_custom_biomarkers(session)
        all_canonical_options = sorted(set(registry_names + custom_names))

        col_r1, col_r2, col_r3 = st.columns([2, 2, 1])
        with col_r1:
            old_name = st.selectbox(t("select_old_name"), unique_canonicals, key="mapping_old_name")
        with col_r2:
            new_name = st.selectbox(t("select_new_name"), all_canonical_options, key="mapping_new_name_select")
        with col_r3:
            free_text = st.text_input(t("input_new_name"), key="mapping_new_name_free", placeholder=t("input_new_name_placeholder"))

        if st.button(t("btn_rename_biomarker"), type="secondary", width='stretch',
                     key="btn_rename_biomarker_action"):
            final_new_name = free_text.strip() if free_text.strip() else new_name
            data_changed = False
            if not final_new_name:
                _flash("mapping_rename", "error", t("error_no_name"))
            elif final_new_name == old_name:
                _flash("mapping_rename", "warning", t("warning_same_name"))
            elif old_name.lower() in {n.lower() for n in registry_names}:
                # Global registry rename: renames the registry row, all data
                # points and biomarker infos across ALL patients, and keeps
                # the old name as an alias for incoming PDF names.
                outcome = update_biomarker_registry(session, old_name, "name", final_new_name)
                if outcome["status"] == "applied":
                    _flash("mapping_rename", "success",
                           t("msg_biomarker_renamed_global", old=old_name, new=final_new_name))
                    data_changed = True
                else:
                    # The only remaining skip reason here is a name collision.
                    _flash("mapping_rename", "warning",
                           t("warning_rename_collision", new=final_new_name))
            else:
                # Custom biomarker without a registry row: keep the old
                # per-patient rename behavior.
                if free_text.strip() and free_text.strip().lower() not in {n.lower() for n in registry_names}:
                    if add_custom_biomarker(session, free_text.strip()):
                        _flash("mapping_rename", "success", t("msg_added_to_master", name=free_text.strip()))
                    else:
                        _flash("mapping_rename", "info", t("msg_already_in_master", name=free_text.strip()))

                count = rename_canonical_name(session, patient.id, old_name, final_new_name)
                if count > 0:
                    plural = "e" if count > 1 else ""
                    _flash("mapping_rename", "success",
                           t("msg_biomarker_renamed", count=count, plural=plural, old=old_name, new=final_new_name))
                    data_changed = True
                else:
                    _flash("mapping_rename", "warning", t("warning_no_data_points", old=old_name))
            # Only reload the data when something actually changed: rerunning on
            # a validation error would discard the message the user needs to see.
            if data_changed:
                st.session_state._rerun_requested = True

        # --- Manual registry edit section ---
        st.divider()
        st.subheader(t("subheader_registry_edit"))
        st.caption(t("caption_registry_edit"))
        _show_flash("registry_edit")

        col_e1, col_e2, col_e3, col_e4 = st.columns([2, 1, 2, 1])
        with col_e1:
            # All known canonical names (registry + custom): editing a custom
            # name promotes it to a full registry row.
            edit_name = st.selectbox(t("select_registry_biomarker"), all_canonical_options, key="registry_edit_name")
        with col_e2:
            edit_field_label = st.selectbox(
                t("select_registry_field"),
                [t("field_name"), t("field_unit"), t("field_ref_range")],
                key="registry_edit_field",
            )
        field_key = {
            t("field_name"): "name",
            t("field_unit"): "unit",
            t("field_ref_range"): "ref_range",
        }[edit_field_label]
        with col_e3:
            edit_value = st.text_input(
                t("input_registry_value"), key="registry_edit_value",
                placeholder={
                    "name": t("placeholder_registry_name"),
                    "unit": t("placeholder_registry_unit"),
                    "ref_range": t("placeholder_registry_ref_range"),
                }[field_key],
            )
        with col_e4:
            apply_edit = st.button(t("btn_apply_registry_edit"), type="secondary",
                                   width='stretch', key="btn_apply_registry_edit")

        if apply_edit:
            if not edit_value.strip():
                st.error(t("error_empty_value"))
            else:
                outcome = update_biomarker_registry(session, edit_name, field_key, edit_value)
                # Queued via _flash: the _rerun_requested below discards this run.
                if outcome["status"] == "applied":
                    _flash("registry_edit", "success",
                           t("msg_registry_edit_applied", field=field_key, proposed=outcome["detail"]))
                else:
                    _flash("registry_edit", "warning",
                           t("msg_suggestion_skipped", detail=outcome.get("detail", "")))
                st.session_state._rerun_requested = True

        # --- Backfill registry defaults onto missing data points -----------
        # Applies the registry default ranges to historical DataPoints that are
        # still missing them. Global across all patients — same operation as
        # the Berichte-tab button (the kind lock in llm_jobs prevents double
        # execution from either tab).

        # Contextual backfill: applies registry defaults to missing data points
        # (all patients).
        st.divider()
        st.caption(t("caption_backfill_from_registry"))
        # Result of the previous click: queued via _flash() because the helper's
        # st.rerun() discards anything rendered in the click run.
        _show_flash("backfill_mapping")
        _job_button(
            session,
            label_key="btn_backfill_ref_ranges", kind="backfill",
            job_label_key="job_kind_backfill", key="btn_backfill_ref_ranges",
            flash_key="backfill_mapping",
            button_type="secondary",
        )

        # --- Data quality check (registry + actual measurements) -----------
        # Feeds the LLM the patient's real data points, so it can flag
        # per-value reference ranges, unit mismatches and implausible values.
        # Findings are stored in their own table (not per-patient
        # BiomarkerInfo) and each carries a machine-applicable fix that is
        # applied on accept.

        st.divider()
        st.subheader(t("subheader_data_quality"))
        st.caption(t("caption_data_quality"))

        _show_flash("data_quality_check")
        _job_button(
            session,
            label_key="btn_data_quality_check", kind="data_quality_check",
            job_label_key="job_kind_data_quality_check", key="btn_data_quality_check",
            params={"patient_id": patient.id},
            flash_key="data_quality_check",
        )

        # Render results queued on the previous run. Placed here, OUTSIDE the
        # findings block, so they still show when a finding was just resolved
        # and the pending list is now empty (the _rerun_requested rerun below
        # would otherwise discard messages rendered in the click-run).
        _show_flash("findings_accept_all")
        _show_flash("finding_result")

        findings = get_data_quality_findings(session, patient.id, status="pending")
        if findings:
            st.markdown(t("subheader_findings"))

            # Accept-all: apply every pending finding in one click. The summary
            # is queued via _flash() because the _rerun_requested below aborts
            # this run and would discard any message rendered here directly.
            if st.button(t("btn_accept_all_findings"), type="secondary"):
                applied = skipped = 0
                for f in findings:
                    outcome = apply_data_quality_finding(session, f.id)
                    if outcome["status"] == "applied":
                        applied += 1
                    else:
                        skipped += 1
                _flash(
                    "findings_accept_all",
                    "success" if skipped == 0 else "info",
                    t("msg_accept_all_findings", applied=applied, skipped=skipped),
                )
                st.session_state._rerun_requested = True

            col_widths = [1.0, 1.2, 0.9, 3.0, 2.6, 1.0, 1.0]
            header_cols = st.columns(col_widths)
            for hc, label in zip(header_cols, (
                t("col_category"), t("col_biomarker"), t("col_target"),
                t("col_description"), t("col_proposal"), "", "",
            )):
                if label:
                    hc.markdown(f"**{label}**")

            any_changed = False
            for idx, f in enumerate(findings):
                c_cat, c_bio, c_target, c_desc, c_prop, c_accept, c_dismiss = st.columns(col_widths)
                c_cat.markdown(t(f"cat_{f.category}"))
                c_bio.markdown(f.biomarker or "—")
                c_target.markdown(
                    t("target_registry") if f.target_type == "registry" else t("target_data_point")
                )
                c_desc.markdown(f.description)
                c_prop.markdown(_finding_proposal(f))
                if c_accept.button(t("btn_accept_finding"), key=f"accept_finding_{f.id}"):
                    outcome = apply_data_quality_finding(session, f.id)
                    # Queue via _flash: the rerun triggered by any_changed
                    # below discards anything rendered in this run.
                    if outcome["status"] == "applied":
                        _flash("finding_result", "success",
                               t("msg_finding_applied", detail=outcome.get("detail", "")))
                    else:
                        _flash("finding_result", "warning",
                               t("msg_finding_skipped", detail=outcome.get("detail", "")))
                    any_changed = True
                if c_dismiss.button(t("btn_dismiss_finding"), key=f"dismiss_finding_{f.id}"):
                    dismiss_data_quality_finding(session, f.id)
                    any_changed = True
                if idx < len(findings) - 1:
                    st.divider()

            if any_changed:
                st.session_state._rerun_requested = True

            # Housekeeping: drop applied/dismissed findings for this patient.
            if st.button(t("btn_clear_resolved_findings"), key="btn_clear_resolved_findings"):
                n = clear_resolved_data_quality_findings(session, patient.id)
                # Queued via _flash: the rerun below discards this run's output.
                _flash("findings_accept_all", "info", t("msg_findings_cleared", n=n))
                st.session_state._rerun_requested = True
        else:
            st.caption(t("caption_no_findings"))
