"""Data Quality tab (moved verbatim from app.py main(), WP3f).

Renders the per-patient data review tab ("🩺 Datenqualität"): editable
measurement table, rename section, reference-range backfill, and the
data-quality check + second-pass column audit. The manual registry edit lived
here until the UX rework — reference-data maintenance moved to the Settings
tab (ui/tabs/llm_settings.py), which is where new names/units/ranges belong.

Since WP6 the long-running operations (backfill, data-quality check) are
submitted as background jobs (llm_jobs.submit_job) — the buttons no longer
block the UI; progress is tracked in the "LLM-Jobs" tab."""

import json

import pandas as pd
import streamlit as st

from analyzer import refresh_patient_risk_flags
from config import resolve_canonical
from database import (
    get_registry, get_registry_names, get_custom_biomarkers, _canonical_base,
    rename_canonical_name, add_custom_biomarker, update_biomarker_registry,
    get_data_quality_findings, apply_data_quality_finding,
    dismiss_data_quality_finding, clear_resolved_data_quality_findings,
    update_data_point,
)
from translations import t
from ui import queries
from ui.helpers import _flash, _job_button, _show_flash


def _unregistered_markers(names, registry: dict) -> list[str]:
    """Canonical names the extractor kept verbatim (no registry entry).

    Mirrors the extractor's own resolution tiers so a marker is flagged
    'unregistered' exactly when the pipeline could not map it: a name counts
    as registered if resolve_canonical matches it (exact canonical or alias)
    OR its base name (unit suffix in parens stripped) matches a canonical's
    base. Unknown markers have no default reference range and are invisible
    to critical-value / cross-biomarker / missing-marker risk detection, so
    surfacing them here is the recovery path (rename or add a registry row).
    """
    bases = {_canonical_base(c).lower() for c in registry}
    unregistered = []
    for name in names:
        if resolve_canonical(name, registry=registry):
            continue
        if _canonical_base(name).lower() in bases:
            continue
        unregistered.append(name)
    return unregistered


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
    if action == "set_value":
        value = params.get("value")
        unit = str(params.get("unit", "") or "").strip()
        val = f"{_fmt_num(value)}" + (f" {unit}" if unit else "")
        return t("proposal_set_value", value=val)
    if action == "clear_ref_range":
        return t("proposal_clear_ref_range")
    if action == "swap_value_ref_range":
        low, high = params.get("low"), params.get("high")
        value = params.get("value")
        unit = str(params.get("unit", "") or "").strip()
        if low is None and high is None:
            # The auditor re-read only the measurement off the correct column and
            # did not re-state a range — a value-only fix.
            val = f"{_fmt_num(value)}" + (f" {unit}" if unit else "")
            return t("proposal_swap_value_only", value=val)
        if low is not None and high is not None:
            rng = f"{_fmt_num(low)}–{_fmt_num(high)}"
        elif high is not None:
            rng = f"< {_fmt_num(high)}"
        else:
            rng = f"> {_fmt_num(low)}"
        if unit:
            rng += f" {unit}"
        if value is None:
            # Only the range can be repaired — the displaced measurement is unknown.
            return t("proposal_swap_ref_only", range=rng)
        val = f"{_fmt_num(value)}" + (f" {unit}" if unit else "")
        return t("proposal_swap_value_ref", value=val, range=rng)
    if action == "acknowledge":
        return t("proposal_acknowledge")
    if action == "merge":
        return t("proposal_merge", target=str(params.get("target", "") or "?"))
    if action == "registry_name":
        return t("proposal_rename", proposed=str(params.get("proposed", "") or "?"))
    if action == "registry_unit":
        return t("proposal_registry_unit", proposed=str(params.get("proposed", "") or "?"))
    if action == "registry_ref_range":
        return t("proposal_registry_ref_range", range=str(params.get("proposed", "") or "?"))
    if action == "register_biomarker":
        hint = str(params.get("ref_range", "") or "").strip()
        if hint:
            return t("proposal_register_biomarker_ref",
                     name=str(f.biomarker or "?"), ref_range=hint)
        return t("proposal_register_biomarker", name=str(f.biomarker or "?"))
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

        # Names the extraction pipeline could not map to the registry (it keeps
        # unknown names verbatim). Such markers get no default reference range
        # and are skipped by critical-value, trend, cross-biomarker and
        # missing-marker detection — so a patient can look entirely healthy
        # while an abnormal odd-named marker sits in the data. Surface them
        # with the recovery path (rename below, or add a registry row).
        registry = get_registry(session)
        custom_names = get_custom_biomarkers(session)
        unregistered = _unregistered_markers(unique_canonicals, registry)
        # Names the user added deliberately as custom biomarkers are not
        # pipeline misses — the rename section below (and the registry editor
        # in the Settings tab) already cover them, so listing them here would
        # be permanent noise.
        unregistered = [n for n in unregistered if n.lower() not in {c.lower() for c in custom_names}]
        if unregistered:
            st.warning(
                t("warning_unregistered_markers", names=", ".join(unregistered))
                + "\n\n" + t("warning_unregistered_markers_hint")
            )
            # The registry editor moved to the Settings tab in the UX rework —
            # point there instead of leaving the user guessing where it went.
            st.caption(t("warning_unregistered_registry_crossref"))

        # Column labels are resolved once, not per row: a mapping table can run
        # into the thousands of rows and each t() hits session state for the
        # language, which measurably dominates the rerun.
        c_date, c_report = t("col_date"), t("col_report")
        c_original, c_canonical = t("col_original"), t("col_canonical")
        c_value, c_unit, c_reference = t("col_value"), t("col_unit"), t("col_reference")
        c_registered = t("col_registered")
        c_corrected = t("col_corrected")
        badge_unregistered = t("badge_unregistered")
        badge_corrected = t("badge_corrected")
        unregistered_lower = {n.lower() for n in unregistered}

        def _ref_text(dp) -> str:
            if dp.ref_low is None and dp.ref_high is None:
                return ""
            if dp.ref_low is not None and dp.ref_high is not None:
                return f"{_fmt_num(dp.ref_low)}–{_fmt_num(dp.ref_high)}"
            if dp.ref_high is not None:
                return f"< {_fmt_num(dp.ref_high)}"
            return f"> {_fmt_num(dp.ref_low)}"

        mapping_rows = []
        for dp in dp_results:
            mapping_rows.append({
                c_date: (dp.report_date.strftime("%d.%m.%Y")
                         if dp.report_date else "?"),
                c_report: dp.filename,
                c_original: dp.original_name or "—",
                c_canonical: dp.canonical_name,
                # Editable cells stay plain strings: the user may type a German
                # decimal ('13,18') and a float column would silently drop it.
                c_value: "" if dp.value is None else _fmt_num(dp.value),
                c_unit: dp.unit or "",
                c_reference: _ref_text(dp),
                c_registered: badge_unregistered if dp.canonical_name.lower() in unregistered_lower else "",
                c_corrected: badge_corrected if dp.is_corrected else "",
            })
        df_mapping = pd.DataFrame(mapping_rows)

        # Editable measurement table. This is the manual counterpart to the
        # findings below: the deterministic guards and the LLM auditor can only
        # propose a fix they recognised, and a misread they missed (or got
        # wrong) had no recovery path at all — the page image showed the user
        # the evidence with nowhere to act on it.
        #
        # What is deliberately NOT editable:
        #  * original_name — the verbatim PDF spelling, and the key
        #    llm_jobs._reapply_corrections uses to find this row again after a
        #    reprocess deletes it. Editing it would silently orphan the
        #    correction.
        #  * canonical_name — a name change belongs to the rename section below,
        #    which records the old name as a registry ALIAS so the next
        #    extraction resolves to the corrected name on its own. A per-row
        #    rename has no such anchor and the next reprocess would undo it.
        # The editor and its snapshot are keyed on the cache token, so a
        # committed edit changes the key and the widget is rebuilt from the
        # freshly-queried df_mapping (DB truth: normalized unit spelling, the
        # '✏️' badge, recomputed range). Without this, the widget would keep its
        # own copy of the user's raw input and diff against a stale snapshot
        # forever — re-applying the same edit each rerun, and for a REJECTED
        # edit (bad unit, range typed into the value cell) re-rejecting it on
        # every rerun in an endless warning loop. Keying on the token makes the
        # rejected input disappear and the reason show once.
        token = queries.cache_token()
        editor_key = f"mapping_editor_{patient.id}_{selected_report_id}_{token}"
        snap_key = f"mapping_snapshot_{patient.id}_{selected_report_id}_{token}"
        _show_flash("mapping_edit")
        st.caption(t("caption_edit_values"))

        edited_df = st.data_editor(
            df_mapping,
            key=editor_key,
            hide_index=True,
            num_rows="fixed",
            width="stretch",
            column_config={
                c_date: st.column_config.TextColumn(disabled=True),
                c_report: st.column_config.TextColumn(disabled=True),
                c_original: st.column_config.TextColumn(disabled=True),
                c_canonical: st.column_config.TextColumn(disabled=True),
                c_value: st.column_config.TextColumn(
                    c_value, help=t("help_edit_value")),
                c_unit: st.column_config.TextColumn(
                    c_unit, help=t("help_edit_unit")),
                c_reference: st.column_config.TextColumn(
                    c_reference, help=t("help_edit_reference")),
                c_registered: st.column_config.TextColumn(disabled=True),
                c_corrected: st.column_config.TextColumn(disabled=True),
            },
        )

        # Snapshot of the stored cell text, keyed by position. The editor keeps
        # row order (num_rows="fixed", and mapping_rows orders by
        # canonical_name, report_date, id), so row i is dp_results[i]; diffing
        # the editor output against this snapshot yields exactly the cells the
        # user changed. Same pattern as the Detail tab's enable/disable column.
        prev = st.session_state.get(snap_key)
        if prev is None or len(prev) != len(dp_results) or any(
            p[0] != dp.id for p, dp in zip(prev, dp_results)
        ):
            st.session_state[snap_key] = [
                (dp.id, "" if dp.value is None else _fmt_num(dp.value),
                 dp.unit or "", _ref_text(dp))
                for dp in dp_results
            ]
        else:
            applied = rejected = 0
            for idx, dp in enumerate(dp_results):
                row = edited_df.iloc[idx]
                _old_id, old_val, old_unit, old_ref = prev[idx]
                new_val = str(row[c_value] or "").strip()
                new_unit = str(row[c_unit] or "").strip()
                new_ref = str(row[c_reference] or "").strip()
                if new_val == old_val and new_unit == old_unit and new_ref == old_ref:
                    continue
                kwargs = {}
                if new_val != old_val:
                    kwargs["value"] = new_val
                if new_unit != old_unit:
                    kwargs["unit"] = new_unit
                if new_ref != old_ref:
                    kwargs["ref_range"] = new_ref
                outcome = update_data_point(session, dp.id, patient.id, **kwargs)
                if outcome["status"] == "applied":
                    applied += 1
                elif outcome["status"] == "skipped":
                    rejected += 1
                    # Rendered directly, not queued: when nothing at all was
                    # applied we do NOT rerun, so a queued message would sit
                    # until some unrelated interaction.
                    st.warning(t("msg_edit_rejected", name=dp.canonical_name,
                                 detail=outcome["detail"]))
            if applied:
                # One rebuild for the whole batch: refreshing per row would
                # re-run detection N times, and the flags are derived data
                # anyway.
                refresh_patient_risk_flags(session, patient.id)
                plural = "e" if applied > 1 else ""
                _flash("mapping_edit", "success",
                       t("msg_edits_applied", count=applied, plural=plural))
                # The commit bumped the cache token, so the rerun rebuilds the
                # editor from DB truth and the user's raw input is replaced by
                # the stored (normalized) form.
                st.session_state._rerun_requested = True

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
                # Registry rename: renames the registry row (shared reference
                # data) and keeps the old name as an alias for incoming PDF
                # names. The DATA migration (this patient's data points +
                # biomarker infos) is scoped to the selected patient — other
                # patients keep their rows under the old name, which still
                # resolves via the alias, and their own Mapping tab surfaces
                # the now-unregistered name so they can decide to apply the
                # same fix themselves.
                outcome = update_biomarker_registry(session, old_name, "name",
                                                    final_new_name,
                                                    patient_id=patient.id)
                if outcome["status"] == "applied":
                    _flash("mapping_rename", "success",
                           t("msg_biomarker_renamed_scoped", old=old_name,
                             new=final_new_name))
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

        # --- Backfill registry defaults onto missing data points -----------
        # Applies the registry default ranges to THIS patient's historical
        # DataPoints that are still missing them. The registry read stays global
        # (reference data), the writes are scoped — writing ranges into another
        # patient's measurements would be editing their data.

        # Contextual backfill: applies registry defaults to missing data points
        # of the selected patient.
        st.divider()
        st.caption(t("caption_backfill_from_registry"))
        # Result of the previous click: queued via _flash() because the helper's
        # st.rerun() discards anything rendered in the click run.
        _show_flash("backfill_mapping")
        _job_button(
            session,
            label_key="btn_backfill_ref_ranges", kind="backfill",
            job_label_key="job_kind_backfill", key="btn_backfill_ref_ranges",
            params={"patient_id": patient.id},
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

        # --- Second-pass column audit --------------------------------------
        # The column-first extraction prompt makes the model declare the table
        # layout before reading any row, but a model can still read a value from
        # the wrong column. This pass re-sends each page image and asks the LLM
        # ONLY whether every stored value came from the right column — a narrow,
        # cheap audit whose disagreements become reviewable findings (never
        # auto-applied). It complements the deterministic checks the extractor
        # already ran at ingestion (those need no LLM and catch the shapes that
        # cannot occur in a correctly read table).
        st.markdown(f"**{t('subheader_verify')}**")
        st.caption(t("caption_verify"))
        _show_flash("verify_extraction")
        _job_button(
            session,
            label_key="btn_verify_extraction", kind="verify_extraction",
            job_label_key="job_kind_verify_extraction", key="btn_verify_extraction",
            params={"patient_id": patient.id},
            flash_key="verify_extraction", button_type="secondary",
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
            # report_id → filename, so a finding can name the PDF its source page
            # came from (the image helper needs the filename, not just the id).
            filename_by_report = {r.id: r.filename for r in patient_reports}
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

                # Show the exact source cell the finding is about: the report
                # page it came from, rendered next to the finding. The user
                # eyeballs the real table instead of guessing which column the
                # model misread. Only extraction/verify findings carry a page;
                # the on-demand check reasons over stored data and has none.
                if f.report_id and getattr(f, "source_page", None):
                    fname = filename_by_report.get(f.report_id)
                    if fname:
                        with st.expander(
                            t("caption_finding_source_page", page=f.source_page, filename=fname)
                        ):
                            img = queries.report_page_image(f.report_id, fname, f.source_page)
                            if img:
                                st.image(f"data:image/jpeg;base64,{img}", use_container_width=True)
                            else:
                                st.caption(t("caption_finding_source_missing", page=f.source_page))
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
