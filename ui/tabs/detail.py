"""Detail tab (moved verbatim from app.py main(), WP3b).

Renders the patient Detailansicht tab: report selectbox and a compact,
editable table of the report's values with an "active" checkbox per row.
"""

import pandas as pd
import streamlit as st

from analyzer import refresh_patient_risk_flags
from database import DataPoint, set_data_point_disabled
from translations import t
from ui.helpers import _flash, _show_flash


def render_detail_tab(session, patient, patient_reports):
    # Confirmation for the last enable/disable toggle (queued via _flash on
    # the run where the checkbox was flipped; that run ends in st.rerun()).
    _show_flash("dp_toggle")

    if patient_reports:
        report_options = {
            f"{r.report_date.strftime('%d.%m.%Y')} — {r.filename}": r
            for r in reversed(patient_reports)
        }
        selected_label = st.selectbox(t("select_report"), list(report_options.keys()))
        selected_report = report_options[selected_label]

        dps_detail = (
            session.query(DataPoint)
            .filter(DataPoint.report_id == selected_report.id)
            .order_by(DataPoint.canonical_name)
            .all()
        )

        if dps_detail:
            st.caption(t("caption_disabled_values"))
            # One compact table instead of ~40 stacked rows. The "active"
            # checkbox column is the only editable one; every other column
            # is read-only and row add/remove is disabled.
            rows = []
            for dp in dps_detail:
                ref_str = (
                    f"{dp.ref_low}–{dp.ref_high}"
                    if dp.ref_low is not None and dp.ref_high is not None
                    else "—"
                )
                status = "🟢" if not dp.is_abnormal else "🔴"
                marker_name = (
                    f"⛔ {dp.canonical_name}" if dp.is_disabled else dp.canonical_name
                )
                value_str = (
                    f"{dp.value} {dp.unit or ''}".strip() if dp.value is not None else "—"
                )
                original = (
                    dp.original_name
                    if dp.original_name and dp.original_name != dp.canonical_name
                    else ""
                )
                rows.append(
                    {
                        t("col_status"): status,
                        t("col_marker"): marker_name,
                        t("col_value"): value_str,
                        t("col_reference"): ref_str,
                        t("col_original"): original,
                        t("col_active"): not dp.is_disabled,
                    }
                )
            df = pd.DataFrame(rows)

            # Snapshot of the enabled state from the previous run, keyed by
            # report id so it survives the st.rerun() dance in app.py. The
            # data_editor widget (keyed by report id too) keeps its checkbox
            # state across reruns, so diffing its output against this
            # snapshot yields exactly the rows the user flipped.
            snap_key = f"dp_snapshot_{selected_report.id}"
            prev_enabled = st.session_state.get(snap_key)

            editor_df = st.data_editor(
                df,
                key=f"dp_editor_{selected_report.id}",
                hide_index=True,
                num_rows="fixed",
                column_config={
                    t("col_status"): st.column_config.TextColumn(disabled=True),
                    t("col_marker"): st.column_config.TextColumn(disabled=True),
                    t("col_value"): st.column_config.TextColumn(disabled=True),
                    t("col_reference"): st.column_config.TextColumn(disabled=True),
                    t("col_original"): st.column_config.TextColumn(disabled=True),
                },
            )

            if prev_enabled is not None and len(editor_df) == len(dps_detail):
                # Diff old vs new checkbox state. Rows are matched by position:
                # the editor keeps row order (num_rows="fixed", no reordering),
                # so editor row i corresponds to dps_detail[i].
                changed = False
                for idx, (_, row) in enumerate(editor_df.iterrows()):
                    dp = dps_detail[idx]
                    was_enabled = prev_enabled.get(dp.id)
                    if was_enabled is None:
                        continue
                    now_enabled = bool(row[t("col_active")])
                    if was_enabled == now_enabled:
                        continue
                    updated = set_data_point_disabled(session, dp.id, not now_enabled)
                    if updated is not None:
                        # The targeted delete inside set_data_point_disabled
                        # matches data_point_id only — flags reassigned to a
                        # different data point by _dedup_by_biomarker would
                        # survive. Rebuild this patient's flags instead.
                        refresh_patient_risk_flags(session, patient.id)
                    msg = (
                        t("msg_value_enabled")
                        if now_enabled
                        else t("msg_value_disabled")
                    )
                    _flash("dp_toggle", "success", f"{msg} ({dp.canonical_name})")
                    changed = True

                # Update the snapshot to the new state BEFORE the rerun so
                # the next run diffs against what was just applied.
                st.session_state[snap_key] = {
                    dps_detail[idx].id: bool(row[t("col_active")])
                    for idx, (_, row) in enumerate(editor_df.iterrows())
                }
                if changed:
                    st.session_state._rerun_requested = True
            else:
                # First render of this report (or unexpected shape change):
                # seed the snapshot from the DB state.
                st.session_state[snap_key] = {
                    dp.id: not dp.is_disabled for dp in dps_detail
                }
        else:
            st.info(t("info_no_data_points"))
    else:
        st.info(t("info_no_reports"))
