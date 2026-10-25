"""Trends tab (moved verbatim from app.py main(), WP3c).

Renders the patient Trends tab: biomarker selectbox, time-range slider,
biomarker description generation, trend chart, data table, and biomarker
interpretation generation.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from database import get_biomarker_info
from translations import t
from ui import queries
from ui.helpers import _job_button, _show_flash, _time_range_slider


def render_trends_tab(session, patient, patient_reports, min_date, max_date, time_range_key, lang):
    if patient_reports:
        all_entries = queries.biomarkers(patient.id)
        # display label -> (canonical name, unit). Resolving the selection with a
        # dict lookup also removes the old `[x for x in …][0]` pair, which shadowed
        # the translation function `t` and raised IndexError on a stale selection.
        options = {}
        for cname, unit in all_entries:
            options[f"{cname} ({unit})" if unit else cname] = (cname, unit)
        # Explicit key: without one, Streamlit derives an auto-key from the label +
        # option list, so adding a report or renaming a marker silently resets the
        # user's current selection.
        selected_label = st.selectbox(t("select_biomarker"), list(options),
                                      key=f"trends_marker_{patient.id}")
        selected_cname, selected_unit = options[selected_label]

        # --- Time range slider (robust against a single report) ---
        start_date, end_date = _time_range_slider(
            min_date, max_date, time_range_key, f"slider_trends_{patient.id}", t("time_range_label"),
        )

        # Cached per (patient, marker, unit); the time window is applied in Python
        # so moving the slider re-slices instead of re-querying.
        dp_data = [d for d in queries.biomarker_series(patient.id, selected_cname, selected_unit)
                   if d.unit == selected_unit]
        if start_date is not None:
            dp_data = [d for d in dp_data if d.report_date >= start_date]
        if end_date is not None:
            dp_data = [d for d in dp_data if d.report_date <= end_date]

        if dp_data:
            dates = [d.report_date for d in dp_data]
            values = [d.value for d in dp_data]
            units = dp_data[0].unit or ""
            ref_low = None
            ref_high = None
            for dp_data_row in reversed(dp_data):
                if dp_data_row.ref_low is not None and dp_data_row.ref_high is not None:
                    ref_low = dp_data_row.ref_low
                    ref_high = dp_data_row.ref_high
                    break

            # ── Biomarker Description (above graph) ──────────────────────────
            st.subheader(t("subheader_biomarker_description"))
            st.caption(t("caption_biomarker_description"))

            # Load existing description from DB
            existing_info = get_biomarker_info(session, patient.id, selected_cname)
            desc_text = existing_info.description if existing_info else None

            # --- Description generation (WP6-aligned) -----------------------
            # Was an inline st.spinner() inside a fragment: the LLM call blocked
            # the whole script for tens of seconds, with no row in the LLM-Jobs
            # tab and no way to abort. Now a background job via _job_button(), so
            # this button behaves like every other long-running operation (toast
            # on submit, disabled "⏳ running" button, progress bar, abort).
            # When the job finishes, the LLM-Jobs poll fires an app rerun, which
            # re-queries existing_info above and renders the new description.
            _show_flash("gen_desc")

            col_desc_btn, _ = st.columns([1, 3])
            with col_desc_btn:
                _job_button(
                    session,
                    label_key="btn_generate_description",
                    kind="gen_description", job_label_key="job_kind_gen_description",
                    key="btn_gen_desc",
                    params={"patient_id": patient.id,
                            "canonical_name": selected_cname,
                            "unit": selected_unit},
                    flash_key="gen_desc",
                )

            if desc_text:
                st.markdown(
                    f"<div class='bb-card bb-card--summary bb-card--warning'>{desc_text}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(t("msg_no_description"))

            st.divider()

            # ── Trend Chart ──────────────────────────────────────────────────
            chart_label = selected_label
            df_chart = pd.DataFrame({
                t("col_date"): dates,
                chart_label: values,
            })

            fig = px.line(
                df_chart, x=t("col_date"), y=chart_label,
                markers=True, title=f"{selected_cname} — {t('chart_trend')}",
            )

            if ref_low is not None and ref_high is not None:
                fig.add_hrect(
                    y0=ref_low, y1=ref_high,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text=t("chart_ref_range"),
                )
                fig.add_hline(y=ref_low, line_dash="dash", line_color="green", opacity=0.5)
                fig.add_hline(y=ref_high, line_dash="dash", line_color="green", opacity=0.5)

            fig.update_layout(yaxis_title=units, xaxis_title="")
            st.plotly_chart(fig, width='stretch')

            # ── Data Table ───────────────────────────────────────────────────
            c_status, c_date, c_value, c_reference = (
                t("col_status"), t("col_date"), t("col_value"), t("col_reference"))
            table_rows = []
            for dp in dp_data:
                ref_s = (f"{dp.ref_low}–{dp.ref_high}"
                         if dp.ref_low is not None and dp.ref_high is not None else "—")
                table_rows.append({
                    c_status: "🔴" if dp.is_abnormal else "🟢",
                    c_date: dp.report_date.strftime("%d.%m.%Y") if dp.report_date else "?",
                    c_value: f"{dp.value} {dp.unit or ''}",
                    c_reference: ref_s,
                })
            st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True)

            # ── Biomarker Interpretation (after table) ───────────────────────
            st.divider()
            st.subheader(t("subheader_biomarker_interpretation"))
            st.caption(t("caption_biomarker_interpretation"))

            interp_text = existing_info.interpretation if existing_info else None

            # --- Interpretation generation (WP6-aligned) --------------------
            # Background job, same pattern as the description above. The time
            # range the user selected is passed as ISO strings in the job params
            # (params_json is plain JSON), so the worker reproduces exactly the
            # window the button was clicked with — it cannot rely on main's
            # dp_data, which is detached once main's session closes.
            _show_flash("gen_interp")

            col_interp_btn, _ = st.columns([1, 3])
            with col_interp_btn:
                _job_button(
                    session,
                    label_key="btn_generate_interpretation",
                    kind="gen_interpretation", job_label_key="job_kind_gen_interpretation",
                    key="btn_gen_interp",
                    params={"patient_id": patient.id,
                            "canonical_name": selected_cname,
                            "unit": selected_unit,
                            "start_date": start_date.isoformat() if start_date else None,
                            "end_date": end_date.isoformat() if end_date else None},
                    flash_key="gen_interp",
                )

            if interp_text:
                st.markdown(
                    f"<div class='bb-card bb-card--summary bb-card--info'>{interp_text}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(t("msg_no_interpretation"))
        else:
            st.info(t("info_no_biomarker_data"))
    else:
        st.info(t("info_no_reports"))
