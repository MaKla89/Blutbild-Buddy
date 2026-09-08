"""Trends tab (moved verbatim from app.py main(), WP3c).

Renders the patient Trends tab: biomarker selectbox, time-range slider,
biomarker description generation, trend chart, data table, and biomarker
interpretation generation.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from database import Report, DataPoint, get_biomarker_info, upsert_biomarker_info, get_session
from llm_client import _build_client, generate_biomarker_description, generate_biomarker_interpretation
from translations import t, _get_lang
from ui.helpers import _flash, _show_flash


def render_trends_tab(session, patient, patient_reports, min_date, max_date, time_range_key, lang):
    if patient_reports:
        all_entries = (
            session.query(DataPoint.canonical_name, DataPoint.unit)
            .join(Report)
            .filter(
                Report.patient_id == patient.id,
                DataPoint.value.isnot(None),
                DataPoint.is_disabled == False,  # noqa: E712
            )
            .distinct()
            .order_by(DataPoint.canonical_name)
            .all()
        )
        trend_options = []
        for cname, unit in all_entries:
            if unit:
                trend_options.append((f"{cname} ({unit})", cname, unit))
            else:
                trend_options.append((cname, cname, None))
        display_labels = [t[0] for t in trend_options]

        selected_label = st.selectbox(t("select_biomarker"), display_labels)
        selected_cname, selected_unit = [t[1] for t in trend_options if t[0] == selected_label][0], [t[2] for t in trend_options if t[0] == selected_label][0]

        # --- Time range slider ---
        if min_date and max_date:
            time_range = st.slider(
                t("time_range_label"),
                min_value=min_date,
                max_value=max_date,
                value=st.session_state[time_range_key],
                format="DD.MM.YYYY",
                key=f"slider_trends_{patient.id}",
            )
            st.session_state[time_range_key] = time_range
            start_date, end_date = time_range
        else:
            start_date = None
            end_date = None

        dp_data = (
            session.query(DataPoint, Report.report_date)
            .join(Report)
            .filter(
                DataPoint.canonical_name == selected_cname,
                DataPoint.value.isnot(None),
                DataPoint.unit == selected_unit,
                DataPoint.is_disabled == False,  # noqa: E712
                Report.patient_id == patient.id,
            )
            .order_by(Report.report_date)
        )
        if start_date is not None:
            dp_data = dp_data.filter(Report.report_date >= start_date)
        if end_date is not None:
            dp_data = dp_data.filter(Report.report_date <= end_date)
        dp_data = dp_data.all()

        if dp_data:
            dates = [d.report_date for d in dp_data]
            values = [d.DataPoint.value for d in dp_data]
            units = dp_data[0].DataPoint.unit or ""
            ref_low = None
            ref_high = None
            for dp_data_row in reversed(dp_data):
                if dp_data_row.DataPoint.ref_low is not None and dp_data_row.DataPoint.ref_high is not None:
                    ref_low = dp_data_row.DataPoint.ref_low
                    ref_high = dp_data_row.DataPoint.ref_high
                    break

            # ── Biomarker Description (above graph) ──────────────────────────
            st.subheader(t("subheader_biomarker_description"))
            st.caption(t("caption_biomarker_description"))

            # Load existing description from DB
            existing_info = get_biomarker_info(session, patient.id, selected_cname)
            desc_text = existing_info.description if existing_info else None

            # --- Description generation (WP5-A) -----------------------------
            # Self-contained @st.fragment: button + work live together, so a click
            # re-runs ONLY this fragment. It opens its own DB session via
            # get_session() (main's session is closed by the time the fragment
            # re-runs). upsert_biomarker_info commits internally. st.rerun() at the
            # end escalates to a full app rerun so main reloads existing_info and
            # shows the new description below — replacing the old flag dance.

            @st.fragment
            def _op_gen_desc(patient_id, cname, unit):
                # Result of the previous click: this run's own message is discarded
                # by the st.rerun() below, so it is queued via _flash() and shown
                # here on the following completed run.
                _show_flash("gen_desc")
                if st.button(t("btn_generate_description"), key="btn_gen_desc", type="primary", width='stretch'):
                    with get_session() as op_session:
                        with st.spinner(t("msg_generating_description")):
                            try:
                                llm = _build_client()
                                new_desc = generate_biomarker_description(
                                    llm, cname, unit,
                                    config.CANONICAL_BIOMARKERS, _get_lang(),
                                )
                                if new_desc:
                                    upsert_biomarker_info(op_session, patient_id, cname, description=new_desc)
                                    _flash("gen_desc", "success", t("msg_description_generated"))
                                else:
                                    _flash("gen_desc", "error", t("error_summary_generation"))
                            except Exception as e:
                                _flash("gen_desc", "error", f"{t('error_summary_generation')}: {e}")
                    st.rerun()

            col_desc_btn, _ = st.columns([1, 3])
            with col_desc_btn:
                _op_gen_desc(patient.id, selected_cname, selected_unit)

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
            table_rows = []
            for d, dp in zip(dates, dp_data):
                ref_s = f"{dp.DataPoint.ref_low}–{dp.DataPoint.ref_high}" if dp.DataPoint.ref_low is not None and dp.DataPoint.ref_high is not None else "—"
                flag = "🔴" if dp.DataPoint.is_abnormal else "🟢"
                table_rows.append({
                    t("col_status"): flag,
                    t("col_date"): d.strftime("%d.%m.%Y") if d else "?",
                    t("col_value"): f"{dp.DataPoint.value} {dp.DataPoint.unit or ''}",
                    t("col_reference"): ref_s,
                })
            st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True)

            # ── Biomarker Interpretation (after table) ───────────────────────
            st.divider()
            st.subheader(t("subheader_biomarker_interpretation"))
            st.caption(t("caption_biomarker_interpretation"))

            interp_text = existing_info.interpretation if existing_info else None

            # --- Interpretation generation (WP5-A) --------------------------
            # Same fragment pattern as description. The interpretation prompt is
            # built from the data points in the CURRENT time range, so the
            # fragment re-queries them with its own session using the same date
            # bounds (passed as args — all hashable scalars) rather than relying
            # on main's dp_data (whose ORM objects are detached once main's
            # session closes). st.rerun() refreshes the display below.

            @st.fragment
            def _op_gen_interp(patient_id, cname, unit, lang_arg, start_date, end_date):
                # Same flash pattern as the description fragment above.
                _show_flash("gen_interp")
                if st.button(t("btn_generate_interpretation"), key="btn_gen_interp", type="primary", width='stretch'):
                    with get_session() as op_session:
                        q = (
                            op_session.query(DataPoint, Report.report_date)
                            .join(Report)
                            .filter(
                                DataPoint.canonical_name == cname,
                                DataPoint.value.isnot(None),
                                DataPoint.unit == unit,
                                DataPoint.is_disabled == False,  # noqa: E712
                                Report.patient_id == patient_id,
                            )
                        )
                        if start_date is not None:
                            q = q.filter(Report.report_date >= start_date)
                        if end_date is not None:
                            q = q.filter(Report.report_date <= end_date)
                        rows = q.order_by(Report.report_date).all()
                        dp_tuples = [
                            (d.report_date, d.DataPoint.value, d.DataPoint.ref_low, d.DataPoint.ref_high)
                            for d in rows
                        ]
                        with st.spinner(t("msg_generating_interpretation")):
                            try:
                                llm = _build_client()
                                new_interp = generate_biomarker_interpretation(
                                    llm, cname, unit, dp_tuples, lang_arg,
                                )
                                if new_interp:
                                    upsert_biomarker_info(op_session, patient_id, cname, interpretation=new_interp)
                                    _flash("gen_interp", "success", t("msg_interpretation_generated"))
                                else:
                                    _flash("gen_interp", "error", t("error_summary_generation"))
                            except Exception as e:
                                _flash("gen_interp", "error", f"{t('error_summary_generation')}: {e}")
                    st.rerun()

            col_interp_btn, _ = st.columns([1, 3])
            with col_interp_btn:
                _op_gen_interp(patient.id, selected_cname, selected_unit, lang, start_date, end_date)

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
