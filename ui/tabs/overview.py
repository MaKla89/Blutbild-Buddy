"""Overview tab (moved verbatim from app.py main(), WP3a).

Renders the patient Overview tab: metric cards, health summary, last-abnormal
table, favorite-biomarker trend charts, out-of-range biomarker relative charts,
and the panel-type table.

`render_health_summary` is injected by the caller (app._render_health_summary)
to avoid a circular import between app and ui.tabs.
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from database import Report, DataPoint, RiskFlag
from translations import t


def render_overview_tab(session, patient, patient_reports, min_date, max_date, time_range_key, render_health_summary):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(t("metric_reports"), len(patient_reports))
    with col2:
        latest = patient_reports[-1] if patient_reports else None
        latest_date = latest.report_date.strftime("%d.%m.%Y") if latest and latest.report_date else "—"
        st.metric(t("metric_last_report"), latest_date)

    all_dps = (
        session.query(DataPoint)
        .join(Report)
        .filter(
            Report.patient_id == patient.id,
            DataPoint.is_disabled == False,  # noqa: E712
        )
        .all()
    )
    abnormal_dps = [dp for dp in all_dps if dp.is_abnormal]
    with col3:
        n_abnormal = len(abnormal_dps)
        # The headline number spans ALL reports; the delta shows how many of
        # the abnormalities are in the *latest* report — new information.
        latest_abnormal = (
            sum(1 for dp in abnormal_dps if dp.report_id == latest.id) if latest else 0
        )
        st.metric(
            t("metric_abnormal"),
            n_abnormal,
            delta=t("metric_delta_latest", n=latest_abnormal) if n_abnormal else t("metric_delta_none"),
            delta_color="red" if n_abnormal else "green",
            delta_arrow="off",
        )
    with col4:
        # Match the Risks tab: when the LLM ranked the flags, only the ones it
        # selected (with an explanation) are shown as cards; leftovers without
        # an explanation are hidden. Without an LLM assessment all flags are
        # shown, so count them all.
        patient_flags = (
            session.query(RiskFlag)
            .join(Report)
            .filter(Report.patient_id == patient.id)
            .all()
        )
        if any(rf.explanation is not None for rf in patient_flags):
            counted_flags = [rf for rf in patient_flags if rf.explanation is not None]
        else:
            counted_flags = list(patient_flags)
        flags_count = len(counted_flags)
        # Delta shows the critical subset of the counted flags.
        critical_count = sum(1 for rf in counted_flags if rf.severity == 3)
        st.metric(
            t("metric_risk_flags"),
            flags_count,
            delta=t("metric_delta_critical", n=critical_count) if flags_count else t("metric_delta_none"),
            delta_color="red" if flags_count else "green",
            delta_arrow="off",
        )

    # --- Health Summary ---
    render_health_summary(session, patient, t)

    if patient_reports:
        st.subheader(t("subheader_last_abnormal"))
        if latest:
            latest_dps = (
                session.query(DataPoint)
                .filter(
                    DataPoint.report_id == latest.id,
                    DataPoint.is_disabled == False,  # noqa: E712
                )
                .all()
            )
            abnormal = [dp for dp in latest_dps if dp.is_abnormal]
            if abnormal:
                rows = []
                for dp in abnormal:
                    if dp.ref_high is not None and dp.value > dp.ref_high:
                        direction = "➕"
                    elif dp.ref_low is not None and dp.value < dp.ref_low:
                        direction = "➖"
                    else:
                        direction = "❓"
                    rows.append({
                        t("col_marker"): f"{direction} {dp.canonical_name}",
                        t("col_value"): f"{dp.value} {dp.unit or ''}",
                        t("col_reference"): f"{dp.ref_low}–{dp.ref_high}" if dp.ref_low is not None and dp.ref_high is not None else "—",
                    })
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            else:
                st.success(t("msg_no_abnormal"))

        # --- Out-of-range biomarker trend charts (relative) ---
        # Candidate pool: every biomarker with at least one out-of-range
        # measurement within the last 2 years (relative to the patient's most
        # recent report). Deliberately independent of risk flags and of whether
        # the latest value is back in range — persistently elevated markers
        # (e.g. triglycerides) stay visible even when no flag fires for them.
        from datetime import timedelta

        out_of_range_entries: set = set()
        two_years_ago = None
        if latest and latest.report_date:
            two_years_ago = latest.report_date - timedelta(days=2 * 365)
            oor_rows = (
                session.query(DataPoint.canonical_name, DataPoint.unit)
                .join(Report)
                .filter(
                    Report.patient_id == patient.id,
                    DataPoint.value.isnot(None),
                    DataPoint.is_disabled == False,  # noqa: E712
                    Report.report_date >= two_years_ago,
                    ((DataPoint.ref_high.isnot(None)) & (DataPoint.value > DataPoint.ref_high))
                    | ((DataPoint.ref_low.isnot(None)) & (DataPoint.value < DataPoint.ref_low)),
                )
                .distinct()
                .all()
            )
            out_of_range_entries = {(cname, unit) for cname, unit in oor_rows}

        stored_favs = []
        if patient.favorites:
            try:
                stored_favs = json.loads(patient.favorites)
            except (json.JSONDecodeError, ValueError):
                stored_favs = []
        all_cnames_units = (
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
        all_options = []
        for cname, unit in all_cnames_units:
            label = f"{cname} ({unit})" if unit else cname
            all_options.append((label, cname, unit))
        selected_fav_labels = st.multiselect(
            t("multiselect_favorites"),
            options=[o[0] for o in all_options],
            default=stored_favs,
            key=f"fav_select_{patient.id}",
        )
        if selected_fav_labels != stored_favs:
            patient.favorites = json.dumps(selected_fav_labels)
            session.commit()

        # --- Time range slider ---
        if min_date and max_date:
            time_range = st.slider(
                t("time_range_label"),
                min_value=min_date,
                max_value=max_date,
                value=st.session_state[time_range_key],
                format="DD.MM.YYYY",
                key=f"slider_{patient.id}",
            )
            st.session_state[time_range_key] = time_range
            start_date, end_date = time_range
        else:
            start_date = None
            end_date = None

          # --- Favorite biomarker charts (absolute values, one per biomarker) ---
        for label in selected_fav_labels:
            st.subheader(f"{label} — {t('chart_trend')}")
            cname, unit = [(t[1], t[2]) for t in all_options if t[0] == label][0]
            dp_query = (
                session.query(DataPoint, Report.report_date)
                .join(Report)
                .filter(
                    DataPoint.canonical_name == cname,
                    DataPoint.unit == unit,
                    DataPoint.value.isnot(None),
                    DataPoint.is_disabled == False,  # noqa: E712
                    Report.patient_id == patient.id,
                )
                .order_by(Report.report_date)
            )
            if start_date is not None:
                dp_query = dp_query.filter(Report.report_date >= start_date)
            if end_date is not None:
                dp_query = dp_query.filter(Report.report_date <= end_date)
            dps_entry = dp_query.all()
            if not dps_entry:
                continue

            dates = [d.report_date for d in dps_entry]
            values = [d.DataPoint.value for d in dps_entry]
            units = dps_entry[0].DataPoint.unit or ""
            ref_low = None
            ref_high = None
            for dp_data in reversed(dps_entry):
                if dp_data.DataPoint.ref_low is not None and dp_data.DataPoint.ref_high is not None:
                    ref_low = dp_data.DataPoint.ref_low
                    ref_high = dp_data.DataPoint.ref_high
                    break

            chart_label = label
            df_chart = pd.DataFrame({
                t("col_date"): dates,
                chart_label: values,
            })

            fig = px.line(
                df_chart, x=t("col_date"), y=chart_label,
                markers=True,
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

        # --- Out-of-range biomarker charts (relative to violated boundary) ---
        risk_entries = out_of_range_entries

        if risk_entries:
            series = {}
            for cname, unit in risk_entries:
                risk_query = (
                    session.query(DataPoint, Report.report_date)
                    .join(Report)
                    .filter(
                        DataPoint.canonical_name == cname,
                        DataPoint.unit == unit,
                        DataPoint.value.isnot(None),
                        DataPoint.is_disabled == False,  # noqa: E712
                        Report.patient_id == patient.id,
                    )
                    .order_by(Report.report_date)
                )
                # The charts only consider the last 2 years (same window as
                # the candidate pool), so clamp the slider's start date to it.
                effective_start = start_date
                if two_years_ago is not None:
                    effective_start = max(start_date, two_years_ago) if start_date is not None else two_years_ago
                if effective_start is not None:
                    risk_query = risk_query.filter(Report.report_date >= effective_start)
                if end_date is not None:
                    risk_query = risk_query.filter(Report.report_date <= end_date)
                dps_entry = risk_query.all()
                if not dps_entry:
                    continue
                series[(cname, unit)] = [
                    (d.report_date, d.DataPoint.value, d.DataPoint.ref_low, d.DataPoint.ref_high)
                    for d in dps_entry
                ]

            if series:
                above_high_series = {}
                below_low_series = {}

                for (cname, unit), pts in series.items():
                    # A marker belongs to a chart if ANY measurement in its
                    # series violated that bound — not only the latest one.
                    # This keeps persistently-elevated markers visible even when
                    # the most recent report is back in range or didn't measure
                    # the marker at all. A marker can appear in both charts if it
                    # violated both bounds at different times.
                    any_above = any(
                        ref_high is not None and value > ref_high
                        for (_, value, _, ref_high) in pts
                    )
                    any_below = any(
                        ref_low is not None and value < ref_low
                        for (_, value, ref_low, _) in pts
                    )
                    if any_above:
                        above_high_series[(cname, unit)] = pts
                    if any_below:
                        below_low_series[(cname, unit)] = pts

                # Optional view: the relative deviation on a logarithmic
                # y-axis. Useful when deviations span orders of magnitude
                # (the linear % view is dominated by markers with small
                # reference limits). Only violating measurements are shown,
                # as their deviation magnitude — a log axis has no sign.
                show_log_scale = st.toggle(
                    t("chart_log_scale_toggle"),
                    value=False,
                    key=f"deviation_log_scale_{patient.id}",
                )
                if show_log_scale:
                    st.caption(t("chart_log_scale_caption"))

                def build_deviation_chart(series_dict, above=True):
                    rows = []
                    for (cname, unit), pts in series_dict.items():
                        # Use the most recent point that actually carries the
                        # relevant bound as the 0% reference line — a marker's
                        # latest measurement may lack a reference range even
                        # though an earlier one had it.
                        boundary = None
                        for (_, _, ref_low, ref_high) in reversed(pts):
                            b = ref_high if above else ref_low
                            if b is not None:
                                boundary = b
                                break
                        marker_label = f"{cname} ({unit})" if unit else cname
                        for report_date, value, ref_low, ref_high in pts:
                            if show_log_scale:
                                # Log mode plots the deviation magnitude of
                                # violating measurements only: a log axis has
                                # no sign, and within-range points have no
                                # violation to plot.
                                if boundary in (None, 0) or value is None:
                                    continue
                                pct = ((value - boundary) / boundary) * 100
                                if not (pct > 0 if above else pct < 0):
                                    continue
                                rows.append({
                                    t("col_date"): report_date,
                                    t("col_marker"): marker_label,
                                    "pct": abs(pct),
                                })
                            else:
                                pct = ((value - boundary) / boundary) * 100 if boundary not in (None, 0) else 0.0
                                rows.append({
                                    t("col_date"): report_date,
                                    t("col_marker"): marker_label,
                                    # Neutral internal column name — the visible
                                    # axis title comes from t("chart_relative_ylabel").
                                    "pct": pct,
                                })
                    df = pd.DataFrame(rows)
                    if df.empty:
                        return None
                    fig = px.line(
                        df, x=t("col_date"), y="pct",
                        color=t("col_marker"), markers=True,
                    )
                    if show_log_scale:
                        # tickformat renders plain numbers (200, 50, 0.5)
                        # instead of Plotly's default mantissa+superscript log
                        # ticks ("2·10²"), which read like a scrambled axis.
                        fig.update_layout(
                            yaxis_type="log",
                            yaxis_tickformat=".4~g",
                            yaxis_title=t("chart_relative_ylabel"),
                            xaxis_title="",
                            legend_title="",
                        )
                        # The reference boundary (0%) sits at the bottom edge
                        # of a log axis (log(0) = -inf), so mark it with an
                        # annotation instead of a line.
                        fig.add_annotation(
                            x=0.98, y=0.02, xref="paper", yref="paper",
                            text=t("chart_ref_boundary"), showarrow=False,
                            font=dict(size=10, color="gray"),
                        )
                    else:
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text=t("chart_ref_boundary"))
                        fig.update_layout(
                            yaxis_title=t("chart_relative_ylabel"),
                            xaxis_title="",
                            legend_title="",
                        )
                    return fig

                if above_high_series:
                    st.subheader(t("chart_above_high"))
                    fig = build_deviation_chart(above_high_series, above=True)
                    if fig:
                        st.plotly_chart(fig, width='stretch')

                if below_low_series:
                    st.subheader(t("chart_below_low"))
                    fig = build_deviation_chart(below_low_series, above=False)
                    if fig:
                        st.plotly_chart(fig, width='stretch')

        st.subheader(t("subheader_panel_types"))
        panel_data = [
            {t("col_date"): r.report_date.strftime("%d.%m.%Y") if r.report_date else "?", t("col_panel"): r.panel_type or t("panel_unknown")}
            for r in patient_reports
        ]
        st.dataframe(pd.DataFrame(panel_data), width='stretch', hide_index=True)
