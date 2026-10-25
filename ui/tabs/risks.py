"""Risks tab (moved verbatim from app.py main(), WP3d).

Renders the patient Risks tab: LLM-ranked risk cards and the flags dataframe.
Ranking, urgency classification and explanations come from the LLM assessment
(analyzer.refresh_risk_flags); without an LLM all flags are shown in random
order without explanations. When the LLM was used, only the findings it
selected (up to 10) are displayed — unselected leftovers have no explanation
and are hidden instead of being shown as unexplained cards.
"""

import html

import pandas as pd
import streamlit as st

from translations import t, translate_description, category_label, category_label_plain
from ui import queries


def render_risks_tab(session, patient, patient_reports, lang):
    patient_flags = queries.risk_flags(patient.id)

    if patient_flags:
        # When the LLM ranked the flags, it selected at most 10 findings and
        # wrote an explanation for each of them. Flags it did not select are
        # leftovers without an explanation — show those only in the table
        # below, not as unexplained cards. Without an LLM assessment no flag
        # has an explanation; show all of them as cards.
        llm_ranked = any(rf.explanation is not None for rf in patient_flags)
        shown = (
            [rf for rf in patient_flags if rf.explanation is not None]
            if llm_ranked else patient_flags
        )

        c_rank, c_severity, c_category = t("col_rank"), t("col_severity"), t("col_category")
        c_date, c_marker, c_description = t("col_date"), t("col_marker"), t("col_description")

        def _row(rf):
            date_str = rf.report_date.strftime("%d.%m.%Y") if rf.report_date else "?"
            sev_label = {3: t("severity_critical"), 2: t("severity_warning"),
                         1: t("severity_info")}.get(rf.severity, f"Level {rf.severity}")
            return {
                c_rank: rf.rank if rf.rank is not None else "—",
                c_severity: sev_label,
                c_category: category_label(rf.category),
                c_date: date_str,
                c_marker: rf.canonical_name or "—",
                c_description: rf.description,
            }

        # The card loop needs the flag itself (severity/category/explanation), so
        # it iterates the namedtuples and the display dict side by side instead of
        # smuggling the object through a DataFrame column.
        for rf, row in zip(shown, [_row(rf) for rf in shown]):
            sev = rf.severity if isinstance(rf.severity, int) else 0
            # Severity → modifier class of the shared card CSS (ui/cards.py):
            # critical=red, warning=orange, everything else info=blue.
            sev_class = {3: "critical", 2: "warning"}.get(sev, "info")
            # Severity badge and urgency label are both derived from rf.severity
            # so the border color, badge and urgency always agree on the level.
            sev_badge = {3: t("severity_critical"), 2: t("severity_warning")}.get(sev, t("severity_info"))
            if sev == 3:
                urgency_label = t("urgency_immediate")
            elif sev == 2:
                urgency_label = t("urgency_long_term")
            else:
                urgency_label = t("urgency_info")
            description = (translate_description(rf.description) if lang == "en"
                           else rf.description)
            rank_str = str(row[c_rank])
            # Plain descriptive category (no severity wording) — the severity is
            # carried by the badge/border, so e.g. "critical_value" renders as
            # "Außerhalb Referenzbereich" instead of "Kritischer Wert".
            plain_cat = category_label_plain(rf.category)
            # Escape data-derived values before interpolating into HTML: the
            # card is rendered with unsafe_allow_html=True, so any <, > or & in
            # LLM/DB text (e.g. "Troponin < 0.01") would otherwise corrupt the
            # markup. Translation labels and the formatted date are trusted.
            esc_rank = html.escape(rank_str)
            esc_marker = html.escape(str(row[c_marker]))
            esc_description = html.escape(description) if description else ""
            esc_explanation = html.escape(rf.explanation) if rf.explanation else ""
            # Build the card with opening/closing div in one call (trusted HTML),
            # but keep description/explanation separate without unsafe_allow_html
            inner = (
                f"<strong>{esc_rank}. {sev_badge}</strong> · {urgency_label} · "
                f"{plain_cat} · {row[c_date]} · {esc_marker}"
            )
            if esc_description:
                # The finding is the headline of the card: normal weight, full
                # text color, on its own line below the header.
                inner += f"<br>{esc_description}"
            if esc_explanation:
                # The LLM's clinical interpretation is secondary context: a
                # muted sub-block (smaller font, dimmed via opacity so it stays
                # readable in both light and dark themes) separated from the
                # finding by a thin theme-colored top border.
                inner += (
                    f"<div class='bb-card__explanation'>"
                    f"<em>{t('explanation_label')}: {esc_explanation}</em></div>"
                )
            st.markdown(
                f"<div class='bb-card bb-card--{sev_class}'>{inner}</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader(t("subheader_all_flags"))
        # The table below keeps showing every flag, LLM leftovers included.
        st.dataframe(pd.DataFrame([
            {k: v for k, v in _row(rf).items() if k != c_severity} for rf in patient_flags
        ]), width='stretch', hide_index=True)

        st.caption(t("caption_disclaimer"))

    else:
        st.success(t("msg_no_risk_flags"))
