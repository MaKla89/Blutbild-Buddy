"""Chat tab — "chat with your data" (free-form LLM Q&A over lab values).

Renders the patient chat tab: the LLM receives every non-disabled
measurement of the selected patient within a user-selectable time range
(the same date-range slider as the Trends/Overview tabs; default = the
patient's full range) — measurement date, value, unit, reference range —
plus the patient profile as system-prompt context, then answers free-form
questions with full conversation history.

Data collection and context formatting are plain functions (no Streamlit)
so tests can verify the exact prompt payload without a Streamlit runtime.
"""

from datetime import datetime

import streamlit as st

from database import Report, DataPoint, patient_age_years
from llm_client import _build_client, chat_with_data
from translations import t
from ui.helpers import _time_range_slider


def collect_chat_measurements(session, patient_id, start_date=None, end_date=None):
    """All non-disabled measurements of a patient within [start_date, end_date].

    The window is the chat tab's date-range slider (default: the patient's
    full range). Unlike the summary prompt (which caps per-biomarker history
    via _collect_historical_data), the chat context deliberately includes
    EVERY non-disabled measurement in the window — the whole point is that
    the model can answer arbitrary questions about any value.

    Returns ``(report_date, canonical_name, value, unit, ref_low, ref_high,
    is_abnormal)`` tuples — the same shape the summary pipeline uses —
    oldest first. Excluded on purpose:
      - disabled values (``is_disabled``, same rule as every chart/analysis),
      - null-valued qualitative entries (``value IS NULL`` — "negativ" etc.
        carry no number the model could reason about),
      - reports without a date (no timestamp to place in the window),
      - anything outside the selected date range.
    """
    query = (
        session.query(DataPoint, Report.report_date)
        .join(Report)
        .filter(
            Report.patient_id == patient_id,
            DataPoint.value.isnot(None),
            DataPoint.is_disabled == False,  # noqa: E712
            Report.report_date.isnot(None),
        )
    )
    if start_date is not None:
        query = query.filter(Report.report_date >= start_date)
    if end_date is not None:
        query = query.filter(Report.report_date <= end_date)
    rows = query.order_by(Report.report_date).all()
    return [
        (rd, dp.canonical_name, dp.value, dp.unit, dp.ref_low, dp.ref_high, dp.is_abnormal)
        for dp, rd in rows
    ]


def format_chat_data_context(patient, measurements, lang, range_label=""):
    """Format the patient dataset injected into the chat system prompt.

    Two sections, following the conventions of the summary/interpretation
    prompts:
      - patient profile (age / sex / clinical context — reference ranges are
        age- and sex-dependent, the model needs these to judge values);
      - every measurement grouped per (biomarker, unit), chronological,
        each line ``date: value unit (Ref: low-high) [ABNORMAL]``.
    """
    de = lang == "de"
    parts = []

    # ── Patient profile ─────────────────────────────────────────────────────
    prof_lines = []
    age = patient_age_years(patient)
    if age is not None:
        prof_lines.append(f"{'Alter' if de else 'Age'}: {age}")
    if patient.sex in ("f", "m", "d"):
        sex_labels = {"f": ("weiblich", "female"), "m": ("männlich", "male"), "d": ("divers", "diverse")}
        prof_lines.append(f"{'Geschlecht' if de else 'Sex'}: {sex_labels[patient.sex][0 if de else 1]}")
    if patient.clinical_context:
        prof_lines.append(
            ("Klinischer Kontext (Vorerkrankungen/Medikamente): " if de
             else "Clinical context (conditions/medication): ")
            + str(patient.clinical_context).strip()
        )
    if prof_lines:
        parts.append(("=== PATIENTENPROFIL ===" if de else "=== PATIENT PROFILE ===")
                     + "\n" + "\n".join(prof_lines))

    # ── Measurements grouped per biomarker (chronological) ─────────────────
    groups: dict[tuple[str, str], list] = {}
    for rdate, cname, value, unit, ref_low, ref_high, is_abn in measurements:
        groups.setdefault((cname, unit or ""), []).append(
            (rdate, value, unit, ref_low, ref_high, is_abn)
        )
    blocks = []
    for key in sorted(groups):
        cname, unit = key
        series = sorted(groups[key], key=lambda r: r[0] or datetime.min)
        lines = []
        for rdate, value, u, ref_low, ref_high, is_abn in series:
            date_str = rdate.strftime("%d.%m.%Y") if rdate else "?"
            if ref_low is not None and ref_high is not None:
                ref_str = f" (Ref: {ref_low}-{ref_high} {u or ''})".rstrip()
            elif ref_high is not None:
                ref_str = f" (Ref: < {ref_high} {u or ''})".rstrip()
            elif ref_low is not None:
                ref_str = f" (Ref: > {ref_low} {u or ''})".rstrip()
            else:
                ref_str = ""
            marker = " [ABNORMAL]" if is_abn else ""
            lines.append(f"  - {date_str}: {value} {u or ''}{ref_str}{marker}".rstrip())
        title = f"{cname}{' (' + unit + ')' if unit else ''}"
        blocks.append(f"{title}:\n" + "\n".join(lines))

    span = f" {range_label}" if range_label else ""
    header = (f"=== MESSWERTE{span} "
              "(gruppiert nach Biomarker, chronologisch) ===" if de else
              f"=== MEASUREMENTS{span} "
              "(grouped per biomarker, chronological) ===")
    parts.append(header + "\n" + "\n".join(blocks))
    return "\n\n".join(parts)


def render_chat_tab(session, patient, lang, min_date, max_date, time_range_key):
    """Render the chat tab: time-range slider, transcript, and the input line.

    The date-range slider (shared helper with the Trends/Overview tabs, but a
    chat-scoped value key so the chat window stays independent) selects which
    measurements feed the LLM; the default is the patient's full range. The
    conversation lives in session state keyed by patient id, so switching
    patients never mixes transcripts and a rerun keeps the chat. The data
    context is re-collected on every send, so values disabled, reports
    uploaded, or the slider moved since the last question are all reflected.
    """
    st.subheader(t("subheader_chat"))
    st.caption(t("caption_chat"))

    start_date, end_date = _time_range_slider(
        min_date, max_date, time_range_key, f"slider_chat_{patient.id}",
        t("time_range_label"),
    )

    measurements = collect_chat_measurements(session, patient.id, start_date, end_date)
    if not measurements:
        st.info(t("chat_no_data"))
        return

    # Transparency: show exactly what window the LLM is answering from.
    n_markers = len({(m[1], m[3] or "") for m in measurements})
    st.caption(t(
        "chat_data_context_info",
        n=len(measurements),
        markers=n_markers,
        first=measurements[0][0].strftime("%d.%m.%Y"),
        last=measurements[-1][0].strftime("%d.%m.%Y"),
    ))

    history_key = f"chat_history_{patient.id}"
    history = st.session_state.setdefault(history_key, [])

    if history:
        if st.button(t("btn_clear_chat"), key=f"chat_clear_{patient.id}"):
            st.session_state[history_key] = []
            st.rerun()

    # Transcript (rendered before the input is processed — st.chat_input
    # renders as a fixed bar at the bottom of the page regardless of script
    # position, so the visual order stays natural).
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input(t("chat_input_placeholder"), key=f"chat_input_{patient.id}")
    if user_msg:
        with st.chat_message("user"):
            st.markdown(user_msg)
        history.append({"role": "user", "content": user_msg})

        range_label = (
            f"{start_date.strftime('%d.%m.%Y')} – {end_date.strftime('%d.%m.%Y')}"
            if start_date and end_date else ""
        )
        context = format_chat_data_context(patient, measurements, lang, range_label=range_label)
        with st.chat_message("assistant"):
            with st.spinner(t("chat_thinking")):
                try:
                    answer = chat_with_data(_build_client(), context, history, lang=lang)
                except Exception as e:
                    # Drop the unanswered question so a retry doesn't duplicate
                    # it in the history; the error stays visible in this run.
                    history.pop()
                    st.error(t("error_chat_generation", error=f"{type(e).__name__}: {e}"))
                else:
                    st.markdown(answer)
                    history.append({"role": "assistant", "content": answer})

    st.caption(t("chat_disclaimer"))
