"""Cached read-only queries for the UI (performance layer).

Every helper here replaces a query the tabs used to issue inline on *every*
Streamlit rerun. Streamlit re-executes the whole script on each interaction (and
the LLM-Jobs/Chat fragments re-run every 2 s), so a patient with a few thousand
measurements paid for the same SELECTs over and over — a measurable share of the
"janky" feeling.

Two rules make the caching safe:

1. **Plain data only.** Nothing here returns ORM objects. A cached ORM object
   would outlive its ``Session`` and raise on attribute access (or silently
   serve detached, stale rows). Everything comes back as ``NamedTuple`` rows,
   tuples or dicts, which stay valid after the session closes. Field names match
   the model attributes the UI already reads, so call sites barely change.
2. **Explicit invalidation.** Each cached helper takes a *cache token* as its
   first argument, built by :func:`cache_token`. The token changes as soon as
   anything in the database changes, so the next rerun re-queries.

Cache token
-----------
``database.commit_version()`` counts every commit made through an ORM session in
this process (UI code *and* the LLM job worker threads), which covers all writes
the app itself performs. ``PRAGMA data_version`` additionally changes when a
*different* process writes the file — measured to be unreliable on its own for
commits made through the same session/connection pool, which is why both are
combined.
"""

from collections import namedtuple

import streamlit as st
from sqlalchemy import func, text

from database import DataPoint, Report, RiskFlag, commit_version, get_session

# Lightweight, ORM-free stand-ins for the rows the tabs render.
ReportRow = namedtuple("ReportRow", "id filename report_date panel_type")
ValueRow = namedtuple(
    "ValueRow",
    "id report_id canonical_name original_name value unit ref_low ref_high is_abnormal is_disabled",
)
SeriesRow = namedtuple(
    "SeriesRow", "report_date value unit ref_low ref_high is_abnormal",
)
RiskRow = namedtuple(
    "RiskRow", "rank severity category description explanation report_date canonical_name",
)
MappingRow = namedtuple(
    "MappingRow",
    "id report_date filename original_name canonical_name value unit ref_low ref_high",
)


def cache_token() -> tuple:
    """Token that changes whenever the database changed.

    Pass it as the first argument of every cached helper below. Cheap: one
    in-process counter read plus one PRAGMA round-trip.
    """
    with get_session() as s:
        data_version = s.execute(text("PRAGMA data_version")).scalar()
    return (commit_version(), data_version)


# ── Reports ───────────────────────────────────────────────────────────────────

@st.cache_data(max_entries=32, show_spinner=False)
def _reports(token, patient_id: int):
    with get_session() as s:
        return [
            ReportRow(r.id, r.filename, r.report_date, r.panel_type)
            for r in s.query(Report)
            .filter(Report.patient_id == patient_id)
            .order_by(Report.report_date)
            .all()
        ]


def patient_reports(patient_id: int) -> list:
    """All reports of a patient, oldest first."""
    return _reports(cache_token(), patient_id)


@st.cache_data(max_entries=8, show_spinner=False)
def _report_stats(token):
    with get_session() as s:
        return {
            pid: (count, latest)
            for pid, count, latest in s.query(
                Report.patient_id, func.count(Report.id), func.max(Report.report_date)
            )
            .filter(Report.patient_id.isnot(None))
            .group_by(Report.patient_id)
            .all()
        }


def report_stats() -> dict:
    """patient_id → (report count, latest report date) for the sidebar labels."""
    return _report_stats(cache_token())


@st.cache_data(max_entries=32, show_spinner=False)
def _date_range(token, patient_id: int):
    with get_session() as s:
        row = (
            s.query(func.min(Report.report_date), func.max(Report.report_date))
            .filter(Report.patient_id == patient_id)
            .filter(Report.report_date.isnot(None))
            .first()
        )
    return (row[0], row[1]) if row else (None, None)


def patient_date_range(patient_id: int):
    """(earliest, latest) report date of a patient."""
    return _date_range(cache_token(), patient_id)


@st.cache_data(max_entries=32, show_spinner=False)
def _report_dates(token, patient_id: int):
    with get_session() as s:
        return [
            r[0]
            for r in s.query(Report.report_date)
            .filter(Report.patient_id == patient_id, Report.report_date.isnot(None))
            .distinct()
            .order_by(Report.report_date)
            .all()
        ]


def report_dates(patient_id: int) -> list:
    """Distinct non-null report dates of a patient, ascending."""
    return _report_dates(cache_token(), patient_id)


# ── Biomarker option lists ────────────────────────────────────────────────────

@st.cache_data(max_entries=32, show_spinner=False)
def _biomarkers(token, patient_id: int, include_disabled: bool):
    with get_session() as s:
        q = s.query(DataPoint.canonical_name, DataPoint.unit).join(Report).filter(
            Report.patient_id == patient_id, DataPoint.value.isnot(None)
        )
        if not include_disabled:
            q = q.filter(DataPoint.is_disabled == False)  # noqa: E712
        return q.distinct().order_by(DataPoint.canonical_name).all()


def biomarkers(patient_id: int, include_disabled: bool = False) -> list:
    """(canonical_name, unit) pairs a patient has measurements for."""
    return _biomarkers(cache_token(), patient_id, include_disabled)


@st.cache_data(max_entries=32, show_spinner=False)
def _out_of_range(token, patient_id: int, since):
    with get_session() as s:
        q = (
            s.query(DataPoint.canonical_name, DataPoint.unit)
            .join(Report)
            .filter(
                Report.patient_id == patient_id,
                DataPoint.value.isnot(None),
                DataPoint.is_disabled == False,  # noqa: E712
                ((DataPoint.ref_high.isnot(None)) & (DataPoint.value > DataPoint.ref_high))
                | ((DataPoint.ref_low.isnot(None)) & (DataPoint.value < DataPoint.ref_low)),
            )
        )
        if since is not None:
            q = q.filter(Report.report_date >= since)
        return set(q.distinct().all())


def out_of_range_biomarkers(patient_id: int, since) -> set:
    """(canonical_name, unit) pairs with at least one out-of-range value since `since`."""
    return _out_of_range(cache_token(), patient_id, since)


# ── Measurements ──────────────────────────────────────────────────────────────

@st.cache_data(max_entries=32, show_spinner=False)
def _patient_values(token, patient_id: int):
    with get_session() as s:
        return [
            ValueRow(dp.id, dp.report_id, dp.canonical_name, dp.original_name, dp.value,
                     dp.unit, dp.ref_low, dp.ref_high, dp.is_abnormal, dp.is_disabled)
            for dp in s.query(DataPoint)
            .join(Report)
            .filter(Report.patient_id == patient_id, DataPoint.is_disabled == False)  # noqa: E712
            .all()
        ]


def patient_values(patient_id: int) -> list:
    """Every enabled measurement of a patient (Overview metric cards)."""
    return _patient_values(cache_token(), patient_id)


@st.cache_data(max_entries=32, show_spinner=False)
def _report_values(token, report_id: int):
    with get_session() as s:
        return [
            ValueRow(dp.id, dp.report_id, dp.canonical_name, dp.original_name, dp.value,
                     dp.unit, dp.ref_low, dp.ref_high, dp.is_abnormal, dp.is_disabled)
            for dp in s.query(DataPoint)
            .filter(DataPoint.report_id == report_id)
            .order_by(DataPoint.canonical_name)
            .all()
        ]


def report_values(report_id: int) -> list:
    """All measurements of one report, ordered by marker (Detail tab).

    Deliberately includes disabled values — the Detail tab is where they are
    re-enabled.
    """
    return _report_values(cache_token(), report_id)


@st.cache_data(max_entries=256, show_spinner=False)
def _series(token, patient_id: int, cname, unit):
    with get_session() as s:
        return [
            SeriesRow(r.report_date, r.value, r.unit, r.ref_low, r.ref_high,
                      r.is_abnormal)
            for r in s.query(Report.report_date, DataPoint.value, DataPoint.unit,
                             DataPoint.ref_low, DataPoint.ref_high, DataPoint.is_abnormal)
            .join(DataPoint, DataPoint.report_id == Report.id)
            .filter(
                Report.patient_id == patient_id,
                DataPoint.canonical_name == cname,
                DataPoint.unit == unit,
                DataPoint.value.isnot(None),
                DataPoint.is_disabled == False,  # noqa: E712
            )
            .order_by(Report.report_date)
            .all()
        ]


def biomarker_series(patient_id: int, cname, unit) -> list:
    """One marker's time series (oldest first) — Trends and Overview charts."""
    return _series(cache_token(), patient_id, cname, unit)


@st.cache_data(max_entries=32, show_spinner=False)
def _matrix_rows(token, patient_id: int):
    """Flat (marker, unit, date, value, disabled) rows for the All-Values matrix.

    Also yields the latest reference range per (marker, unit): walking the rows
    in date order and overwriting gives the newest range in one query instead of
    one query per marker (the old N+1 loop).
    """
    with get_session() as s:
        rows = (
            s.query(DataPoint.canonical_name, DataPoint.unit, Report.report_date,
                    DataPoint.value, DataPoint.is_disabled,
                    DataPoint.ref_low, DataPoint.ref_high)
            .join(Report)
            .filter(Report.patient_id == patient_id, DataPoint.value.isnot(None))
            .order_by(Report.report_date)
            .all()
        )
    lookup = {}
    disabled_lookup = {}
    ref_lookup = {}
    for cname, unit, rdate, value, is_dis, ref_low, ref_high in rows:
        key = (cname, unit, rdate)
        lookup[key] = value
        disabled_lookup[key] = bool(is_dis)
        ref_lookup[(cname, unit)] = (ref_low, ref_high)
    return lookup, disabled_lookup, ref_lookup


def measurement_matrix(patient_id: int):
    """(value lookup, disabled lookup, latest ref-range lookup) for All-Values."""
    return _matrix_rows(cache_token(), patient_id)


@st.cache_data(max_entries=32, show_spinner=False)
def _mapping_rows(token, patient_id: int, report_id):
    with get_session() as s:
        q = (
            s.query(DataPoint.id, Report.report_date, Report.filename,
                    DataPoint.original_name, DataPoint.canonical_name, DataPoint.value,
                    DataPoint.unit, DataPoint.ref_low, DataPoint.ref_high)
            .join(Report, DataPoint.report_id == Report.id)
            .filter(Report.patient_id == patient_id)
        )
        if report_id is not None:
            q = q.filter(DataPoint.report_id == report_id)
        return [
            MappingRow(*r)
            for r in q.order_by(DataPoint.canonical_name, Report.report_date).all()
        ]


def mapping_rows(patient_id: int, report_id=None) -> list:
    """Rows of the Mapping tab (all reports, or one when `report_id` is given)."""
    return _mapping_rows(cache_token(), patient_id, report_id)


# ── Risk flags ────────────────────────────────────────────────────────────────

@st.cache_data(max_entries=32, show_spinner=False)
def _risk_flags(token, patient_id: int):
    with get_session() as s:
        return [
            RiskRow(rf.rank, rf.severity, rf.category, rf.description, rf.explanation,
                    rdate, cname)
            for rf, rdate, cname in s.query(
                RiskFlag, Report.report_date, DataPoint.canonical_name
            )
            .join(Report, RiskFlag.report_id == Report.id)
            .outerjoin(DataPoint, RiskFlag.data_point_id == DataPoint.id)
            .filter(Report.patient_id == patient_id)
            .order_by(RiskFlag.rank.asc().nullslast(), RiskFlag.severity.desc())
            .all()
        ]


def risk_flags(patient_id: int) -> list:
    """All risk flags of a patient, ranked (Risks tab and Overview counters)."""
    return _risk_flags(cache_token(), patient_id)
