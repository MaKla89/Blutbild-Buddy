import random
from datetime import timedelta

from sqlalchemy.orm import Session as SASession

from sqlalchemy import func

from config import RISK_CONFIG, get_critical_direction
from database import Patient, Report, DataPoint, RiskFlag, normalize_canonical_name, normalize_unit
from translations import t


def _compute_trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = list(range(len(values)))
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def _get_range_size(ref_low: float, ref_high: float) -> float:
    return ref_high - ref_low


# Cap for "% beyond reference range" shown in descriptions. Narrow ranges
# (e.g. PO2 80–90 mmHg) would otherwise produce absurd figures like 451%.
PCT_BEYOND_CAP = 200.0


def _compute_pct_beyond(value: float, ref_low: float | None, ref_high: float | None) -> float:
    """Return positive % beyond range (negative if within range). Capped at PCT_BEYOND_CAP to avoid absurd values for narrow ranges.

    One-sided bounds are supported (R3-08): with only a high bound the value is
    compared against it, with only a low bound likewise. With no usable bound
    (or zero-width range) returns 0.0.
    """
    if ref_high is not None and value > ref_high:
        if ref_low is not None:
            range_size = _get_range_size(ref_low, ref_high)
            if range_size == 0:
                return 0.0
            return min(((value - ref_high) / range_size) * 100, PCT_BEYOND_CAP)
        # No low bound — fall back to the limit itself as the scale.
        if ref_high > 0:
            return min(((value - ref_high) / ref_high) * 100, PCT_BEYOND_CAP)
        return 0.0
    if ref_low is not None and value < ref_low:
        if ref_high is not None:
            range_size = _get_range_size(ref_low, ref_high)
            if range_size == 0:
                return 0.0
            return min(((ref_low - value) / range_size) * 100, PCT_BEYOND_CAP)
        # No high bound — fall back to the limit itself as the scale.
        if ref_low > 0:
            return min(((ref_low - value) / ref_low) * 100, PCT_BEYOND_CAP)
        return 0.0
    return 0.0


def _pct_deviation_from_limit(value: float, ref_low: float | None, ref_high: float | None) -> float | None:
    """Percentage deviation relative to the violated limit itself (for display).

    "56% unter der Untergrenze" means the value is 56% below the lower bound —
    a number a reader can actually interpret. Naturally bounded at 100% for
    below-range values (a value of 0 is exactly 100% under). Falls back to the
    range-width-based percentage when the limit is zero or negative (e.g. pH),
    where a ratio against the limit is meaningless.

    One-sided bounds are supported (R3-08): with only one bound set, the value
    is compared against that bound and the deviation is relative to it.
    """
    if ref_high is not None and value > ref_high:
        if ref_high > 0:
            return ((value - ref_high) / ref_high) * 100
        if ref_low is not None:
            range_size = _get_range_size(ref_low, ref_high)
            return min(((value - ref_high) / range_size) * 100, PCT_BEYOND_CAP) if range_size > 0 else None
        return None
    if ref_low is not None and value < ref_low:
        if ref_low > 0:
            return ((ref_low - value) / ref_low) * 100
        if ref_high is not None:
            range_size = _get_range_size(ref_low, ref_high)
            return min(((ref_low - value) / range_size) * 100, PCT_BEYOND_CAP) if range_size > 0 else None
        return None
    return None


def _violation_extremity(dp) -> float:
    """Score how extreme a data point's violation is (higher = more extreme).

    Prefers the uncapped limit-relative deviation so that values which both
    saturate the capped range-width percentage still rank by their true
    distance from the violated limit. Falls back to the capped range-width
    percentage when no limit-relative value exists (in-range or zero/negative
    limits without a fallback). Non-abnormal / missing values score -1 so they
    never win a max().
    """
    if dp.value is None or not dp.is_abnormal:
        return -1.0
    dev = _pct_deviation_from_limit(dp.value, dp.ref_low, dp.ref_high)
    if dev is not None:
        return abs(dev)
    return abs(_compute_pct_beyond(dp.value, dp.ref_low, dp.ref_high))


def _detect_critical_values(dps: list) -> list[RiskFlag]:
    """Flag the LATEST report's value as critical — not historical ones.

    A marker that was critical in an older report but has improved (back to
    normal, or below the critical margin) in the most recent report is no
    longer a critical finding. It still surfaces via concerning_trend /
    value_changed at warning level, and _dedup_by_biomarker annotates the
    description with the historical extreme for context.
    """
    flags = []
    if not dps:
        return flags
    margin_pct = RISK_CONFIG["critical_margin_pct"]

    # A report can contain several values of the same biomarker; take the most
    # extreme violation among the latest report's data points.
    last_date = dps[-1].report_date
    best_dp = None
    best_pct = margin_pct
    for dp_data in dps:
        if dp_data.report_date != last_date:
            continue
        dp = dp_data.DataPoint
        # One-sided bounds count too (R3-08): a value beyond the single bound
        # is critical as well. _compute_pct_beyond handles both cases.
        if dp.value is None or not dp.is_abnormal:
            continue
        # Direction-awareness: some markers are only dangerous in one direction
        # (e.g. COHb — elevated = CO poisoning, a low value is not). A violation
        # on the non-meaningful side must not be flagged as critical.
        crit_dir = get_critical_direction(dp.canonical_name)
        above = dp.ref_high is not None and dp.value > dp.ref_high
        below = dp.ref_low is not None and dp.value < dp.ref_low
        if crit_dir == "high" and not above:
            continue
        if crit_dir == "low" and not below:
            continue
        pct_beyond = abs(_compute_pct_beyond(dp.value, dp.ref_low, dp.ref_high))
        if pct_beyond > best_pct:
            best_pct = pct_beyond
            best_dp = dp
        elif pct_beyond == best_pct and best_dp is not None:
            # Tie on the capped percentage — prefer the truly more extreme
            # value (uncapped limit-relative deviation).
            if _violation_extremity(dp) > _violation_extremity(best_dp):
                best_dp = dp

    if best_dp is None:
        return flags

    dp = best_dp
    # Display the deviation relative to the violated limit itself
    # ("56% unter der Untergrenze"), not relative to the range width.
    dev = _pct_deviation_from_limit(dp.value, dp.ref_low, dp.ref_high)
    if dev is None:
        return flags
    above = dp.ref_high is not None and dp.value > dp.ref_high
    direction = t("risk_above_upper_limit") if above else t("risk_below_lower_limit")
    # Format the reference range; one-sided bounds (R3-08) render as
    # "< high" / "> low" instead of a dash range.
    if dp.ref_low is not None and dp.ref_high is not None:
        ref_str = f"{dp.ref_low}\u2013{dp.ref_high}"
    elif dp.ref_high is not None:
        ref_str = f"< {dp.ref_high}"
    else:
        ref_str = f"> {dp.ref_low}"
    flags.append(RiskFlag(
        report_id=getattr(dp, "report_id", None),
        data_point_id=dp.id,
        category="critical_value",
        severity=3,
        description=(
            f"{dp.canonical_name} = {dp.value} {dp.unit or ''} "
            f"liegt {dev:.0f}% {direction} "
            f"(Referenzbereich {ref_str} {dp.unit or ''})."
        ),
    ))
    return flags


def _detect_concerning_trends(dps: list) -> list[RiskFlag]:
    flags = []
    min_points = RISK_CONFIG["concerning_trend_min_points"]
    min_pct = RISK_CONFIG["concerning_trend_pct"]

    if len(dps) < min_points:
        return flags

    values = [dp.DataPoint.value for dp in dps]
    latest_dp = dps[-1].DataPoint
    latest_report_date = dps[-1].report_date
    first_val = values[0]
    last_val = values[-1]

    # Use reference-range-aware percentage change instead of raw value change.
    # A 2% absolute change on a 0-100 scale is less meaningful than a 20%
    # relative change on a 4-8 scale, but being ABOVE the reference range
    # matters more than the raw percentage delta.
    ref_low = latest_dp.ref_low
    ref_high = latest_dp.ref_high

    if first_val == 0:
        if last_val == 0:
            return flags
        pct_change = abs(last_val) * 100
    else:
        pct_change = ((last_val - first_val) / abs(first_val)) * 100

    # Also compute how far the latest value is from the reference range center.
    # One-sided bounds (R3-08): use the single bound as the "center" so a trend
    # moving away from it still registers.
    if ref_low is not None and ref_high is not None:
        ref_center = (ref_low + ref_high) / 2
        ref_range_size = _get_range_size(ref_low, ref_high)
    elif ref_high is not None:
        ref_center = ref_high
        ref_range_size = abs(ref_high)
    elif ref_low is not None:
        ref_center = ref_low
        ref_range_size = abs(ref_low)
    else:
        ref_center = None
        ref_range_size = 0
    pct_from_center = abs(last_val - ref_center) / ref_range_size * 100 if (ref_center and ref_range_size > 0) else 0

    # Flag if either the raw trend is significant OR the latest value is
    # moving away from the reference range center by a meaningful amount
    if abs(pct_change) < min_pct and pct_from_center < min_pct:
        return flags

    slope = _compute_trend(values)
    if abs(slope) < 1e-10:
        return flags

    direction = t("risk_trend_rising") if slope > 0 else t("risk_trend_falling")
    flags.append(RiskFlag(
        report_id=latest_dp.report_id,
        data_point_id=latest_dp.id,
        category="concerning_trend",
        severity=2,
        description=(
            f"{latest_dp.canonical_name} zeigt {direction}: "
            f"von {first_val:.1f} auf {last_val:.1f} {latest_dp.unit or ''} "
            f"({pct_change:+.0f}%)."
        ),
    ))
    return flags


def _detect_value_changes(dps: list, reports: list) -> list[RiskFlag]:
    flags = []
    if len(reports) < 2:
        return flags

    latest_report = reports[-1]
    latest_dp = dps[-1].DataPoint if dps else None
    if not latest_dp or latest_dp.value is None:
        return flags

    if not latest_dp.is_abnormal:
        return flags

    all_dps = dps
    earlier_dps = []
    for rep in reports[:-1]:
        rep_dps = [dp for dp in all_dps if dp.DataPoint.report_id == rep.id]
        earlier_dps.extend(rep_dps)

    if not earlier_dps:
        return flags

    earlier_normal = [dp for dp in earlier_dps if not dp.DataPoint.is_abnormal]
    if not earlier_normal:
        return flags

    latest_report_date = reports[-1].report_date if reports else None
    if latest_report_date is None:
        return flags
    improvement_cutoff = latest_report_date - timedelta(days=RISK_CONFIG["improvement_lookback_years"] * 365)
    recent_abnormal = False
    for dp_data in dps:
        dp = dp_data.DataPoint
        if dp.value is not None and dp.is_abnormal and dp_data.report_date and dp_data.report_date > improvement_cutoff:
            recent_abnormal = True
            break

    if not recent_abnormal:
        return flags

    flags.append(RiskFlag(
        report_id=latest_dp.report_id,
        data_point_id=latest_dp.id,
        category="value_changed",
        severity=2,
        description=(
            f"{latest_dp.canonical_name} ist im aktuellen Bericht abnormal "
            f"({latest_dp.value} {latest_dp.unit or ''}, Referenz: "
            f"{latest_dp.ref_low}\u2013{latest_dp.ref_high} {latest_dp.unit or ''}), "
            f"war in fr\u00fcheren Berichten jedoch im Normbereich."
        ),
    ))
    return flags


def _extract_display_name(description: str) -> str:
    """Extract a short display name from a risk-flag description.

    Fallback for flags whose data point cannot be resolved (e.g. missing_marker
    flags have data_point_id=None). Handles the known formats:
      "PO2 = 34.9 mmHg liegt ..."        -> "PO2"
      "Kreatinin zeigt steigend: ..."    -> "Kreatinin"
      "📋 Ferritin wurde vor 400 Tagen..."-> "Ferritin"
    """
    text = description.lstrip("📋 ").strip()
    if "=" in text:
        return text.split("=")[0].strip()
    for marker in (" zeigt ", " ist im aktuellen Bericht", " wurde vor "):
        if marker in text:
            return text.split(marker)[0].strip()
    return text.split(":")[0].strip() if ":" in text else text


def _dedup_by_biomarker(flags: list[RiskFlag], session: SASession, max_flags: int = 10) -> list[RiskFlag]:
    """Keep at most one flag per biomarker, selecting the most severe one.

    Groups flags by lowered canonical_name and picks the highest-severity flag
    per biomarker (first detected wins ties). Then returns the top N across all
    biomarkers, most severe first — a deterministic pre-selection only; the final
    ranking is done by the LLM in refresh_risk_flags().
    Updates the winning flag's data_point_id to the latest report's data point
    and rebuilds the description to show both worst historical and latest values.
    """
    # Batch-load all referenced data points to avoid N+1 queries
    flag_ids = [f.data_point_id for f in flags if f.data_point_id]
    dp_map: dict[int, DataPoint] = {}
    if flag_ids:
        for dp in session.query(DataPoint).filter(DataPoint.id.in_(flag_ids)).all():
            dp_map[dp.id] = dp

    # Group flags by normalized canonical name (base name, case-insensitive)
    biomarker_flags: dict[str, list[RiskFlag]] = {}

    for flag in flags:
        cname = None
        if flag.data_point_id and flag.data_point_id in dp_map:
            cname = normalize_canonical_name(dp_map[flag.data_point_id].canonical_name)
        if cname is None:
            cname = flag.description[:30].lower()

        if cname not in biomarker_flags:
            biomarker_flags[cname] = []
        biomarker_flags[cname].append(flag)

    # Load all valued data points for the involved patients once and group them
    # in Python by the same normalized name used above. A per-biomarker SQL
    # filter on the full lowered name would never match parenthesized names
    # like "Glukose (nüchtern)" whose normalized key is just "glukose".
    patient_ids: set[int] = set()
    for flag_list in biomarker_flags.values():
        for flag in flag_list:
            if flag.data_point_id and flag.data_point_id in dp_map:
                patient_ids.add(dp_map[flag.data_point_id].report.patient_id)

    dps_by_patient_name: dict[tuple[int, str], list] = {}
    for pid in patient_ids:
        rows = (
            session.query(DataPoint, Report.report_date)
            .join(Report)
            .filter(
                DataPoint.value.isnot(None),
                DataPoint.is_disabled == False,  # noqa: E712
                Report.patient_id == pid,
            )
            .order_by(Report.report_date)
            .all()
        )
        for row in rows:
            cname = row.DataPoint.canonical_name
            if not cname:
                continue
            key = (pid, normalize_canonical_name(cname))
            dps_by_patient_name.setdefault(key, []).append(row)

    winners: list[RiskFlag] = []
    for norm_cname, biomarker_flag_list in biomarker_flags.items():
        # Use the canonical name of the first resolvable data point for display.
        # Parsing the description prefix is unreliable: critical_value
        # descriptions contain no colon, so a naive split(':') returned the
        # whole sentence ("PO2 = 34.9 mmHg liegt 451% unter ...").
        display_name = None
        for flag in biomarker_flag_list:
            if flag.data_point_id and flag.data_point_id in dp_map:
                display_name = dp_map[flag.data_point_id].canonical_name
                break
        if display_name is None:
            display_name = _extract_display_name(biomarker_flag_list[0].description)
        latest_report_date = None
        patient_id = None
        for flag in biomarker_flag_list:
            if flag.data_point_id and flag.data_point_id in dp_map:
                dp = dp_map[flag.data_point_id]
                if patient_id is None:
                    patient_id = dp.report.patient_id
                dp_date = dp.report.report_date
                if latest_report_date is None or (dp_date is not None and dp_date > latest_report_date):
                    latest_report_date = dp_date

        all_dps_for_biomarker = dps_by_patient_name.get((patient_id, norm_cname), []) if patient_id is not None else []

        # Pick the most severe flag for this biomarker (stable: first detected
        # wins ties). No composite scoring — final ranking is done by the LLM.
        best_flag = max(biomarker_flag_list, key=lambda f: f.severity or 0)
        if len(all_dps_for_biomarker) == 0 or latest_report_date is None or patient_id is None:
            winners.append(best_flag)
            continue

        # Find the latest data point and the worst historical violation.
        # If several values share the latest date (multiple measurements in one
        # report), prefer the most extreme violation so the headline matches
        # what _detect_critical_values flagged.
        latest_dp_data = all_dps_for_biomarker[-1]
        last_date = latest_dp_data.report_date
        same_date = [d for d in all_dps_for_biomarker if d.report_date == last_date]
        if len(same_date) > 1:
            latest_dp_data = max(
                same_date, key=lambda dp_data: _violation_extremity(dp_data.DataPoint)
            )
        latest_dp = latest_dp_data.DataPoint

        worst_dp = None
        worst_dp_data = None
        worst_pct = 0.0
        for dp_data in all_dps_for_biomarker:
            dp = dp_data.DataPoint
            # One-sided bounds count too (R3-08) — _compute_pct_beyond handles them.
            if dp.value is not None and dp.is_abnormal:
                if latest_report_date is not None and dp_data.report_date is not None:
                    if dp_data.report_date > latest_report_date:
                        continue
                pct = abs(_compute_pct_beyond(dp.value, dp.ref_low, dp.ref_high))
                if pct > worst_pct:
                    worst_pct = pct
                    worst_dp = dp
                    worst_dp_data = dp_data

        # Update flag to point to the latest data point
        best_flag.data_point_id = latest_dp.id
        best_flag.report_id = latest_dp.report_id

        # Build the enhanced description. The CURRENT value is the headline;
        # the historical extreme is context only — a marker that has improved
        # in the most recent report must not be presented as if it were still
        # at its worst. The reference range is stated exactly once.
        unit = latest_dp.unit or ""
        ref_low = latest_dp.ref_low
        ref_high = latest_dp.ref_high

        def _ref_range_str(lo, hi):
            """Format a reference range; one-sided bounds (R3-08) render as '< high' / '> low'."""
            if lo is not None and hi is not None:
                return f"{lo}\u2013{hi}"
            if hi is not None:
                return f"< {hi}"
            if lo is not None:
                return f"> {lo}"
            return "?"

        def _deviation_str(value, lo, hi):
            """'36% über der Obergrenze' / '56% unter der Untergrenze' / '' (in range)."""
            above = hi is not None and value > hi
            below = lo is not None and value < lo
            if not (above or below):
                return ""
            dev = _pct_deviation_from_limit(value, lo, hi)
            direction = t("risk_above_upper_limit") if above else t("risk_below_lower_limit")
            if dev is not None:
                return f"{dev:.0f}% {direction}"
            # Limit is zero/negative (e.g. pH) — a ratio against it is meaningless.
            return "au\u00dferhalb des Referenzbereichs"

        latest_date = latest_dp_data.report_date.strftime("%d.%m.%Y") if latest_dp_data.report_date else "?"
        range_str = _ref_range_str(ref_low, ref_high)
        latest_dev_str = _deviation_str(latest_dp.value, ref_low, ref_high)
        if latest_dev_str:
            status_str = f"{latest_dp.value} {unit} ({latest_date}, {latest_dev_str})"
        else:
            status_str = f"{latest_dp.value} {unit} ({latest_date}, im Bereich)"

        # Historical extreme as context — only when it is a different data
        # point than the latest one (otherwise it would just repeat the
        # headline).
        context_str = ""
        if worst_dp is not None and worst_dp_data is not None and worst_dp.id != latest_dp.id:
            worst_date = worst_dp_data.report_date.strftime("%d.%m.%Y") if worst_dp_data.report_date else "?"
            worst_dev_str = _deviation_str(worst_dp.value, ref_low, ref_high)
            if worst_dev_str:
                context_str = f"Bisheriger Extremwert: {worst_dp.value} {unit} ({worst_date}, {worst_dev_str})"
            else:
                context_str = f"Bisheriger Extremwert: {worst_dp.value} {unit} ({worst_date})"

        parts = [f"{display_name}: {status_str} (Referenzbereich {range_str} {unit})"]
        if context_str:
            parts.append(context_str)
        best_flag.description = " | ".join(parts)
        winners.append(best_flag)

    # Deterministic pre-selection: most severe first (stable), capped at N.
    # The final ranking is done by the LLM in refresh_risk_flags(); this order
    # only decides which candidates survive when there are more than max_flags.
    top = sorted(winners, key=lambda f: (f.severity or 0), reverse=True)[:max_flags]
    return top


def _get_latest_report_date(session: SASession, patient_id: int) -> datetime | None:
    """Get the most recent report date for a patient."""
    result = session.query(func.max(Report.report_date)).filter(
        Report.patient_id == patient_id
    ).scalar()
    return result


def _get_biomarker_latest(session: SASession, patient_id: int, canonical_name: str) -> tuple[DataPoint | None, datetime | None]:
    """Get the latest data point for a biomarker for a patient.

    Returns (DataPoint, report_date) or (None, None).
    """
    norm_name = normalize_canonical_name(canonical_name)
    result = (
        session.query(DataPoint, Report.report_date)
        .join(Report)
        .filter(
            func.lower(DataPoint.canonical_name) == norm_name,
            DataPoint.value.isnot(None),
            DataPoint.is_disabled == False,  # noqa: E712
            Report.patient_id == patient_id,
        )
        .order_by(Report.report_date.desc())
        .first()
    )
    if result:
        return result.DataPoint, result.report_date
    return None, None


def _get_biomarker_latest_value(session: SASession, patient_id: int, canonical_name: str) -> tuple[float | None, bool | None, datetime | None, float | None, float | None]:
    """Get the latest value, abnormality status, and reference bounds for a biomarker.

    Returns (value, is_abnormal, report_date, ref_low, ref_high) or
    (None, None, None, None, None).
    """
    dp, rpt_date = _get_biomarker_latest(session, patient_id, canonical_name)
    if dp is None:
        return None, None, None, None, None
    return float(dp.value), dp.is_abnormal, rpt_date, dp.ref_low, dp.ref_high


def _detect_cross_biomarker_patterns(
    session: SASession,
    patient_id: int,
    latest_report_date: datetime | None,
) -> list[RiskFlag]:
    """Detect clinically meaningful patterns across multiple biomarkers.

    Uses the cross_biomarker_rules from RISK_CONFIG to identify combinations
    of biomarker values that are more significant than any single marker alone.

    Examples:
      - High CK + normal Troponin → likely muscular, not cardiac (downgrade)
      - High CK + high Troponin → myocardial injury pattern (upgrade)
      - Low TSH + high fT3/fT4 → hyperthyroidism (composite flag)
      - High LDL + low HDL → elevated cardiovascular risk
      - High CRP + high BSG → systemic inflammation
    """
    flags = []
    rules = RISK_CONFIG.get("cross_biomarker_rules", [])

    for rule in rules:
        marker_results = []
        all_have_data = True

        for mdef in rule["markers"]:
            canonical = mdef["canonical"]
            condition = mdef["condition"]

            value, is_abnormal, rpt_date, ref_low, ref_high = _get_biomarker_latest_value(session, patient_id, canonical)

            if value is None:
                # Missing data — treat as "missing" condition match for some rules
                if condition == "normal_or_missing":
                    marker_results.append({"canonical": canonical, "condition": condition, "matched": True, "value": None})
                else:
                    all_have_data = False
                    marker_results.append({"canonical": canonical, "condition": condition, "matched": False, "value": None})
                continue

            # Determine if the condition matches. "high"/"low" are evaluated against
            # the data point's own reference bounds (not 0.0 — lab values are positive).
            matched = False
            if condition == "abnormal":
                matched = bool(is_abnormal)
            elif condition == "normal_or_missing":
                matched = not bool(is_abnormal)
            elif condition == "high":
                matched = ref_high is not None and value > ref_high
            elif condition == "low":
                matched = ref_low is not None and value < ref_low

            marker_results.append({"canonical": canonical, "condition": condition, "matched": matched, "value": value})

        if not all_have_data:
            # If a rule requires specific markers and one is completely missing
            # (not just normal), skip unless the rule explicitly allows it
            for mr in marker_results:
                if mr["value"] is None and mr["condition"] != "normal_or_missing":
                    continue  # This marker was truly absent, check if rule requires it

        # Check if all conditions are met
        all_matched = all(mr["matched"] for mr in marker_results)
        if not all_matched:
            continue

        # Build description based on rule type
        desc = _build_cross_biomarker_description(rule, marker_results)
        if not desc:
            continue

        severity = 2  # default warning
        mod = rule.get("severity_modifier", 0)
        if mod == 1:
            severity = 3  # upgrade to critical
        elif mod == -1:
            severity = 1  # downgrade to info

        flags.append(RiskFlag(
            data_point_id=None,
            category=rule.get("category", "cross_biomarker_correlation"),
            severity=severity,
            description=desc,
        ))

    # Set report_id to the latest report for this patient
    if flags and latest_report_date:
        latest_report = (
            session.query(Report.id)
            .filter(Report.patient_id == patient_id, Report.report_date == latest_report_date)
            .order_by(Report.id.desc())
            .first()
        )
        if latest_report:
            for flag in flags:
                flag.report_id = latest_report.id

    return flags


def _build_cross_biomarker_description(rule: dict, marker_results: list) -> str:
    """Build a human-readable description for a cross-biomarker pattern."""
    template = rule.get("description_template", "")
    rule_name = rule.get("name", "Korrelationsmuster")

    # Gather which markers matched and their values
    matched_info = []
    for mr in marker_results:
        canonical = mr["canonical"]
        value = mr["value"]
        condition = mr["condition"]

        if value is not None:
            matched_info.append(f"{canonical} = {value}")
        else:
            matched_info.append(f"{canonical} = nicht gemessen")

    info_str = ", ".join(matched_info)

    # German descriptions for known patterns
    descriptions = {
        "risk_ck_troponin": (
            f"⚠️ {rule_name}: "
            f"{info_str}. "
            f"Gleichzeitige Erhöhung von CK und Troponin T deutet auf eine "
            f"Muskelschädigung mit möglicher Beteiligung des Herzmuskels hin. "
            f"Klinische Korrelation empfohlen."
        ),
        "risk_ck_isolated": (
            f"ℹ️ {rule_name}: "
            f"{info_str}. "
            f"Isolierte CK-Erhöhung bei normalem Troponin T spricht eher für "
            f"skelettmuskuläre als für kardiale Ursache."
        ),
        "risk_ckmb_troponin": (
            f"⚠️ {rule_name}: "
            f"{info_str}. "
            f"Gleichzeitige Erhöhung von CK-MB und Troponin T ist ein starkes "
            f"Indiz für eine akute Herzmuskelverletzung. Sofortige klinische "
            f"Bewertung empfohlen."
        ),
        "risk_ldl_hdl": (
            f"⚠️ {rule_name}: "
            f"{info_str}. "
            f"Hohes LDL bei niedrigem HDL erhöht das kardiovaskuläre Risiko "
            f"signifikant. Lebensstilmaßnahmen und ggf. medikamentöse Therapie "
            f"überprüfen."
        ),
        "risk_tsh_ft": (
            f"⚠️ {rule_name}: "
            f"{info_str}. "
            f"Gleichzeitige Abweichung von TSH und freien Schilddrüsenhormonen "
            f"spricht für eine klinisch relevante Schilddrüsenfunktionsstörung. "
            f"Klinische Bewertung empfohlen."
        ),
        "risk_crp_bsg": (
            f"⚠️ {rule_name}: "
            f"{info_str}. "
            f"Gleichzeitig erhöhte Entzündungsmarker (CRP und BSG) deuten auf "
            f"eine systemische Entzündungsreaktion hin. Ursache abklären."
        ),
        "risk_crea_egfr": (
            f"⚠️ {rule_name}: "
            f"{info_str}. "
            f"Gleichzeitig erhöhtes Kreatinin und vermindertes eGFR bestätigen "
            f"eine eingeschränkte Nierenfunktion. Staging und Verlaufskontrolle "
            f"empfohlen."
        ),
        "risk_glucose_hba1c": (
            f"⚠️ {rule_name}: "
            f"{info_str}. "
            f"Gleichzeitig erhöhte Glukose und HbA1c bestätigen eine "
            f"gestörte Glukoseregulation bzw. Diabetes. Therapieanpassung "
            f"überprüfen."
        ),
        "risk_iron_tsat": (
            f"ℹ️ {rule_name}: "
            f"{info_str}. "
            f"Gleichzeitig abnormale Eisenwerte und Transferrinsättigung "
            f"können auf einen Eisenmangel oder eine Eisenüberladung hinweisen. "
            f"Klinische Einordnung empfohlen."
        ),
    }

    return descriptions.get(template, f"⚠️ {rule_name}: {info_str}")


def _detect_missing_markers(
    session: SASession,
    patient_id: int,
    latest_report_date: datetime | None,
) -> list[RiskFlag]:
    """Detect biomarkers that should be monitored but haven't been tested recently.

    Checks if required markers (from RISK_CONFIG) have been measured within the
    lookback period. Missing key markers in a patient with known conditions is
    itself a risk indicator.
    """
    flags = []
    required_markers = RISK_CONFIG.get("missing_marker_required_markers", [])
    lookback_days = RISK_CONFIG.get("missing_marker_lookback_days", 180)

    if not required_markers or latest_report_date is None:
        return flags

    for marker_name in required_markers:
        value, is_abnormal, rpt_date, _ref_low, _ref_high = _get_biomarker_latest_value(session, patient_id, marker_name)

        # Marker has never been measured — skip (not all patients get all tests)
        if value is None:
            continue

        # Marker was measured within the lookback period — OK
        if rpt_date is not None and (latest_report_date - rpt_date).days <= lookback_days:
            continue

        # Marker hasn't been tested in too long
        days_since = (latest_report_date - rpt_date).days
        flags.append(RiskFlag(
            data_point_id=None,
            category="missing_marker",
            severity=2,
            description=(
                f"📋 {marker_name} wurde vor {days_since} Tagen gemessen "
                f"(letzter Wert: {value}). Überprüfung empfohlen."
            ),
        ))

    # Set report_id to the latest report for this patient
    if flags and latest_report_date:
        latest_report = (
            session.query(Report.id)
            .filter(Report.patient_id == patient_id, Report.report_date == latest_report_date)
            .order_by(Report.id.desc())
            .first()
        )
        if latest_report:
            for flag in flags:
                flag.report_id = latest_report.id

    return flags


def _analyze_patient(session: SASession, patient: Patient) -> list[RiskFlag]:
    """Run all per-patient detection (pre-dedup) for a single patient.

    Shared by analyze_trends() (all patients) and refresh_patient_risk_flags()
    (one patient), so a single-value toggle no longer re-runs detection for
    every patient in the database.
    """
    flags = []
    reports = (
        session.query(Report)
        .filter(Report.patient_id == patient.id)
        .order_by(Report.report_date)
        .all()
    )
    if not reports:
        return flags

    latest_report = reports[-1]
    latest_report_date = latest_report.report_date

    # 1. Per-biomarker detection (critical values, trends, value changes)
    biomarker_groups = (
        session.query(
            func.lower(DataPoint.canonical_name).label("lname"),
            func.lower(DataPoint.unit).label("lunit"),
        )
        .join(Report)
        .filter(
            Report.patient_id == patient.id,
            DataPoint.value.isnot(None),
            DataPoint.is_disabled == False,  # noqa: E712
        )
        .distinct()
        .all()
    )

    for lname, lunit in biomarker_groups:
        dps = (
            session.query(DataPoint, Report.report_date)
            .join(Report)
            .filter(
                func.lower(DataPoint.canonical_name) == lname,
                func.lower(DataPoint.unit) == lunit,
                DataPoint.value.isnot(None),
                DataPoint.is_disabled == False,  # noqa: E712
                Report.patient_id == patient.id,
            )
            .order_by(Report.report_date.asc())
            .all()
        )

        if not dps:
            continue

        flags.extend(_detect_critical_values(dps))
        flags.extend(_detect_concerning_trends(dps))
        flags.extend(_detect_value_changes(dps, reports))

    # 2. Cross-biomarker pattern detection
    flags.extend(
        _detect_cross_biomarker_patterns(session, patient.id, latest_report_date)
    )

    # 3. Missing marker detection
    flags.extend(
        _detect_missing_markers(session, patient.id, latest_report_date)
    )

    return flags


def analyze_trends(session: SASession) -> list[RiskFlag]:
    all_flags = []
    patients = session.query(Patient).all()

    for patient in patients:
        all_flags.extend(_analyze_patient(session, patient))

    deduped_flags = _dedup_by_biomarker(all_flags, session, RISK_CONFIG["max_flags"])
    return deduped_flags


def refresh_patient_risk_flags(session: SASession, patient_id: int) -> int:
    """Rebuild risk flags for ONE patient only (deterministic, no LLM).

    Used after single-value enable/disable toggles so stale flags caused by
    _dedup_by_biomarker's data_point_id reassignment are removed. Flags keep
    detection severities and get sequential ranks; the next full
    refresh_risk_flags() run may re-rank them with the LLM.
    """
    report_ids = [
        rid
        for (rid,) in session.query(Report.id).filter(Report.patient_id == patient_id).all()
    ]
    if not report_ids:
        return 0

    # Delete this patient's existing flags via their reports. Matching by
    # report (not data_point_id) matters: _dedup_by_biomarker reassigns the
    # surviving flag's data_point_id to the latest data point, so a targeted
    # delete by data_point_id can miss it.
    session.query(RiskFlag).filter(RiskFlag.report_id.in_(report_ids)).delete(
        synchronize_session=False
    )

    # Rebuild ONLY this patient's flags (pre-dedup), then dedup scoped to this
    # patient. The full analyze_trends() would re-run detection for every
    # patient just to rebuild one — and its global cross-patient dedup could
    # suppress this patient's flag behind another patient's same-name flag.
    patient = session.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        return 0
    flags = _dedup_by_biomarker(
        _analyze_patient(session, patient), session, RISK_CONFIG["max_flags"]
    )

    # Deterministic order: severity desc, stable — mirrors the deterministic
    # pre-selection in _dedup_by_biomarker (no LLM, no random shuffle).
    flags = sorted(flags, key=lambda f: (f.severity or 0), reverse=True)
    for rank, flag in enumerate(flags, start=1):
        flag.rank = rank
        session.add(flag)
    session.commit()
    return len(flags)


def refresh_risk_flags(session: SASession, llm=None) -> int:
    """Rebuild all risk flags and let the LLM rank + explain them.

    Detection (analyze_trends) is deterministic and LLM-free. The LLM then
    selects up to 10 findings, ranks them by urgency and classifies each as
    "critical" (immediate action recommended) or "long_term" (observation
    recommended). If the LLM is unavailable or fails, flags keep their
    detection order/severity and are shuffled randomly instead.

    IMPORTANT: the DELETE + INSERT (write lock) happens AFTER the LLM call,
    not before. Holding SQLite's single write lock across a multi-second/
    minute HTTP request would block every other writer (UI reruns, progress
    updates) until busy_timeout expires → "database is locked" crash.
    """
    # Detection + LLM ranking are read-only — no write lock held during the
    # (slow) LLM HTTP call. analyze_trends creates in-memory RiskFlag objects
    # and does not persist them, so nothing reads the old rows in between.
    flags = analyze_trends(session)

    ranked: list[RiskFlag] | None = None
    if llm and flags:
        findings_data = [{"category": f.category, "description": f.description} for f in flags]
        from llm_client import assess_risks
        try:
            risks = assess_risks(llm, findings_data)
        except Exception:
            risks = []
        if risks:
            ranked = _apply_llm_ranking(flags, risks)

    if ranked is None:
        # No LLM assessment: keep detection severities, random order.
        ranked = list(flags)
        random.shuffle(ranked)

    from llm_client import is_job_aborted
    if is_job_aborted():
        # User aborted: skip the write phase so old flags stay untouched.
        return 0

    # Only now take the write lock — for as short a time as possible.
    session.query(RiskFlag).delete(synchronize_session=False)
    for rank, flag in enumerate(ranked, start=1):
        flag.rank = rank
        session.add(flag)
    session.commit()
    return len(ranked)


def _apply_llm_ranking(flags: list[RiskFlag], risks: list[dict]) -> list[RiskFlag]:
    """Order flags by LLM rank, apply urgency-based severity and explanations.

    `risks` is the LLM's ranked list (rank 1 first) with 1-based indices into
    `flags`. Flags not referenced by the LLM are appended after the ranked
    ones in their original order.
    """
    by_index = {i: f for i, f in enumerate(flags, start=1)}
    ranked: list[RiskFlag] = []
    used: set[int] = set()
    for risk in risks:
        idx = risk.get("index")
        if not isinstance(idx, int) or idx in used or idx not in by_index:
            continue
        flag = by_index[idx]
        urgency = risk.get("urgency")
        if urgency == "critical":
            flag.severity = 3
        elif urgency == "long_term":
            # Long-term observation is explicitly *not* critical.
            if (flag.severity or 0) >= 3:
                flag.severity = 2
        flag.explanation = risk.get("explanation")
        ranked.append(flag)
        used.add(idx)

    for i, flag in enumerate(flags, start=1):
        if i not in used:
            ranked.append(flag)
    return ranked
