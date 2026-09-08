from datetime import datetime
import json
import logging
import re

from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, ForeignKey, Boolean, inspect, text, func, select
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session as SASession

import config as _config

_logger = logging.getLogger(__name__)

_base = declarative_base()
_engine = None
_SessionLocal = None
# DB path the current engine was created for. Streamlit re-executes app.py
# (and thus init_db()) on every rerun; when the path is unchanged we can skip
# engine recreation and all migration checks entirely.
_engine_db_path = None


def _ensure_engine():
    """Ensure the SQLAlchemy engine is created once and reused."""
    global _engine
    if _engine is None:
        from sqlalchemy import event

        _engine = create_engine(f"sqlite:///{_config.DB_PATH}", echo=False, connect_args={"check_same_thread": False})

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return _engine


def _ensure_session_local():
    """Ensure the sessionmaker is bound to the shared engine."""
    global _SessionLocal
    engine = _ensure_engine()
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal


def normalize_patient_name(raw: str) -> str:
    """Normalize a patient name to 'LastName, FirstName' canonical form.

    Handles formats like:
      'Doe J.' → 'Doe, J.'
      'J., Doe' → 'J., Doe'
      'Doe_J.' → 'Doe, J.'
      'Doe J' → 'Doe, J'
    """
    if not raw:
        return raw

    # Replace underscores with spaces
    name = raw.replace("_", " ")

    # Replace non-breaking spaces and collapse multiple spaces
    name = re.sub(r"\s+", " ", name).strip()

    # Already in "Last, First" format (contains a comma)
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        last_name = parts[0]
        first_name = parts[1] if len(parts) > 1 else ""
        return f"{last_name}, {first_name}".strip()

    # Split by space — last word is last name, rest is first name
    parts = name.split()
    if len(parts) == 1:
        return f"{parts[0]}, "
    if len(parts) == 2:
        return f"{parts[1]}, {parts[0]}"
    # More than 2 words: last word = last name, rest = first name
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def _get_engine():
    """Return the shared engine instance. Created once, reused."""
    return _ensure_engine()


def _get_session_local():
    """Return the shared sessionmaker. Created once, reused."""
    return _ensure_session_local()


Base = _base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    name_normalized = Column(String(255), nullable=False, index=True, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    favorites = Column(Text, nullable=True)

    reports = relationship("Report", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient {self.id}: {self.name}>"


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True)
    report_date = Column(DateTime, nullable=True)
    panel_type = Column(String(255), nullable=True)
    raw_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="reports")
    data_points = relationship("DataPoint", back_populates="report", cascade="all, delete-orphan")
    risk_flags = relationship("RiskFlag", back_populates="report", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Report {self.id}: {self.filename}>"


class DataPoint(Base):
    __tablename__ = "data_points"

    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    canonical_name = Column(String(255), nullable=False, index=True)
    original_name = Column(String(255), nullable=True)
    value = Column(Float, nullable=True)
    unit = Column(String(100), nullable=True)
    ref_low = Column(Float, nullable=True)
    ref_high = Column(Float, nullable=True)
    is_abnormal = Column(Boolean, default=False)
    # Disabled values are excluded from all charts and LLM analyses
    # (e.g. measurements distorted by medical procedures).
    is_disabled = Column(Boolean, default=False)

    report = relationship("Report", back_populates="data_points")
    risk_flags = relationship("RiskFlag", back_populates="data_point", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DataPoint {self.canonical_name}: {self.value} {self.unit}>"


class RiskFlag(Base):
    __tablename__ = "risk_flags"

    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    data_point_id = Column(Integer, ForeignKey("data_points.id"), nullable=True)
    category = Column(String(50), nullable=False)  # out-of-range, trending, volatility, missing
    severity = Column(Integer, default=1)  # 1=info, 2=warning, 3=critical
    criticality_score = Column(Float, nullable=True)  # legacy — no longer written, kept for old DBs
    rank = Column(Integer, nullable=True)  # LLM-assigned urgency rank (1 = most urgent)
    description = Column(Text, nullable=False)
    explanation = Column(Text, nullable=True)

    report = relationship("Report", back_populates="risk_flags")
    data_point = relationship("DataPoint", back_populates="risk_flags")

    def __repr__(self):
        return f"<RiskFlag {self.category} severity={self.severity}>"


class CustomBiomarker(Base):
    __tablename__ = "custom_biomarkers"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True, index=True)

    def __repr__(self):
        return f"<CustomBiomarker {self.name}>"


class PatientSummary(Base):
    __tablename__ = "patient_summaries"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=True)
    summary_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PatientSummary patient_id={self.patient_id} created_at={self.created_at}>"

class BiomarkerInfo(Base):
    __tablename__ = "biomarker_infos"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    canonical_name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)  # General biomarker explanation from LLM
    interpretation = Column(Text, nullable=True)  # Patient-specific interpretation from LLM
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<BiomarkerInfo patient_id={self.patient_id} name={self.canonical_name}>"


class BiomarkerRegistry(Base):
    """Active biomarker registry — the single source of truth at runtime.

    Seeded from config._CANONICAL_BIOMARKERS on init (fresh installs and
    new baseline entries), then mutable: accepted LLM validation proposals
    and manual edits update rows here instead of touching config.py or
    historical data_points.
    """
    __tablename__ = "biomarker_registry"

    id = Column(Integer, primary_key=True)
    canonical_name = Column(String(255), nullable=False, unique=True, index=True)
    units = Column(Text, nullable=False, default="[]")  # JSON list of str
    aliases = Column(Text, nullable=False, default="[]")  # JSON list of str
    ref_low = Column(Float, nullable=True)
    ref_high = Column(Float, nullable=True)
    ref_unit = Column(String(100), nullable=True)
    source = Column(String(50), nullable=False, default="seed")  # seed | llm_accepted | manual
    updated_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<BiomarkerRegistry {self.canonical_name}>"


class LlmJob(Base):
    """A background LLM job (see llm_jobs.py).

    Jobs run in worker threads OUTSIDE the Streamlit script lifecycle, so a
    page refresh or close never loses progress. Status/progress/result are
    persisted here; the UI polls this table via an auto-refreshing fragment.
    """
    __tablename__ = "llm_jobs"

    id = Column(Integer, primary_key=True)
    kind = Column(String(50), nullable=False, index=True)  # process_pdfs | refresh_risks | ...
    label = Column(String(255), nullable=False)  # human-readable description (translated at submit time)
    status = Column(String(20), nullable=False, default="pending", index=True)  # pending|running|done|error|interrupted
    progress_current = Column(Integer, nullable=True)
    progress_total = Column(Integer, nullable=True)
    progress_message = Column(Text, nullable=True)  # human-readable current step (translated at update time)
    params_json = Column(Text, nullable=True)  # job parameters (JSON): patient_id, patient name, ...
    result_text = Column(Text, nullable=True)  # translated summary on success
    error = Column(Text, nullable=True)
    aborted = Column(Boolean, nullable=False, default=False)  # user-initiated abort (LLM-Jobs tab)
    config_json = Column(Text, nullable=True)  # LLM/render settings snapshot (JSON) captured at submit time
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<LlmJob {self.id} {self.kind} {self.status}>"


class DataQualityFinding(Base):
    """One finding from the data-quality check (see llm_client.run_biomarker_quality_check / run_duplicate_check).

    Unlike LLM registry suggestions (stored per patient+biomarker in
    BiomarkerInfo), a quality finding may target individual DATA POINTS as
    well as registry rows, so it lives in its own table. Each finding carries
    a machine-applicable fix (action + params) that the user can accept from
    the mapping tab; apply_data_quality_finding() validates and executes it.

    status: pending | applied | dismissed
    """
    __tablename__ = "data_quality_findings"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)  # run context (the check runs per selected patient)
    category = Column(String(50), nullable=False)  # ref_range | unit | name | merge | value | missing
    target_type = Column(String(20), nullable=False)  # registry | data_point
    biomarker = Column(String(255), nullable=False)  # canonical name the finding is about
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=True)  # set for data_point findings
    data_point_ids = Column(Text, nullable=True)  # JSON list of DataPoint ids (data_point findings)
    description = Column(Text, nullable=False)  # human-readable explanation (translated at creation)
    action = Column(String(50), nullable=True)  # machine-applicable fix, e.g. set_ref_range | set_unit | disable_value | registry_ref_range | ...
    params_json = Column(Text, nullable=True)  # JSON params for the fix (e.g. {"low":..,"high":..,"unit":..})
    status = Column(String(20), nullable=False, default="pending", index=True)  # pending|applied|dismissed
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<DataQualityFinding {self.id} {self.category}/{self.target_type} {self.status}>"


def init_db():
    """Create tables, run migrations, and seed the biomarker registry.

    Idempotent and cheap on repeat calls: when the engine already points at
    the same DB file (the normal case — Streamlit re-executes app.py, which
    calls init_db(), on every rerun), all work is skipped. A path change
    (tests create several temp DBs per process) forces a full re-init.
    """
    global _engine, _SessionLocal, _engine_db_path
    db_path = str(_config.DB_PATH)
    if _engine is not None and _engine_db_path == db_path:
        return
    _engine = None
    _SessionLocal = None
    engine = _ensure_engine()
    _engine_db_path = db_path
    Base.metadata.create_all(engine)
    _migrate_risk_flags_explanation(engine)
    _migrate_risk_flags_criticality_score(engine)
    _migrate_risk_flags_rank(engine)
    _migrate_patient_favorites(engine)
    _migrate_custom_biomarkers(engine)
    _migrate_patient_summaries(engine)
    _migrate_biomarker_infos(engine)
    _migrate_data_point_disabled(engine)
    _migrate_biomarker_registry(engine)
    _migrate_llm_jobs(engine)
    _migrate_data_quality_findings(engine)
    _migrate_fix_implausible_ref_ranges(engine)
    # Invalidate the registry cache — init_db() may have pointed the engine
    # at a different DB file (tests create several temp DBs per process).
    _registry_cache_invalidate()


def _migrate_risk_flags_explanation(engine):
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("risk_flags")]
    if "explanation" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE risk_flags ADD COLUMN explanation TEXT"))
            conn.commit()


def _migrate_risk_flags_criticality_score(engine):
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("risk_flags")]
    if "criticality_score" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE risk_flags ADD COLUMN criticality_score FLOAT"))
            conn.commit()


def _migrate_risk_flags_rank(engine):
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("risk_flags")]
    if "rank" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE risk_flags ADD COLUMN rank INTEGER"))
            conn.commit()


def _migrate_patient_favorites(engine):
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("patients")]
    if "favorites" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE patients ADD COLUMN favorites TEXT"))
            conn.commit()


def _migrate_custom_biomarkers(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "custom_biomarkers" not in tables:
        CustomBiomarker.__table__.create(engine)


def _migrate_patient_summaries(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "patient_summaries" not in tables:
        PatientSummary.__table__.create(engine)


def _migrate_biomarker_infos(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "biomarker_infos" not in tables:
        BiomarkerInfo.__table__.create(engine)
        return
    # Migrate: drop legacy validation columns (removed with the registry-
    # validation feature) from databases created before that change.
    columns = [col["name"] for col in inspector.get_columns("biomarker_infos")]
    with engine.connect() as conn:
        for col in ("validation_comment", "validated_at", "validation_suggestions"):
            if col in columns:
                conn.execute(text(f"ALTER TABLE biomarker_infos DROP COLUMN {col}"))
        conn.commit()


def _migrate_data_point_disabled(engine):
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("data_points")]
    if "is_disabled" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE data_points ADD COLUMN is_disabled BOOLEAN DEFAULT 0"))
            conn.commit()


def _migrate_biomarker_registry(engine):
    """Create the biomarker_registry table and seed it from config.

    Seeding is idempotent: only canonical names that have no row yet are
    inserted, so customized rows (accepted LLM proposals, manual edits)
    are never overwritten — even if the baseline in config.py changes.
    New baseline entries added to config.py in later versions are picked
    up automatically on the next init.
    """
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "biomarker_registry" not in tables:
        BiomarkerRegistry.__table__.create(engine)

    session = _get_session_local()()
    try:
        existing = {r.canonical_name for r in session.query(BiomarkerRegistry.canonical_name).all()}
        now = datetime.utcnow()
        inserted = 0
        for name, info in _config._CANONICAL_BIOMARKERS.items():
            if name in existing:
                continue
            default_ref = info.get("default_ref") or {}
            session.add(BiomarkerRegistry(
                canonical_name=name,
                units=json.dumps(info.get("units", [])),
                aliases=json.dumps(info.get("aliases", [])),
                ref_low=default_ref.get("low"),
                ref_high=default_ref.get("high"),
                ref_unit=default_ref.get("unit") or None,
                source="seed",
                updated_at=now,
            ))
            inserted += 1
        if inserted:
            session.commit()
    finally:
        session.close()


def _migrate_llm_jobs(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "llm_jobs" not in tables:
        LlmJob.__table__.create(engine)
        return
    columns = [c["name"] for c in inspector.get_columns("llm_jobs")]
    missing = {
        col: ddl for col, ddl in (
            ("progress_message", "ALTER TABLE llm_jobs ADD COLUMN progress_message TEXT"),
            ("params_json", "ALTER TABLE llm_jobs ADD COLUMN params_json TEXT"),
            ("config_json", "ALTER TABLE llm_jobs ADD COLUMN config_json TEXT"),
            ("aborted", "ALTER TABLE llm_jobs ADD COLUMN aborted BOOLEAN NOT NULL DEFAULT 0"),
        ) if col not in columns
    }
    if missing:
        with engine.connect() as conn:
            for ddl in missing.values():
                conn.execute(text(ddl))
            conn.commit()


def _migrate_data_quality_findings(engine):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "data_quality_findings" not in tables:
        DataQualityFinding.__table__.create(engine)


def _migrate_fix_implausible_ref_ranges(engine):
    """One-time cleanup of reference ranges corrupted by an older code version.

    Before the unit-safety guard in ``get_default_ref_range`` and the strict
    matching in ``_find_biomarker_info`` existed, the extractor's fallback
    could stamp a registry default range onto a DataPoint whose unit did not
    match (Testosteron measured in ng/ml receiving the ng/dl default 280–800)
    or whose name was unrelated to the matched entry (COHb/MetHb in %
    receiving the Hämoglobin g/dl range 12–17.5).

    The safe, unambiguous signature of that corruption is a value far BELOW
    the stored lower bound (``value < ref_low/5``, i.e. below 20% of the bound)
    — physiologically impossible for a living patient. Only the too-low
    direction is corrected: a genuine extreme *high* value (e.g. PSA = 100)
    must never be touched.

    The corrupted bounds are reset to NULL so a later backfill or
    re-extraction can fill them with a correct, unit-consistent range;
    ``is_abnormal`` is recomputed from whatever bounds remain. Idempotent:
    once fixed, no row matches the predicate anymore.
    """
    inspector = inspect(engine)
    if "data_points" not in inspector.get_table_names():
        return
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, canonical_name, value, ref_low FROM data_points "
            "WHERE value IS NOT NULL AND ref_low IS NOT NULL AND ref_low > 0 "
            "AND value < ref_low / 5.0"
        )).fetchall()
        if not rows:
            return
        for dp_id, name, value, ref_low in rows:
            _logger.info(
                "Fixing implausible reference range on data_point %s (%s): "
                "value %s < ref_low %s / 5 — resetting bounds to NULL",
                dp_id, name, value, ref_low,
            )
        conn.execute(text(
            "UPDATE data_points SET ref_low = NULL, ref_high = NULL, is_abnormal = 0 "
            "WHERE value IS NOT NULL AND ref_low IS NOT NULL AND ref_low > 0 "
            "AND value < ref_low / 5.0"
        ))
        conn.commit()


# ── Registry loader (cached) ─────────────────────────────────────────────────

_registry_cache_key = None
_registry_cache_value: dict | None = None


def _registry_cache_invalidate():
    """Drop the cached registry. Called by init_db() and every registry write."""
    global _registry_cache_key, _registry_cache_value
    _registry_cache_key = None
    _registry_cache_value = None


def get_registry(session: SASession) -> dict:
    """Load the active biomarker registry from the DB.

    Returns the same shape as config._CANONICAL_BIOMARKERS:
        {name: {"units": [...], "aliases": [...],
                "default_ref": {"low": float|None, "high": float|None, "unit": str}}}

    Cached per DB file (config.DB_PATH) because tests create several temp
    DBs in one process; invalidated by init_db() and registry writes.
    """
    global _registry_cache_key, _registry_cache_value
    db_path = str(_config.DB_PATH)
    if _registry_cache_key == db_path and _registry_cache_value is not None:
        return _registry_cache_value

    registry: dict[str, dict] = {}
    for row in session.query(BiomarkerRegistry).order_by(BiomarkerRegistry.canonical_name).all():
        try:
            units = json.loads(row.units) if row.units else []
        except (json.JSONDecodeError, TypeError):
            units = []
        try:
            aliases = json.loads(row.aliases) if row.aliases else []
        except (json.JSONDecodeError, TypeError):
            aliases = []
        registry[row.canonical_name] = {
            "units": units,
            "aliases": aliases,
            "default_ref": {
                "low": row.ref_low,
                "high": row.ref_high,
                "unit": row.ref_unit or "",
            },
        }

    _registry_cache_key = db_path
    _registry_cache_value = registry
    return registry


def get_registry_names(session: SASession) -> list[str]:
    """Sorted list of canonical names in the active registry."""
    return sorted(get_registry(session).keys())


def set_data_point_disabled(session, dp_id: int, disabled: bool) -> DataPoint | None:
    """Enable/disable a single measurement value.

    Disabled values are excluded from all charts and LLM analyses.
    Risk flags that reference the data point are removed immediately;
    remaining flags are regenerated on the next risk refresh.
    """
    dp = session.get(DataPoint, dp_id)
    if dp is None:
        return None
    dp.is_disabled = bool(disabled)
    if disabled:
        session.query(RiskFlag).filter(RiskFlag.data_point_id == dp_id).delete()
    session.commit()
    return dp


def get_custom_biomarkers(session: SASession) -> list[str]:
    return [cb.name for cb in session.query(CustomBiomarker).order_by(CustomBiomarker.name).all()]


def add_custom_biomarker(session: SASession, name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    existing = session.query(CustomBiomarker).filter(CustomBiomarker.name == name).first()
    if existing:
        return False
    session.add(CustomBiomarker(name=name))
    session.commit()
    return True


def _canonical_base(name: str) -> str:
    """Extract the base name from a canonical biomarker name by stripping the unit suffix in parens.

    'Hämoglobin (Hb)' → 'Hämoglobin'
    'HbA1c (%)' → 'HbA1c'
    'CRP' → 'CRP'
    """
    if "(" in name and name.endswith(")"):
        return name[:name.rfind("(")].strip()
    return name


def normalize_canonical_name(name: str) -> str:
    """Normalize a canonical name for case-insensitive comparison."""
    return _canonical_base(name.strip().lower())


def normalize_unit(unit: str | None) -> str | None:
    """Normalize a unit for case-insensitive comparison."""
    if unit is None:
        return None
    return unit.strip().lower()


def get_unique_biomarker_groups(session: SASession, patient_id: int) -> list[dict]:
    """Get groups of biomarker names that should potentially be merged.

    Groups by (lowered base name, lowered unit) to ensure only truly
    duplicate variants are grouped. Variants with genuinely different
    units (e.g. '%' vs 'mmol/mol' for HbA1c) are kept in separate groups
    and will NOT be merged.

    A merge is only suggested when:
    - The base name (canonical name without unit suffix) matches case-insensitively AND
    - The unit is the same when lowered (only casing differences allowed)

    This means:
    - "HB" (g/dl) and "Hb" (g/dl) → 2 variants, same base + same unit → MERGE suggested
    - "HbA1c (%)" and "HbA1c (%)" (MG/dl) → 2 variants, same base + same unit → MERGE suggested
    - "HbA1c (%)" and "HbA1c (mmol/mol)" → 2 groups, different units → NO MERGE
    - "Glucose" (mg/dl) and "Glucose" (mmol/l) → 2 groups, different units → NO MERGE
    - "Glucose" (mg/dl) and "Glu" (mg/dl) → 2 variants, same base + same unit → MERGE suggested
    """
    # Get all distinct (canonical_name, unit) pairs with counts
    pairs = (
        session.query(
            DataPoint.canonical_name,
            DataPoint.unit,
            func.count(DataPoint.id).label("cnt"),
        )
        .join(Report)
        .filter(Report.patient_id == patient_id)
        .group_by(DataPoint.canonical_name, DataPoint.unit)
        .all()
    )

    # Group by (lowered base name, lowered unit) — only group when both match
    groups: dict[str, dict] = {}
    for canon_name, unit, cnt in pairs:
        base = _canonical_base(canon_name).lower()
        norm_unit = normalize_unit(unit)
        group_key = f"{base}|{norm_unit}"
        if group_key not in groups:
            groups[group_key] = {"base_key": base, "variants": [], "total_count": 0}
        groups[group_key]["variants"].append({
            "name": canon_name,
            "unit": unit,
            "count": cnt,
        })
        groups[group_key]["total_count"] += cnt

    return [g for g in groups.values() if len(g["variants"]) > 1]


def merge_canonical_names(session: SASession, patient_id: int, source_names: list[str], target_name: str) -> int:
    """Merge all data points matching any of source_names (case-insensitive) into target_name.

    Returns total count of updated rows.
    """
    source_lower = [n.strip().lower() for n in source_names if n.strip()]
    if not source_lower:
        return 0

    # Use a subquery to avoid join+update conflict in SQLAlchemy
    subq = (
        select(DataPoint.id)
        .join(Report)
        .where(
            Report.patient_id == patient_id,
            func.lower(DataPoint.canonical_name).in_(source_lower),
        )
    ).scalar_subquery()

    count = (
        session.query(DataPoint)
        .filter(DataPoint.id.in_(subq))
        .update({DataPoint.canonical_name: target_name}, synchronize_session="fetch")
    )
    session.commit()
    return count


def merge_two_biomarkers(session: SASession, patient_id: int, source_name: str, target_name: str) -> int:
    """Merge all data points from source_name (case-insensitive) into target_name.

    Returns count of updated rows.
    """
    source_lower = source_name.strip().lower()

    subq = (
        select(DataPoint.id)
        .join(Report)
        .where(
            Report.patient_id == patient_id,
            func.lower(DataPoint.canonical_name) == source_lower,
        )
    ).scalar_subquery()

    count = (
        session.query(DataPoint)
        .filter(DataPoint.id.in_(subq))
        .update({DataPoint.canonical_name: target_name}, synchronize_session="fetch")
    )
    session.commit()
    return count


def auto_merge_biomarkers(session: SASession, patient_id: int, llm_client=None) -> dict:
    """Perform automatic biomarker merging using the LLM to suggest canonical targets.

    All variant groups are sent to the LLM in a SINGLE call (see
    llm_client.suggest_merge_targets); per-group gaps fall back to
    _fallback_target/_fallback_unit heuristics.

    Returns a dict with:
        - groups: list of {variants, suggested_target} dicts (before merging)
        - merge_plan: list of {"variants", "suggested_target", "suggested_unit"}
        - merged_count: total data points migrated (after merging; 0 here —
          apply_merge_plan() performs the actual migration)
    """
    groups = get_unique_biomarker_groups(session, patient_id)
    if not groups:
        return {"groups": [], "merged_count": 0}

    from llm_client import _build_client, suggest_merge_targets

    if llm_client is None:
        llm_client = _build_client()

    # Active registry (honors accepted LLM proposals) for the canonical list
    registry = get_registry(session)
    canonical_names = sorted(registry.keys())

    # One LLM call for ALL groups; per-group gaps come back as None.
    try:
        llm_results = suggest_merge_targets(llm_client, groups, canonical_names)
    except Exception:
        llm_results = [None] * len(groups)

    merge_plan = []
    for group, suggestion in zip(groups, llm_results):
        units = list({v["unit"] for v in group["variants"] if v["unit"]})
        if suggestion is not None:
            target = suggestion["target"]
            target_unit = suggestion["unit"]
        else:
            # Fallback: use the most common variant's base name matched to canonical list
            target = _fallback_target([v["name"] for v in group["variants"]], canonical_names)
            target_unit = _fallback_unit(units)

        if target and target in canonical_names:
            merge_plan.append({
                "variants": group["variants"],
                "suggested_target": target,
                "suggested_unit": target_unit,
            })

    return {"groups": groups, "merge_plan": merge_plan, "merged_count": 0}


def _fallback_target(variant_names: list[str], canonical_names: list[str] | None = None) -> str:
    """Fallback: try to match variant base names to the canonical list."""
    if canonical_names is None:
        canonical_names = _config.CANONICAL_BIOMARKERS
    for variant in variant_names:
        base = _canonical_base(variant).lower()
        for canonical in canonical_names:
            if _canonical_base(canonical).lower() == base:
                return canonical
    # Last resort: return the first variant as-is
    return variant_names[0] if variant_names else ""


def _fallback_unit(units: list[str]) -> str:
    """Fallback: pick the most common unit or empty string."""
    if not units:
        return ""
    # Return the first non-empty unit
    for u in units:
        if u:
            return u
    return ""


# Units that represent fundamentally different measurement systems for the same biomarker
# If variants contain units from different groups, the merge is rejected
_INCOMPATIBLE_UNIT_GROUPS = [
    {"mg/dl", "mg%", "mg/dl"},
    {"mmol/l", "mmol/L"},
    {"mmol/mol", "%"},       # HbA1c: different systems
    {"IU/l", "IE/l", "U/l"}, # International vs. units
    {"ng/ml", "μg/l"},       # ng/ml = μg/l (same, kept together)
    {"μg/l", "ng/ml", "mcg/l"},
    {"g/dl", "g/100ml"},
    {"mmol/l", "mEq/l"},     # sometimes equivalent, sometimes not — keep separate
]


def _units_compatible(units: list[str]) -> bool:
    """Check whether all given units are compatible (same measurement system).

    Returns True if all units belong to the same compatibility group,
    or if there is only one non-empty unit, or if all units are empty.
    Returns False if units span different incompatible groups.
    """
    normalized = {normalize_unit(u).lower().strip() for u in units if u and u.strip()}
    if len(normalized) <= 1:
        return True

    for group in _INCOMPATIBLE_UNIT_GROUPS:
        group_lower = {g.lower() for g in group}
        has_in_group = bool(normalized & group_lower)
        has_out_group = bool(normalized - group_lower)
        if has_in_group and has_out_group:
            return False
    return True


def apply_merge_plan(session: SASession, patient_id: int, merge_plan: list[dict]) -> int:
    """Apply a merge plan generated by auto_merge_biomarkers.

    Merges both canonical_name and unit for each group.
    Only merges names when they are truly different (case-insensitive).
    Only merges units when they are truly different (case-insensitive).
    Returns total count of updated rows.
    """
    total = 0
    for plan in merge_plan:
        variants = plan["variants"]
        source_units = [v["unit"] for v in variants if v["unit"]]
        target = plan["suggested_target"]
        target_unit = plan.get("suggested_unit", "")

        # Validate unit compatibility before merging
        if source_units and not _units_compatible(source_units):
            # Skip this merge group — variants have incompatible measurement units
            continue

        # Group sources by lowered canonical name — only merge names that differ
        name_groups: dict[str, list[str]] = {}
        for v in variants:
            ln = v["name"].strip().lower()
            if ln not in name_groups:
                name_groups[ln] = []
            name_groups[ln].append(v["name"])

        # Only merge names if there are truly different canonical names
        if len(name_groups) > 1:
            all_sources = [n for names in name_groups.values() for n in names]
            total += merge_canonical_names(session, patient_id, all_sources, target)
        elif len(name_groups) == 1:
            # All names are the same (case-insensitive), update them to target casing
            original_names = list(name_groups.values())[0]
            if target not in original_names:
                total += merge_canonical_names(session, patient_id, original_names, target)

        # Merge units if there are truly different units (case-insensitive)
        # Also merge when original casing differs from target
        unique_units = set()
        all_source_units = []
        for u in source_units:
            lu = u.strip().lower()
            unique_units.add(lu)
            all_source_units.append(u)

        if target_unit and len(unique_units) > 1:
            _merge_units(session, patient_id, all_source_units, target_unit)
            total += len(unique_units)  # count unit merges
        elif target_unit and all_source_units:
            # Check if any source unit differs from target (case-insensitive)
            target_lower = target_unit.strip().lower()
            needs_merge = any(u.strip().lower() != target_lower for u in all_source_units)
            if needs_merge:
                _merge_units(session, patient_id, all_source_units, target_unit)
                total += 1
            else:
                # All normalize to same value as target — still count as merge
                # since original casing may differ (normalization)
                _merge_units(session, patient_id, all_source_units, target_unit)
                total += 1

    return total


def _merge_units(session: SASession, patient_id: int, source_units: list[str], target_unit: str) -> int:
    """Merge units for data points matching source units into target unit."""
    source_lower = [u.strip().lower() for u in source_units if u.strip()]
    if not source_lower:
        return 0
    subq = (
        select(DataPoint.id)
        .join(Report)
        .where(
            Report.patient_id == patient_id,
            func.lower(DataPoint.unit).in_(source_lower),
        )
    ).scalar_subquery()
    count = (
        session.query(DataPoint)
        .filter(DataPoint.id.in_(subq))
        .update({DataPoint.unit: target_unit}, synchronize_session="fetch")
    )
    session.commit()
    return count


def rename_canonical_name(session: SASession, patient_id: int, old_name: str, new_name: str) -> int:
    """Rename canonical_name for all data points of a patient. Returns count of updated rows."""
    count = (
        session.query(DataPoint)
        .join(Report)
        .filter(
            Report.patient_id == patient_id,
            DataPoint.canonical_name == old_name,
        )
        .update({DataPoint.canonical_name: new_name}, synchronize_session="fetch")
    )
    session.commit()
    return count


def merge_patients(session: SASession, source_patient_id: int, target_patient_id: int) -> dict:
    """Merge source patient into target patient.

    Moves all reports (and their data points + risk flags) from source to target,
    re-points the source's LLM summaries and biomarker info onto the target, then
    deletes the source patient row.

    Returns a dict with merge stats: { 'reports_moved': int, 'dps_moved': int,
    'flags_moved': int, 'summaries_moved': int, 'infos_moved': int,
    'source_deleted': bool }
    """
    source = session.query(Patient).filter(Patient.id == source_patient_id).first()
    target = session.query(Patient).filter(Patient.id == target_patient_id).first()
    if not source or not target:
        return {"error": "One or both patients not found"}

    # Count data points and risk flags BEFORE moving reports
    dps_moved = (
        session.query(func.count(DataPoint.id))
        .join(Report)
        .filter(Report.patient_id == source_patient_id)
        .scalar()
    ) or 0

    flags_moved = (
        session.query(func.count(RiskFlag.id))
        .join(Report)
        .filter(Report.patient_id == source_patient_id)
        .scalar()
    ) or 0

    # Move all reports from source to target (in place — report ids are kept,
    # so PatientSummary.report_id references stay valid).
    reports_moved = (
        session.query(Report)
        .filter(Report.patient_id == source_patient_id)
        .update({Report.patient_id: target_patient_id}, synchronize_session="fetch")
    )

    # Move LLM-generated summaries and biomarker info from source to target so
    # they are not orphaned when the source patient row is deleted below. SQLite
    # does not enforce the FKs, so without this the rows would silently point at
    # a non-existent patient (unreachable data).
    summaries_moved = (
        session.query(PatientSummary)
        .filter(PatientSummary.patient_id == source_patient_id)
        .update({PatientSummary.patient_id: target_patient_id}, synchronize_session="fetch")
    )

    # BiomarkerInfo is keyed by (patient_id, canonical_name). Where the target
    # already has a row for the same biomarker, the target's row wins and the
    # source row is dropped; otherwise the source row is re-pointed to target.
    target_info_names = {
        name
        for (name,) in session.query(BiomarkerInfo.canonical_name)
        .filter(BiomarkerInfo.patient_id == target_patient_id)
        .all()
    }
    infos_moved = 0
    for info in (
        session.query(BiomarkerInfo)
        .filter(BiomarkerInfo.patient_id == source_patient_id)
        .all()
    ):
        if info.canonical_name in target_info_names:
            session.delete(info)
        else:
            info.patient_id = target_patient_id
            infos_moved += 1

    # Delete the source patient — cascade will remove its reports, dps, and flags.
    # We already moved those above, so this is safe.
    session.delete(source)
    session.commit()

    return {
        "reports_moved": reports_moved,
        "dps_moved": dps_moved,
        "flags_moved": flags_moved,
        "summaries_moved": summaries_moved,
        "infos_moved": infos_moved,
        "source_deleted": True,
    }


def get_session() -> SASession:
    return _get_session_local()()


def get_or_create_patient(session: SASession, name: str) -> Patient:
    normalized = normalize_patient_name(name)

    # Try to find existing patient by normalized name first
    patient = session.query(Patient).filter(Patient.name_normalized == normalized).first()
    if patient is None:
        patient = Patient(name=name, name_normalized=normalized)
        session.add(patient)
        session.flush()
    return patient


def rename_patient(session: SASession, current_name: str, new_name: str) -> bool:
    """Rename a patient, updating all associated reports and data points."""
    patient = session.query(Patient).filter(Patient.name == current_name).first()
    if not patient:
        return False

    patient.name = new_name
    patient.name_normalized = normalize_patient_name(new_name)
    session.commit()
    return True


def get_latest_patient_summary(session: SASession, patient_id: int):
    """Get the most recent summary for a patient."""
    return (
        session.query(PatientSummary)
        .filter(PatientSummary.patient_id == patient_id)
        .order_by(PatientSummary.created_at.desc())
        .first()
    )


def create_patient_summary(session: SASession, patient_id: int, report_id: int | None, text: str):
    """Store a new patient summary."""
    summary = PatientSummary(
        patient_id=patient_id,
        report_id=report_id,
        summary_text=text,
    )
    session.add(summary)
    session.commit()
    return summary


# ── LlmJob CRUD (background jobs — see llm_jobs.py) ─────────────────────────

def create_llm_job(session: SASession, kind: str, label: str,
                   config: dict | None = None, params: dict | None = None) -> LlmJob:
    """Insert a new job row in 'pending' state and return it.

    ``config`` is the LLM/render settings snapshot (base_url, model, api_key,
    timeout, dpi, batch_size, lang) captured at submit time — workers run
    outside the Streamlit script context and must not read session state.
    ``params`` holds job-specific inputs (patient_id, patient name, ...).
    """
    job = LlmJob(kind=kind, label=label, status="pending",
                 config_json=json.dumps(config) if config else None,
                 params_json=json.dumps(params) if params else None)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_llm_job(session: SASession, job_id: int) -> LlmJob | None:
    return session.query(LlmJob).filter(LlmJob.id == job_id).first()


def list_llm_jobs(session: SASession, limit: int = 20) -> list[LlmJob]:
    """Most recent jobs first (newest id = newest job)."""
    return (
        session.query(LlmJob)
        .order_by(LlmJob.id.desc())
        .limit(limit)
        .all()
    )


def get_active_llm_jobs(session: SASession) -> list[LlmJob]:
    """Jobs in 'pending' or 'running' state."""
    return (
        session.query(LlmJob)
        .filter(LlmJob.status.in_(["pending", "running"]))
        .order_by(LlmJob.id)
        .all()
    )


def clear_finished_llm_jobs(session: SASession) -> int:
    """Delete finished job rows (done/error/interrupted); returns the count.

    User action from the LLM-Jobs tab — keeps the list short. Pending/running
    jobs are never touched.
    """
    result = (
        session.query(LlmJob)
        .filter(LlmJob.status.in_(["done", "error", "interrupted"]))
        .delete(synchronize_session=False)
    )
    session.commit()
    return result


def mark_stale_llm_jobs_interrupted(session: SASession, exclude_ids=None) -> int:
    """Mark pending/running jobs as 'interrupted' (app restart cleanup).

    Called at startup (and on reruns): any job that was active when the
    previous process died can no longer be running, so it is surfaced as
    interrupted instead of spinning forever in the UI. ``exclude_ids`` are
    job ids with a live worker thread in THIS process — they must never be
    marked interrupted (Streamlit re-runs the script on every interaction,
    so this runs while jobs are legitimately running). Returns the number of
    affected rows.
    """
    now = datetime.utcnow()
    q = session.query(LlmJob).filter(LlmJob.status.in_(["pending", "running"]))
    if exclude_ids:
        q = q.filter(~LlmJob.id.in_(list(exclude_ids)))
    result = q.update(
        {LlmJob.status: "interrupted", LlmJob.finished_at: now},
        synchronize_session=False,
    )
    session.commit()
    return result


# ── BiomarkerInfo CRUD ────────────────────────────────────────────────────────

def get_biomarker_info(session: SASession, patient_id: int, canonical_name: str):
    """Get existing BiomarkerInfo for a patient+biomarker, or None."""
    return (
        session.query(BiomarkerInfo)
        .filter_by(patient_id=patient_id, canonical_name=canonical_name)
        .first()
    )


def upsert_biomarker_info(
    session: SASession,
    patient_id: int,
    canonical_name: str,
    description: str | None = None,
    interpretation: str | None = None,
):
    """Insert or update BiomarkerInfo for a patient+biomarker."""
    info = get_biomarker_info(session, patient_id, canonical_name)
    if info is None:
        info = BiomarkerInfo(
            patient_id=patient_id,
            canonical_name=canonical_name,
        )
        session.add(info)
    if description is not None:
        info.description = description
    if interpretation is not None:
        info.interpretation = interpretation
    session.commit()
    return info


def delete_biomarker_info(session: SASession, patient_id: int, canonical_name: str):
    """Delete BiomarkerInfo for a patient+biomarker."""
    info = get_biomarker_info(session, patient_id, canonical_name)
    if info:
        session.delete(info)
        session.commit()


def delete_all_biomarker_infos(session: SASession, patient_id: int):
    """Delete all BiomarkerInfo rows for a patient."""
    session.query(BiomarkerInfo).filter_by(patient_id=patient_id).delete()
    session.commit()


def _parse_ref_range(proposed: str) -> tuple[float | None, float | None, str | None]:
    """Parse a proposed ref range string into (low, high, unit).

    Supported forms (separators: en-dash –, hyphen -, tilde ~):
      "100–200 U/l"  -> (100.0, 200.0, "U/l")
      "< 5.2 mg/dl"  -> (None, 5.2, "mg/dl")
      "> 90 %"       -> (90.0, None, "%")
    Returns (None, None, None) when the string cannot be parsed.
    """
    s = proposed.strip()
    if not s:
        return (None, None, None)
    m = re.match(r"^([<>])\s*([\d.,]+)\s*(.*)$", s)
    if m:
        op, num, unit = m.group(1), m.group(2), m.group(3).strip()
        try:
            val = float(num.replace(",", "."))
        except ValueError:
            return (None, None, None)
        return ((None, val, unit) if op == "<" else (val, None, unit))
    m = re.match(r"^([\d.,]+)\s*[–\-~]\s*([\d.,]+)\s*(.*)$", s)
    if m:
        try:
            low = float(m.group(1).replace(",", "."))
            high = float(m.group(2).replace(",", "."))
        except ValueError:
            return (None, None, None)
        return (low, high, m.group(3).strip() or None)
    return (None, None, None)


def _mutate_registry_row(session: SASession, row: BiomarkerRegistry, field: str, proposed: str) -> tuple[bool, str]:
    """Apply one field change to a registry row (no commit).

    Shared by the manual edit path (update_biomarker_registry) and the
    data-quality fix path (_apply_registry_fix) so both behave identically.
    All validation happens before any mutation, so a failure leaves the row
    unchanged. Returns (ok, detail).
    """
    if field == "ref_range":
        low, high, unit = _parse_ref_range(proposed)
        if low is None and high is None:
            return False, f"unparseable ref range '{proposed}'"
        row.ref_low = low
        row.ref_high = high
        if unit:
            row.ref_unit = unit
    elif field == "unit":
        units = json.loads(row.units) if row.units else []
        if not isinstance(units, list):
            units = []
        # Case-insensitive replace-or-append so re-applying a corrected unit
        # doesn't accumulate duplicates.
        new_units = [u for u in units if str(u).strip().lower() != proposed.lower()]
        new_units.append(proposed)
        row.units = json.dumps(new_units)
    elif field == "name":
        old_name = row.canonical_name
        if proposed.lower() != old_name.lower():
            clash = (
                session.query(BiomarkerRegistry)
                .filter(
                    func.lower(BiomarkerRegistry.canonical_name) == proposed.lower(),
                    BiomarkerRegistry.id != row.id,
                )
                .first()
            )
            if clash is not None:
                return False, f"registry already contains '{proposed}'"
        # Global rename: every data point and biomarker info row across all
        # patients, mirroring the per-patient rename_canonical_name().
        session.query(DataPoint).filter(
            DataPoint.canonical_name == old_name
        ).update({DataPoint.canonical_name: proposed}, synchronize_session="fetch")
        session.query(BiomarkerInfo).filter(
            BiomarkerInfo.canonical_name == old_name
        ).update({BiomarkerInfo.canonical_name: proposed}, synchronize_session="fetch")
        # Keep the old name resolvable as an alias for incoming PDF names.
        aliases = json.loads(row.aliases) if row.aliases else []
        if not isinstance(aliases, list):
            aliases = []
        if old_name.lower() not in [str(a).lower() for a in aliases]:
            aliases.append(old_name)
        row.aliases = json.dumps(aliases)
        row.canonical_name = proposed
    else:
        return False, "invalid field"
    return True, proposed


def _merge_registry_rows(session: SASession, source_name: str, target_row: BiomarkerRegistry) -> tuple[bool, str]:
    """Fold a biomarker name into another registry row (no commit).

    Used by the data-quality "merge" fix (_apply_registry_fix) to resolve
    cross-entry duplicates/typos. Everything named after ``source_name`` is
    migrated onto the target:
      - every DataPoint and BiomarkerInfo row named ``source_name`` is renamed
        to the target (across ALL patients), mirroring the name-rename branch of
        _mutate_registry_row();
      - if a registry row exists for ``source_name``, its units are merged into
        the target's (case-insensitive dedup) and the row is deleted;
      - ``source_name`` is appended to the target's aliases so
        resolve_canonical() still maps incoming PDF names onto the survivor.

    A missing source registry row is NOT an error: the duplicate check feeds the
    LLM the patient's raw data-point names, so a variant often has data points
    but no registry row yet — only the canonical target does. Returns (ok, detail).
    """
    # Migrate data points and info rows globally, source -> target.
    session.query(DataPoint).filter(
        DataPoint.canonical_name == source_name
    ).update({DataPoint.canonical_name: target_row.canonical_name}, synchronize_session="fetch")
    session.query(BiomarkerInfo).filter(
        BiomarkerInfo.canonical_name == source_name
    ).update({BiomarkerInfo.canonical_name: target_row.canonical_name}, synchronize_session="fetch")

    # Merge units into the target (case-insensitive dedup, keep target order).
    def _as_list(raw):
        try:
            val = json.loads(raw) if raw else []
        except (json.JSONDecodeError, TypeError):
            return []
        return val if isinstance(val, list) else []

    source_row = (
        session.query(BiomarkerRegistry)
        .filter(func.lower(BiomarkerRegistry.canonical_name) == source_name.lower())
        .first()
    )
    target_units = _as_list(target_row.units)
    source_units = _as_list(source_row.units) if source_row is not None else []
    have = {str(u).strip().lower() for u in target_units}
    for u in source_units:
        key = str(u).strip().lower()
        if key and key not in have:
            target_units.append(u)
            have.add(key)
    target_row.units = json.dumps(target_units)

    # Keep the source name resolvable as an alias of the survivor.
    aliases = _as_list(target_row.aliases)
    if source_name.lower() not in {str(a).lower() for a in aliases}:
        aliases.append(source_name)
    target_row.aliases = json.dumps(aliases)

    if source_row is not None:
        session.delete(source_row)
    return True, target_row.canonical_name


def update_biomarker_registry(session: SASession, canonical_name: str, field: str, proposed: str) -> dict:
    """Manually edit one field of a registry row (source='manual').

    If the name has no registry row yet (e.g. a custom biomarker), 'unit' and
    'ref_range' edits create the row; 'name' edits still require an existing
    row. Returns {'status': 'applied'|'skipped', 'field': str, 'detail': str}.
    """
    field = str(field).strip()
    proposed = str(proposed).strip()
    if field not in ("name", "unit", "ref_range") or not proposed:
        return {"status": "skipped", "field": field, "detail": "invalid suggestion"}

    row = (
        session.query(BiomarkerRegistry)
        .filter(func.lower(BiomarkerRegistry.canonical_name) == canonical_name.lower())
        .first()
    )
    if row is None:
        if field == "name":
            return {
                "status": "skipped", "field": field,
                "detail": f"registry row not found for '{canonical_name}' — rename requires an existing registry entry",
            }
        # Promote the name to a full registry row so unit/ref_range can be set.
        row = BiomarkerRegistry(
            canonical_name=canonical_name,
            units="[]",
            aliases="[]",
            source="manual",
            updated_at=datetime.utcnow(),
        )
        session.add(row)

    ok, detail = _mutate_registry_row(session, row, field, proposed)
    if not ok:
        return {"status": "skipped", "field": field, "detail": detail}

    row.source = "manual"
    row.updated_at = datetime.utcnow()
    session.commit()
    _registry_cache_invalidate()
    return {"status": "applied", "field": field, "detail": proposed}


# ── Reference Range Backfill ──────────────────────────────────────────────────

def backfill_reference_ranges(session: SASession) -> dict:
    """Backfill missing ref_low/ref_high for DataPoints using registry defaults.

    For each DataPoint where either ref_low or ref_high is None, looks up the
    canonical name in the active DB registry (seeded from config, mutable via
    accepted LLM proposals) using flexible matching:
      1. Exact key match (case-insensitive)
      2. Alias match (case-insensitive)
      3. Partial/substring match for common abbreviations
    
    Applies the default_ref range (with unit conversion if needed).

    Returns a dict with 'updated', 'skipped_unknown', 'skipped_no_range',
    and 'skipped_unconvertible' (DataPoints whose unit differs from the default
    but has no supported conversion).
    """
    import config as _config

    registry = get_registry(session)
    updated = 0
    skipped_unknown = 0
    skipped_no_range = 0
    skipped_unconvertible = 0

    # Find all DataPoints where EITHER ref_low or ref_high is NULL
    missing_dps = (
        session.query(DataPoint)
        .filter(DataPoint.ref_low.is_(None) | DataPoint.ref_high.is_(None))
        .all()
    )

    for dp in missing_dps:
        # Try flexible lookup: exact match, alias match, or partial match
        info = _find_biomarker_info(dp.canonical_name, registry)
        if not info:
            skipped_unknown += 1
            continue

        default_ref = info.get("default_ref")
        if not default_ref:
            skipped_no_range += 1
            continue

        ref_unit = default_ref.get("unit", "")
        ref_low = default_ref.get("low")
        ref_high = default_ref.get("high")

        # Skip biomarkers with no meaningful range (e.g., urine dipsticks)
        if ref_low is None and ref_high is None:
            skipped_no_range += 1
            continue

        # Convert units if the DataPoint's unit differs from the default.
        # Uses the single name-aware converter in config.py. If the conversion is
        # unsupported we skip rather than applying a range in the wrong unit
        # (which would produce false abnormal flags).
        dp_unit = dp.unit.strip().lower() if dp.unit else ""
        if dp_unit and ref_unit and dp_unit != ref_unit.strip().lower():
            converted_low, converted_high = _config._convert_ref_range(
                dp.canonical_name, ref_low, ref_high, ref_unit, dp_unit
            )
            if converted_low is None and converted_high is None:
                skipped_unconvertible += 1
                continue
            ref_low = converted_low
            ref_high = converted_high

        # Only set values that are actually None (preserve existing partial data)
        if dp.ref_low is None and ref_low is not None:
            dp.ref_low = ref_low
        if dp.ref_high is None and ref_high is not None:
            dp.ref_high = ref_high
        # Recompute is_abnormal if at least one bound is set (R3-08:
        # one-sided ranges like "> 90" must flag values below the bound).
        if dp.value is not None and (dp.ref_low is not None or dp.ref_high is not None):
            below = dp.ref_low is not None and dp.value < dp.ref_low
            above = dp.ref_high is not None and dp.value > dp.ref_high
            dp.is_abnormal = below or above
        updated += 1

    if updated > 0:
        session.commit()

    return {
        "updated": updated,
        "skipped_unknown": skipped_unknown,
        "skipped_no_range": skipped_no_range,
        "skipped_unconvertible": skipped_unconvertible,
    }


def _find_biomarker_info(canonical_name: str, biomarkers_dict: dict) -> dict | None:
    """Find biomarker info using flexible matching.

    Tries in order:
      1. Exact key match (case-insensitive)
      2. Alias match (case-insensitive)
      3. Partial/substring match of the name against config keys
      4. Partial/substring match of the name against aliases

    Substring tiers (3 and 4) require BOTH sides to have length >= 3 so that
    short lab abbreviations (e.g. "Cr" for creatinine) cannot accidentally
    match unrelated entries such as "CRP". When a substring tier matches
    multiple candidates, the LONGEST candidate wins (most specific name);
    ties are broken by first-seen order. Tier 3 logs a warning naming all
    ambiguous keys; tier 4 logs a warning when multiple distinct registry
    entries matched.
    """
    name_lower = canonical_name.strip().lower()

    # 1. Exact match
    for key, info in biomarkers_dict.items():
        if key.lower() == name_lower:
            return info

    # 2. Alias match
    for key, info in biomarkers_dict.items():
        for alias in info.get("aliases", []):
            if alias.lower() == name_lower:
                return info

    # 3. Partial match - check if the DB name is a substring of a config key or vice versa.
    # Both sides must be >= 3 chars so short abbreviations ("Cr") can't hit
    # unrelated entries ("CRP"). Collect ALL matches; the longest key wins.
    if len(name_lower) >= 3:
        matches = []  # (key, info), first-seen order
        for key, info in biomarkers_dict.items():
            key_lower = key.lower()
            if len(key_lower) >= 3 and (name_lower in key_lower or key_lower in name_lower):
                matches.append((key, info))
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            best_key, best_info = max(matches, key=lambda m: len(m[0]))
            _logger.warning(
                "Ambiguous substring match for %r against keys %s; using longest key %r",
                canonical_name, [k for k, _ in matches], best_key,
            )
            return best_info

    # 4. Abbreviation match - check if DB name matches any alias abbreviation.
    # Same minimum-length rule as tier 3: both sides >= 3 chars. Collect all
    # matches across all registry entries; the longest alias wins (ties broken
    # by first-seen order).
    if len(name_lower) >= 3:
        matches = []  # (alias, key, info), first-seen order
        for key, info in biomarkers_dict.items():
            for alias in info.get("aliases", []):
                alias_lower = alias.lower()
                if len(alias_lower) >= 3 and (name_lower in alias_lower or alias_lower in name_lower):
                    matches.append((alias, key, info))
        if len(matches) == 1:
            return matches[0][2]
        if len(matches) > 1:
            best_alias, best_key, best_info = max(matches, key=lambda m: len(m[0]))
            if len({k for _, k, _ in matches}) > 1:
                _logger.warning(
                    "Ambiguous alias substring match for %r against aliases %s; using longest alias %r (key %r)",
                    canonical_name, [a for a, _, _ in matches], best_alias, best_key,
                )
            return best_info

    return None


# ── Data Quality Findings ─────────────────────────────────────────────────────

def get_data_quality_findings(
    session: SASession,
    patient_id: int | None = None,
    status: str | None = "pending",
) -> list[DataQualityFinding]:
    """Return data-quality findings, optionally filtered by patient and status."""
    q = session.query(DataQualityFinding)
    if patient_id is not None:
        q = q.filter(DataQualityFinding.patient_id == patient_id)
    if status:
        q = q.filter(DataQualityFinding.status == status)
    return q.order_by(DataQualityFinding.id).all()


def clear_patient_data_quality_findings(session: SASession, patient_id: int) -> int:
    """Delete ALL findings for a patient (used before re-running the check so
    stale findings from a previous run don't linger)."""
    n = (
        session.query(DataQualityFinding)
        .filter(DataQualityFinding.patient_id == patient_id)
        .delete(synchronize_session=False)
    )
    if n:
        session.commit()
    return n


def dismiss_data_quality_finding(session: SASession, finding_id: int) -> bool:
    """Mark a pending finding as dismissed (user rejected the suggested fix)."""
    f = session.get(DataQualityFinding, finding_id)
    if f is None or f.status != "pending":
        return False
    f.status = "dismissed"
    session.commit()
    return True


def clear_resolved_data_quality_findings(session: SASession, patient_id: int | None = None) -> int:
    """Delete applied/dismissed findings (optionally scoped to one patient).

    Pending findings are never touched."""
    q = session.query(DataQualityFinding).filter(
        DataQualityFinding.status.in_(["applied", "dismissed"])
    )
    if patient_id is not None:
        q = q.filter(DataQualityFinding.patient_id == patient_id)
    n = q.delete(synchronize_session=False)
    if n:
        session.commit()
    return n


def _recompute_abnormal(dp: DataPoint) -> None:
    """Recompute is_abnormal from value + bounds (one-sided ranges aware)."""
    if dp.value is None or (dp.ref_low is None and dp.ref_high is None):
        return
    below = dp.ref_low is not None and dp.value < dp.ref_low
    above = dp.ref_high is not None and dp.value > dp.ref_high
    dp.is_abnormal = below or above


def _apply_data_point_fix(session: SASession, finding: DataQualityFinding, params: dict) -> tuple[bool, str]:
    """Apply a data-point-level fix. Returns (ok, detail). No commit.

    Supported actions:
      - set_ref_range: correct the stored ref bounds on the named data points
        (converting to each point's own unit when the proposed range is in a
        different unit), then recompute is_abnormal. The measured value is
        never touched.
      - set_unit: change the stored unit of the named data points, converting
        the measured value and any stored ref bounds with the name-aware
        conversion table; unsupported conversions skip the point. Recomputes
        is_abnormal afterwards.
      - disable_value: mark the measurements disabled so they are excluded from
        charts and analyses; their risk flags are removed immediately.
    """
    action = finding.action
    try:
        ids = json.loads(finding.data_point_ids) if finding.data_point_ids else []
    except (json.JSONDecodeError, TypeError):
        return False, "invalid data_point_ids"
    dps = [session.get(DataPoint, i) for i in ids]
    dps = [d for d in dps if d is not None]
    if not dps:
        return False, "no matching data points (already removed?)"

    if action == "set_ref_range":
        low = params.get("low")
        high = params.get("high")
        unit = str(params.get("unit", "") or "").strip()
        if low is None and high is None:
            return False, "no ref range bounds provided"
        import config as _config
        updated = 0
        for dp in dps:
            c_low, c_high = low, high
            # If the proposed range is in a different unit than this data point,
            # convert it (name-aware). An unsupported conversion skips the point
            # rather than stamping a range in the wrong unit.
            if unit and dp.unit and unit.lower() != dp.unit.strip().lower():
                conv_low, conv_high = _config._convert_ref_range(
                    dp.canonical_name, low, high, unit, dp.unit
                )
                if conv_low is None and conv_high is None:
                    continue
                c_low, c_high = conv_low, conv_high
            if c_low is not None:
                dp.ref_low = float(c_low)
            if c_high is not None:
                dp.ref_high = float(c_high)
            _recompute_abnormal(dp)
            updated += 1
        if not updated:
            return False, "no data point could be updated (unsupported unit conversion?)"
        return True, f"{updated} data point(s) updated"

    if action == "set_unit":
        new_unit = str(params.get("unit", "") or "").strip()
        if not new_unit:
            return False, "no target unit provided"
        import config as _config
        updated = 0
        for dp in dps:
            old_unit = (dp.unit or "").strip()
            if old_unit and old_unit.lower() == new_unit.lower():
                # Same unit — nothing to convert, just normalize the label.
                dp.unit = new_unit
                _recompute_abnormal(dp)
                updated += 1
                continue
            if dp.value is None:
                continue
            # Convert the measured value (name-aware). An unsupported
            # conversion skips the point rather than storing a value in the
            # wrong unit.
            conv_value, _ = _config._convert_ref_range(
                dp.canonical_name, dp.value, dp.value, old_unit, new_unit
            )
            if conv_value is None:
                continue
            dp.value = float(conv_value)
            # Convert stored ref bounds when present (assumed to be in the
            # old unit); an unsupported conversion drops them.
            if dp.ref_low is not None or dp.ref_high is not None:
                c_low, c_high = _config._convert_ref_range(
                    dp.canonical_name, dp.ref_low, dp.ref_high, old_unit, new_unit
                )
                dp.ref_low, dp.ref_high = c_low, c_high
            dp.unit = new_unit
            _recompute_abnormal(dp)
            updated += 1
        if not updated:
            return False, "no data point could be updated (unsupported unit conversion?)"
        return True, f"{updated} data point(s) updated"

    if action == "disable_value":
        for dp in dps:
            dp.is_disabled = True
            session.query(RiskFlag).filter(RiskFlag.data_point_id == dp.id).delete()
        return True, f"{len(dps)} value(s) disabled"

    return False, f"unknown action '{action}'"


def _apply_registry_fix(session: SASession, finding: DataQualityFinding, params: dict) -> tuple[bool, str]:
    """Apply a registry-level fix by reusing the existing mutation helpers.

    Returns (ok, detail). No commit — the caller commits once both the mutation
    and the status update are staged.
    """
    if finding.action == "merge":
        target = str(params.get("target", "") or "").strip()
        source_name = finding.biomarker
        # The source (the redundant variant) often has data points but no
        # registry row — only the canonical target does. A missing source row
        # is fine: its data points are still migrated and the name becomes an
        # alias of the survivor.
        if not source_name or source_name.lower() == target.lower():
            return False, "merge source and target are identical"
        target_row = (
            session.query(BiomarkerRegistry)
            .filter(func.lower(BiomarkerRegistry.canonical_name) == target.lower())
            .first()
        )
        if target_row is None:
            # The LLM proposes merge targets from the patient's measurement
            # names, so a target that isn't a registry row yet may still be a
            # real biomarker (e.g. two Borrelia spellings, neither seeded).
            # Promote it to a registry row so the merge can run and the
            # survivor becomes a proper canonical entry. A target with no
            # measurements at all is a hallucination — skip it.
            has_measurements = (
                session.query(DataPoint.id)
                .filter(func.lower(DataPoint.canonical_name) == target.lower())
                .first()
                is not None
            )
            if not has_measurements:
                return False, f"merge target '{target}' not found in registry"
            target_row = BiomarkerRegistry(
                canonical_name=target, units="[]", aliases="[]",
                source="llm_accepted", updated_at=datetime.utcnow(),
            )
            session.add(target_row)
        ok, detail = _merge_registry_rows(session, source_name, target_row)
        if ok:
            target_row.source = "llm_accepted"
            target_row.updated_at = datetime.utcnow()
        return ok, detail

    field_map = {
        "registry_ref_range": ("ref_range", str(params.get("proposed", "") or "")),
        "registry_unit": ("unit", str(params.get("proposed", "") or "")),
        "registry_name": ("name", str(params.get("proposed", "") or "")),
    }
    mapped = field_map.get(finding.action)
    if mapped is None:
        return False, f"unknown registry action '{finding.action}'"
    field, proposed = mapped
    if not proposed:
        return False, "no proposed value"

    row = (
        session.query(BiomarkerRegistry)
        .filter(func.lower(BiomarkerRegistry.canonical_name) == finding.biomarker.lower())
        .first()
    )
    if row is None:
        # Promote the name to a registry row so unit/ref_range can be set,
        # mirroring update_biomarker_registry(). 'name' needs an existing row.
        if field in ("unit", "ref_range"):
            row = BiomarkerRegistry(
                canonical_name=finding.biomarker, units="[]", aliases="[]",
                source="manual", updated_at=datetime.utcnow(),
            )
            session.add(row)
        else:
            return False, f"registry row not found for '{finding.biomarker}'"

    ok, detail = _mutate_registry_row(session, row, field, proposed)
    if ok:
        row.source = "llm_accepted"
        row.updated_at = datetime.utcnow()
    return ok, detail


def apply_data_quality_finding(session: SASession, finding_id: int) -> dict:
    """Apply one accepted data-quality finding to the DB.

    Validates before mutating and commits only on success. Returns
    {'status': 'applied'|'skipped', 'detail': str}. A skipped finding stays
    pending so the user can retry or dismiss it.
    """
    f = session.get(DataQualityFinding, finding_id)
    if f is None or f.status != "pending":
        return {"status": "skipped", "detail": "finding not found or already resolved"}

    try:
        params = json.loads(f.params_json) if f.params_json else {}
    except (json.JSONDecodeError, TypeError):
        params = {}
    if not isinstance(params, dict):
        params = {}

    if f.target_type == "data_point":
        ok, detail = _apply_data_point_fix(session, f, params)
    elif f.target_type == "registry":
        ok, detail = _apply_registry_fix(session, f, params)
    else:
        return {"status": "skipped", "detail": f"unknown target_type '{f.target_type}'"}

    if not ok:
        return {"status": "skipped", "detail": detail}

    session.commit()
    _registry_cache_invalidate()
    f.status = "applied"
    session.commit()
    return {"status": "applied", "detail": detail}
