# Blutbild-Buddy — Agent Instructions

## Quick commands

```bash
streamlit run app.py                        # Dev server
python test_blutbild_buddy.py               # All tests (unittest)
docker compose up -d                        # Docker deployment
```

## Gotchas (easy to get wrong)

- **Streamlit + Starlette pin**: `streamlit==1.58.0` must stay paired with `starlette==1.3.1` in `requirements.txt`. Newer Starlette requires a `thread_minimum_size` param that older Streamlit doesn't pass.
- **DB lives at `data/blutbild_buddy.db`** (relative to project root), NOT `blutbild_buddy.db` in the repo root. `.gitignore` covers both paths but they're different files.
- **Docker LLM default**: `LLM_BASE_URL` defaults to `http://172.17.0.1:8888/v1` (Docker bridge network), not `localhost`. Override via `.env`.
- **Tests patch by value**: Test setup does `config.DB_PATH = temp_path`. This works because other modules import specific names from config (`from config import DB_PATH`) rather than the module reference. Don't change this pattern.
- **LLM env vars are read lazily** in `config.py` (via getter functions). Changes to `os.environ` after import *are* picked up. This is intentional for the in-app Settings dialog.
- **Risk flags are idempotent**: `refresh_risk_flags()` does DELETE + INSERT from scratch each call.
- **Patient names normalize to "LastName, FirstName"** via `database.normalize_patient_name()`. Handles formats like `Doe J.` → `J., Doe`, underscores, commas.
- **Biomarker merge groups** are keyed by `(case-insensitive base name, case-insensitive unit)`. Different units (e.g. `mg/dl` vs `mmol/l`) prevent grouping even for the same biomarker.

## Architecture (one file per concern)

| File | Role |
|------|------|
| `app.py` | Streamlit UI — sidebar, data tabs (overview/detail/trends/risks/all_values) + management tabs (reports/patients/llm_settings/llm_jobs/mapping; nested under a "Verwaltung" tab when a patient is selected), session state management |
| `llm_jobs.py` | Background LLM job queue: worker threads for all long-running ops, kind locks, config snapshots, progress persistence |
| `extractor.py` | PDF pipeline: hash check → pypdfium2 render → LLM extract → normalize → DB insert |
| `analyzer.py` | Risk flag engine: out-of-range, trending, volatility, missing markers, cross-biomarker correlation rules |
| `llm_client.py` | OpenAI-compatible client calls (extract + normalize + summarize), embedded system prompts |
| `pdf_processor.py` | pypdfium2 → JPEG images, base64 encoding, SHA-256 hashing |
| `database.py` | SQLAlchemy models: `Patient → Report → DataPoint → RiskFlag`, `LlmJob`. Also: merge logic, migrations, patient name normalization |
| `schemas.py` | Pydantic models for LLM JSON responses |
| `config.py` | Paths, lazy env-var getters, ~94 biomarker registry (canonical names, aliases, default ref ranges), risk thresholds & cross-biomarker rules |
| `translations.py` | DE/EN translation dictionaries and helpers |
| `test_blutbild_buddy.py` | Single-file unittest suite — mocks LLM calls for extractor/analyzer tests |

## Key patterns

- **Biomarker normalization**: 3-tier registry in `config.py` — canonical name → aliases + units + default ref range. Resolution order: exact match → alias match → fallback to original name.
- **Unit conversion**: Single name-aware converter in `config._convert_ref_range(canonical_name, low, high, from_unit, to_unit)` — factors are keyed by (biomarker, unit pair) so shared pairs like mg/dl↔mmol/l resolve per biomarker (glucose ÷18 vs cholesterol ÷38.67). Returns `(None, None)` when unsupported; callers must treat that as "no range". Used by both `get_default_ref_range()` and `database.backfill_reference_ranges()`.
- **Cross-biomarker patterns**: Config-driven rules in `RISK_CONFIG["cross_biomarker_rules"]` detect correlated abnormalities (e.g., CK + Troponin → cardiac pattern). Severity can be upgraded or downgraded based on the pattern.
- **Deduplication**: PDFs skipped if SHA-256 hash already exists in `Report.file_hash`.
- **Auto-merge**: `database.auto_merge_biomarkers()` uses LLM to suggest canonical targets for variant groups; `apply_merge_plan()` executes (only merges when units are compatible).
- **Background LLM jobs** (`llm_jobs.py`): All long-running operations (process_pdfs, refresh_risks, regen_summary, bulk_descriptions, bulk_interpretations, backfill) run in daemon worker threads, NOT inline in the UI. Buttons call `submit_job(kind, label_key, params=...)` and return immediately; progress is persisted to the `llm_jobs` table and polled by the "LLM-Jobs" tab (`ui/tabs/llm_jobs.py`, `@st.fragment(run_every=2)`). Rules:
  - **Worker threads must NEVER touch `st.session_state`** — settings are captured as a snapshot at submit time into `LlmJob.config_json` and applied via `llm_client.set_job_config()` (thread-local); translations use `t(key, lang=...)` with the language from the snapshot.
  - **Kind locks**: one job per kind at a time (`_active_kinds`), so duplicate buttons across tabs (e.g. backfill in Reports + Mapping) can't double-run; `submit_job` returns `None` → UI shows `msg_job_already_running`.
  - **Stale marking**: `app.py` calls `mark_stale_llm_jobs_interrupted(session, exclude_ids=active_job_ids())` at module level — Streamlit re-executes it on EVERY rerun, so jobs with a live worker in this process are excluded (`llm_jobs._active_job_ids`). Best-effort (try/except): a DB lock must never crash the app view.
  - **Write locks & LLM calls**: `analyzer.refresh_risk_flags()` does its DELETE+INSERT AFTER the LLM call — never hold SQLite's single write lock across a slow HTTP request (UI reruns would hit "database is locked" after busy_timeout). Keep it that way in new work functions.
  - **Rerun trigger**: when a job in `DATA_AFFECTING_KINDS` transitions to done/error, the LLM-Jobs tab sets `st.session_state._rerun_requested = True` so main() reloads data.
  - **Job abort** (user action from the LLM-Jobs tab): per-job, NOT a global flag. `llm_jobs.abort_job(job_id)` marks the DB row (`LlmJob.aborted`) and calls `llm_client.request_job_abort(job_id)`, which adds the id to a module-level `_aborted_job_ids` set. Workers check `is_job_aborted()` (thread-local current job id, set via `set_current_job_id()`) between steps — `extractor.process_new_pdfs` breaks its PDF loop, `_run_bulk` breaks its item loop, `analyzer.refresh_risk_flags` skips the write phase. In-flight LLM calls fail fast: every call site passes `timeout=_abort_timeout() or _get_timeout()` (1s while aborted → `APITimeoutError`). `_run_job` then records status `"interrupted"` (not done/error) with `job_aborted_note`; partial results are kept. Job ids are unique, so finished entries never match a new job — no clearing needed.
  - **Finished-job cleanup**: `database.clear_finished_llm_jobs(session)` deletes rows in done/error/interrupted state (returns count); the LLM-Jobs tab's "Clear finished" button calls it. Pending/running jobs are never touched.
  - **Testing**: `submit_job(..., runner=llm_jobs._run_job)` runs the worker synchronously (no thread) — see `TestLlmJobs` and `TestLlmJobAbort`.

## Testing

```bash
python test_blutbild_buddy.py -v    # verbose
```

- Single file, `unittest`. Tests use temp dirs and mock LLM calls.
- No fixture files or external services needed beyond Python packages.
- Test PDFs live in `reports_input/` (gitignored real data — a sample PDF is used for hash/page/image tests).

## Docker specifics

- **Health check**: HTTP probe on `/_stcore/health` every 30s.
- **Volumes**: Named volumes `blutbild-data` (DB) and `blutbild-reports` (PDFs).
- **Resource limits**: 4G max / 512M reserved. Security: `no-new-privileges`, tmpfs for `/tmp` and `__pycache__`.
- `.dockerignore` excludes `AGENTS.md`, `test_blutbild_buddy.py`, README, `.env`, data dirs.

## What's NOT present

No lint, typecheck, formatter, CI, pre-commit, or release workflow. Python 3.14 only. Single venv at `.venv/`.
