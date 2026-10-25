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
- **Biomarker merge groups** are keyed by `(case-insensitive base name, normalized unit)`. `normalize_unit()` lowercases AND collapses internal whitespace, so `ml/ min` == `ml/min` (a stray OCR space no longer splits one marker into two). Genuinely different units (e.g. `mg/dl` vs `mmol/l`) still prevent grouping. `clean_unit()` (storage-side) strips all whitespace runs; the extractor cleans units at ingestion, and `_migrate_clean_unit_whitespace()` fixes legacy rows once. Every other unit-writing path also cleans first (`clean_unit`): `set_unit`/`set_ref_range` data-quality fixes, `apply_merge_plan` target units, `_parse_ref_range`, registry `unit` edits (`_mutate_registry_row`). `config._norm_unit()` also collapses whitespace, so the conversion/equivalence tables need no per-spacing entries (`_UNIT_EQUIV` has no `ml/ min` band-aid).

## Architecture (one file per concern)

| File | Role |
|------|------|
| `app.py` | Streamlit UI — sidebar, data tabs (overview/detail/trends/risks/all_values/chat) + management tabs (reports/patients/llm_settings/llm_jobs/mapping; nested under a "Verwaltung" tab when a patient is selected), session state management |
| `llm_jobs.py` | Background LLM job queue: worker threads for all long-running ops, kind locks, config snapshots, progress persistence |
| `extractor.py` | PDF pipeline: hash check → pypdfium2 render → LLM extract → normalize → DB insert |
| `analyzer.py` | Risk flag engine: out-of-range, trending, volatility, missing markers, cross-biomarker correlation rules |
| `llm_client.py` | OpenAI-compatible client calls (extract + normalize + summarize + chat), embedded system prompts |
| `pdf_processor.py` | pypdfium2 → JPEG images, base64 encoding, SHA-256 hashing |
| `database.py` | SQLAlchemy models: `Patient → Report → DataPoint → RiskFlag`, `LlmJob`. Also: merge logic, migrations, patient name normalization, `commit_version()` (in-process write counter) |
| `ui/queries.py` | Cached read-only query layer for the tabs: `@st.cache_data` helpers returning ORM-free namedtuples, keyed on `cache_token()` |
| `schemas.py` | Pydantic models for LLM JSON responses |
| `config.py` | Paths, lazy env-var getters, ~94 biomarker registry (canonical names, aliases, default ref ranges), risk thresholds & cross-biomarker rules |
| `translations.py` | DE/EN translation dictionaries and helpers |
| `test_blutbild_buddy.py` | Single-file unittest suite — mocks LLM calls for extractor/analyzer tests |

## Key patterns

- **Cached UI reads** (`ui/queries.py`): every tab reads through cached helpers instead of inline ORM queries. Two rules: (1) helpers return **ORM-free namedtuples/scalars** — `st.cache_data` deep-copies its return value, so a detached ORM object raises on the second access; (2) every helper takes `cache_token()` as its first argument, which is `(database.commit_version(), PRAGMA data_version)`. `commit_version()` is an in-process counter bumped by an `after_commit` event listener; `data_version` alone misses commits made through the same connection, `commit_version` alone misses other processes — both are needed. Time-range sliders filter the cached rows **in Python** so dragging a slider never busts the cache. Column labels (`t("col_*")`) must be hoisted out of row loops: they were translated per row (31k `t()` calls per rerun at 7.2k data points, each one a session-state read).
- **Biomarker normalization**: 3-tier registry in `config.py` — canonical name → aliases + units + default ref range. Resolution order: exact match → alias match → fallback to original name.
- **Unit conversion**: Single name-aware converter in `config._convert_ref_range(canonical_name, low, high, from_unit, to_unit)` — factors are keyed by (biomarker, unit pair) so shared pairs like mg/dl↔mmol/l resolve per biomarker (glucose ÷18 vs cholesterol ÷38.67). Returns `(None, None)` when unsupported; callers must treat that as "no range". Used by both `get_default_ref_range()` and `database.backfill_reference_ranges()`.
- **Cross-biomarker patterns**: Config-driven rules in `RISK_CONFIG["cross_biomarker_rules"]` detect correlated abnormalities (e.g., CK + Troponin → cardiac pattern). Severity can be upgraded or downgraded based on the pattern.
- **Deduplication**: PDFs skipped if SHA-256 hash already exists in `Report.file_hash`.
- **Auto-merge**: `database.auto_merge_biomarkers()` uses LLM to suggest canonical targets for variant groups; `apply_merge_plan()` executes (only merges when units are compatible).
- **Chat with your data** (`ui/tabs/chat.py`): free-form LLM Q&A per patient. A date-range slider (shared `_time_range_slider` helper, chat-scoped value key `chat_time_range_{patient.id}` so it's independent of the Trends/Overview slider; default = patient's full range) selects the window. `collect_chat_measurements()` returns ALL non-disabled, non-null measurements in that range (undated reports excluded) — deliberately uncapped, unlike the summary's `_collect_historical_data`. `format_chat_data_context()` renders profile + per-(biomarker,unit) chronological series into the system prompt. **Transcripts are DB-persisted** (`ChatConversation`/`ChatMessage`, cascade with patient delete, re-pointed on patient merge) — NOT session state: multiple conversations per patient (selectbox picker + "new conversation" + delete dialog), F5/restart-proof. The answer runs as an LlmJob kind `chat_answer` (question + context captured into `params_json` at submit; deliberately NOT in `DATA_AFFECTING_KINDS` — an app rerun would yank the user's tab). The tab's own polling fragment (`_poll_chat_answer`, wrapped in `@st.fragment(run_every=2)` **only while an answer is in flight** — see the polling rule under Background LLM jobs) shows progress and calls `adopt_chat_answer()` (idempotent via `ChatMessage.job_id`, unique per conversation) when the job finishes → survives tab switches AND F5. Failed/aborted answers leave the question as an unanswered user row (persistent warning; `build_llm_messages()` drops unanswered turns from the LLM payload except the trailing one). `chat_with_data()` still raises — the worker records it as job `error`. `clear_finished_llm_jobs()` adopts un-adopted done `chat_answer` results before deleting rows. Two selection keys: `chat_conv_{id}` (plain state, writable from body/callbacks) + `chat_conv_widget_{id}` (selectbox widget key) — Streamlit forbids writing a widget key from the body after the widget rendered.
- **Background LLM jobs** (`llm_jobs.py`): All long-running operations (process_pdfs, refresh_risks, regen_summary, bulk_descriptions, bulk_interpretations, backfill) run in daemon worker threads, NOT inline in the UI. Buttons call `submit_job(kind, label_key, params=...)` and return immediately; progress is persisted to the `llm_jobs` table and polled by the "LLM-Jobs" tab (`ui/tabs/llm_jobs.py`). Rules:
  - **Polling is gated**: both `run_every=2` fragments (LLM-Jobs list, chat answer) are *defined* only while a job is actually pending/running — `get_active_llm_jobs()` for the jobs tab, the tab's own `pending` flag for chat. `run_every` is baked into the decorator at definition time, so this needs two branches (polling fragment vs. plain render of the same body). An unconditional `run_every` re-executed forever AND re-registered a new browser `setInterval` on every full rerun (`cleanupAutoReruns` only runs on page change / new session), so an idle screen got steadily busier. When the last job finishes, the still-live poll triggers an app rerun; that rerun defines no polling fragment, and a leftover browser interval resolves to an unknown fragment id and is dropped server-side.
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
- **Behavioural UI tests** (`TestUiBehaviourAppTest`): run the real `app.py` with `streamlit.testing.v1.AppTest` against a temp DB, so a widget-level crash (stale multiselect default, unkeyed selectbox) shows up in `at.exception` instead of silently breaking the browser view. They set `config.DB_PATH`/`config.REPORTS_DIR` in `setUp` and restore them in `tearDown` — `app.py` runs in-process, so it reads the module-level paths. `AppTest` models `download_button` as an untyped element with no key, so match it by label.

## Docker specifics

- **Health check**: HTTP probe on `/_stcore/health` every 30s.
- **Volumes**: Named volumes `blutbild-data` (DB) and `blutbild-reports` (PDFs).
- **Resource limits**: 4G max / 512M reserved. Security: `no-new-privileges`, tmpfs for `/tmp` and `__pycache__`.
- `.dockerignore` excludes `AGENTS.md`, `test_blutbild_buddy.py`, README, `.env`, data dirs.

## What's NOT present

No lint, typecheck, formatter, CI, pre-commit, or release workflow. Python 3.14 only. Single venv at `.venv/`.
