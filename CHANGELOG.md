# Changelog

All notable changes to Blutbild-Buddy are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — since v0.5 (2026-11-16)

### Added

#### Unit-qualified marker variants (`HbA1c (%)` vs `HbA1c (mmol/mol)`)
*Some marker names embed their unit in a trailing parenthesis. Previously the LLM name-mapping pass could resolve a bare `HbA1c` to the wrong variant, storing a `%` value under the name `HbA1c (mmol/mol)` — the name contradicted the data and the marker pickers showed a phantom duplicate.*

- New registry-driven helpers in `config.py`: `marker_base()` (strips the trailing parenthetical), `embedded_unit()` (returns the unit only when the parenthetical really is a unit — not for abbreviations like `ALT (GPT)` or `Vitamin D (25-OH)`), `name_has_unit()` (equivalence-aware), and `resolve_unit_qualified_variant()` (snaps a row to the sibling variant whose embedded unit matches the row's stored unit — rename only, never converts value/unit/range).
- The variant guard runs in **both** extraction phases: phase 1 (per-PDF alias resolution) and phase 2 (the LLM name-normalisation pass, which is the load-bearing half).
- **Two-level Trends picker**: the marker selectbox lists each base name once; a second `Einheit / Variante` selectbox appears only when a marker actually has multiple stored variants. Variants are labelled honestly (`— ohne Einheit`, `— gespeicherte Einheit: <unit>` for legacy drifted rows).
- New DE/EN translations: `select_unit_variant`, `label_unit_missing`, `label_unit_stored`.

#### Unified display labels
- New `ui/helpers.marker_label(cname, unit)` is the single source of truth for biomarker display labels: it appends `(<unit>)` only when the name does not already embed it, fixing the doubled `HbA1c (%) (%)` label. Overview favorites, the deviation chart, the All-Values matrix and the XLSX export all route through it.

#### Data-quality findings: manual correction in the Mapping tab
- **Every data-point finding now gets an inline correction form** (`mapping._render_finding_correction`): one prefilled value/unit/reference row per affected measurement, shown under the source-page image, saved through the same patient-scoped `update_data_point` primitive as the manual editor. A hand-fix supersedes (dismisses) the machine suggestion instead of deleting it.
- The machine "accept fix" button is suppressed for `acknowledge` findings (low-confidence reads, the systematic value==reference-bound misread, a dropped result column) — for those no automated fix exists, so the form is the action path. Machine-applicable findings show **both** the fix button and the correction form, so a proposal that fails to apply no longer dead-ends.
- New cached read helper `ui/queries.finding_data_points(finding_id)` shows the suspect stored number beside the scan it must be checked against.
- New DE/EN strings: `caption_finding_correct`, `caption_finding_original`, `btn_save_finding_correction`, `msg_finding_corrections_applied`, `msg_finding_correction_nothing`.

#### Stale unit proposals after a rename
- A pending `set_unit` proposal can go stale when the marker is renamed in between (e.g. `Testosteron gesamt` → `Testosteron frei`, where the stored unit becomes correct and the proposed one foreign). `_apply_data_point_fix` now detects this via a strict `_unit_matches_marker_units()` check and reports an honest explanation ("the marker was likely renamed after this finding was created — use the manual correction instead") instead of the misleading "unsupported unit conversion". Stored value/unit are left untouched.

#### Low-confidence findings show the actual reading
- The description of a low-confidence finding now quotes the exact reading from the scan (preferring the verbatim `value_raw`, e.g. `13,18`) plus its reference range — previously it named only the marker and a confidence percentage, making the source-page image useless for comparison.

### Fixed

- **Tab selection no longer resets when a background job starts or finishes.** The running-jobs strip used to render a variable number of blocks above the tab bar; appearing/vanishing shifted the tab bar's delta path, React remounted it, and the app jumped back to the first tab (Übersicht). The strip now renders into an always-created container slot, and both tab bars carry stable keys (`_main_tab_bar`, `_mgmt_tab_bar`) so the frontend persists the active tab. Pinned by a new AppTest regression test.

### Tests

- New regression coverage: tab-bar position stability on job start, acknowledge-finding correction form (renders form, not fix button; applies and dismisses; rejects a range typed into a value cell), machine findings showing both action paths, stale post-rename `set_unit` reporting, unit-qualified variant extraction in both phases, and the two-level Trends picker (multi-variant grouping, no sub-picker for single-variant markers).

---

## [v0.5] — 2026-11-06

See `git tag -n99 v0.5`: global running-jobs strip, consolidated Einstellungen tab, per-tab job buttons, gated polling fragments.

## [v0.4.3] — 2026-10-26

Deterministic typo detection for biomarker variants, surfaced as data-quality merge findings.
